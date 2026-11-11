"""HTTP client for the mock travel APIs.

Responsibilities that the LLM must never handle itself:
- OAuth2 client-credentials token acquisition and caching;
- retries with exponential backoff on transient failures (5xx, timeouts, connection errors);
- fallback to a secondary mirror when the primary one is down.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from vacation_agent.config import Settings, get_settings

log = logging.getLogger(__name__)


class ToolError(Exception):
    """A tool call failed in a way the agent should know about (not a crash)."""

    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, ToolError):
        return exc.retryable
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


class MockApiClient:
    def __init__(self, settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings or get_settings()
        self._transport = transport
        self._token: str | None = None
        self._token_expiry: float = 0.0
        self._token_lock = asyncio.Lock()

    @property
    def base_urls(self) -> list[str]:
        urls = [self.settings.mock_api_base_url, self.settings.mock_api_fallback_url]
        return [u.rstrip("/") for u in urls if u]

    def _client(self, base_url: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=base_url, timeout=self.settings.http_timeout_s, transport=self._transport)

    async def _send_once(self, base_url: str, method: str, path: str, **kwargs: Any) -> Any:
        async with self._client(base_url) as client:
            response = await client.request(method, path, **kwargs)
        if response.status_code >= 500 or response.status_code == 429:
            raise ToolError(
                f"{method} {path} returned {response.status_code}", retryable=True, status=response.status_code
            )
        if response.status_code >= 400:
            detail = response.text[:300]
            raise ToolError(
                f"{method} {path} returned {response.status_code}: {detail}",
                retryable=False,
                status=response.status_code,
            )
        return response.json()

    async def request(self, method: str, path: str, *, auth: bool = False, **kwargs: Any) -> Any:
        """Send a request with retry on the primary mirror, then fall back to the next mirror."""
        last_exc: BaseException | None = None
        for base_url in self.base_urls:
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(self.settings.http_max_attempts),
                    wait=wait_exponential(multiplier=0.3, max=3),
                    retry=retry_if_exception(_is_transient),
                    reraise=True,
                ):
                    with attempt:
                        headers = dict(kwargs.pop("headers", {}) or {})
                        if auth:
                            headers["Authorization"] = f"Bearer {await self.token(base_url)}"
                        return await self._send_once(base_url, method, path, headers=headers, **kwargs)
            except ToolError as exc:
                if not exc.retryable:
                    raise
                last_exc = exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
            log.warning("Mirror %s failed for %s %s: %s; trying fallback", base_url, method, path, last_exc)
        raise ToolError(f"All mirrors failed for {method} {path}: {last_exc}", retryable=True)

    async def token(self, base_url: str | None = None) -> str:
        async with self._token_lock:
            if self._token and time.monotonic() < self._token_expiry - 30:
                return self._token
            base = base_url or self.base_urls[0]
            data = await self._send_once(
                base,
                "POST",
                "/security/oauth2/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.settings.mock_client_id,
                    "client_secret": self.settings.mock_client_secret,
                },
            )
            self._token = data["access_token"]
            self._token_expiry = time.monotonic() + float(data.get("expires_in", 1799))
            return self._token


_client: MockApiClient | None = None


def get_client() -> MockApiClient:
    global _client
    if _client is None:
        _client = MockApiClient()
    return _client


def set_client(client: MockApiClient | None) -> None:
    """Swap the client (used by tests)."""
    global _client
    _client = client
