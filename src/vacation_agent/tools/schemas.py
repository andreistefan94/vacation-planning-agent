"""Input/output schemas of the tools.

Inputs are what the LLM (tool selection) is allowed to pass; outputs are the normalized,
compact shapes the agents and the deterministic code consume (never the raw API payload).
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

# ---------- inputs ----------


class ResolveLocationInput(BaseModel):
    keyword: str = Field(
        description="City name in ENGLISH (e.g. 'Rome', 'Chisinau', 'Athens') or a 3-letter IATA code."
    )


class SearchFlightsInput(BaseModel):
    origin_iata: str = Field(description="3-letter IATA code of the departure city, e.g. 'KIV'.")
    destination_iata: str = Field(description="3-letter IATA code of the arrival city, e.g. 'ROM'.")
    departure_date: date = Field(description="Departure date, YYYY-MM-DD.")
    adults: int = Field(default=1, ge=1, le=9)
    children: int = Field(default=0, ge=0, le=9)


class ListHotelsInput(BaseModel):
    city_code: str = Field(description="3-letter IATA city code, e.g. 'PAR'.")


class HotelOffersInput(BaseModel):
    hotel_ids: list[str] = Field(description="Hotel ids returned by list_hotels_in_city (max 8).")
    check_in_date: date
    check_out_date: date
    adults: int = Field(default=1, ge=1, le=9)
    children: int = Field(default=0, ge=0, le=9)


class WeatherInput(BaseModel):
    latitude: float
    longitude: float
    start_date: date | None = Field(default=None, description="First forecast day, YYYY-MM-DD.")


class CalendarCheckInput(BaseModel):
    start_date: date
    end_date: date


class CalendarEventInput(BaseModel):
    summary: str
    start_date: date
    end_date: date = Field(description="Last day of the trip (inclusive).")
    description: str = ""


class BudgetInput(BaseModel):
    flight_total: float = Field(ge=0, description="Total price of all flights for all travellers.")
    hotel_total: float = Field(ge=0, description="Total accommodation price for the whole stay.")
    budget: float | None = Field(default=None, ge=0)
    extras: float = Field(default=0, ge=0)
    currency: str = "EUR"


# ---------- outputs ----------


class Location(BaseModel):
    iata_code: str
    name: str
    city_name: str
    city_code: str
    country_code: str
    latitude: float
    longitude: float
    sub_type: str


class FlightOffer(BaseModel):
    offer_id: str
    carrier: str
    flight_number: str
    origin: str
    destination: str
    departure_at: str
    arrival_at: str
    duration: str
    stops: int
    direct: bool
    price_per_person: float
    travellers: int
    total_price: float
    currency: str
    seats_available: int | None = None


class HotelSummary(BaseModel):
    hotel_id: str
    name: str
    chain_code: str = ""


class HotelOffer(BaseModel):
    hotel_id: str
    name: str
    available: bool
    room_type: str
    price_per_night: float
    nights: int
    rooms: int
    total_price: float
    currency: str
    offer_id: str = ""


class DailyForecast(BaseModel):
    date: str
    t_max: float
    t_min: float
    precipitation_mm: float


class BudgetResult(BaseModel):
    flight_total: float
    hotel_total: float
    extras: float
    total: float
    budget: float | None
    remaining: float | None
    within_budget: bool | None
    currency: str
