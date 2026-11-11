from datetime import date

from vacation_agent.graph import planning as P
from vacation_agent.graph.requirements import merge_request, missing_slots, pick_free_window, search_window


def _f(num, price, direct=True, duration="PT2H"):
    return {
        "flight_number": num,
        "origin": "KIV",
        "destination": "ROM",
        "departure_at": "2026-12-18T09:00:00",
        "arrival_at": "2026-12-18T11:00:00",
        "duration": duration,
        "stops": 0 if direct else 1,
        "direct": direct,
        "price_per_person": price / 2,
        "travellers": 2,
        "total_price": price,
        "currency": "EUR",
    }


def _h(hid, price, room="STANDARD", rank=0):
    return {
        "hotel_id": hid,
        "name": hid,
        "available": True,
        "room_type": room,
        "price_per_night": price / 7,
        "nights": 7,
        "rooms": 1,
        "total_price": price,
        "currency": "EUR",
        "room_rank": rank,
    }


def test_merge_request_overlays_and_resets_destination_fields():
    req = merge_request({}, {"destination": "Italia", "destination_kind": "country", "candidate_cities_en": ["Rome"]})
    req = merge_request(req, {"destination": "Roma", "destination_city_en": "Rome"})
    assert req["destination_city_en"] == "Rome" and req["candidate_cities_en"] == []
    req = merge_request(req, {"start_date": "2026-12-18", "end_date": "2026-12-25"})
    assert req["duration_days"] == 7


def test_missing_slots_and_month_window():
    today = date(2026, 11, 11)
    req = {"destination": "Paris", "adults": 2, "month": 2, "duration_days": 5}
    assert missing_slots(req, today) == ["origin", "budget"]
    start, end = search_window(req, today)
    assert start == date(2027, 2, 1) and end == date(2027, 2, 28)


def test_pick_free_window_skips_busy_days():
    busy = {"2026-12-01", "2026-12-02"}
    start, end = pick_free_window((date(2026, 12, 1), date(2026, 12, 31)), 3, busy)
    assert start == date(2026, 12, 3) and end == date(2026, 12, 6)


def test_build_variants_respects_direct_preference_and_rejections():
    out = [_f("A1", 300), _f("A2", 200, direct=False)]
    inn = [_f("B1", 250)]
    hotels = [_h("H1", 500), _h("H2", 700, "SUPERIOR", 1), _h("H3", 1400, "SUITE", 3)]
    variants, unmet = P.build_variants(out, inn, hotels, prefer_direct=True, rejected=[])
    assert not unmet and all(v["outbound"]["direct"] for v in variants)
    assert [v["label"] for v in variants] == ["Economic", "Echilibrat", "Confort"]
    again, _ = P.build_variants(out, inn, hotels, prefer_direct=True, rejected=[variants[0]["signature"]])
    assert variants[0]["signature"] not in {v["signature"] for v in again}


def test_build_variants_flags_unmet_direct_preference():
    _, unmet = P.build_variants(
        [_f("A2", 200, direct=False)], [_f("B1", 250)], [_h("H1", 500)], prefer_direct=True, rejected=[]
    )
    assert unmet


def test_verifier_rejects_invented_amounts_and_booking_claims():
    v = {
        "outbound": _f("A1", 300),
        "inbound": _f("B1", 250),
        "hotel": _h("H1", 500),
        "total": 1050,
        "flight_total": 550,
        "hotel_total": 500,
        "remaining": 450,
    }
    start, end = date(2026, 12, 18), date(2026, 12, 25)
    assert P.verify_text("Total 1050.00 EUR, hotel 500 EUR.", v, 1500, start, end) == []
    issues = P.verify_text("O cină costă 80 EUR. Am rezervat hotelul. Pe 2027-01-05 plecăm.", v, 1500, start, end)
    assert len(issues) == 3
