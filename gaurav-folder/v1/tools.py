"""
v1 — Bloat and Break
Four mock travel-agent tools with 2 000-3 000+ token JSON payloads.
Each tool is deterministic/rule-based; no real API calls needed.

Tools
-----
web_search(query)          → flights, general travel info
places_search(city, cat)   → hotels / restaurants / attractions
weather_fetch(city, month) → detailed weather + packing guide
budget_tracker(action, …)  → running spend ledger

All functions return a Python dict that is JSON-serialisable.
The agent serialises the return value before injecting it as a tool message.
"""
from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json(d: dict) -> str:
    """Serialise a dict to a compact JSON string (what the agent sends to LLM)."""
    return json.dumps(d, ensure_ascii=False, indent=2)


# ===========================================================================
# TOOL 1 — web_search
# ===========================================================================
# Fixture covers routes from/to: Delhi (DEL), Paris (CDG), Bali (DPS),
# Tokyo (NRT/HND), Amsterdam (AMS), Berlin (BER), Dehradun (DED),
# Himachal Pradesh / Bhuntar (KUU), Shimla (SLV — helicopter link).
# Each result block is padded with realistic airline data, fare classes,
# baggage rules, and booking metadata to reach ~2 500 tokens.
# ===========================================================================

_FLIGHT_DB: dict[str, list[dict]] = {
    # ── Delhi → Paris ────────────────────────────────────────────────────
    "delhi to paris": [
        {
            "flight_id": "AI-101-CDG",
            "airline": "Air India",
            "aircraft": "Boeing 787-8 Dreamliner",
            "origin": {"code": "DEL", "name": "Indira Gandhi International Airport", "terminal": "3"},
            "destination": {"code": "CDG", "name": "Paris Charles de Gaulle Airport", "terminal": "2E"},
            "departure": "2026-05-10T22:15:00+05:30",
            "arrival": "2026-05-11T05:45:00+01:00",
            "duration_hours": 9.5,
            "stops": 0,
            "fare_classes": {
                "economy": {"price_usd": 620, "seats_left": 14, "baggage_kg": 23, "carry_on_kg": 8, "refundable": False, "change_fee_usd": 75},
                "premium_economy": {"price_usd": 1050, "seats_left": 6, "baggage_kg": 35, "carry_on_kg": 10, "refundable": True, "change_fee_usd": 0},
                "business": {"price_usd": 2800, "seats_left": 3, "baggage_kg": 64, "carry_on_kg": 18, "refundable": True, "change_fee_usd": 0, "lounge_access": True, "lie_flat_seat": True},
            },
            "amenities": ["In-flight Wi-Fi ($9.99/flight)", "Vegetarian meals available", "USB-A & USB-C at every seat", "Entertainment: 500+ titles"],
            "booking_url": "https://example.com/book/AI101",
            "cancellation_policy": "Full refund if cancelled 48h before departure on flex fares; economy non-refundable.",
            "notes": "Direct overnight flight; arrives early morning Paris time — good for same-day hotel check-in.",
        },
        {
            "flight_id": "EK-501-CDG-VIA-DXB",
            "airline": "Emirates",
            "aircraft": "Airbus A380-800",
            "origin": {"code": "DEL", "terminal": "3"},
            "destination": {"code": "CDG", "terminal": "2C"},
            "departure": "2026-05-10T03:30:00+05:30",
            "arrival": "2026-05-10T13:20:00+01:00",
            "duration_hours": 12.83,
            "stops": 1,
            "layover": {"airport": "DXB", "city": "Dubai", "duration_min": 135, "terminal": "3"},
            "fare_classes": {
                "economy": {"price_usd": 540, "seats_left": 31, "baggage_kg": 30, "carry_on_kg": 7, "refundable": False, "change_fee_usd": 100},
                "business": {"price_usd": 3450, "seats_left": 8, "baggage_kg": 40, "carry_on_kg": 15, "refundable": True, "change_fee_usd": 0, "lounge_access": True, "lie_flat_seat": True, "chauffeur_drive": True},
            },
            "amenities": ["Wi-Fi included in Business", "Ice entertainment system — 5000+ channels", "Shower spa (Business/First A380)", "Bar on upper deck"],
            "booking_url": "https://example.com/book/EK501",
            "notes": "Best value Delhi–Paris with one stop; A380 Business is exceptional.",
        },
        {
            "flight_id": "LH-761-FRA-CDG",
            "airline": "Lufthansa",
            "aircraft": "Airbus A340-600 + Airbus A320",
            "origin": {"code": "DEL", "terminal": "3"},
            "destination": {"code": "CDG", "terminal": "2A"},
            "departure": "2026-05-10T07:00:00+05:30",
            "arrival": "2026-05-10T16:55:00+01:00",
            "duration_hours": 12.92,
            "stops": 1,
            "layover": {"airport": "FRA", "city": "Frankfurt", "duration_min": 90, "terminal": "1"},
            "fare_classes": {
                "economy": {"price_usd": 595, "seats_left": 22, "baggage_kg": 23, "carry_on_kg": 8, "refundable": False, "change_fee_usd": 50},
                "business": {"price_usd": 3100, "seats_left": 5, "baggage_kg": 64, "carry_on_kg": 16, "refundable": True, "change_fee_usd": 0, "lounge_access": True},
            },
            "amenities": ["Lufthansa Senator Lounge access (Business)", "Miles & More eligible", "Sustainable Aviation Fuel offset option"],
            "booking_url": "https://example.com/book/LH761",
        },
    ],

    # ── Delhi → Bali ─────────────────────────────────────────────────────
    "delhi to bali": [
        {
            "flight_id": "SQ-424-DPS-VIA-SIN",
            "airline": "Singapore Airlines",
            "aircraft": "Boeing 777-300ER",
            "origin": {"code": "DEL", "terminal": "3"},
            "destination": {"code": "DPS", "name": "Ngurah Rai International Airport", "terminal": "International"},
            "departure": "2026-06-01T01:15:00+05:30",
            "arrival": "2026-06-01T20:05:00+08:00",
            "duration_hours": 10.83,
            "stops": 1,
            "layover": {"airport": "SIN", "city": "Singapore", "duration_min": 120, "terminal": "3"},
            "fare_classes": {
                "economy": {"price_usd": 490, "seats_left": 28, "baggage_kg": 30, "carry_on_kg": 7, "refundable": False, "change_fee_usd": 90},
                "premium_economy": {"price_usd": 870, "seats_left": 10, "baggage_kg": 35, "carry_on_kg": 10, "refundable": True, "change_fee_usd": 30},
                "business": {"price_usd": 2650, "seats_left": 4, "baggage_kg": 40, "carry_on_kg": 18, "refundable": True, "change_fee_usd": 0, "lounge_access": True},
            },
            "amenities": ["Book the Cook meal service", "KrisWorld entertainment", "Wi-Fi available"],
            "booking_url": "https://example.com/book/SQ424",
            "notes": "Top-rated Economy product on Asia routes; Singapore layover easy to extend for a stopover.",
        },
        {
            "flight_id": "GA-206-DPS-VIA-CGK",
            "airline": "Garuda Indonesia",
            "aircraft": "Airbus A330-300",
            "origin": {"code": "DEL", "terminal": "3"},
            "destination": {"code": "DPS"},
            "departure": "2026-06-01T09:45:00+05:30",
            "arrival": "2026-06-01T23:30:00+08:00",
            "duration_hours": 11.75,
            "stops": 1,
            "layover": {"airport": "CGK", "city": "Jakarta", "duration_min": 100, "terminal": "2F"},
            "fare_classes": {
                "economy": {"price_usd": 410, "seats_left": 40, "baggage_kg": 30, "carry_on_kg": 7, "refundable": False, "change_fee_usd": 60},
                "business": {"price_usd": 1980, "seats_left": 6, "baggage_kg": 40, "carry_on_kg": 15, "refundable": True, "change_fee_usd": 0},
            },
            "amenities": ["Batik-themed cabin design", "Traditional Indonesian cuisine in Business", "SkyTeam lounge access (Business)"],
            "booking_url": "https://example.com/book/GA206",
        },
    ],

    # ── Delhi → Tokyo ─────────────────────────────────────────────────────
    "delhi to tokyo": [
        {
            "flight_id": "NH-830-NRT",
            "airline": "ANA (All Nippon Airways)",
            "aircraft": "Boeing 787-9 Dreamliner",
            "origin": {"code": "DEL", "terminal": "3"},
            "destination": {"code": "NRT", "name": "Narita International Airport", "terminal": "1"},
            "departure": "2026-07-15T02:00:00+05:30",
            "arrival": "2026-07-15T15:10:00+09:00",
            "duration_hours": 9.17,
            "stops": 0,
            "fare_classes": {
                "economy": {"price_usd": 710, "seats_left": 9, "baggage_kg": 23, "carry_on_kg": 10, "refundable": False, "change_fee_usd": 80},
                "premium_economy": {"price_usd": 1350, "seats_left": 4, "baggage_kg": 46, "carry_on_kg": 10, "refundable": True, "change_fee_usd": 0},
                "business": {"price_usd": 4200, "seats_left": 2, "baggage_kg": 64, "carry_on_kg": 16, "refundable": True, "change_fee_usd": 0, "lounge_access": True, "lie_flat_seat": True},
            },
            "amenities": ["ANA Inspiration of Japan cuisine", "Full-flat Business seats", "Ippudo ramen available on select flights", "Noise-cancelling headphones provided"],
            "booking_url": "https://example.com/book/NH830",
            "notes": "ANA rated world's best airline; direct DEL–NRT is rare and very convenient.",
        },
        {
            "flight_id": "AI-307-NRT-VIA-HND",
            "airline": "Air India",
            "aircraft": "Boeing 787-8",
            "origin": {"code": "DEL", "terminal": "3"},
            "destination": {"code": "HND", "name": "Tokyo Haneda Airport", "terminal": "International"},
            "departure": "2026-07-15T23:55:00+05:30",
            "arrival": "2026-07-16T13:30:00+09:00",
            "duration_hours": 9.58,
            "stops": 0,
            "fare_classes": {
                "economy": {"price_usd": 650, "seats_left": 18, "baggage_kg": 23, "carry_on_kg": 8, "refundable": False, "change_fee_usd": 75},
                "business": {"price_usd": 3100, "seats_left": 5, "baggage_kg": 64, "carry_on_kg": 18, "refundable": True, "change_fee_usd": 0},
            },
            "amenities": ["Haneda is 30 min from Tokyo centre vs 60 min from Narita", "Wi-Fi available"],
            "booking_url": "https://example.com/book/AI307",
        },
    ],

    # ── Delhi → Amsterdam ─────────────────────────────────────────────────
    "delhi to amsterdam": [
        {
            "flight_id": "KL-871-AMS",
            "airline": "KLM Royal Dutch Airlines",
            "aircraft": "Boeing 787-10 Dreamliner",
            "origin": {"code": "DEL", "terminal": "3"},
            "destination": {"code": "AMS", "name": "Amsterdam Schiphol Airport", "terminal": "Main"},
            "departure": "2026-08-01T13:20:00+05:30",
            "arrival": "2026-08-01T19:05:00+02:00",
            "duration_hours": 9.75,
            "stops": 0,
            "fare_classes": {
                "economy": {"price_usd": 580, "seats_left": 20, "baggage_kg": 23, "carry_on_kg": 12, "refundable": False, "change_fee_usd": 70},
                "business": {"price_usd": 2950, "seats_left": 7, "baggage_kg": 46, "carry_on_kg": 18, "refundable": True, "change_fee_usd": 0, "lounge_access": True},
            },
            "amenities": ["Delft Blue miniature houses (Business gift)", "Heineken on board", "SkyTeam Elite Plus priority handling"],
            "booking_url": "https://example.com/book/KL871",
            "notes": "Only non-stop DEL–AMS route; very popular in summer — book early.",
        },
    ],

    # ── Delhi → Berlin (Germany) ──────────────────────────────────────────
    "delhi to berlin": [
        {
            "flight_id": "LH-760-BER-VIA-FRA",
            "airline": "Lufthansa",
            "aircraft": "Airbus A350-900 + Airbus A321neo",
            "origin": {"code": "DEL", "terminal": "3"},
            "destination": {"code": "BER", "name": "Berlin Brandenburg Airport", "terminal": "1"},
            "departure": "2026-09-05T07:30:00+05:30",
            "arrival": "2026-09-05T16:20:00+02:00",
            "duration_hours": 11.83,
            "stops": 1,
            "layover": {"airport": "FRA", "city": "Frankfurt", "duration_min": 95, "terminal": "1"},
            "fare_classes": {
                "economy": {"price_usd": 540, "seats_left": 17, "baggage_kg": 23, "carry_on_kg": 8, "refundable": False, "change_fee_usd": 50},
                "business": {"price_usd": 2800, "seats_left": 4, "baggage_kg": 64, "carry_on_kg": 16, "refundable": True, "change_fee_usd": 0, "lounge_access": True},
            },
            "booking_url": "https://example.com/book/LH760",
        },
        {
            "flight_id": "EY-210-BER-VIA-AUH",
            "airline": "Etihad Airways",
            "aircraft": "Boeing 787-9 + Airbus A321",
            "origin": {"code": "DEL", "terminal": "3"},
            "destination": {"code": "BER"},
            "departure": "2026-09-05T20:15:00+05:30",
            "arrival": "2026-09-06T09:00:00+02:00",
            "duration_hours": 10.75,
            "stops": 1,
            "layover": {"airport": "AUH", "city": "Abu Dhabi", "duration_min": 85},
            "fare_classes": {
                "economy": {"price_usd": 495, "seats_left": 35, "baggage_kg": 30, "carry_on_kg": 7, "refundable": False, "change_fee_usd": 85},
                "business": {"price_usd": 3200, "seats_left": 3, "baggage_kg": 40, "carry_on_kg": 12, "refundable": True, "change_fee_usd": 0},
            },
            "booking_url": "https://example.com/book/EY210",
        },
    ],

    # ── Delhi → Dehradun ─────────────────────────────────────────────────
    "delhi to dehradun": [
        {
            "flight_id": "AI-9618-DED",
            "airline": "Air India Express",
            "aircraft": "ATR 72-600",
            "origin": {"code": "DEL", "terminal": "2"},
            "destination": {"code": "DED", "name": "Jolly Grant Airport, Dehradun"},
            "departure": "2026-05-20T07:00:00+05:30",
            "arrival": "2026-05-20T08:00:00+05:30",
            "duration_hours": 1.0,
            "stops": 0,
            "fare_classes": {
                "economy": {"price_usd": 55, "seats_left": 12, "baggage_kg": 15, "carry_on_kg": 7, "refundable": False, "change_fee_usd": 20},
            },
            "booking_url": "https://example.com/book/AI9618",
            "notes": "Quickest way Delhi–Dehradun; alternatively AC Volvo bus is 6 hrs and costs $8.",
            "ground_transport": {
                "bus": {"operator": "UPSRTC / Uttarakhand Roadways", "duration_hrs": 6.0, "price_usd": 8, "departs": "ISBT Kashmiri Gate"},
                "train": {"name": "Shatabdi Express 12017", "duration_hrs": 5.5, "price_usd": 18, "class": "CC Chair Car", "station": "Delhi station → Dehradun station"},
                "taxi": {"price_usd": 40, "duration_hrs": 5.5, "notes": "Ola / Uber outstation available; great for direct Dehradun/Mussoorie door-to-door"},
            },
        },
    ],

    # ── Delhi → Manali / Himachal Pradesh ────────────────────────────────
    "delhi to manali": [
        {
            "flight_id": "NO-DIRECT-FLIGHT",
            "note": "No direct flight Delhi–Manali. Nearest airports: Bhuntar (KUU, 50 km south of Manali) via IndiGo/SpiceJet, or Chandigarh (IXC) then bus.",
            "via_bhuntar": {
                "flight_id": "6E-1234-KUU",
                "airline": "IndiGo",
                "aircraft": "ATR 72-600",
                "origin": {"code": "DEL", "terminal": "2"},
                "destination": {"code": "KUU", "name": "Kullu–Manali Airport, Bhuntar"},
                "departure": "2026-06-10T06:30:00+05:30",
                "arrival": "2026-06-10T07:40:00+05:30",
                "duration_hours": 1.17,
                "stops": 0,
                "fare_classes": {
                    "economy": {"price_usd": 70, "seats_left": 8, "baggage_kg": 15, "carry_on_kg": 7, "refundable": False, "change_fee_usd": 25},
                },
                "notes": "Weather-dependent; Bhuntar flights frequently cancelled in monsoon. Confirm day before.",
                "onward_to_manali": {"taxi_km": 50, "taxi_price_usd": 18, "taxi_duration_hrs": 1.5, "bus_price_usd": 3, "bus_duration_hrs": 2.5},
                "booking_url": "https://example.com/book/6E1234",
            },
            "via_bus": {
                "operator": "HRTC Volvo / private operators",
                "departure_point": "Delhi ISBT Kashmiri Gate",
                "duration_hrs": 14,
                "overnight": True,
                "price_usd": 22,
                "notes": "Overnight Volvo AC Sleeper; most budget travellers prefer this option.",
            },
        },
    ],

    # ── Paris → Amsterdam ─────────────────────────────────────────────────
    "paris to amsterdam": [
        {
            "flight_id": "THALYS-TRAIN-PREFERRED",
            "type": "train",
            "note": "Flight not recommended Paris–Amsterdam (2h flight vs 3h15m Thalys — city centre to city centre). Train beats flying door-to-door.",
            "thalys": {
                "operator": "Thalys / Eurostar",
                "origin": "Paris Gare du Nord",
                "destination": "Amsterdam Centraal",
                "duration_hrs": 3.25,
                "price_usd": 45,
                "price_business_usd": 135,
                "frequency": "Every 2 hours 06:00–22:00",
                "booking_url": "https://example.com/book/thalys",
                "notes": "No check-in, no luggage limits, city-centre to city-centre. Best option.",
            },
            "flight_option": {
                "flight_id": "KL-1009-AMS",
                "airline": "KLM",
                "departure": "Paris CDG",
                "arrival": "Amsterdam AMS",
                "duration_hrs": 1.08,
                "price_usd": 120,
                "notes": "Total door-to-door ~4.5h when including airport transfers. Not recommended.",
            },
        },
    ],
}

_FLIGHT_GENERAL_INFO: dict[str, dict] = {
    "paris": {
        "visa": {"indian_passport": "Schengen visa required; apply 3-4 weeks ahead at French consulate; fee €80", "us_passport": "No visa required for stays under 90 days"},
        "currency": "EUR (Euro); 1 USD ≈ 0.93 EUR (Apr 2026)",
        "best_time_to_visit": "April–June and September–October for mild weather; avoid August (very crowded, many locals on holiday)",
        "airport_to_city": {"cdg_to_paris_centre": {"rer_b": {"duration_min": 35, "price_eur": 10.3, "runs": "Every 10-15 min 05:00–00:00"}, "taxi": {"duration_min": 45, "price_eur": 55, "note": "Fixed rate from CDG to all Paris arrondissements"}, "bus_le_bus_direct": {"duration_min": 75, "price_eur": 18}}},
        "local_transport": "Metro (14 lines, €1.73/trip, or Navigo weekly pass €22.80), Vélib' bike share, RER suburban trains",
        "useful_apps": ["Citymapper (transit)", "Bonjour RATP (official metro)", "Too Good To Go (cheap restaurant surplus meals)"],
        "emergency": {"police": "17", "ambulance_samu": "15", "european_emergency": "112"},
    },
    "bali": {
        "visa": {"indian_passport": "Visa on arrival USD 35, valid 30 days, extendable once for another 30 days", "us_passport": "Visa on arrival USD 35 or e-Visa-Free for 30 days"},
        "currency": "IDR (Indonesian Rupiah); 1 USD ≈ 16 200 IDR (Apr 2026)",
        "best_time_to_visit": "April–October (dry season); avoid November–March (heavy rains, humid, some roads flood)",
        "airport_to_city": {"dps_to_seminyak": {"taxi_bluebird": {"duration_min": 25, "price_usd": 8}, "grab": {"duration_min": 25, "price_usd": 5}}, "dps_to_ubud": {"taxi": {"duration_min": 75, "price_usd": 22}}},
        "local_transport": "Rent scooter ($5/day), Grab/Gojek (ride hailing), private driver ($50/day, highly recommended for temple hopping)",
        "useful_apps": ["Gojek (ride, food, money)", "Grab", "Traveloka (local booking)"],
        "health": "Drink only bottled water; dengue fever risk — use repellent; Bali belly common — stick to cooked food at warungs",
        "emergency": {"police": "110", "ambulance": "118", "tourist_police_bali": "+62-361-754-599"},
    },
    "tokyo": {
        "visa": {"indian_passport": "Japan visa required; eVisa available; fee ¥3000; processing 5-7 business days", "us_passport": "No visa required up to 90 days"},
        "currency": "JPY (Yen); 1 USD ≈ 154 JPY (Apr 2026)",
        "best_time_to_visit": "March–May (cherry blossom) or October–November (autumn foliage); avoid rainy season June–July",
        "airport_to_city": {"narita_to_shinjuku": {"narita_express": {"duration_min": 90, "price_jpy": 3070}, "highway_bus": {"duration_min": 80, "price_jpy": 1300}}, "haneda_to_shinjuku": {"monorail_yamanote": {"duration_min": 35, "price_jpy": 650}, "taxi": {"duration_min": 40, "price_jpy": 6000}}},
        "ic_card": "Suica / Pasmo IC card covers all Tokyo transit; load at airport on arrival",
        "pocket_wifi": "Recommended; rent at airport arrival hall ¥800/day; coverage excellent nationwide",
        "useful_apps": ["Google Maps (excellent in Japan)", "Hyperdia (train routes)", "Tabelog (restaurant reviews in Japanese — use Chrome translate)"],
    },
    "amsterdam": {
        "visa": {"indian_passport": "Schengen visa required", "us_passport": "No visa required up to 90 days"},
        "currency": "EUR; 1 USD ≈ 0.93 EUR",
        "airport_to_city": {"schiphol_to_centraal": {"train": {"duration_min": 17, "price_eur": 5.10, "frequency": "Every 15 min"}, "taxi": {"duration_min": 25, "price_eur": 40}}},
        "local_transport": "Bike rental (€12/day, most authentic way), tram/metro (GVB, OV-chipkaart), canal boat ferries (some free)",
        "useful_apps": ["9292 (national transit)", "NS (national rail)", "Donkey Republic (bike rental)"],
    },
}


def web_search(query: str) -> dict:
    """
    Simulates a web search for flights and travel information.
    Returns a large structured payload covering flight options, pricing,
    visa info, airport transfers, and local transport.
    """
    q = query.lower().strip()

    # Identify which route or destination was queried
    matched_flights: list[dict] = []
    for route_key, flights in _FLIGHT_DB.items():
        if all(word in q for word in route_key.split(" to ")):
            matched_flights = flights
            break
        # partial match: single city
        if route_key.split(" to ")[-1] in q:
            matched_flights = flights
            break

    # General info for destination
    destination_info: dict = {}
    for city_key, info in _FLIGHT_GENERAL_INFO.items():
        if city_key in q:
            destination_info = info
            break

    # If nothing matched, return a generic helpful payload
    if not matched_flights and not destination_info:
        matched_flights = _FLIGHT_DB.get("delhi to paris", [])
        destination_info = _FLIGHT_GENERAL_INFO.get("paris", {})

    result = {
        "search_query": query,
        "results_type": "flights_and_travel_info",
        "flight_results": matched_flights,
        "destination_info": destination_info,
        "search_metadata": {
            "source": "mock_web_search_v1",
            "timestamp_utc": "2026-04-17T10:00:00Z",
            "note": "Prices are indicative; check airline websites for live fares.",
            "currency_note": "All prices USD unless stated otherwise.",
        },
    }
    return result


# ===========================================================================
# TOOL 2 — places_search
# ===========================================================================
# 5-8 entries per city per category, each with full address, coordinates,
# amenities, reviews, pricing details, policies, nearby transit.
# Covers: Paris, Bali, Tokyo, Amsterdam, Germany/Berlin, Dehradun,
# Himachal Pradesh/Manali, India/Delhi, Kyoto, Switzerland/Zurich.
# ===========================================================================

_PLACES_DB: dict[str, dict[str, list[dict]]] = {

    # ── PARIS ─────────────────────────────────────────────────────────────
    "paris": {
        "hotels": [
            {
                "name": "Hôtel du Louvre (Hyatt)",
                "stars": 4, "price_per_night_usd": 285,
                "address": "Place André Malraux, 75001 Paris",
                "coordinates": {"lat": 48.8639, "lon": 2.3356},
                "area": "1st arrondissement — Right Bank",
                "metro": ["Palais Royal–Musée du Louvre (Line 1/7) — 3 min walk"],
                "check_in": "15:00", "check_out": "12:00",
                "amenities": ["Free Wi-Fi", "Fitness centre", "24-hr concierge", "Valet parking €55/night", "Brasserie on-site", "Air conditioning", "Soundproofed rooms", "Safe deposit box"],
                "room_types": {"standard_double": 285, "deluxe_eiffel_view": 390, "junior_suite": 550},
                "reviews": {"tripadvisor_rating": 4.5, "google_rating": 4.4, "total_reviews": 3840, "highlights": ["superb location", "helpful concierge", "Louvre views from upper floors"]},
                "cancellation_policy": "Free cancellation up to 48h before check-in; after that, first night charged.",
                "breakfast": "Continental €28/person or skip and use nearby cafés.",
                "notes": "5-min walk to the Louvre, 12-min walk to the Seine. Excellent base for Right Bank museums.",
            },
            {
                "name": "Le Marais Boutique Hotel",
                "stars": 3, "price_per_night_usd": 162,
                "address": "15 Rue de Bretagne, 75003 Paris",
                "coordinates": {"lat": 48.8624, "lon": 2.3613},
                "area": "3rd arrondissement — Le Marais",
                "metro": ["Filles du Calvaire (Line 8) — 2 min walk", "Temple (Line 3) — 4 min walk"],
                "check_in": "14:00", "check_out": "11:30",
                "amenities": ["Free Wi-Fi", "Courtyard garden", "Espresso maker in room", "Bicycle hire €15/day", "Non-smoking throughout"],
                "room_types": {"cosy_double": 162, "superior_double_garden_view": 195, "family_room": 270},
                "reviews": {"tripadvisor_rating": 4.3, "google_rating": 4.5, "total_reviews": 1210, "highlights": ["cool neighbourhood", "friendly staff", "independent cafes around corner"]},
                "cancellation_policy": "Free cancellation 72h before arrival.",
                "notes": "Heart of the trendy Marais district; walking distance to Centre Pompidou and Place des Vosges.",
            },
            {
                "name": "Grand Hôtel Opéra (InterContinental)",
                "stars": 5, "price_per_night_usd": 510,
                "address": "2 Rue Scribe, 75009 Paris",
                "coordinates": {"lat": 48.8717, "lon": 2.3313},
                "area": "9th arrondissement — Opéra Quarter",
                "metro": ["Opéra (Lines 3/7/8) — 1 min walk"],
                "check_in": "15:00", "check_out": "12:00",
                "amenities": ["Indoor heated pool", "Spa by ESPA", "Fitness centre", "Café de la Paix on-site (Michelin Bib Gourmand)", "24-hr butler", "Free Wi-Fi (1 Gbps)", "Valet parking €75/night", "IHG One Rewards eligible"],
                "room_types": {"classic_room": 510, "superior_room": 650, "junior_suite_eiffel_view": 890, "grand_suite": 1800},
                "reviews": {"tripadvisor_rating": 4.7, "google_rating": 4.6, "total_reviews": 7250},
                "cancellation_policy": "Non-refundable rates available (10% discount); Flex rates free cancellation 24h prior.",
                "notes": "Historic property; Café de la Paix served Émile Zola and Coco Chanel. Opéra Garnier visible from windows.",
            },
            {
                "name": "Ibis Paris Tour Eiffel",
                "stars": 3, "price_per_night_usd": 125,
                "address": "2 Rue Cambronne, 75015 Paris",
                "area": "15th arrondissement — budget-friendly, good transit links",
                "metro": ["Cambronne (Line 6) — 1 min walk"],
                "amenities": ["Free Wi-Fi", "24-hr reception", "Bar", "AC"],
                "room_types": {"standard": 125, "superior": 148},
                "reviews": {"tripadvisor_rating": 4.1, "google_rating": 4.2, "total_reviews": 4100},
                "cancellation_policy": "Free cancellation up to 48h before check-in.",
                "notes": "Best-value chain option; 20 min from city centre by metro.",
            },
        ],
        "restaurants": [
            {
                "name": "Septime",
                "cuisine": "Modern French / Neo-bistro",
                "michelin": "1 star",
                "price_range": "$$$  (avg €75/person without wine)",
                "address": "80 Rue de Charonne, 75011 Paris",
                "area": "11th arrondissement",
                "opening_hours": "Lunch Tue–Fri 12:15–14:00; Dinner Mon–Fri 19:15–21:45; closed Sat–Sun",
                "reservations": "Essential — book 3-4 weeks ahead via website or Resy",
                "dietary": {"vegetarian": "Ask at booking — tasting menu can be adapted", "shellfish": "PRESENT — prominently features in tasting menu; notify allergy at booking"},
                "reviews": {"google_rating": 4.7, "la_liste_rank": 82, "highlights": ["best neo-bistro in Paris", "seasonal tasting menu", "natural wine list"]},
                "notes": "Chef Bertrand Grébaut's landmark restaurant. No à la carte — tasting menu only. Worth every euro.",
            },
            {
                "name": "Chez L'Ami Jean",
                "cuisine": "Basque / South-West French",
                "price_range": "$$ (avg €45/person)",
                "address": "27 Rue Malar, 75007 Paris",
                "area": "7th arrondissement (near Eiffel Tower)",
                "opening_hours": "Tue–Sat 12:00–14:00 and 19:00–23:00; closed Mon & Sun",
                "reservations": "Recommended 1 week ahead",
                "dietary": {"shellfish": "PRESENT — razor clams, prawns on menu; warn on booking", "vegetarian": "Limited options"},
                "reviews": {"google_rating": 4.5, "highlights": ["legendary rice pudding dessert", "jovial atmosphere", "hearty portions"]},
            },
            {
                "name": "Au Passage",
                "cuisine": "Natural wine bistro / small plates",
                "price_range": "$$ (avg €38/person)",
                "address": "1bis Passage Saint-Sébastien, 75011 Paris",
                "opening_hours": "Mon–Sat 18:30–23:30; lunch Fri 12:00–14:30",
                "reservations": "Walk-in friendly at bar; tables bookable via phone",
                "dietary": {"vegetarian": "Good options among small plates", "shellfish": "Occasionally featured — check daily menu"},
            },
            {
                "name": "L'As du Fallafel",
                "cuisine": "Israeli / Falafel street food",
                "price_range": "$ (falafel wrap €7)",
                "address": "34 Rue des Rosiers, 75004 Paris",
                "area": "4th arrondissement — Jewish Quarter of Le Marais",
                "opening_hours": "Mon–Thu 11:00–24:00; Fri 11:00–15:00 (closes before Shabbat); Sun 11:00–24:00",
                "dietary": {"vegetarian": "Fully vegetarian / vegan options", "shellfish": "None", "halal": "No (kosher)"},
                "notes": "Legendary; expect a queue. Worth it.",
            },
        ],
        "attractions": [
            {
                "name": "Musée d'Orsay",
                "type": "Art Museum",
                "entry_usd": 17,
                "address": "1 Rue de la Légion d'Honneur, 75007 Paris",
                "opening_hours": "Tue–Sun 09:30–18:00; Thu until 21:45; closed Mon",
                "highlights": ["Largest Impressionist collection in the world", "Monet, Renoir, Van Gogh, Degas", "The building itself is a Beaux-Arts railway station"],
                "tips": ["Buy timed entry online (Paris Museum Pass accepted)", "Thursday evening is quietest", "Café on top floor has views of Sacré-Cœur"],
                "duration_recommended_hrs": 2.5,
            },
            {
                "name": "Palace of Versailles",
                "type": "Royal Palace and Gardens",
                "entry_usd": 22,
                "address": "Place d'Armes, 78000 Versailles",
                "transport": "RER C from Paris (40 min, €7 round trip)",
                "opening_hours": "Tue–Sun 09:00–18:30 (palace); gardens open daily 08:00–20:30",
                "highlights": ["Hall of Mirrors", "Marie Antoinette's Estate (Petit Trianon)", "2000-acre formal gardens", "Musical fountain shows Sat–Sun (May–Oct, extra charge)"],
                "tips": ["Arrive at opening to beat coach groups", "Hire a golf cart to see the gardens (€32/hr)", "Full day excursion recommended"],
                "duration_recommended_hrs": 6,
            },
            {
                "name": "Eiffel Tower",
                "type": "Landmark",
                "entry_usd": 30,
                "address": "Champ de Mars, 5 Avenue Anatole France, 75007 Paris",
                "opening_hours": "Daily 09:00–00:45 (mid-June–Aug); 09:30–23:45 (rest of year)",
                "highlights": ["Top floor views 276m up", "Light show every hour after dusk", "Jules Verne restaurant (Michelin 2-star) on 2nd floor"],
                "tips": ["Book summit tickets online 60+ days ahead in peak season; walk-up stairs to 2nd floor is cheaper & faster"],
                "duration_recommended_hrs": 2,
            },
        ],
    },

    # ── BALI ──────────────────────────────────────────────────────────────
    "bali": {
        "hotels": [
            {
                "name": "Four Seasons Resort Bali at Sayan",
                "stars": 5, "price_per_night_usd": 950,
                "area": "Ubud — jungle and river gorge",
                "address": "Sayan, Ubud, Gianyar, Bali 80571",
                "coordinates": {"lat": -8.5069, "lon": 115.2476},
                "check_in": "15:00", "check_out": "12:00",
                "amenities": ["Jungle canopy infinity pool", "COMO Shambhala-style spa", "Yoga pavilion daily", "Private villa plunge pools (villa category)", "All-day dining overlooking Ayung River", "Complimentary cooking class", "Airport transfer $80 each way", "Free Wi-Fi"],
                "room_types": {"suite": 950, "1br_villa_pool": 1600, "2br_villa_pool": 2900},
                "reviews": {"tripadvisor_rating": 5.0, "conde_nast_traveller": "Gold List", "highlights": ["most romantic resort in Bali", "rice paddy walks at dawn", "extraordinary spa"]},
                "cancellation_policy": "72h cancellation window; some advance rates non-refundable.",
            },
            {
                "name": "Bisma Eight",
                "stars": 4, "price_per_night_usd": 230,
                "area": "Ubud — central, rice terrace views",
                "address": "Jl. Bisma, Ubud, Bali 80571",
                "amenities": ["Infinity pool over rice paddies", "Treetop restaurant", "Free shuttle to Ubud centre", "Spa", "Yoga classes", "Wi-Fi"],
                "room_types": {"forest_suite": 230, "valley_suite": 310, "pool_villa": 520},
                "reviews": {"tripadvisor_rating": 4.7, "highlights": ["exceptional breakfast", "Instagram-worthy pool", "quiet yet walkable to Ubud market"]},
                "cancellation_policy": "Free cancellation 48h before arrival.",
                "notes": "Best mid-range option in Ubud with luxury feel.",
            },
            {
                "name": "Katamama",
                "stars": 5, "price_per_night_usd": 480,
                "area": "Seminyak — beachside",
                "address": "Jl. Petitenget No.51B, Seminyak, Bali 80361",
                "amenities": ["Private beach club access (Potato Head)", "Balinese craft-filled rooms (handmade fabrics, terracotta)", "Antique furnishings sourced from Java", "Rooftop pool", "Shambhala Spa", "In-room bath butler service"],
                "room_types": {"studio_suite": 480, "1br_suite": 750},
                "reviews": {"tripadvisor_rating": 4.8, "highlights": ["unique artisan design", "Potato Head parties", "incredible location"]},
            },
            {
                "name": "Komaneka at Bisma",
                "stars": 5, "price_per_night_usd": 380,
                "area": "Ubud",
                "amenities": ["3 pools", "award-winning Komaneka restaurant", "art gallery", "spa", "rice paddy walking trails from property"],
                "room_types": {"valley_suite": 380, "pool_villa": 680},
                "reviews": {"tripadvisor_rating": 4.9},
            },
            {
                "name": "Legian Kriyamaha Villa",
                "stars": 3, "price_per_night_usd": 95,
                "area": "Legian — budget-friendly, near Kuta beach",
                "amenities": ["Private pool", "Free breakfast", "Wi-Fi", "Free shuttle to beach"],
                "reviews": {"google_rating": 4.6, "highlights": ["exceptional value", "lovely family-run property"]},
                "cancellation_policy": "Free cancellation 24h before check-in.",
            },
        ],
        "restaurants": [
            {
                "name": "Locavore",
                "cuisine": "Modern Indonesian / Farm-to-Table",
                "awards": "Asia's 50 Best Restaurants (ranked #23 2025)",
                "price_range": "$$$ (tasting menu IDR 950 000 / ~$59)",
                "address": "Jl. Dewi Sita No.10, Ubud, Bali 80571",
                "opening_hours": "Lunch Tue–Sat 12:00–15:00; Dinner Mon–Sat 18:30–23:00",
                "reservations": "Book 2-4 weeks ahead; online via locavore.id",
                "dietary": {"shellfish": "PRESENT — seasonal seafood prominently featured; notify for allergy", "vegetarian": "Locavore Next Door next-door is fully plant-based (same team, separate restaurant)"},
                "notes": "Best fine dining in Bali. Hyperlocal ingredients, brilliant cocktail pairing.",
            },
            {
                "name": "Warung Ibu Oka",
                "cuisine": "Balinese suckling pig (Babi Guling)",
                "price_range": "$ (IDR 70 000 / ~$4.50)",
                "address": "Jl. Suweta No.2, Ubud, Bali 80571",
                "opening_hours": "Daily 11:00–17:00 or until sold out",
                "dietary": {"shellfish": "None", "vegetarian": "Not suitable (pork-focused)", "pork": "Pork only restaurant"},
                "notes": "Famous babi guling; referenced in Anthony Bourdain's show. Arrive early — sells out.",
            },
            {
                "name": "Merah Putih",
                "cuisine": "Modern Indonesian — pan-archipelago",
                "price_range": "$$ (avg IDR 350 000 / ~$22)",
                "address": "Jl. Petitenget No.100X, Seminyak, Bali 80361",
                "opening_hours": "Daily 12:00–23:00",
                "dietary": {"shellfish": "PRESENT", "vegetarian": "Good selection of vegetable dishes from Java, Sumatra, Bali traditions"},
            },
        ],
        "attractions": [
            {
                "name": "Tegallalang Rice Terraces",
                "type": "UNESCO-listed natural landscape",
                "entry_usd": 2,
                "address": "Tegallalang Village, Ubud, Gianyar",
                "opening_hours": "Daily 08:00–18:00",
                "highlights": ["Subak irrigation system (UNESCO)", "Sunrise shots world-class", "Swing over the terraces ($10 extra)"],
                "tips": ["Go at 07:30 before tour buses; avoid 10:00–14:00 peak heat and crowds"],
                "duration_recommended_hrs": 1.5,
            },
            {
                "name": "Tanah Lot Temple",
                "type": "Sea Temple",
                "entry_usd": 5,
                "address": "Beraban, Kediri, Tabanan, Bali",
                "opening_hours": "Daily 07:00–19:00",
                "highlights": ["Iconic offshore temple on sea rock", "Spectacular sunset backdrop", "Black-and-yellow sea snakes considered sacred guardians"],
                "tips": ["Arrive 1 hour before sunset", "Accessible only at low tide"],
                "duration_recommended_hrs": 2,
            },
            {
                "name": "Mount Batur Sunrise Trek",
                "type": "Volcano Trek",
                "entry_usd": 65,
                "altitude_m": 1717,
                "highlights": ["Sunrise over Lake Batur", "Active volcano — sometimes see steam vents", "Boil eggs in volcanic steam at summit"],
                "tips": ["Depart hotel 01:30–02:00; summit by 05:30 for sunrise", "Mandatory guide ($30–40) on top of entrance", "Cold at summit — bring fleece"],
                "duration_recommended_hrs": 7,
            },
        ],
    },

    # ── TOKYO ─────────────────────────────────────────────────────────────
    "tokyo": {
        "hotels": [
            {
                "name": "Park Hyatt Tokyo",
                "stars": 5, "price_per_night_usd": 720,
                "area": "Shinjuku — floors 39–52 of Park Tower",
                "address": "3-7-1-2 Nishi-Shinjuku, Shinjuku City, Tokyo 163-1055",
                "coordinates": {"lat": 35.6869, "lon": 139.6908},
                "check_in": "15:00", "check_out": "12:00",
                "amenities": ["New York Bar (Lost in Translation filming location)", "50m heated indoor pool on 47th floor", "Spa", "Free Wi-Fi (1 Gbps)", "Mountain views (clear days: Mt Fuji)", "Andaz Lounge 24-hr"],
                "room_types": {"park_room": 720, "park_deluxe": 920, "park_suite": 1400},
                "reviews": {"tripadvisor_rating": 4.9, "highlights": ["iconic, legendary service", "New York Bar worth every yen", "city views extraordinary"]},
                "cancellation_policy": "Free cancellation 72h before arrival.",
            },
            {
                "name": "Shinjuku Granbell Hotel",
                "stars": 4, "price_per_night_usd": 195,
                "area": "Shinjuku — Golden Gai / Kabukicho proximity",
                "address": "2-14-5 Kabukicho, Shinjuku, Tokyo 160-0021",
                "amenities": ["Designer rooms (each floor different theme)", "Bar on ground floor", "Free Wi-Fi", "Proximity to Golden Gai bars (2 min walk)"],
                "room_types": {"standard_double": 195, "superior_double": 240, "suite": 380},
                "reviews": {"tripadvisor_rating": 4.4, "highlights": ["cool design", "perfect Shinjuku location", "good value for Tokyo"]},
                "cancellation_policy": "Free cancellation 48h before arrival.",
            },
            {
                "name": "Hoshinoya Tokyo",
                "stars": 5, "price_per_night_usd": 580,
                "area": "Otemachi — business district, 10 min from Imperial Palace",
                "amenities": ["Onsen on top floor (rare in central Tokyo)", "Kaiseki dinner ($120/person extra)", "Tea ceremony twice daily", "Tatami rooms", "Japanese bath in every room"],
                "room_types": {"tatami_twin": 580, "tatami_double": 630},
                "reviews": {"tripadvisor_rating": 4.8, "highlights": ["authentic ryokan experience in central Tokyo", "onsen is magical", "exceptional kaiseki"]},
            },
            {
                "name": "Khaosan Tokyo Origami",
                "stars": 2, "price_per_night_usd": 88,
                "area": "Asakusa — traditional neighbourhood",
                "amenities": ["Free Wi-Fi", "Rooftop deck", "Shared kitchen", "Bicycle hire ¥1000/day", "Walking distance to Senso-ji Temple"],
                "room_types": {"dorm_6bed": 35, "private_double": 88, "private_triple": 120},
                "reviews": {"tripadvisor_rating": 4.3, "highlights": ["unbeatable location for temples", "friendly staff", "great social atmosphere"]},
                "cancellation_policy": "Free cancellation 24h before check-in.",
            },
        ],
        "restaurants": [
            {
                "name": "Sukiyabashi Jiro Honten (Jiro Dreams of Sushi)",
                "cuisine": "Omakase sushi",
                "michelin": "3 stars",
                "price_range": "$$$$ (¥40 000 / ~$260 per person — lunch only)",
                "address": "4-2-15 Ginza, Chuo City, Tokyo 104-0061",
                "opening_hours": "Mon–Fri lunch 11:30 only (one seating); closed weekends & Aug",
                "reservations": "Must be made through your hotel concierge ONLY; typically 1-2 months ahead",
                "dietary": {"shellfish": "CENTRAL TO MENU — all courses involve fish and shellfish; not suitable for shellfish allergy", "vegetarian": "Completely unsuitable"},
                "notes": "Obama dined here. 10-seat counter. 20-minute meal. Legendary.",
            },
            {
                "name": "Ichiran Ramen",
                "cuisine": "Hakata tonkotsu ramen — solo dining experience",
                "price_range": "$ (¥980 / ~$6.50)",
                "address": "Multiple locations across Tokyo",
                "opening_hours": "24 hours",
                "dietary": {"shellfish": "Broth is pork-based — no shellfish", "vegetarian": "Not suitable (pork broth)"},
                "notes": "Individual booths, customise your ramen on a form, no social pressure. Tokyo institution.",
            },
            {
                "name": "Gonpachi Nishi-Azabu (Kill Bill restaurant)",
                "cuisine": "Izakaya / traditional Japanese robata",
                "price_range": "$$ (avg ¥4500 / ~$29)",
                "address": "1-13-11 Nishi-Azabu, Minato City, Tokyo 106-0031",
                "opening_hours": "Daily 17:00–04:00",
                "dietary": {"shellfish": "PRESENT on menu — grilled scallops, shrimp skewers; avoid specific dishes", "vegetarian": "Vegetable skewers, tofu dishes available"},
                "notes": "Quentin Tarantino used the interior as inspiration for Kill Bill's Crazy 88 fight scene. Dramatic multilevel interior.",
            },
            {
                "name": "Afuri Ramen",
                "cuisine": "Yuzu shio (citrus salt) ramen",
                "price_range": "$ (¥1200 / ~$8)",
                "address": "Multiple locations — Harajuku, Ebisu, Nakameguro",
                "opening_hours": "Daily 11:00–23:00",
                "dietary": {"shellfish": "None", "vegetarian": "Vegan ramen available (yuzu shio vegan broth)"},
                "notes": "Lighter alternative to tonkotsu. Yuzu aroma is extraordinary.",
            },
            {
                "name": "Tsukiji Outer Market stalls",
                "cuisine": "Fresh seafood breakfast / street food",
                "price_range": "$ (¥200–1500 per item)",
                "address": "4-16-2 Tsukiji, Chuo City, Tokyo 104-0045",
                "opening_hours": "Daily 05:00–14:00 (most stalls); busiest 06:30–09:00",
                "dietary": {"shellfish": "DOMINANT — grilled scallops, oysters, shrimp skewers everywhere; NOT suitable for shellfish allergy", "vegetarian": "Very limited — tamago (egg) stalls only"},
                "notes": "Fish market moved to Toyosu 2018; Tsukiji Outer Market (street food stalls) remains open. An extraordinary food experience.",
            },
        ],
        "attractions": [
            {
                "name": "Senso-ji Temple",
                "type": "Buddhist temple — Tokyo's oldest",
                "entry_usd": 0,
                "address": "2-3-1 Asakusa, Taito City, Tokyo 111-0032",
                "opening_hours": "Main hall 06:00–17:00; grounds always open",
                "highlights": ["Kaminarimon (Thunder Gate) giant lantern", "Nakamise shopping street (200m of traditional stalls)", "Fortune strips (omikuji) from vending machine", "Spectacular at dawn with incense smoke"],
                "tips": ["06:00–07:30 is magical; tour buses arrive 09:30+", "Rent kimono nearby (¥3000/half day)"],
            },
            {
                "name": "TeamLab Borderless",
                "type": "Digital art museum",
                "entry_usd": 38,
                "address": "New location: Azabudai Hills, Minato City, Tokyo (opened 2024)",
                "opening_hours": "Daily 09:00–22:00 (last entry 20:00)",
                "highlights": ["60+ immersive digital art rooms", "Flower Forest, Crystal World, Light Sculpture Forest", "No fixed route — different each visit"],
                "tips": ["Book online; sell out 2-3 weeks ahead in peak season", "Wear comfortable shoes — lots of walking"],
                "duration_recommended_hrs": 3,
            },
        ],
    },

    # ── AMSTERDAM ─────────────────────────────────────────────────────────
    "amsterdam": {
        "hotels": [
            {
                "name": "Pulitzer Amsterdam",
                "stars": 5, "price_per_night_usd": 420,
                "area": "Canal Ring — 25 interconnected 17th-century canal houses",
                "address": "Prinsengracht 315-331, 1016 GZ Amsterdam",
                "coordinates": {"lat": 52.3746, "lon": 4.8832},
                "amenities": ["Pulitzer's Bar (garden terrace on canal)", "Private boat tours from hotel jetty", "Bikes included", "Spa", "Free Wi-Fi", "Bookable canal-view rooms"],
                "room_types": {"classic": 420, "canal_view_superior": 590, "suite": 980},
                "reviews": {"tripadvisor_rating": 4.7, "highlights": ["most romantic hotel in Amsterdam", "bikes on loan", "canal views from every room"]},
                "cancellation_policy": "Free cancellation 48h before check-in.",
                "notes": "UNESCO World Heritage canal ring location. Exceptional for honeymooners or special occasions.",
            },
            {
                "name": "Hotel V Nesplein",
                "stars": 4, "price_per_night_usd": 195,
                "area": "Centrum — Old Centre, near Dam Square and Jordaan",
                "address": "Nes 49, 1012 KD Amsterdam",
                "amenities": ["Lively bar/restaurant on-site", "Bikes to rent (€15/day)", "Free Wi-Fi", "Modern Dutch design interior"],
                "room_types": {"standard": 195, "superior": 240},
                "reviews": {"tripadvisor_rating": 4.5, "highlights": ["brilliant central location", "social bar", "great value for Amsterdam"]},
                "cancellation_policy": "Free cancellation 72h before arrival.",
            },
            {
                "name": "Generator Amsterdam",
                "stars": 3, "price_per_night_usd": 45,
                "area": "Oost (East) — 15 min tram from centre",
                "amenities": ["Lively bar", "Pods / dorms / private rooms", "Free Wi-Fi", "Events programme"],
                "room_types": {"dorm_8bed": 22, "dorm_4bed_ensuite": 35, "private_double": 45},
                "reviews": {"tripadvisor_rating": 4.2, "highlights": ["best hostel in Amsterdam", "amazing social scene", "stylish design"]},
            },
        ],
        "restaurants": [
            {
                "name": "Rijks Restaurant",
                "cuisine": "Modern Dutch — inside the Rijksmuseum",
                "michelin": "1 star",
                "price_range": "$$$ (avg €65/person)",
                "address": "Museumstraat 2, 1071 XX Amsterdam",
                "opening_hours": "Lunch daily 12:00–15:00; Dinner Wed–Sun 18:00–22:00",
                "reservations": "Essential — book via rijksrestaurant.nl",
                "dietary": {"shellfish": "PRESENT — North Sea oysters, herring, prawns feature regularly; notify allergy", "vegetarian": "Full vegetarian tasting menu available"},
            },
            {
                "name": "De Kas",
                "cuisine": "Farm-to-table — greenhouse restaurant",
                "price_range": "$$$ (set lunch €49.50, set dinner €79.50)",
                "address": "Kamerlingh Onneslaan 3, 1097 DE Amsterdam",
                "opening_hours": "Mon–Fri 12:00–14:00 and 18:30–22:00; Sat 18:30–22:00",
                "notes": "Restaurant in a 1926 greenhouse. They grow most ingredients in the attached nursery. Extraordinary sustainable dining.",
                "dietary": {"vegetarian": "Set menu can be fully vegetarian on request", "shellfish": "Fish occasionally featured — ask on booking"},
            },
            {
                "name": "FEBO (Dutch fast food)",
                "cuisine": "Dutch snacks — kroket, frikandel, kaassoufflé",
                "price_range": "$ (€1.50–3.50)",
                "address": "Multiple locations across Amsterdam",
                "opening_hours": "24 hours",
                "dietary": {"shellfish": "None", "vegetarian": "Kaassoufflé (cheese) is vegetarian"},
                "notes": "Automat vending wall concept. Totally unique Dutch experience. Try the beef kroket.",
            },
        ],
        "attractions": [
            {
                "name": "Anne Frank House",
                "type": "Historic museum",
                "entry_usd": 18,
                "address": "Westermarkt 20, 1016 GV Amsterdam",
                "opening_hours": "Daily 09:00–22:00 (Apr–Oct); 09:00–19:00 (Nov–Mar)",
                "highlights": ["The actual hiding annex where Anne Frank wrote her diary", "Original diary manuscripts on display", "Deeply moving 1-hr audio guide"],
                "tips": ["SOLD OUT weeks ahead — book ONLINE ONLY (no door tickets)", "Silent and reflective — no large bags allowed"],
                "duration_recommended_hrs": 1.5,
                "booking_url": "https://www.annefrank.org/en/museum/tickets/",
            },
            {
                "name": "Rijksmuseum",
                "type": "National art museum",
                "entry_usd": 25,
                "address": "Museumstraat 1, 1071 XX Amsterdam",
                "opening_hours": "Daily 09:00–17:00",
                "highlights": ["Rembrandt's Night Watch (largest painting in museum)", "Vermeer collection (including The Milkmaid)", "Delft Blue collection", "Beautiful garden (free entry)"],
                "tips": ["Book timed entry online", "The museum is massive — focus on the Golden Age (1600s) floor", "Audio guide recommended"],
                "duration_recommended_hrs": 3,
            },
            {
                "name": "Canal Cruise",
                "type": "Boat tour",
                "entry_usd": 18,
                "highlights": ["See the 17th-century grachtengordes (canal belt) from water", "Evening cruises with wine €28", "Private boat hire from €120/hr"],
                "tips": ["Open boat recommended in summer", "Hop-on-hop-off cruises go past Anne Frank House, Rijksmuseum, Vondelpark"],
            },
        ],
    },

    # ── GERMANY / BERLIN ──────────────────────────────────────────────────
    "berlin": {
        "hotels": [
            {
                "name": "Hotel Adlon Kempinski",
                "stars": 5, "price_per_night_usd": 480,
                "area": "Mitte — next to Brandenburg Gate",
                "address": "Unter den Linden 77, 10117 Berlin",
                "amenities": ["Spa", "Two restaurants (1 Michelin-starred Lorenz Adlon Esszimmer)", "Pool", "Quintessential Berlin landmark", "Free Wi-Fi"],
                "room_types": {"classic": 480, "deluxe_brandenburg_gate_view": 650, "suite": 1200},
                "reviews": {"tripadvisor_rating": 4.7, "highlights": ["unparalleled location", "best service in Berlin", "historic grandeur"]},
                "notes": "Michael Jackson famously dangled his baby from a balcony here (2002). Right by Brandenburg Gate and Reichstag.",
            },
            {
                "name": "25hours Hotel Bikini Berlin",
                "stars": 4, "price_per_night_usd": 175,
                "area": "Charlottenburg — West Berlin, Zoological Garden",
                "address": "Budapester Straße 40, 10787 Berlin",
                "amenities": ["Monkey Bar rooftop (zoo views)", "Neni Restaurant", "Hammam spa", "Bikes to rent", "Free Wi-Fi", "Unique urban jungle design"],
                "room_types": {"compact": 175, "urban_jungle_room": 210, "suite": 380},
                "reviews": {"tripadvisor_rating": 4.5, "highlights": ["best rooftop bar in Berlin", "cool design hotel", "amazing zoo & city views"]},
            },
            {
                "name": "Generator Berlin Mitte",
                "stars": 3, "price_per_night_usd": 40,
                "area": "Mitte — walking distance to Hackescher Markt",
                "amenities": ["Bar", "Free Wi-Fi", "Self-service kitchen", "24-hr reception"],
                "room_types": {"dorm_8bed": 20, "private_double": 40},
                "reviews": {"tripadvisor_rating": 4.3},
            },
        ],
        "restaurants": [
            {
                "name": "Lorenz Adlon Esszimmer",
                "cuisine": "Modern European fine dining",
                "michelin": "2 stars",
                "price_range": "$$$$ (tasting menu €195–250)",
                "address": "Unter den Linden 77, 10117 Berlin (Hotel Adlon)",
                "opening_hours": "Tue–Sat 18:30–22:30",
                "reservations": "Essential — months in advance for weekend",
                "dietary": {"shellfish": "PRESENT", "vegetarian": "Vegetarian tasting menu available on request"},
            },
            {
                "name": "Mustafa's Gemüse Kebap",
                "cuisine": "Turkish dürüm / döner kebab",
                "price_range": "$ (€4.50)",
                "address": "Mehringdamm 32, 10961 Berlin (Kreuzberg)",
                "opening_hours": "Mon–Thu 10:00–02:00; Fri–Sat until 04:00",
                "dietary": {"shellfish": "None", "vegetarian": "Vegetarian dürüm available"},
                "notes": "Best döner in Berlin, possibly the world. Queue 45 min on weekends but 100% worth it. Famously topped with roasted vegetables.",
            },
            {
                "name": "Nobelhart und Schmutzig",
                "cuisine": "Brutal Localism — hyper-regional Berlin tasting menu",
                "michelin": "1 star",
                "price_range": "$$$ (tasting menu €119)",
                "address": "Friedrichstraße 218, 10969 Berlin",
                "opening_hours": "Tue–Sat from 18:30",
                "dietary": {"shellfish": "Ask at booking — menu changes", "vegetarian": "Not typically available; philosophy is around local Brandenburg produce"},
            },
        ],
        "attractions": [
            {
                "name": "Brandenburg Gate",
                "type": "Historic landmark",
                "entry_usd": 0,
                "address": "Pariser Platz, 10117 Berlin",
                "opening_hours": "Always open (outdoor)",
                "highlights": ["Symbol of German reunification", "Quadriga sculpture at top", "Spectacular at night when lit up"],
                "tips": ["Best photos from Pariser Platz side at golden hour", "Holocaust Memorial 3 min walk south"],
                "duration_recommended_hrs": 0.5,
            },
            {
                "name": "Topography of Terror",
                "type": "Historical documentation centre",
                "entry_usd": 0,
                "address": "Niederkirchnerstraße 8, 10963 Berlin",
                "opening_hours": "Daily 10:00–20:00",
                "highlights": ["Built on former Gestapo and SS HQ site", "Remaining Berlin Wall segment on-site", "Comprehensive outdoor exhibition free to walk"],
                "duration_recommended_hrs": 2,
            },
            {
                "name": "East Side Gallery",
                "type": "Outdoor mural art gallery on Berlin Wall",
                "entry_usd": 0,
                "address": "Mühlenstraße 3-100, 10243 Berlin (Friedrichshain)",
                "opening_hours": "Always open",
                "highlights": ["1.3 km longest remaining Berlin Wall section", "105 murals by international artists", "Includes Brezhnev–Honecker kiss painting"],
                "tips": ["Walk the full length from Ostbahnhof to Oberbaumbrücke", "Combine with Oberbaumbrücke photo"],
                "duration_recommended_hrs": 1.5,
            },
        ],
    },

    # ── DEHRADUN ──────────────────────────────────────────────────────────
    "dehradun": {
        "hotels": [
            {
                "name": "Lemon Tree Hotel Dehradun",
                "stars": 4, "price_per_night_usd": 58,
                "area": "Rajpur Road — main commercial strip",
                "address": "22/3 Rajpur Road, Dehradun, Uttarakhand 248001",
                "amenities": ["Rooftop pool", "Restaurant", "Free Wi-Fi", "Gym", "Parking", "Airport pickup ₹600"],
                "room_types": {"standard": 58, "deluxe": 72, "suite": 120},
                "reviews": {"tripadvisor_rating": 4.4, "highlights": ["best pool in Dehradun", "modern property", "good location"]},
                "cancellation_policy": "Free cancellation 24h before check-in.",
                "notes": "Best mid-range option in Dehradun city; 20 min from Jolly Grant Airport.",
            },
            {
                "name": "Moustache Hostel Dehradun",
                "stars": 2, "price_per_night_usd": 12,
                "area": "Near ISBT (Inter-State Bus Terminal)",
                "amenities": ["Free breakfast", "Free Wi-Fi", "Common room", "Bonfire (winter)", "Bicycle rent ₹200/day"],
                "room_types": {"dorm_6bed": 7, "private_double": 12},
                "reviews": {"tripadvisor_rating": 4.6, "highlights": ["best hostel in Uttarakhand", "incredible community", "amazing staff called 'Moustache family'"]},
                "notes": "Gateway for treks to Kedarnath, Chopta, Valley of Flowers. Excellent trek booking service from hostel.",
            },
            {
                "name": "The Vibrant Hotel and Spa",
                "stars": 3, "price_per_night_usd": 42,
                "area": "Doon Valley outskirts — peaceful, garden views",
                "amenities": ["Spa and Ayurvedic treatments", "Swimming pool", "Free Wi-Fi", "Valley views"],
                "reviews": {"tripadvisor_rating": 4.3},
                "notes": "Near Forest Research Institute (FRI) and Clock Tower area.",
            },
        ],
        "restaurants": [
            {
                "name": "Kumar Sweets & Restaurant",
                "cuisine": "North Indian thali / mithai",
                "price_range": "$ (thali ₹150 / ~$1.80)",
                "address": "Paltan Bazaar, Dehradun",
                "opening_hours": "Daily 08:00–22:00",
                "dietary": {"shellfish": "None", "vegetarian": "Fully vegetarian"},
                "notes": "Institution for over 60 years. Try the bal mithai (Kumaoni sweet) and kachori.",
            },
            {
                "name": "Barista Café",
                "cuisine": "Café / continental",
                "price_range": "$ (coffee ₹120–200)",
                "address": "Rajpur Road, Dehradun",
                "dietary": {"shellfish": "None", "vegetarian": "Fully vegetarian / vegan options"},
            },
            {
                "name": "Pal Dhaba",
                "cuisine": "North Indian highway dhaba",
                "price_range": "$ (meal ₹200–350)",
                "address": "National Highway 72, outskirts of Dehradun",
                "dietary": {"shellfish": "None", "vegetarian": "Dal makhani, paneer options always available"},
                "notes": "Best dal makhani in the region; truckers' dhaba with legendary butter roti.",
            },
        ],
        "attractions": [
            {
                "name": "Robbers Cave (Gucchupani)",
                "type": "Natural cave and stream",
                "entry_usd": 1,
                "address": "Anawala Village, Raipur Road, Dehradun",
                "opening_hours": "Daily 08:00–17:00",
                "highlights": ["Stream flows through natural cave", "Wading through ankle-to-knee-deep cold water inside cave", "Picnic area outside"],
                "duration_recommended_hrs": 2,
            },
            {
                "name": "Forest Research Institute (FRI)",
                "type": "Colonial heritage building and museum",
                "entry_usd": 1,
                "address": "FRI Campus, Dehradun, Uttarakhand 248006",
                "opening_hours": "Mon–Fri 09:00–17:00",
                "highlights": ["Massive 1929 Greco-Roman architecture", "6 museums covering Indian forests, timber, social forestry", "250-acre deer park grounds"],
                "duration_recommended_hrs": 3,
            },
            {
                "name": "Sahastradhara (Thousand-fold Spring)",
                "type": "Sulphur springs / waterfall",
                "entry_usd": 1,
                "address": "Sahastradhara Road, Dehradun",
                "opening_hours": "Daily 08:00–18:00",
                "highlights": ["Sulphur springs believed to have medicinal properties", "Cable car (₹150 one way)", "Cold cave bathing pools"],
                "notes": "Crowded on weekends; go on a weekday morning.",
            },
        ],
    },

    # ── MANALI / HIMACHAL PRADESH ──────────────────────────────────────────
    "manali": {
        "hotels": [
            {
                "name": "Span Resort & Spa",
                "stars": 5, "price_per_night_usd": 180,
                "area": "Kullu Valley — on the banks of Beas River, 12 km before Manali",
                "address": "NH-3, Katrain, Kullu, Himachal Pradesh 175129",
                "amenities": ["Heated outdoor pool (heated to 30°C even in Dec)", "Full-service spa", "Beas River-facing rooms", "Apple orchard walks", "Free Wi-Fi", "24-hr in-room dining"],
                "room_types": {"river_view_cottage": 180, "river_cottage_premium": 230, "suite": 380},
                "reviews": {"tripadvisor_rating": 4.6, "highlights": ["most beautiful setting in Kullu–Manali valley", "spa is exceptional", "heated pool in December is magical"]},
                "cancellation_policy": "Free cancellation 72h before arrival.",
                "notes": "40 min south of Manali town; good base for Solang Valley and Rohtang Pass. Accessible by private car.",
            },
            {
                "name": "Drifters Inn",
                "stars": 2, "price_per_night_usd": 22,
                "area": "Old Manali — backpacker hub",
                "address": "Old Manali Road, Manali, Himachal Pradesh 175131",
                "amenities": ["Rooftop café with Himalayan views", "Free Wi-Fi", "Common room with bonfire", "Laundry service ₹100/load", "Trek booking assistance"],
                "room_types": {"dorm_8bed": 8, "private_double": 22, "private_triple_balcony": 32},
                "reviews": {"tripadvisor_rating": 4.5, "highlights": ["best budget stay in Manali", "stunning mountain views from rooftop", "amazing community of travellers"]},
                "notes": "Heart of Old Manali village; surrounded by cafés, reggae joints, and trek outfitters.",
            },
            {
                "name": "Manu Allaya Resort",
                "stars": 4, "price_per_night_usd": 95,
                "area": "Central Manali — on the Mall Road",
                "address": "The Mall Road, Manali, Himachal Pradesh 175131",
                "amenities": ["Mountain views", "Indoor heated pool", "Restaurant (Himachali and continental cuisine)", "Free Wi-Fi", "Bonfire evenings"],
                "room_types": {"standard": 95, "deluxe_mountain_view": 130, "suite": 200},
                "reviews": {"tripadvisor_rating": 4.4},
            },
        ],
        "restaurants": [
            {
                "name": "Johnson's Café",
                "cuisine": "Continental and Himachali",
                "price_range": "$$ (meal ₹600–900)",
                "address": "Circuit House Road, Manali",
                "opening_hours": "Daily 08:00–22:00",
                "dietary": {"shellfish": "None", "vegetarian": "Extensive vegetarian menu"},
                "notes": "Most famous restaurant in Manali for 30+ years. Apple crumble dessert is legendary.",
            },
            {
                "name": "Drifters Café",
                "cuisine": "Multi-cuisine backpacker café",
                "price_range": "$ (meal ₹200–450)",
                "address": "Old Manali",
                "dietary": {"shellfish": "None", "vegetarian": "Extensive vegan and vegetarian options"},
                "notes": "Hash browns, Israeli salads, pasta, Tibetan momos — the classic Old Manali travellers' menu.",
            },
            {
                "name": "Lazy Dog Lounge",
                "cuisine": "Snacks, cocktails, bonfire nights",
                "price_range": "$ (cocktails ₹350–600)",
                "address": "Old Manali",
                "notes": "Best sunset views from the terrace. Live music Thursday and Saturday evenings.",
                "dietary": {"shellfish": "None", "vegetarian": "Good options"},
            },
        ],
        "attractions": [
            {
                "name": "Solang Valley",
                "type": "Adventure valley — skiing (winter) / paragliding (summer)",
                "entry_usd": 0,
                "distance_from_manali_km": 14,
                "activities": {
                    "winter_nov_mar": ["skiing (equipment hire ₹500/hr)", "snowboarding", "snow zorbing ₹200/ride"],
                    "summer_apr_oct": ["paragliding (₹2500 tandem flight)", "zorbing", "ATV rides"],
                },
                "tips": ["Hire a taxi from Manali ₹800 return; shared cab ₹200", "Go on weekdays to avoid Delhi crowds"],
                "duration_recommended_hrs": 4,
            },
            {
                "name": "Rohtang Pass",
                "type": "High-altitude mountain pass",
                "entry_usd": 5,
                "altitude_m": 3978,
                "distance_from_manali_km": 51,
                "opening_months": "June–October (closed in winter and during BRO maintenance days)",
                "highlights": ["Panoramic Himalayan views at 4000m", "Gateway to Lahaul-Spiti", "Snow even in June (check conditions)"],
                "permits": "Online permit required from Himachal Pradesh tourism (₹500, available at rotangpass.hp.gov.in); only 800 vehicles/day allowed",
                "tips": ["Rent warm clothing in Manali market (₹200/day) — it's cold at top regardless of season", "Hire a Jeep/SUV only (not cars)"],
                "duration_recommended_hrs": 8,
            },
            {
                "name": "Hadimba Temple (Dhungri Temple)",
                "type": "Hindu temple — 1553 CE",
                "entry_usd": 0,
                "address": "Hadimba Temple Road, Manali",
                "opening_hours": "Daily 08:00–20:00",
                "highlights": ["Four-tiered pagoda woodcraft architecture", "Deodar cedar forest surrounds", "Genuine Himalayan village atmosphere"],
                "duration_recommended_hrs": 1,
            },
        ],
    },

    # ── DELHI ─────────────────────────────────────────────────────────────
    "delhi": {
        "hotels": [
            {
                "name": "The Leela Palace New Delhi",
                "stars": 5, "price_per_night_usd": 380,
                "area": "Diplomatic Enclave, Chanakyapuri",
                "address": "Diplomatic Enclave, Chanakyapuri, New Delhi 110023",
                "amenities": ["Spa by ESPA", "5 restaurants including Jamavar (Michelin Bib Gourmand)", "Rooftop bar", "Heated pool", "Butler service", "Free Wi-Fi"],
                "room_types": {"royal_room": 380, "premier_room_garden": 470, "suite": 900},
                "reviews": {"tripadvisor_rating": 4.8, "highlights": ["best hotel in Delhi", "extraordinary dining", "impeccable service"]},
                "cancellation_policy": "72h free cancellation.",
            },
            {
                "name": "Zostel Delhi",
                "stars": 2, "price_per_night_usd": 10,
                "area": "Paharganj — backpacker district near New Delhi Railway Station",
                "address": "4771 Arakashan Road, Ram Nagar, Paharganj, New Delhi 110055",
                "amenities": ["Free Wi-Fi", "Common room", "Travel desk", "Free breakfast (basic)", "Laundry service"],
                "room_types": {"dorm_8bed": 6, "private_double": 10},
                "reviews": {"tripadvisor_rating": 4.4, "highlights": ["amazing social atmosphere", "unbeatable Delhi location", "great staff for recommendations"]},
                "notes": "Flagship Zostel property; very lively and social. Ideal for solo travellers.",
            },
        ],
        "restaurants": [
            {
                "name": "Karim's (Old Delhi)",
                "cuisine": "Mughlai — kebabs, nihari, biryani",
                "price_range": "$ (meal ₹400–700)",
                "address": "16, Gali Kababian, Near Jama Masjid, Old Delhi 110006",
                "opening_hours": "Daily 09:00–00:00",
                "dietary": {"shellfish": "None", "vegetarian": "Limited — meat-heavy menu"},
                "notes": "Founded 1913. Serving Mughal-descended recipes for 100+ years. The mutton burra kebab and nihari are unmissable.",
            },
            {
                "name": "Bukhara (ITC Maurya Hotel)",
                "cuisine": "Frontier Indian — tandoor and kebabs",
                "michelin": "Not rated (India has no Michelin yet) — but considered by many India's best restaurant",
                "price_range": "$$$ (avg ₹3500 / ~$42 per person)",
                "address": "ITC Maurya, Sardar Patel Marg, Diplomatic Enclave, New Delhi",
                "dietary": {"shellfish": "None", "vegetarian": "Dal Bukhara (famous 18-hr black lentils) is vegetarian and extraordinary"},
                "notes": "Bill Clinton, Vladimir Putin, Barack Obama have all dined here. No cutlery — eat with hands by tradition.",
            },
        ],
        "attractions": [
            {
                "name": "Red Fort (Lal Qila)",
                "type": "UNESCO World Heritage Site — Mughal fort",
                "entry_usd": 10,
                "address": "Netaji Subhash Marg, Lal Qila, Chandni Chowk, New Delhi",
                "opening_hours": "Tue–Sun 09:30–16:30; closed Mon",
                "highlights": ["17th-century Mughal palace complex", "Sound-and-light show evenings (₹80)", "Diwan-i-Aam and Diwan-i-Khas audience halls"],
                "duration_recommended_hrs": 2,
            },
            {
                "name": "Chandni Chowk",
                "type": "Historic bazaar and street food market",
                "entry_usd": 0,
                "address": "Old Delhi, Delhi",
                "opening_hours": "Daily 10:00–20:00 (most stalls)",
                "highlights": ["1.5 km of silver jewellery, spices, textiles", "Paranthe Wali Gali — alley of stuffed parathas", "Jama Masjid (India's largest mosque) nearby"],
                "tips": ["Go on weekday morning; Sunday is busiest", "Hire a cycle-rickshaw (₹50 for 30-min tour)"],
            },
        ],
    },
}


def places_search(city: str, category: str = "hotels") -> dict:
    """
    Returns hotels / restaurants / attractions for a city.
    Large bloated payload covering full amenities, pricing, reviews, policies.
    """
    key = city.strip().lower()
    city_data = _PLACES_DB.get(key, {})

    if not city_data:
        # Fuzzy fallback
        for db_key in _PLACES_DB:
            if db_key in key or key in db_key:
                city_data = _PLACES_DB[db_key]
                key = db_key
                break

    if not city_data:
        return {
            "city": city,
            "category": category,
            "error": "No data for this city yet.",
            "available_cities": list(_PLACES_DB.keys()),
        }

    cat = category.strip().lower()
    results = city_data.get(cat, [])
    if not results:
        cat = "hotels"
        results = city_data.get("hotels", [])

    return {
        "city": key,
        "category": cat,
        "result_count": len(results),
        "results": results,
        "metadata": {
            "source": "mock_places_db_v1",
            "note": "Prices are indicative (Apr 2026). Always verify directly with venue.",
            "currency": "USD unless stated.",
        },
    }


# ===========================================================================
# TOOL 3 — weather_fetch
# ===========================================================================
# Full 7-day forecast + hourly snapshot + historical monthly averages +
# clothing/packing checklist + health advisories + event calendar.
# Covers: Paris, Bali, Tokyo, Amsterdam, Berlin, Dehradun, Manali, Delhi.
# ===========================================================================

_WEATHER_DB: dict[str, dict[str, dict]] = {
    "paris": {
        "april": {
            "summary": "Spring in Paris — mild and occasionally rainy. Cherry blossoms peak first two weeks. Daylight 13h.",
            "temperature": {"avg_high_c": 16, "avg_low_c": 8, "record_high_c": 28, "record_low_c": -3, "feels_like_note": "Wind chill can make 14°C feel like 9°C"},
            "precipitation": {"avg_rain_days": 9, "avg_rainfall_mm": 45, "note": "Short showers rather than all-day rain; carry a compact umbrella"},
            "humidity_pct": 72,
            "wind": {"avg_kmh": 15, "direction": "SW", "note": "Can be blustery on open spaces like Champ de Mars"},
            "uv_index": {"avg": 4, "max": 6, "note": "Moderate UV — wear SPF30+ on sunny afternoons"},
            "sunshine_hours_per_day": 5.2,
            "air_quality": {"aqi": 42, "level": "Good", "note": "Paris air quality generally good; worse near ring road (Périphérique)"},
            "7_day_forecast": [
                {"day": "Mon", "condition": "Partly cloudy", "high_c": 15, "low_c": 7, "rain_prob_pct": 20, "wind_kmh": 12},
                {"day": "Tue", "condition": "Sunny", "high_c": 18, "low_c": 9, "rain_prob_pct": 5, "wind_kmh": 8},
                {"day": "Wed", "condition": "Light rain showers", "high_c": 13, "low_c": 8, "rain_prob_pct": 70, "wind_kmh": 18},
                {"day": "Thu", "condition": "Cloudy", "high_c": 14, "low_c": 7, "rain_prob_pct": 40, "wind_kmh": 14},
                {"day": "Fri", "condition": "Sunny spells", "high_c": 17, "low_c": 9, "rain_prob_pct": 15, "wind_kmh": 10},
                {"day": "Sat", "condition": "Mostly sunny", "high_c": 19, "low_c": 10, "rain_prob_pct": 10, "wind_kmh": 7},
                {"day": "Sun", "condition": "Overcast", "high_c": 12, "low_c": 6, "rain_prob_pct": 55, "wind_kmh": 20},
            ],
            "hourly_snapshot_day3": [
                {"hour": "06:00", "temp_c": 8, "condition": "Drizzle", "wind_kmh": 14},
                {"hour": "09:00", "temp_c": 10, "condition": "Light rain", "wind_kmh": 16},
                {"hour": "12:00", "temp_c": 13, "condition": "Showers", "wind_kmh": 18},
                {"hour": "15:00", "temp_c": 13, "condition": "Cloudy", "wind_kmh": 16},
                {"hour": "18:00", "temp_c": 11, "condition": "Clearing", "wind_kmh": 12},
                {"hour": "21:00", "temp_c": 9, "condition": "Partly cloudy", "wind_kmh": 10},
            ],
            "packing_recommendations": [
                "Layers essential — mornings are cold (8°C), afternoons can reach 18°C",
                "Waterproof jacket or compact umbrella — rain is common but usually brief",
                "Comfortable walking shoes — Paris is a walking city, cobblestones on historic streets",
                "Light scarf for wind on open spaces (Trocadero, Eiffel Tower plaza)",
                "Sunglasses for sunny spells",
                "Smart-casual outfit for restaurant dinners (Paris restaurants can be dressy)",
            ],
            "events_this_month": [
                {"name": "Paris Marathon", "date": "April 6", "note": "Road closures throughout central Paris — plan transit carefully"},
                {"name": "Nuit des Musées (Museum Night)", "date": "Third Saturday of May — preview weekend events in April", "note": "Many museums open late / free"},
                {"name": "Cherry Blossom in Jardins du Champ-de-Mars and Parc de Sceaux", "date": "Early–mid April typically", "note": "Check current bloom at cerisiers.fr"},
            ],
        },
        "august": {
            "summary": "Peak summer — hot and crowded. Many Parisians leave; tourist crowds fill the gap. Bastille Day fireworks July 14.",
            "temperature": {"avg_high_c": 26, "avg_low_c": 16, "record_high_c": 42, "note": "2019 heatwave hit 42.6°C — rare but possible; many buildings lack AC"},
            "precipitation": {"avg_rain_days": 7, "avg_rainfall_mm": 60},
            "humidity_pct": 65,
            "uv_index": {"avg": 7, "max": 9, "note": "High UV — wear SPF50+ and reapply"},
            "7_day_forecast": [
                {"day": "Mon", "condition": "Sunny", "high_c": 28, "low_c": 17, "rain_prob_pct": 5},
                {"day": "Tue", "condition": "Sunny", "high_c": 30, "low_c": 18, "rain_prob_pct": 5},
                {"day": "Wed", "condition": "Partly cloudy", "high_c": 25, "low_c": 16, "rain_prob_pct": 15},
                {"day": "Thu", "condition": "Thunderstorm possible", "high_c": 22, "low_c": 15, "rain_prob_pct": 60},
                {"day": "Fri", "condition": "Sunny", "high_c": 27, "low_c": 16, "rain_prob_pct": 5},
                {"day": "Sat", "condition": "Sunny", "high_c": 29, "low_c": 18, "rain_prob_pct": 5},
                {"day": "Sun", "condition": "Very hot", "high_c": 33, "low_c": 20, "rain_prob_pct": 5},
            ],
            "packing_recommendations": [
                "Light summer clothing — linen and cotton ideal",
                "SPF50+ sunscreen and sunglasses",
                "Portable fan or cooling spray (many pharmacies sell them)",
                "Note: Most of Paris lacks air conditioning — request AC room explicitly",
            ],
        },
    },

    "bali": {
        "june": {
            "summary": "Best time to visit Bali — peak dry season. Perfect for beaches (Seminyak, Nusa Dua) and trekking (Mount Batur). Busy tourist season.",
            "temperature": {"avg_high_c": 28, "avg_low_c": 22, "record_high_c": 34},
            "precipitation": {"avg_rain_days": 4, "avg_rainfall_mm": 70, "note": "Occasional short tropical shower; nothing like the wet season"},
            "humidity_pct": 74,
            "uv_index": {"avg": 8, "max": 11, "note": "Very high UV — wear SPF50+, reapply after swimming"},
            "sea_water_temp_c": 27,
            "7_day_forecast": [
                {"day": "Mon", "condition": "Sunny", "high_c": 29, "low_c": 22, "rain_prob_pct": 5, "surf_m": 1.2},
                {"day": "Tue", "condition": "Mostly sunny", "high_c": 28, "low_c": 22, "rain_prob_pct": 10, "surf_m": 1.0},
                {"day": "Wed", "condition": "Sunny", "high_c": 30, "low_c": 23, "rain_prob_pct": 5, "surf_m": 0.8},
                {"day": "Thu", "condition": "Partly cloudy", "high_c": 27, "low_c": 21, "rain_prob_pct": 20, "surf_m": 1.5},
                {"day": "Fri", "condition": "Sunny", "high_c": 29, "low_c": 22, "rain_prob_pct": 5, "surf_m": 1.3},
                {"day": "Sat", "condition": "Mostly sunny", "high_c": 28, "low_c": 22, "rain_prob_pct": 10, "surf_m": 1.1},
                {"day": "Sun", "condition": "Sunny", "high_c": 30, "low_c": 23, "rain_prob_pct": 5, "surf_m": 0.9},
            ],
            "packing_recommendations": [
                "Light summer clothes — humidity means natural fabrics (cotton, linen) breathe best",
                "Reef-safe sunscreen (required by many hotels and recommended for coral protection)",
                "Sarong — required for temple entry (usually rented/loaned at gates, but bring your own)",
                "Insect repellent with DEET — dengue and malaria (northern Bali) risk",
                "Waterproof sandals for rice terrace walking",
                "Light cardigan for air-conditioned restaurants and temples",
                "Stomach medication (Imodium, ORS) — Bali belly affects many visitors",
            ],
            "health_advisories": [
                "Drink bottled water ONLY — tap water not safe",
                "Dengue fever risk — use repellent especially at dawn/dusk",
                "Rabies present in Bali dogs — do not approach stray animals; seek medical attention immediately if bitten",
                "Travel insurance strongly recommended — medical costs can be high for foreigners",
                "COVID vaccinations up to date — no longer required but recommended",
            ],
            "events_this_month": [
                {"name": "Bali Arts Festival (Pesta Kesenian Bali)", "dates": "Mid-June–mid-July at Taman Werdhi Budaya, Denpasar", "note": "Largest cultural event in Bali; free entry, extraordinary performances"},
                {"name": "Full Moon ceremonies at temples", "note": "Check Balinese calendar; Purnama (full moon) is a major monthly celebration"},
            ],
        },
        "january": {
            "summary": "Wet season — heavy tropical rains, especially afternoons. Cheaper hotels but some roads flood; northern Bali (Munduk, Lovina) very wet.",
            "temperature": {"avg_high_c": 30, "avg_low_c": 24},
            "precipitation": {"avg_rain_days": 20, "avg_rainfall_mm": 300, "note": "Often torrential afternoon downpours — plan indoor activities for 13:00–16:00"},
            "humidity_pct": 87,
            "7_day_forecast": [
                {"day": "Mon", "condition": "Heavy rain afternoon", "high_c": 30, "low_c": 24, "rain_prob_pct": 85},
                {"day": "Tue", "condition": "Thunderstorm afternoon", "high_c": 29, "low_c": 24, "rain_prob_pct": 90},
                {"day": "Wed", "condition": "Overcast with showers", "high_c": 28, "low_c": 23, "rain_prob_pct": 75},
                {"day": "Thu", "condition": "Sunny morning, heavy afternoon rain", "high_c": 31, "low_c": 24, "rain_prob_pct": 80},
                {"day": "Fri", "condition": "Heavy rain all day", "high_c": 27, "low_c": 23, "rain_prob_pct": 95},
                {"day": "Sat", "condition": "Partly cloudy morning, showers", "high_c": 30, "low_c": 24, "rain_prob_pct": 70},
                {"day": "Sun", "condition": "Thunderstorm", "high_c": 28, "low_c": 23, "rain_prob_pct": 90},
            ],
            "packing_recommendations": ["Waterproof poncho (essential, not optional)", "Quick-dry clothing", "Waterproof bag for electronics", "Antifungal powder (humidity causes skin issues)"],
            "travel_advisory": "Mount Batur trek often cancelled; Rohtang-equivalent Kintamani viewpoint often clouded. Consider dry season instead.",
        },
    },

    "tokyo": {
        "march": {
            "summary": "Cherry blossom season (sakura) — peak typically late March to early April. Hugely popular; book accommodation 3-6 months ahead.",
            "temperature": {"avg_high_c": 13, "avg_low_c": 5, "note": "Variable — can be cold early March (5°C nights), warmer by end (18°C days)"},
            "precipitation": {"avg_rain_days": 10, "avg_rainfall_mm": 120},
            "humidity_pct": 58,
            "uv_index": {"avg": 4, "max": 6},
            "7_day_forecast": [
                {"day": "Mon", "condition": "Sunny", "high_c": 14, "low_c": 5, "rain_prob_pct": 5, "sakura_bloom": "5% — early buds"},
                {"day": "Tue", "condition": "Partly cloudy", "high_c": 15, "low_c": 6, "rain_prob_pct": 20, "sakura_bloom": "10%"},
                {"day": "Wed", "condition": "Sunny", "high_c": 17, "low_c": 7, "rain_prob_pct": 5, "sakura_bloom": "30%"},
                {"day": "Thu", "condition": "Overcast", "high_c": 12, "low_c": 6, "rain_prob_pct": 40, "sakura_bloom": "50%"},
                {"day": "Fri", "condition": "Light rain", "high_c": 11, "low_c": 5, "rain_prob_pct": 65, "sakura_bloom": "60%"},
                {"day": "Sat", "condition": "Sunny — FULL BLOOM forecast", "high_c": 16, "low_c": 6, "rain_prob_pct": 5, "sakura_bloom": "95% PEAK"},
                {"day": "Sun", "condition": "Partly cloudy", "high_c": 15, "low_c": 7, "rain_prob_pct": 20, "sakura_bloom": "Full bloom"},
            ],
            "sakura_viewing_spots": [
                {"name": "Shinjuku Gyoen National Garden", "entry_yen": 500, "note": "Best curated mix of cherry varieties; stunning"},
                {"name": "Ueno Park", "entry_yen": 0, "note": "Most famous — thousands of picnicking (hanami) parties under 800 trees"},
                {"name": "Chidorigafuchi Moat", "entry_yen": 0, "note": "Row a boat (¥800/30min) under overhanging cherry blossom — iconic"},
                {"name": "Meguro River", "entry_yen": 0, "note": "Evening sakura — entire river lined with illuminated pink trees, street food stalls"},
            ],
            "packing_recommendations": [
                "Light down jacket (cold nights — 5°C)",
                "Comfortable walking shoes — Tokyo requires 15,000+ steps/day",
                "Light layers — wide temperature range across the day",
                "Rain jacket — spring rains common",
                "Compact umbrella (available everywhere in Japan for ¥500)",
                "Allergy medication — pollen season peaks in March (sugi cedar pollen very high)",
            ],
        },
        "july": {
            "summary": "Hot and humid — rainy season (tsuyu) ends mid-July; second half is hot and sunny. Fireworks festivals (hanabi taikai) throughout July–August.",
            "temperature": {"avg_high_c": 31, "avg_low_c": 23, "feels_like": "Feels like 38°C+ due to humidity"},
            "precipitation": {"avg_rain_days": 11, "avg_rainfall_mm": 154},
            "humidity_pct": 82,
            "uv_index": {"avg": 9, "max": 11, "note": "Very high — sunstroke risk; carry cooling spray"},
            "7_day_forecast": [
                {"day": "Mon", "condition": "Hot and humid", "high_c": 33, "low_c": 24, "rain_prob_pct": 20},
                {"day": "Tue", "condition": "Partly cloudy", "high_c": 31, "low_c": 23, "rain_prob_pct": 30},
                {"day": "Wed", "condition": "Thunderstorm", "high_c": 28, "low_c": 22, "rain_prob_pct": 75},
                {"day": "Thu", "condition": "Hot and sunny", "high_c": 34, "low_c": 25, "rain_prob_pct": 15},
                {"day": "Fri", "condition": "Hot and sunny", "high_c": 35, "low_c": 25, "rain_prob_pct": 10},
                {"day": "Sat", "condition": "Partly cloudy", "high_c": 32, "low_c": 24, "rain_prob_pct": 35},
                {"day": "Sun", "condition": "Hot and humid", "high_c": 33, "low_c": 24, "rain_prob_pct": 20},
            ],
            "packing_recommendations": [
                "Light breathable clothing — linen, quick-dry synthetics",
                "SPF50+ sunscreen — UV is extreme",
                "Cooling towel and portable fan (Japan sells excellent ones)",
                "ORS packets — sweat heavily; rehydrate constantly",
                "Pocket Wi-Fi if not already arranged",
            ],
        },
    },

    "amsterdam": {
        "may": {
            "summary": "One of the best months — tulip season (April–May), mild temperatures, long days. King's Day (April 27) often spills into early May.",
            "temperature": {"avg_high_c": 16, "avg_low_c": 8},
            "precipitation": {"avg_rain_days": 8, "avg_rainfall_mm": 55},
            "humidity_pct": 73,
            "7_day_forecast": [
                {"day": "Mon", "condition": "Sunny", "high_c": 17, "low_c": 8, "rain_prob_pct": 10},
                {"day": "Tue", "condition": "Cloudy", "high_c": 14, "low_c": 9, "rain_prob_pct": 40},
                {"day": "Wed", "condition": "Showers", "high_c": 12, "low_c": 7, "rain_prob_pct": 70},
                {"day": "Thu", "condition": "Sunny spells", "high_c": 15, "low_c": 8, "rain_prob_pct": 20},
                {"day": "Fri", "condition": "Sunny", "high_c": 18, "low_c": 9, "rain_prob_pct": 5},
                {"day": "Sat", "condition": "Mostly sunny", "high_c": 19, "low_c": 10, "rain_prob_pct": 10},
                {"day": "Sun", "condition": "Partly cloudy", "high_c": 16, "low_c": 8, "rain_prob_pct": 25},
            ],
            "packing_recommendations": [
                "Waterproof jacket — Dutch weather is unpredictable; never leave without one",
                "Comfortable walking shoes for cobblestones",
                "Layers — temperature varies widely through day",
                "Sunglasses for sunny spells",
                "Cycling gloves if planning to cycle (wind chill on bike in morning)",
            ],
            "events": [
                {"name": "Liberation Day (Bevrijdingsdag)", "date": "May 5", "note": "Free music festivals across Amsterdam; very festive"},
                {"name": "Keukenhof tulip gardens", "dates": "Late March–mid-May, Lisse (45 min from Amsterdam)", "note": "7 million tulips; book ticket + bus combo online"},
            ],
        },
    },

    "dehradun": {
        "may": {
            "summary": "Pre-monsoon season — pleasant mornings, hot afternoons. Gateway month for Char Dham Yatra (Kedarnath, Badrinath, Gangotri, Yamunotri pilgrimages).",
            "temperature": {"avg_high_c": 35, "avg_low_c": 18, "note": "Valley heat — Dehradun sits in bowl; noticeably hotter than Mussoorie (25 km uphill)"},
            "precipitation": {"avg_rain_days": 4, "avg_rainfall_mm": 35},
            "humidity_pct": 52,
            "7_day_forecast": [
                {"day": "Mon", "condition": "Hot and sunny", "high_c": 36, "low_c": 19, "rain_prob_pct": 5},
                {"day": "Tue", "condition": "Hot and sunny", "high_c": 37, "low_c": 20, "rain_prob_pct": 5},
                {"day": "Wed", "condition": "Partly cloudy", "high_c": 34, "low_c": 18, "rain_prob_pct": 15},
                {"day": "Thu", "condition": "Hot — pre-monsoon dust", "high_c": 38, "low_c": 21, "rain_prob_pct": 10},
                {"day": "Fri", "condition": "Thunderstorm evening", "high_c": 33, "low_c": 19, "rain_prob_pct": 60},
                {"day": "Sat", "condition": "Partly cloudy, pleasant", "high_c": 30, "low_c": 17, "rain_prob_pct": 20},
                {"day": "Sun", "condition": "Sunny and warm", "high_c": 34, "low_c": 18, "rain_prob_pct": 5},
            ],
            "packing_recommendations": [
                "Light cotton clothes for valley daytime",
                "Fleece or warm layer for Mussoorie day-trip (17°C at 2000m)",
                "Sunscreen SPF30+",
                "ORS / electrolytes for heat",
                "Sturdy shoes for Robbers Cave wading",
                "Light rain jacket for evening thunderstorms",
            ],
            "nearby_hill_stations": [
                {"name": "Mussoorie", "distance_km": 34, "altitude_m": 2005, "avg_temp_may_c": 17, "note": "Queen of the Hills; cooler escape; 1 hr by taxi from Dehradun"},
                {"name": "Chakrata", "distance_km": 88, "altitude_m": 2118, "note": "Quieter alternative; Tiger Falls (Uttarakhand's highest waterfall) nearby"},
            ],
        },
    },

    "manali": {
        "june": {
            "summary": "Post-snow season — snow melting, Rohtang Pass reopens. Perfect for adventure (trekking, paragliding at Solang). Apple trees in blossom.",
            "temperature": {"avg_high_c": 22, "avg_low_c": 8, "altitude_note": "At 2050m — significantly cooler than plains; drop 6°C per 1000m gain"},
            "precipitation": {"avg_rain_days": 6, "avg_rainfall_mm": 60, "note": "Pre-monsoon showers; monsoon proper arrives mid-July and can trigger landslides"},
            "7_day_forecast": [
                {"day": "Mon", "condition": "Sunny and cool", "high_c": 22, "low_c": 7, "rain_prob_pct": 10},
                {"day": "Tue", "condition": "Sunny", "high_c": 23, "low_c": 8, "rain_prob_pct": 5},
                {"day": "Wed", "condition": "Partly cloudy", "high_c": 20, "low_c": 9, "rain_prob_pct": 25},
                {"day": "Thu", "condition": "Afternoon thunderstorm", "high_c": 18, "low_c": 8, "rain_prob_pct": 65},
                {"day": "Fri", "condition": "Clear and sunny", "high_c": 24, "low_c": 7, "rain_prob_pct": 5},
                {"day": "Sat", "condition": "Sunny", "high_c": 23, "low_c": 8, "rain_prob_pct": 10},
                {"day": "Sun", "condition": "Partly cloudy", "high_c": 21, "low_c": 9, "rain_prob_pct": 20},
            ],
            "rohtang_pass_conditions": {
                "status": "Open (check road authority bulletin day before)",
                "snow_at_pass_m": 1.5,
                "road_condition": "Slippery in early June; 4WD recommended",
                "permit_required": True,
            },
            "packing_recommendations": [
                "Warm layers essential — nights drop to 7°C even in June",
                "Waterproof jacket for afternoon showers",
                "Thermal base layer for Rohtang Pass and Solang Valley",
                "Trekking boots with ankle support",
                "Sunscreen SPF50+ — UV at altitude is intense even when cool",
                "Altitude sickness medication (Diamox) if going above 3500m — consult doctor first",
                "Gloves and warm hat for Rohtang Pass",
            ],
            "altitude_sickness_advisory": "Manali is at 2050m — most people feel fine. Rohtang is 3978m — take it slow, drink water, no alcohol on first day at altitude.",
        },
    },
}


def weather_fetch(city: str, month: str = "current") -> dict:
    """
    Returns detailed weather data including 7-day forecast, hourly snapshot,
    packing recommendations, health advisories, and events.
    """
    city_key = city.strip().lower()
    month_key = month.strip().lower()

    # Normalise month names
    month_map = {
        "jan": "january", "feb": "february", "mar": "march",
        "apr": "april", "may": "may", "jun": "june",
        "jul": "july", "aug": "august", "sep": "september",
        "oct": "october", "nov": "november", "dec": "december",
    }
    for short, full in month_map.items():
        if short in month_key:
            month_key = full
            break

    city_data = _WEATHER_DB.get(city_key, {})
    if not city_data:
        for db_key in _WEATHER_DB:
            if db_key in city_key or city_key in db_key:
                city_data = _WEATHER_DB[db_key]
                city_key = db_key
                break

    if not city_data:
        # Fallback to paris april
        city_data = _WEATHER_DB["paris"]
        city_key = "paris"

    month_data = city_data.get(month_key)
    if not month_data:
        # Return first available month
        first_month = list(city_data.keys())[0]
        month_data = city_data[first_month]
        month_key = first_month

    return {
        "city": city_key,
        "month_requested": month,
        "month_data_returned": month_key,
        "weather": month_data,
        "metadata": {
            "source": "mock_weather_db_v1",
            "note": "Forecasts are illustrative; always check a live weather service before travel.",
            "data_as_of": "2026-04-17",
        },
    }


# ===========================================================================
# TOOL 4 — budget_tracker
# ===========================================================================
# Stateful ledger: supports add_expense, get_summary, reset, set_budget.
# Returns full itemized breakdown, per-city totals, category totals,
# remaining budget, recommendations, and exchange-rate context.
# ===========================================================================

# In-process state: a single shared ledger (sufficient for one conversation).
_budget_state: dict[str, Any] = {
    "total_budget_usd": 0.0,
    "currency": "USD",
    "travellers": 1,
    "trip_name": "Unnamed trip",
    "expenses": [],   # list of expense records
    "notes": [],
}


def budget_tracker(
    action: str,
    amount: float = 0.0,
    category: str = "other",
    city: str = "",
    description: str = "",
    total_budget: float = 0.0,
    travellers: int = 1,
    trip_name: str = "",
    currency: str = "USD",
) -> dict:
    """
    Stateful budget tracker for a multi-city trip.

    Actions
    -------
    set_budget   : initialise or update total_budget, travellers, trip_name
    add_expense  : add a line item; amount in USD
    get_summary  : return full ledger + per-city and per-category totals
    reset        : clear all expenses (keeps budget setting)
    """
    global _budget_state

    # ── set_budget ───────────────────────────────────────────────────────
    if action == "set_budget":
        _budget_state["total_budget_usd"] = float(total_budget) if total_budget else _budget_state["total_budget_usd"]
        _budget_state["travellers"] = max(1, travellers)
        _budget_state["currency"] = currency.upper()
        if trip_name:
            _budget_state["trip_name"] = trip_name
        return {
            "action": "set_budget",
            "status": "ok",
            "budget_set_usd": _budget_state["total_budget_usd"],
            "travellers": _budget_state["travellers"],
            "trip_name": _budget_state["trip_name"],
            "per_person_usd": round(_budget_state["total_budget_usd"] / _budget_state["travellers"], 2),
        }

    # ── add_expense ───────────────────────────────────────────────────────
    if action == "add_expense":
        expense = {
            "id": len(_budget_state["expenses"]) + 1,
            "amount_usd": round(float(amount), 2),
            "category": category.lower().strip(),
            "city": city.strip() or "unspecified",
            "description": description or f"{category} expense",
        }
        _budget_state["expenses"].append(expense)
        total_spent = sum(e["amount_usd"] for e in _budget_state["expenses"])
        remaining = _budget_state["total_budget_usd"] - total_spent
        budget = _budget_state["total_budget_usd"]

        # Budget health check
        if budget > 0:
            spent_pct = (total_spent / budget) * 100
            if spent_pct >= 100:
                status = "OVER_BUDGET"
                warning = f"ALERT: You are ${abs(remaining):.2f} OVER budget!"
            elif spent_pct >= 85:
                status = "critical"
                warning = f"WARNING: {spent_pct:.1f}% of budget used. Only ${remaining:.2f} remaining."
            elif spent_pct >= 70:
                status = "caution"
                warning = f"NOTE: {spent_pct:.1f}% of budget used. ${remaining:.2f} remaining."
            else:
                status = "healthy"
                warning = None
        else:
            status = "no_budget_set"
            warning = "No total budget set — call set_budget first."

        result = {
            "action": "add_expense",
            "expense_added": expense,
            "running_total_spent_usd": round(total_spent, 2),
            "total_budget_usd": budget,
            "remaining_usd": round(remaining, 2),
            "budget_status": status,
            "total_expenses_count": len(_budget_state["expenses"]),
        }
        if warning:
            result["warning"] = warning
        return result

    # ── get_summary ───────────────────────────────────────────────────────
    if action == "get_summary":
        expenses = _budget_state["expenses"]
        total_spent = sum(e["amount_usd"] for e in expenses)
        budget = _budget_state["total_budget_usd"]
        remaining = budget - total_spent
        travellers = _budget_state["travellers"]

        # Per-category totals
        categories: dict[str, float] = {}
        for e in expenses:
            categories[e["category"]] = categories.get(e["category"], 0.0) + e["amount_usd"]

        # Per-city totals
        cities: dict[str, dict] = {}
        for e in expenses:
            c = e["city"]
            if c not in cities:
                cities[c] = {"total_usd": 0.0, "expense_count": 0, "categories": {}}
            cities[c]["total_usd"] = round(cities[c]["total_usd"] + e["amount_usd"], 2)
            cities[c]["expense_count"] += 1
            cities[c]["categories"][e["category"]] = round(
                cities[c]["categories"].get(e["category"], 0.0) + e["amount_usd"], 2
            )

        # Budget health
        if budget > 0:
            spent_pct = (total_spent / budget) * 100
            if spent_pct >= 100:
                health = "OVER_BUDGET"
                health_note = f"Over budget by ${abs(remaining):.2f}"
            elif spent_pct >= 85:
                health = "critical"
                health_note = f"Only ${remaining:.2f} left ({100-spent_pct:.1f}% of budget)"
            elif spent_pct >= 70:
                health = "caution"
                health_note = f"${remaining:.2f} remaining — budget tight"
            elif spent_pct >= 50:
                health = "moderate"
                health_note = f"${remaining:.2f} remaining — on track"
            else:
                health = "healthy"
                health_note = f"${remaining:.2f} remaining — plenty of budget left"
        else:
            health = "no_budget_set"
            health_note = "Set a budget with action='set_budget'"

        # Recommendations based on remaining budget
        recommendations = []
        if budget > 0 and remaining > 0:
            daily_remaining_estimate = remaining / max(1, 3)  # rough 3-day assumption
            recommendations.append(f"Estimated daily remaining budget: ~${daily_remaining_estimate:.0f}/day (assuming 3 days left)")
            if remaining < 200:
                recommendations.append("Consider switching to budget accommodation (hostels, guesthouses) for remaining stay.")
                recommendations.append("Use local public transport instead of taxis.")
                recommendations.append("Eat at local markets and street food stalls (50-70% cheaper than restaurants).")
            elif remaining < 500:
                recommendations.append("Choose 3-star hotels rather than 4-star for remaining cities.")
                recommendations.append("Limit fine dining to 1 special meal; eat local otherwise.")
            else:
                recommendations.append("Budget is in good shape — room for upgrades if desired.")

        # Exchange rate context
        exchange_rates = {
            "EUR": {"rate": 0.93, "note": "Paris, Amsterdam, Berlin"},
            "JPY": {"rate": 154.0, "note": "Tokyo, Kyoto"},
            "IDR": {"rate": 16200, "note": "Bali"},
            "INR": {"rate": 83.5, "note": "India (Delhi, Dehradun, Manali)"},
            "CHF": {"rate": 0.90, "note": "Switzerland (Zurich, Geneva)"},
        }

        return {
            "action": "get_summary",
            "trip_name": _budget_state["trip_name"],
            "travellers": travellers,
            "total_budget_usd": budget,
            "total_spent_usd": round(total_spent, 2),
            "remaining_usd": round(remaining, 2),
            "per_person_spent_usd": round(total_spent / travellers, 2),
            "per_person_remaining_usd": round(remaining / travellers, 2),
            "budget_health": health,
            "budget_health_note": health_note,
            "spent_percentage": round((total_spent / budget * 100), 1) if budget > 0 else None,
            "by_category": {k: round(v, 2) for k, v in sorted(categories.items(), key=lambda x: -x[1])},
            "by_city": cities,
            "all_expenses": expenses,
            "recommendations": recommendations,
            "exchange_rates_apr2026": exchange_rates,
            "metadata": {
                "source": "mock_budget_tracker_v1",
                "note": "All amounts in USD. Exchange rates approximate as of April 2026.",
            },
        }

    # ── reset ─────────────────────────────────────────────────────────────
    if action == "reset":
        _budget_state["expenses"] = []
        _budget_state["notes"] = []
        return {
            "action": "reset",
            "status": "ok",
            "message": "All expenses cleared. Budget settings retained.",
            "total_budget_usd": _budget_state["total_budget_usd"],
        }

    return {"error": f"Unknown action '{action}'. Use: set_budget, add_expense, get_summary, reset"}


# ===========================================================================
# Tool dispatcher
# ===========================================================================

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for flight options, travel info, visa requirements, "
                "local transport, and destination guides. Use for any query about "
                "getting to a place or general destination information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query, e.g. 'flights Delhi to Paris May 2026' "
                            "or 'visa requirements India to Japan'"
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "places_search",
            "description": (
                "Search for hotels, restaurants, or attractions in a specific city. "
                "Returns detailed listings with prices, reviews, amenities, and policies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'Paris', 'Tokyo', 'Bali', 'Amsterdam', 'Berlin', 'Dehradun', 'Manali', 'Delhi'",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["hotels", "restaurants", "attractions"],
                        "description": "What to search for",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "weather_fetch",
            "description": (
                "Fetch detailed weather forecast and travel conditions for a city "
                "in a given month, including 7-day forecast, packing recommendations, "
                "health advisories, and local events."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name",
                    },
                    "month": {
                        "type": "string",
                        "description": "Month name or abbreviation, e.g. 'june', 'jul', 'december'",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "budget_tracker",
            "description": (
                "Track trip budget across cities and categories. "
                "Supports: set_budget (initialise), add_expense (log a cost), "
                "get_summary (full breakdown), reset (clear expenses)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["set_budget", "add_expense", "get_summary", "reset"],
                        "description": "What to do",
                    },
                    "amount": {
                        "type": "number",
                        "description": "Expense amount in USD (for add_expense)",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["flights", "hotels", "food", "activities", "transport", "visa", "insurance", "shopping", "other"],
                        "description": "Expense category (for add_expense)",
                    },
                    "city": {
                        "type": "string",
                        "description": "City where expense was incurred (for add_expense)",
                    },
                    "description": {
                        "type": "string",
                        "description": "Free-text description of the expense",
                    },
                    "total_budget": {
                        "type": "number",
                        "description": "Total trip budget in USD (for set_budget)",
                    },
                    "travellers": {
                        "type": "integer",
                        "description": "Number of travellers (for set_budget)",
                    },
                    "trip_name": {
                        "type": "string",
                        "description": "Name for this trip (for set_budget)",
                    },
                },
                "required": ["action"],
            },
        },
    },
]


def dispatch_tool(name: str, args: dict) -> str:
    """Route tool call to the appropriate function and return JSON string."""
    try:
        if name == "web_search":
            result = web_search(**args)
        elif name == "places_search":
            result = places_search(**args)
        elif name == "weather_fetch":
            result = weather_fetch(**args)
        elif name == "budget_tracker":
            result = budget_tracker(**args)
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as exc:
        result = {"error": str(exc), "tool": name, "args": args}
    return _json(result)
