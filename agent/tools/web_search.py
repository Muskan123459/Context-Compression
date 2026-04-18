"""
TOOL 1 — web_search

Fixture covers routes from/to: Delhi (DEL), Paris (CDG), Bali (DPS),
Tokyo (NRT/HND), Amsterdam (AMS), Berlin (BER), Dehradun (DED),
Himachal Pradesh / Bhuntar (KUU), Shimla (SLV — helicopter link).
Each result block is padded with realistic airline data, fare classes,
baggage rules, and booking metadata to reach ~2 500 tokens.
"""
from __future__ import annotations


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

    matched_flights: list[dict] = []
    for route_key, flights in _FLIGHT_DB.items():
        if all(word in q for word in route_key.split(" to ")):
            matched_flights = flights
            break
        if route_key.split(" to ")[-1] in q:
            matched_flights = flights
            break

    destination_info: dict = {}
    for city_key, info in _FLIGHT_GENERAL_INFO.items():
        if city_key in q:
            destination_info = info
            break

    if not matched_flights and not destination_info:
        matched_flights = _FLIGHT_DB.get("delhi to paris", [])
        destination_info = _FLIGHT_GENERAL_INFO.get("paris", {})

    return {
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


SCHEMA: dict = {
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
}