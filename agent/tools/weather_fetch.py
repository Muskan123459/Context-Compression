"""
TOOL 3 — weather_fetch

Full 7-day forecast + hourly snapshot + historical monthly averages +
clothing/packing checklist + health advisories + event calendar.
Covers: Paris, Bali, Tokyo, Amsterdam, Berlin, Dehradun, Manali, Delhi.
"""
from __future__ import annotations


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
        city_data = _WEATHER_DB["paris"]
        city_key = "paris"

    month_data = city_data.get(month_key)
    if not month_data:
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


SCHEMA: dict = {
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
}