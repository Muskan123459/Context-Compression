"""
tools/weather_fetch.py — Live weather with 3-tier fallback.

Tier 1: wttr.in JSON API         (fast, rich, no key)
Tier 2: Open-Meteo + geocoding   (backup when wttr is down)
Tier 3: Mock estimate             (always works)

Public interface:
    run(city: str) -> dict
"""

import json
import random
import urllib.parse
import urllib.request

SCHEMA = {
    "name": "weather_fetch",
    "description": (
        "Get current weather conditions and a 3-day forecast for any city. "
        "Use this for packing advice, activity planning, and weather-dependent decisions."
    ),
    "parameters": {
        "city": "string — city name (e.g. 'Paris', 'Tokyo', 'New York')",
    },
}

# ---------------------------------------------------------------------------
# WMO weather code → human description
# ---------------------------------------------------------------------------

_WMO_DESC: dict[int, str] = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Icy fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Heavy drizzle",
    61: "Light rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Light snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Light showers", 81: "Moderate showers", 82: "Heavy showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail",
}


def _wmo(code: int) -> str:
    return _WMO_DESC.get(code, f"Weather code {code}")


# ---------------------------------------------------------------------------
# Tier 1: wttr.in
# ---------------------------------------------------------------------------

def _wttr(city: str) -> dict | None:
    try:
        q   = urllib.parse.quote(city)
        url = f"https://wttr.in/{q}?format=j1"
        req = urllib.request.Request(url, headers={"User-Agent": "travel-agent/1.0"})
        raw = urllib.request.urlopen(req, timeout=5).read()
        d   = json.loads(raw)
        cur = d["current_condition"][0]
        return {
            "city":          city,
            "temperature_c": int(cur["temp_C"]),
            "temperature_f": int(cur["temp_F"]),
            "feels_like_c":  int(cur["FeelsLikeC"]),
            "description":   cur["weatherDesc"][0]["value"],
            "humidity_pct":  int(cur["humidity"]),
            "wind_kph":      int(cur["windspeedKmph"]),
            "uv_index":      int(cur.get("uvIndex", 0)),
            "source":        "wttr.in",
            "forecast_3day": [
                {
                    "date":        day["date"],
                    "max_c":       int(day["maxtempC"]),
                    "min_c":       int(day["mintempC"]),
                    "description": day["hourly"][4]["weatherDesc"][0]["value"],
                    "rain_chance": int(day["hourly"][4].get("chanceofrain", 0)),
                }
                for day in d["weather"]
            ],
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tier 2: Open-Meteo (geocode → forecast)
# ---------------------------------------------------------------------------

def _open_meteo(city: str) -> dict | None:
    try:
        # Geocode
        q       = urllib.parse.quote(city)
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={q}&count=1&language=en&format=json"
        geo_raw = urllib.request.urlopen(geo_url, timeout=5).read()
        geo     = json.loads(geo_raw)
        if not geo.get("results"):
            return None
        loc     = geo["results"][0]
        lat, lon = loc["latitude"], loc["longitude"]
        tz      = loc.get("timezone", "auto")

        # Forecast
        fc_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code,apparent_temperature"
            f"&daily=temperature_2m_max,temperature_2m_min,weather_code,precipitation_probability_max"
            f"&forecast_days=3&timezone={urllib.parse.quote(tz)}"
        )
        fc_raw  = urllib.request.urlopen(fc_url, timeout=5).read()
        fc      = json.loads(fc_raw)
        cur     = fc["current"]
        daily   = fc["daily"]

        temp_c  = round(cur["temperature_2m"])
        temp_f  = round(temp_c * 9 / 5 + 32)
        feel_c  = round(cur.get("apparent_temperature", temp_c))

        forecast = []
        for i in range(len(daily["time"])):
            forecast.append({
                "date":        daily["time"][i],
                "max_c":       round(daily["temperature_2m_max"][i]),
                "min_c":       round(daily["temperature_2m_min"][i]),
                "description": _wmo(daily["weather_code"][i]),
                "rain_chance": daily["precipitation_probability_max"][i],
            })

        return {
            "city":          city,
            "temperature_c": temp_c,
            "temperature_f": temp_f,
            "feels_like_c":  feel_c,
            "description":   _wmo(cur["weather_code"]),
            "humidity_pct":  cur["relative_humidity_2m"],
            "wind_kph":      round(cur["wind_speed_10m"]),
            "source":        "open-meteo",
            "forecast_3day": forecast,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tier 3: Mock estimate
# ---------------------------------------------------------------------------

_MOCK_CONDITIONS = [
    "Partly cloudy", "Sunny", "Overcast", "Light rain", "Clear sky",
]


def _mock(city: str) -> dict:
    temp_c = random.randint(10, 28)
    temp_f = round(temp_c * 9 / 5 + 32)
    return {
        "city":          city,
        "temperature_c": temp_c,
        "temperature_f": temp_f,
        "feels_like_c":  temp_c - random.randint(0, 3),
        "description":   random.choice(_MOCK_CONDITIONS),
        "humidity_pct":  random.randint(40, 80),
        "wind_kph":      random.randint(5, 30),
        "source":        "mock (all live sources unavailable)",
        "forecast_3day": [],
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(city: str) -> dict:
    """wttr.in → Open-Meteo → mock."""
    return _wttr(city) or _open_meteo(city) or _mock(city)
