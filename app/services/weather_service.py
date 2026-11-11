"""
Weather integrations (Open-Meteo + OpenWeather) with mock fallbacks.
"""
from __future__ import annotations

import math
import os
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .system_defaults import DEFAULT_SYSTEM_CONFIG
from .weather_cache_service import weather_cache
from .weather_source_service import get_active_weather_data, get_active_weather_source

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
CACHE_TTL_SECONDS = 60

_cached_snapshot: Optional[Dict[str, Any]] = None


def _spanish_day_name(date: datetime) -> str:
    names = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    return names[date.weekday()].capitalize()


def _weather_code_to_condition(code: int) -> str:
    if code == 0:
        return "sunny"
    if code <= 3:
        return "partly-cloudy" if code == 1 else "cloudy"
    if code <= 48:
        return "cloudy"
    if code <= 67:
        return "rainy"
    return "rainy"


def _weather_description(code: int) -> str:
    descriptions = {
        0: "Cielo despejado",
        1: "Principalmente despejado",
        2: "Parcialmente nublado",
        3: "Nublado",
        45: "Niebla",
        48: "Niebla con escarcha",
        51: "Llovizna ligera",
        53: "Llovizna moderada",
        55: "Llovizna intensa",
        61: "Lluvia ligera",
        63: "Lluvia moderada",
        65: "Lluvia intensa",
        71: "Nevada ligera",
        73: "Nevada moderada",
        75: "Nevada intensa",
        95: "Tormenta eléctrica",
        96: "Tormenta con granizo ligero",
        99: "Tormenta con granizo intenso",
    }
    return descriptions.get(code, "Condiciones variables")


def _calculate_solar_production(radiation: float, capacity_kw: float, efficiency: float = 0.17) -> float:
    standard_radiation = 1000
    production_factor = radiation / standard_radiation
    return round(capacity_kw * production_factor * efficiency * 24, 2)


async def fetch_open_meteo_weather(
    lat: float,
    lon: float,
    capacity_kw: float,
    location_name: str = "Ubicación",
) -> Dict[str, Any]:
    # El JSON crudo de Open-Meteo (current + daily) viene del caché global, que
    # hace una sola petición cada 5 min y la comparte con toda la app.
    raw = await weather_cache.get_display_raw(lat, lon)
    return _build_display_payload(raw, capacity_kw, location_name)


def _build_display_payload(
    raw: Dict[str, Any], capacity_kw: float, location_name: str
) -> Dict[str, Any]:
    daily_data = raw
    forecast: List[Dict[str, Any]] = []
    for idx, date_str in enumerate(daily_data["daily"]["time"]):
        date_obj = datetime.fromisoformat(date_str)
        # Open-Meteo entrega shortwave_radiation_sum en MJ/m²/día.
        radiation_sum_mj = daily_data["daily"]["shortwave_radiation_sum"][idx] or 0.0
        avg_radiation_wm2 = radiation_sum_mj * 1_000_000 / 86_400  # -> W/m² (media 24 h)
        peak_sun_hours = radiation_sum_mj / 3.6                     # MJ/m² -> kWh/m²/día
        predicted_production = round(capacity_kw * peak_sun_hours * 0.75, 1)  # PR ~0.75
        forecast.append(
            {
                "date": date_str,
                "dayOfWeek": _spanish_day_name(date_obj),
                "maxTemp": daily_data["daily"]["temperature_2m_max"][idx],
                "minTemp": daily_data["daily"]["temperature_2m_min"][idx],
                "solarRadiation": round(avg_radiation_wm2),
                "cloudCover": round(daily_data["daily"]["cloud_cover_mean"][idx]),
                "predictedProduction": predicted_production,
                "condition": _weather_code_to_condition(daily_data["daily"]["weather_code"][idx]),
            }
        )

    current = raw["current"]
    is_mock = bool(raw.get("_is_mock"))
    return {
        "temperature": current["temperature_2m"],
        "solarRadiation": round(current["shortwave_radiation"]),
        "cloudCover": round(current["cloud_cover"]),
        "humidity": round(current["relative_humidity_2m"]),
        "windSpeed": current["wind_speed_10m"],
        "forecast": forecast,
        "provider": "Datos de prueba (sin conexión)" if is_mock else "Open-Meteo API",
        "locationName": location_name,
        "lastUpdated": datetime.utcnow().isoformat(),
        "description": (
            "Datos simulados por falta de conexión con el servicio meteorológico"
            if is_mock else _weather_description(current["weather_code"])
        ),
        "weatherCode": current["weather_code"],
        "isMock": is_mock,
    }


def _estimate_solar_radiation(uvi: float, clouds: float) -> float:
    base = min(1.0, max(0.0, uvi / 11))
    cloud_penalty = 1 - (clouds / 100) * 0.6
    return round(max(0.0, 950 * base * cloud_penalty))


def _estimate_daily_production(radiation: float, daylight_seconds: float, capacity_kw: float) -> float:
    daylight_hours = max(0.0, daylight_seconds / 3600)
    efficiency_factor = 0.72
    production = (radiation / 1000) * capacity_kw * daylight_hours * efficiency_factor
    return round(max(0.0, production), 1)


def _map_openweather_forecast(daily: List[Dict[str, Any]], solar_capacity_kw: float) -> List[Dict[str, Any]]:
    formatter = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    forecast = []
    for entry in daily[:7]:
        dt = datetime.fromtimestamp(entry["dt"])
        condition = entry.get("weather", [{}])[0].get("main")
        pop = entry.get("pop")
        cloud_cover = entry.get("clouds") or 0
        weather_condition = _map_weather_condition(condition, cloud_cover, pop)
        solar_radiation = _estimate_solar_radiation(entry.get("uvi") or 0, cloud_cover)
        forecast.append(
            {
                "date": datetime.fromtimestamp(entry["dt"]).isoformat(),
                "dayOfWeek": formatter[dt.weekday()].capitalize(),
                "maxTemp": entry["temp"]["max"],
                "minTemp": entry["temp"]["min"],
                "solarRadiation": solar_radiation,
                "cloudCover": cloud_cover,
                "predictedProduction": _estimate_daily_production(
                    solar_radiation,
                    entry["sunset"] - entry["sunrise"],
                    solar_capacity_kw,
                ),
                "condition": weather_condition,
            }
        )
    return forecast


def _map_weather_condition(main: Optional[str], clouds: Optional[float], pop: Optional[float]) -> str:
    if pop and pop >= 0.5:
        return "rainy"
    if not main:
        if (clouds or 0) < 20:
            return "sunny"
        if (clouds or 0) < 55:
            return "partly-cloudy"
        if (clouds or 0) < 80:
            return "cloudy"
        return "rainy"
    mapping = {
        "Clear": "sunny",
        "Clouds": "cloudy",
        "Rain": "rainy",
        "Drizzle": "rainy",
        "Thunderstorm": "rainy",
        "Snow": "rainy",
        "Mist": "partly-cloudy",
        "Smoke": "partly-cloudy",
        "Haze": "partly-cloudy",
        "Dust": "partly-cloudy",
        "Fog": "partly-cloudy",
        "Sand": "partly-cloudy",
        "Ash": "partly-cloudy",
        "Squall": "rainy",
        "Tornado": "rainy",
    }
    mapped = mapping.get(main)
    if mapped == "cloudy" and (clouds or 0) < 50:
        return "partly-cloudy"
    return mapped or ("sunny" if (clouds or 0) < 25 else "cloudy")


async def _fetch_openweather_snapshot(force: bool = False) -> Dict[str, Any]:
    global _cached_snapshot
    if not force and _cached_snapshot:
        age = datetime.utcnow().timestamp() - _cached_snapshot["fetchedAt"]
        if age < CACHE_TTL_SECONDS:
            return _cached_snapshot

    if not OPENWEATHER_API_KEY:
        raise RuntimeError("OPENWEATHER_API_KEY environment variable is not set")

    params = {
        "lat": DEFAULT_SYSTEM_CONFIG["location"]["lat"],
        "lon": DEFAULT_SYSTEM_CONFIG["location"]["lon"],
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
        "lang": "es",
        "exclude": "minutely,alerts",
    }

    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get("https://api.openweathermap.org/data/2.5/onecall", params=params)
    response.raise_for_status()

    payload = response.json()
    _cached_snapshot = {"raw": payload, "fetchedAt": datetime.utcnow().timestamp()}
    return _cached_snapshot


async def get_live_weather_data(solar_capacity_kw: float) -> Dict[str, Any]:
    snapshot = await _fetch_openweather_snapshot()
    payload = snapshot["raw"]
    current = payload["current"]
    solar_radiation = _estimate_solar_radiation(current.get("uvi") or 0, current.get("clouds") or 0)
    forecast = _map_openweather_forecast(payload["daily"], solar_capacity_kw)
    return {
        "temperature": round(current["temp"], 1),
        "solarRadiation": solar_radiation,
        "cloudCover": current.get("clouds") or 0,
        "humidity": current.get("humidity"),
        "windSpeed": round(((current.get("wind_speed") or 0) * 3.6), 1),
        "forecast": forecast,
        "provider": "OpenWeather • One Call 2.5",
        "locationName": DEFAULT_SYSTEM_CONFIG["location"]["name"],
        "lastUpdated": datetime.utcfromtimestamp(current.get("dt") or datetime.utcnow().timestamp()).isoformat(),
        "description": (current.get("weather") or [{}])[0].get("description"),
    }


def _generate_fallback_forecast(solar_capacity_kw: float) -> List[Dict[str, Any]]:
    scenarios = ["sunny", "sunny", "partly-cloudy", "cloudy", "partly-cloudy", "sunny", "cloudy"]
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    forecast: List[Dict[str, Any]] = []
    for idx in range(7):
        date = today + timedelta(days=idx)
        scenario = scenarios[idx % len(scenarios)]
        if scenario == "sunny":
            cloud_cover = 5 + random.random() * 15
            avg_radiation = 850 + random.random() * 120
        elif scenario == "partly-cloudy":
            cloud_cover = 30 + random.random() * 20
            avg_radiation = 600 + random.random() * 150
        else:
            cloud_cover = 70 + random.random() * 20
            avg_radiation = 300 + random.random() * 170

        predicted = (avg_radiation / 1000) * solar_capacity_kw * 8 * (1 - cloud_cover / 200)
        base_temp = 18 + random.random() * 8
        forecast.append(
            {
                "date": date.isoformat(),
                "dayOfWeek": _spanish_day_name(date),
                "maxTemp": round(base_temp + 5 + random.random() * 3, 1),
                "minTemp": round(base_temp - 3 + random.random() * 2, 1),
                "solarRadiation": round(avg_radiation),
                "cloudCover": round(cloud_cover),
                "predictedProduction": round(max(0.0, predicted), 1),
                "condition": "sunny" if cloud_cover < 20 else ("partly-cloudy" if cloud_cover < 50 else "cloudy"),
            }
        )
    return forecast


def _generate_fallback_weather_data(solar_capacity_kw: float) -> Dict[str, Any]:
    hour = datetime.utcnow().hour
    cloud_cover = 15 + random.random() * 25
    peak_hour = 13
    sigma = 4
    gaussian = math.exp(-((hour - peak_hour) ** 2) / (2 * sigma**2))
    max_radiation = 1000
    radiation = max_radiation * gaussian
    radiation *= 1 - (cloud_cover / 100) * 0.7
    if hour < 6 or hour > 20:
        radiation = 0
    base_temp = 20
    temp_variation = 8 * math.sin(((hour - 6) * math.pi) / 12)
    temperature = base_temp + temp_variation + (random.random() * 2 - 1)

    return {
        "temperature": round(temperature, 1),
        "solarRadiation": round(radiation),
        "cloudCover": round(cloud_cover),
        "humidity": round(50 + random.random() * 30),
        "windSpeed": round(5 + random.random() * 15, 1),
        "forecast": _generate_fallback_forecast(solar_capacity_kw),
        "provider": "Simulación interna (fallback)",
        "locationName": "Ubicación simulada",
        "lastUpdated": datetime.utcnow().isoformat(),
        "description": "Datos simulados por indisponibilidad de OpenWeather",
        "isMock": True,
    }


async def generate_weather_data(solar_capacity_kw: float) -> Dict[str, Any]:
    return await get_weather_with_fallback(
        DEFAULT_SYSTEM_CONFIG["location"]["lat"],
        DEFAULT_SYSTEM_CONFIG["location"]["lon"],
        solar_capacity_kw,
        DEFAULT_SYSTEM_CONFIG["location"]["name"],
    )


async def get_weather_with_fallback(
    lat: float,
    lon: float,
    capacity_kw: float,
    location_name: str = "Ubicación",
) -> Dict[str, Any]:
    source_error: Optional[str] = None

    active_source = get_active_weather_source()
    if active_source and active_source.get("enabled"):
        # Hay una fuente configurada: cualquier fallo debe ser visible al operador.
        source_name = active_source.get("name", "fuente personalizada")
        try:
            custom_weather = await get_active_weather_data(lat, lon, capacity_kw, location_name)
            if custom_weather:
                return custom_weather
        except Exception as exc:
            source_error = f"La fuente '{source_name}' falló: {exc}"

    try:
        weather = await fetch_open_meteo_weather(lat, lon, capacity_kw, location_name)
        if source_error:
            weather["sourceError"] = source_error
        return weather
    except Exception:
        mock_weather = _generate_fallback_weather_data(capacity_kw)
        mock_weather.update(
            {
                "provider": "Datos simulados (Open-Meteo no disponible)",
                "locationName": location_name,
                "lastUpdated": datetime.utcnow().isoformat(),
            }
        )
        if source_error:
            mock_weather["sourceError"] = source_error
        return mock_weather
