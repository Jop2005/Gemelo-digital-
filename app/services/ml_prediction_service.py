"""
ML Prediction Service — capa de compatibilidad.

La lógica real vive ahora en ``production_forecast_service.ProductionForecastService``
(punto único de predicción de producción, con motor intercambiable ML/física).
Este módulo conserva las funciones históricas como wrappers delgados para no
romper a sus consumidores (resolvers GraphQL y battery_discharge_service).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Dict, List

from .production_forecast_service import (
    get_production_service,
    DEFAULT_LAT,
    DEFAULT_LON,
)

LOCAL_TZ = ZoneInfo("America/Havana")


async def predict_solar_production(
    datetimes: List[str],
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
) -> List[Dict[str, Any]]:
    """Predice producción para datetimes ISO. Delega en el servicio central."""
    return await get_production_service().predict(datetimes, lat, lon)


async def predict_next_hours(
    hours: int = 24,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
) -> List[Dict[str, Any]]:
    """Predice producción para las próximas N horas (hora local de La Habana)."""
    now = datetime.now(LOCAL_TZ)
    target_datetimes = [(now + timedelta(hours=h)).isoformat() for h in range(hours)]
    return await predict_solar_production(target_datetimes, lat, lon)


async def predict_for_date_range(
    start_date: str,
    end_date: str,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
) -> List[Dict[str, Any]]:
    """Predice producción para todas las horas en un rango de fechas."""
    try:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
    except Exception as e:
        raise ValueError(f"Invalid date format. Expected 'YYYY-MM-DD': {e}")

    target_datetimes: List[str] = []
    current = start
    while current <= end:
        target_datetimes.append(current.isoformat())
        current += timedelta(hours=1)

    return await predict_solar_production(target_datetimes, lat, lon)


async def predict_for_specific_hours(
    date: str,
    hours: List[int],
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
) -> List[Dict[str, Any]]:
    """Predice producción para horas específicas de un día."""
    try:
        base_date = datetime.fromisoformat(date)
    except Exception as e:
        raise ValueError(f"Invalid date format. Expected 'YYYY-MM-DD': {e}")

    target_datetimes: List[str] = []
    for hour in sorted(set(hours)):
        if 0 <= hour <= 23:
            dt = base_date.replace(hour=hour, minute=0, second=0, microsecond=0)
            target_datetimes.append(dt.isoformat())

    if not target_datetimes:
        return []

    return await predict_solar_production(target_datetimes, lat, lon)
