"""
Battery Discharge Estimation Service - Calculates time until battery depletion.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional

from .ml_prediction_service import predict_for_specific_hours
from .ml_model_service import ml_model_service
from .system_config import get_system_config

LOCAL_TZ = ZoneInfo("America/Havana")


def _consumption_from_appliances(date_str: str, hours: List[int]) -> List[Dict[str, Any]]:
    """Calcula consumo horario (kW) sumando los electrodomésticos marcados como siempre encendidos."""
    from .appliance_service import list_appliances
    from .appliance_measurement_service import forecast_kw
    from datetime import datetime as _dt

    items = list_appliances()
    result = []
    for hour in hours:
        dt = _dt.fromisoformat(date_str).replace(hour=hour)
        total = 0.0
        for appliance in items:
            if not appliance.get("alwaysOn", True):
                continue
            profile = appliance.get("hourlyProfileKw") or []
            qty = max(1, int(appliance.get("quantity", 1)))
            if profile:
                total += forecast_kw(profile, dt) * qty
            else:
                watts = appliance.get("averagePowerW", 0) or 0
                total += (watts / 1000.0) * qty
        result.append({"datetime": f"{date_str}T{hour:02d}:00:00", "consumption_kw": round(total, 3)})
    return result


def simulate_battery_depletion(
    production_kw_series: List[float],
    consumption_kw_series: List[float],
    battery_capacity_kwh: float,
    start_level_kwh: Optional[float] = None,
) -> Optional[int]:
    """
    Simula hora a hora el nivel de la batería y devuelve los minutos
    transcurridos hasta que se agota, o None si nunca se agota.

    Esta función es pura (sin efectos secundarios) y testeable sin BD ni ML.

    Args:
        production_kw_series: Producción solar en kW para cada hora.
        consumption_kw_series: Consumo en kW para cada hora.
        battery_capacity_kwh: Capacidad total de la batería en kWh.
        start_level_kwh: Nivel inicial de la batería (por defecto: capacidad completa).

    Returns:
        Minutos hasta que la batería llega a 0, o None si siempre queda energía.
    """
    if battery_capacity_kwh <= 0:
        raise ValueError("La capacidad de la batería debe ser mayor que cero.")

    level = start_level_kwh if start_level_kwh is not None else battery_capacity_kwh

    for i, (prod_kw, cons_kw) in enumerate(
        zip(production_kw_series, consumption_kw_series)
    ):
        level += prod_kw - cons_kw
        level = max(0.0, min(battery_capacity_kwh, level))
        if level <= 0:
            return (i + 1) * 60

    return None


async def calculate_battery_discharge_time(
    start_hour: int,
    date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate time until battery reaches empty (0%) level.

    Simulates battery discharge/charge based on predicted production and consumption,
    starting from a given hour with batteries at 100% charge.

    Args:
        start_hour: Starting hour (0-23) to begin simulation
        date: Date in ISO format ('YYYY-MM-DD'). If None, uses today.

    Returns:
        Dictionary with:
        - minutesToEmpty: Minutes until battery reaches 0% (or None if never)
        - startHour: The starting hour provided
        - batteryCapacityKwh: Total battery capacity used for calculation

    Raises:
        ValueError: If start_hour is invalid or models not loaded
    """
    # Validate start_hour
    if not (0 <= start_hour <= 23):
        raise ValueError("start_hour must be between 0 and 23")

    # Get system configuration
    try:
        config = get_system_config()
    except Exception as e:
        raise RuntimeError(f"Failed to get system configuration: {e}")

    battery_capacity_kwh = config["battery"]["capacityKwh"]
    if battery_capacity_kwh <= 0:
        raise ValueError("Battery capacity must be greater than 0")

    # El modelo solar devuelve FACTOR DE CAPACIDAD (0-1); para pasar a kW reales
    # se multiplica por la capacidad solar / capacidad de referencia (= 1.0).
    solar_capacity_kw = config["solar"]["capacityKw"]
    reference_capacity_kw = ml_model_service.get_reference_capacity_kw() or 1.0
    production_scale = (
        solar_capacity_kw / reference_capacity_kw if reference_capacity_kw > 0 else 1.0
    )

    # Determine date (hora local de La Habana)
    if date is None:
        target_date = datetime.now(LOCAL_TZ).date()
    else:
        try:
            target_date = datetime.fromisoformat(date).date()
        except Exception as e:
            raise ValueError(f"Invalid date format. Expected 'YYYY-MM-DD': {e}")

    # Generate hours to simulate (from start_hour to end of day + next day)
    hours_to_simulate = list(range(start_hour, 24))

    # Also simulate next day to ensure we have enough data
    next_day = target_date + timedelta(days=1)
    hours_next_day = list(range(0, 24))

    # Get predictions for both days
    date_str = target_date.isoformat()
    next_date_str = next_day.isoformat()

    lat = config["location"]["lat"]
    lon = config["location"]["lon"]

    try:
        # Get production predictions
        production_today = await predict_for_specific_hours(
            date_str,
            hours_to_simulate,
            lat,
            lon
        )
        production_tomorrow = await predict_for_specific_hours(
            next_date_str,
            hours_next_day,
            lat,
            lon
        )

        # Consumo basado en electrodomésticos configurados como siempre encendidos
        consumption_today = _consumption_from_appliances(date_str, hours_to_simulate)
        consumption_tomorrow = _consumption_from_appliances(next_date_str, hours_next_day)
    except Exception as e:
        raise RuntimeError(f"Failed to get predictions: {e}")

    # Combine predictions from both days
    all_production = production_today + production_tomorrow
    all_consumption = consumption_today + consumption_tomorrow

    # Ensure we have matching data
    if len(all_production) != len(all_consumption):
        min_len = min(len(all_production), len(all_consumption))
        all_production = all_production[:min_len]
        all_consumption = all_consumption[:min_len]

    # Simulate battery discharge/charge using the pure helper
    prod_kw_list = [p["production_kw"] * production_scale for p in all_production]
    cons_kw_list = [c["consumption_kw"] for c in all_consumption]
    minutes_to_empty = simulate_battery_depletion(
        prod_kw_list, cons_kw_list, battery_capacity_kwh
    )

    return {
        "minutesToEmpty": minutes_to_empty,
        "startHour": start_hour,
        "batteryCapacityKwh": round(battery_capacity_kwh, 2),
    }
