"""
High-level service that gathers solar, weather and projection data.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from .analytics import calculate_energy_flow, calculate_system_metrics
from .prediction_service import (
    build_projected_solar_timeline,
    central_production_override,
    generate_battery_projection,
    generate_hourly_predictions,
    _system_efficiency_pct,
)
from .system_config import get_system_config
from .weather_service import get_weather_with_fallback


async def get_solar_snapshot() -> Dict[str, Any]:
    config = get_system_config()
    weather_data = await get_weather_with_fallback(
        config["location"]["lat"],
        config["location"]["lon"],
        config["solar"]["capacityKw"],
        config["location"]["name"],
    )
    override = await central_production_override(24, config)
    predictions = generate_hourly_predictions(
        weather_data["forecast"], config, production_override=override
    )
    projected_timeline = build_projected_solar_timeline(
        predictions,
        config["battery"],
        system_efficiency_pct=_system_efficiency_pct(config),
    )

    if not projected_timeline:
        raise RuntimeError("No hay datos proyectados disponibles.")

    current_projection = projected_timeline[0]
    battery_projection = generate_battery_projection(
        projected_timeline,
        predictions,
        config["battery"]["capacityKwh"],
    )
    metrics = calculate_system_metrics(current_projection, projected_timeline)
    energy_flow = calculate_energy_flow(
        current_projection["production"],
        current_projection["consumption"],
        battery_projection["charging"],
        battery_projection["powerFlow"],
    )

    return {
        "current": current_projection,
        "historical": projected_timeline,
        "battery": battery_projection,
        "metrics": metrics,
        "energyFlow": energy_flow,
        "timestamp": datetime.utcnow().isoformat(),
        "mode": "predictive",
        "weather": weather_data,
        "config": config,
    }
