"""
Prediction and projection helpers migrated from the Next.js implementation.
"""

from __future__ import annotations
import os
import math
import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional

from .system_config import get_system_config
from .weather_service import get_weather_with_fallback
from .inverter_service import list_inverters
from .ml_model_service import ml_model_service

DEFAULT_PANEL_EFFICIENCY = 0.2

# ============================================================================
# MOTOR DE CONSUMO DEL ENSAMBLE (añadido para la integración)
# ============================================================================
from app.services.ensemble_consumption_engine import EnsembleConsumptionEngine

_CONSUMPTION_ENGINE_REGISTRY = {
    "ensemble": EnsembleConsumptionEngine,
    # "profile": ProfileConsumptionEngine,  # se puede añadir después
}

_consumption_engine = None


def get_consumption_engine():
    """Devuelve el motor de consumo activo (singleton con Strategy)."""
    global _consumption_engine
    if _consumption_engine is None:
        name = (os.environ.get("CONSUMPTION_ENGINE") or "ensemble").strip().lower()
        engine_class = _CONSUMPTION_ENGINE_REGISTRY.get(name, EnsembleConsumptionEngine)
        _consumption_engine = engine_class()
    return _consumption_engine


def set_consumption_engine(name: str):
    """Cambia la estrategia de consumo en tiempo de ejecución."""
    global _consumption_engine
    key = str(name).strip().lower()
    if key not in _CONSUMPTION_ENGINE_REGISTRY:
        raise ValueError(
            f"Estrategia de consumo desconocida: {name}. "
            f"Opciones: {sorted(_CONSUMPTION_ENGINE_REGISTRY)}"
        )
    _consumption_engine = _CONSUMPTION_ENGINE_REGISTRY[key]()
    return _consumption_engine
# ============================================================================

def _consumption_from_appliances(dt: datetime) -> float:
    # ========================================================================
    # MOTOR DEL ENSAMBLE (añadido para la integración)
    # ========================================================================
    try:
        engine = get_consumption_engine()
        predicciones = engine.predict([dt.isoformat()], config={})
        print(f"🔍 _consumption_from_appliances({dt}) → {predicciones}")
        if predicciones:
            return round(predicciones[0], 3)
    except Exception as e:
        print(f"⚠️  Ensamble de consumo falló, usando fallback: {e}")
    # ========================================================================

    """Suma el consumo (kW) de electrodomésticos siempre encendidos para una hora dada."""
    from .appliance_service import list_appliances
    from .appliance_measurement_service import forecast_kw
    total = 0.0
    for appliance in list_appliances():
        if not appliance.get("alwaysOn", True):
            continue
        profile = appliance.get("hourlyProfileKw") or []
        qty = max(1, int(appliance.get("quantity", 1)))
        if profile:
            total += forecast_kw(profile, dt) * qty
        else:
            watts = appliance.get("averagePowerW", 0) or 0
            total += (watts / 1000.0) * qty
    return round(total, 3)


def _parse_iso(date_str: str) -> datetime:
    if date_str.endswith("Z"):
        date_str = date_str[:-1] + "+00:00"
    return datetime.fromisoformat(date_str)


def _resolve_inverter_context() -> Dict[str, float]:
    """Suma capacidad AC del inversor y eficiencia ponderada por capacidad.
    Si no hay inversores configurados, se asume sin recorte ni pérdida (legacy).
    """
    try:
        inverters = list_inverters()
    except Exception:  # pragma: no cover - DB failure fallback
        inverters = []
    total_capacity = 0.0
    weighted_eff = 0.0
    for inv in inverters:
        rated = float(inv.get("ratedPowerKw") or 0)
        qty = float(inv.get("quantity") or 0)
        cap = rated * qty
        if cap <= 0:
            continue
        eff = inv.get("efficiencyPercent")
        eff_frac = float(eff) / 100 if eff is not None else 1.0
        total_capacity += cap
        weighted_eff += eff_frac * cap
    if total_capacity <= 0:
        return {"capacityKw": float("inf"), "efficiency": 1.0}
    return {
        "capacityKw": total_capacity,
        "efficiency": weighted_eff / total_capacity,
    }


def _system_efficiency_pct(config: Dict[str, Any]) -> float:
    """Eficiencia del sistema a partir de la ficha técnica del inversor (%).

    Usa la eficiencia de catálogo del/los inversores configurados, ponderada por
    capacidad. Si no hay inversor configurado, cae al estándar PVWatts (pérdidas
    de sistema del 14% → 86%). NO es telemetría real: es un valor de catálogo /
    estándar de ingeniería, no una medición en vivo.
    """
    inverter = _resolve_inverter_context()
    if inverter["capacityKw"] == float("inf"):  # sin inversor configurado
        return 86.0
    return round(inverter["efficiency"] * 100, 2)


def _resolve_solar_context(config: Dict[str, Any]) -> Dict[str, float]:
    solar = config["solar"]
    capacity_kw = solar.get("capacityKw") or 0
    panel_efficiency = (
        (solar.get("panelEfficiencyPercent") or 0) / 100 if solar.get("panelEfficiencyPercent") else DEFAULT_PANEL_EFFICIENCY
    )
    panel_area = solar.get("panelAreaM2") or (solar.get("panelRatedKw") or 0) / max(panel_efficiency, 0.0001)
    array_area = panel_area * (solar.get("panelCount") or 1)
    inverter = _resolve_inverter_context()
    return {
        "capacityKw": capacity_kw,
        "panelEfficiency": panel_efficiency,
        "arrayAreaM2": array_area,
        "inverterCapacityKw": inverter["capacityKw"],
        "inverterEfficiency": inverter["efficiency"],
    }


def _get_hour_efficiency_factor(hour: int) -> float:
    if 12 <= hour <= 14:
        return 1.0
    if 11 <= hour <= 15:
        return 0.95
    if 10 <= hour <= 16:
        return 0.85
    if 8 <= hour <= 17:
        return 0.70
    if 7 <= hour <= 18:
        return 0.50
    if 6 <= hour <= 19:
        return 0.30
    return 0.0


def predict_production(
    solar_radiation: float,
    temperature: float,
    cloud_cover: float,
    hour: int,
    context: Dict[str, float],
) -> float:
    # Potencia DC en el array
    production_dc = (solar_radiation * context["arrayAreaM2"] * context["panelEfficiency"]) / 1000
    temp_factor = 1 - (temperature - 25) * 0.004
    production_dc *= temp_factor
    cloud_factor = 1 - (cloud_cover / 100) * 0.5
    production_dc *= cloud_factor
    production_dc *= _get_hour_efficiency_factor(hour)
    production_dc = min(production_dc, context["capacityKw"])
    # Conversión DC→AC con derrateo y límite del inversor
    production_ac = production_dc * context.get("inverterEfficiency", 1.0)
    production_ac = min(production_ac, context.get("inverterCapacityKw", float("inf")))
    if hour < 6 or hour > 20:
        production_ac = 0
    return round(max(0.0, production_ac), 2)


def _estimate_hourly_solar_radiation(hour: int, daily_avg: float, cloud_cover: float) -> float:
    # daily_avg es la radiación media en W/m² sobre 24 h (incluye noche). El pico
    # de mediodía es ~4-5× esa media; el factor reconstruye la curva diaria.
    peak_hour = 13
    sigma = 4
    gaussian = math.exp(-((hour - peak_hour) ** 2) / (2 * sigma**2))
    radiation = daily_avg * gaussian * 4.0
    cloud_variability = 1 - (random.random() * cloud_cover / 200)
    radiation *= cloud_variability
    return max(0.0, round(radiation))


def _estimate_hourly_temperature(hour: int, max_temp: float, min_temp: float) -> float:
    amplitude = (max_temp - min_temp) / 2
    average = (max_temp + min_temp) / 2
    phase = ((hour - 6) / 24) * 2 * math.pi
    temp = average + amplitude * math.sin(phase)
    return round(temp, 1)


def _predict_consumption(hour: int) -> float:
    base_day = 35
    base_night = 18
    if 7 <= hour <= 9 or 18 <= hour <= 22:
        return base_day * 1.3
    if 6 <= hour < 18:
        return base_day
    return base_night


def _model_confidence_pct() -> int:
    """Confianza = R² real del modelo en el conjunto de test (varianza explicada), en %.

    Es la métrica de generalización medida durante el entrenamiento, no una
    fórmula inventada. Es constante para todas las horas: refleja la calidad del
    modelo, no una falsa certeza que decae con el horizonte. Si el modelo no está
    cargado, devuelve 0 (no se inventa un valor).
    """
    r2 = ml_model_service.get_test_r2()
    if r2 is None:
        return 0
    return int(round(max(0.0, min(1.0, r2)) * 100))


def generate_hourly_predictions(
    weather_forecast: List[Dict[str, Any]],
    config: Dict[str, Any],
    hours: int = 24,
    production_override: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Genera la predicción horaria de producción/consumo.

    Si ``production_override`` se proporciona (producción en kW reales por offset
    horario, calculada por el servicio central de predicción), se usa en lugar de
    la fórmula heredada. Así el dashboard reutiliza el mismo motor (ML/física +
    sombra + orientación) que el resto de la app. Sin override, se mantiene el
    comportamiento anterior intacto.
    """
    predictions: List[Dict[str, Any]] = []
    # Hora local de La Habana (no UTC): la curva diaria y el corte día/noche se
    # calculan sobre la hora solar local, si no la producción "actual" sale
    # desfasada ~5 h (y de noche siempre 0).
    now = datetime.now(ZoneInfo("America/Havana")).replace(tzinfo=None)
    context = _resolve_solar_context(config)
    fallback_forecast = weather_forecast[0] if weather_forecast else None
    hours = max(1, min(int(hours), 24 * 7))
    model_conf = _model_confidence_pct()  # R² real del modelo, igual para todas las horas
    for hour_offset in range(hours):
        timestamp = now + timedelta(hours=hour_offset)
        hour = timestamp.hour
        forecast_idx = min(hour_offset // 12, len(weather_forecast) - 1)
        forecast = (weather_forecast[forecast_idx] if forecast_idx >= 0 else None) or fallback_forecast
        if not forecast:
            continue

        if production_override is not None and hour_offset < len(production_override):
            # Producción del servicio central (ML/física + sombra + orientación).
            production = round(max(0.0, float(production_override[hour_offset])), 2)
        else:
            # Fórmula heredada (fallback si el motor central no está disponible).
            solar_radiation = _estimate_hourly_solar_radiation(hour, forecast["solarRadiation"], forecast["cloudCover"])
            temperature = _estimate_hourly_temperature(hour, forecast["maxTemp"], forecast["minTemp"])
            production = predict_production(solar_radiation, temperature, forecast["cloudCover"], hour, context)
        consumption = _consumption_from_appliances(timestamp)
        confidence = model_conf

        predictions.append(
            {
                "timestamp": timestamp.isoformat(),
                "hour": hour,
                "expectedProduction": round(production, 2),
                "expectedConsumption": round(consumption, 2),
                "confidence": round(confidence),
            }
        )

    return predictions


def _resolve_battery_dynamics(battery: Dict[str, Any]) -> Dict[str, float]:
    """Extrae DoD, tasas y eficiencia round-trip de la config de batería.
    Defaults conservadores: sin DoD (100%), sin límite de potencia, eff 100%.
    """
    capacity = float(battery.get("capacityKwh") or 0)
    dod = battery.get("maxDepthOfDischargePercent")
    soc_min_fraction = max(0.0, 1.0 - (float(dod) / 100)) if dod is not None else 0.0
    round_trip = battery.get("efficiencyPercent")
    round_trip_frac = float(round_trip) / 100 if round_trip is not None else 1.0
    # eff_one_way = sqrt(round_trip): se aplica por igual a carga y descarga.
    one_way = math.sqrt(max(0.0, min(1.0, round_trip_frac)))
    return {
        "capacityKwh": capacity,
        "socMinKwh": soc_min_fraction * capacity,
        "chargeRateKw": float(battery.get("chargeRateKw") or 0) or float("inf"),
        "dischargeRateKw": float(battery.get("dischargeRateKw") or 0) or float("inf"),
        "effOneWay": one_way,
    }


def build_projected_solar_timeline(
    predictions: List[Dict[str, Any]],
    battery: Dict[str, Any],
    initial_battery_level: float = 75,
    system_efficiency_pct: float = 86.0,
) -> List[Dict[str, Any]]:
    timeline: List[Dict[str, Any]] = []
    dyn = _resolve_battery_dynamics(battery)
    battery_capacity_kwh = dyn["capacityKwh"]
    soc_min_kwh = dyn["socMinKwh"]
    charge_rate = dyn["chargeRateKw"]
    discharge_rate = dyn["dischargeRateKw"]
    eff = dyn["effOneWay"]

    # Si el nivel inicial está por debajo del SoC mínimo permitido, súbelo.
    initial_kwh = (initial_battery_level / 100) * battery_capacity_kwh
    stored_energy = max(initial_kwh, soc_min_kwh)

    for prediction in predictions[:24]:
        production = max(0.0, prediction["expectedProduction"])
        consumption = max(0.0, prediction["expectedConsumption"])
        net = production - consumption
        grid_export = 0.0
        grid_import = 0.0
        battery_delta = 0.0

        if net >= 0:
            # Excedente: cargar batería respetando tasa y eficiencia.
            available_capacity = max(0.0, battery_capacity_kwh - stored_energy)
            grid_input = min(net, charge_rate)               # kW desde el array
            energy_to_battery = min(grid_input * eff, available_capacity)  # kWh almacenado
            stored_energy += energy_to_battery
            battery_delta = energy_to_battery
            grid_export = net - (energy_to_battery / eff if eff > 0 else 0)
        else:
            # Déficit: descargar batería respetando DoD, tasa y eficiencia.
            demand = abs(net)
            usable_stored = max(0.0, stored_energy - soc_min_kwh)
            # Energía que la batería puede entregar al bus, limitada por tasa
            from_battery_to_bus = min(usable_stored * eff, demand, discharge_rate)
            # Energía que sale realmente del banco (mayor que la entregada por la pérdida)
            from_battery_internal = from_battery_to_bus / eff if eff > 0 else 0
            stored_energy -= from_battery_internal
            battery_delta = -from_battery_internal
            grid_import = demand - from_battery_to_bus

        denom = battery_capacity_kwh if battery_capacity_kwh > 0 else 1.0
        battery_level = max(0.0, min(100.0, (stored_energy / denom) * 100))
        # Eficiencia real de catálogo (inversor), no una fórmula ligada a la confianza.
        efficiency = system_efficiency_pct

        timeline.append(
            {
                "timestamp": prediction["timestamp"],
                "production": round(production, 2),
                "consumption": round(consumption, 2),
                "batteryLevel": round(battery_level, 2),
                "gridExport": round(grid_export, 2),
                "gridImport": round(grid_import, 2),
                "efficiency": round(efficiency, 2),
                "batteryDelta": round(battery_delta, 2),
            }
        )

    return timeline


def generate_battery_projection(
    timeline: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    battery_capacity_kwh: float,
) -> Dict[str, Any]:
    first_entry = timeline[0] if timeline else None
    stored_energy = (
        (first_entry["batteryLevel"] / 100) * battery_capacity_kwh if first_entry else (55 / 100) * battery_capacity_kwh
    )
    projected_min_level = min((entry["batteryLevel"] for entry in timeline), default=55.0)
    projected_max_level = max((entry["batteryLevel"] for entry in timeline), default=55.0)
    upcoming_consumption = predictions[0]["expectedConsumption"] if predictions else 0
    autonomy_hours = stored_energy / upcoming_consumption if upcoming_consumption else 999
    battery_delta = first_entry["batteryDelta"] if first_entry else 0

    return {
        "chargeLevel": round(first_entry["batteryLevel"] if first_entry else 55.0, 2),
        "capacity": battery_capacity_kwh,
        "current": round(stored_energy, 2),
        "autonomyHours": round(autonomy_hours, 1),
        "charging": battery_delta >= 0,
        "powerFlow": round(battery_delta, 2),
        "projectedMinLevel": round(projected_min_level, 2),
        "projectedMaxLevel": round(projected_max_level, 2),
        "note": "Estimación basada en clima y ficha técnica. No hay telemetría en vivo.",
    }


def generate_alerts(
    predictions: List[Dict[str, Any]],
    battery_status: Dict[str, Any],
    weather_forecast: List[Dict[str, Any]],
    capacity_kw: float = 50.0,
) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    # Umbrales relativos a la capacidad del sistema (kWh/día y kW), para que no
    # estén calibrados a un tamaño fijo. (50 kW daba 150/300/10 originalmente.)
    low_prod_threshold = capacity_kw * 3      # kWh/día
    high_prod_threshold = capacity_kw * 6     # kWh/día
    deficit_threshold = capacity_kw * 0.2     # kW
    starting_level = battery_status["chargeLevel"]
    projected_min = battery_status.get("projectedMinLevel", starting_level)

    if projected_min < 20:
        alerts.append(
            {
                "id": "battery-low",
                "type": "warning",
                "title": "Reserva de Batería Limitada",
                "message": f"Proyección mínima de {projected_min:.1f}% sin telemetría. Considere reducir consumos continuos.",
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
    if projected_min < 10:
        alerts.append(
            {
                "id": "battery-critical",
                "type": "critical",
                "title": "Reserva Crítica Estimada",
                "message": f"La proyección indica un mínimo de {projected_min:.1f}%. Asegure respaldo externo.",
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    tomorrow = weather_forecast[1] if len(weather_forecast) > 1 else None
    if tomorrow and tomorrow["predictedProduction"] < low_prod_threshold:
        alerts.append(
            {
                "id": "low-production-forecast",
                "type": "warning",
                "title": "Baja Producción Esperada",
                "message": f"Se espera baja producción mañana ({tomorrow['predictedProduction']:.0f} kWh) debido a condiciones climáticas.",
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    today_forecast = weather_forecast[0] if weather_forecast else None
    if today_forecast and today_forecast["predictedProduction"] > high_prod_threshold:
        alerts.append(
            {
                "id": "high-production-forecast",
                "type": "info",
                "title": "Excelente Producción Esperada",
                "message": f"Se espera alta producción hoy ({today_forecast['predictedProduction']:.0f} kWh). Buen momento para cargas intensivas.",
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    next_6_hours = predictions[:6]
    if next_6_hours:
        avg_deficit = sum(max(0.0, p["expectedConsumption"] - p["expectedProduction"]) for p in next_6_hours) / len(
            next_6_hours
        )
        if avg_deficit > deficit_threshold and projected_min < 50:
            alerts.append(
                {
                    "id": "deficit-warning",
                    "type": "warning",
                    "title": "Déficit Energético Próximo",
                    "message": f"Se espera déficit promedio de {avg_deficit:.1f} kW en las próximas 6 horas. Considere reducir consumo no esencial.",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

    return alerts


def generate_recommendations(
    predictions: List[Dict[str, Any]],
    battery_status: Dict[str, Any],
    config: Dict[str, Any],
) -> List[str]:
    recommendations: List[str] = []
    solar_capacity = max(config["solar"].get("capacityKw") or 0.0001, 0.0001)
    current_prediction = predictions[0] if predictions else None
    current_production = current_prediction["expectedProduction"] if current_prediction else 0
    current_consumption = current_prediction["expectedConsumption"] if current_prediction else 0
    battery_level = battery_status["chargeLevel"]
    projected_min = battery_status.get("projectedMinLevel", battery_level)
    projected_max = battery_status.get("projectedMaxLevel", battery_level)

    if current_production > current_consumption * 1.2:
        recommendations.append("Se proyecta excedente solar inmediato. Buen momento para programar cargas intensivas supervisadas.")

    next_three = predictions[:3]
    if next_three:
        avg_production = sum(item["expectedProduction"] for item in next_three) / len(next_three)
        if projected_min < 40 and avg_production > 30:
            recommendations.append(
                "Aunque la reserva es limitada, se espera repunte solar en pocas horas. Posponga consumos pesados hasta después del pico solar."
            )

    current_hour = datetime.utcnow().hour
    if 9 <= current_hour <= 11 and projected_max < 85:
        recommendations.append(
            "Se aproximan las horas de mayor producción (12-14h). Coordine actividades de alto consumo dentro de esa ventana."
        )
    if 16 <= current_hour <= 18 and projected_min < 50:
        recommendations.append(
            "La producción solar caerá al atardecer. Asegure carga mínima o reduzca consumos no esenciales esta noche."
        )
    if current_production > solar_capacity * 0.8:
        recommendations.append(
            f"La proyección indica operación cercana al {(current_production / solar_capacity) * 100:.0f}% de la capacidad instalada. Aproveche para tareas que requieran potencia."
        )
    if current_prediction and current_prediction["expectedProduction"] < solar_capacity * 0.3:
        recommendations.append(
            "Día de baja producción estimada. Priorice consumos esenciales y considere apoyo de la red para cargas críticas."
        )

    return recommendations


async def central_production_override(hours: int, config: Dict[str, Any]) -> Optional[List[float]]:
    """Producción horaria (kW reales) desde el servicio central de predicción,
    alineada por offset horario desde "ahora" (hora local de La Habana).

    Devuelve None si el motor no está disponible o la configuración no permite
    escalar (sin capacidad o sin capacidad de referencia). En ese caso el
    llamador cae a la fórmula heredada y nada deja de funcionar.
    """
    from .production_forecast_service import get_production_service
    from .ml_model_service import ml_model_service

    try:
        reference = ml_model_service.get_reference_capacity_kw() or 1.0
        if reference <= 0:
            return None
        capacity = float(config["solar"].get("capacityKw") or 0)
        if capacity <= 0:
            return None
        hours = max(1, min(int(hours), 24 * 7))
        now = datetime.now(ZoneInfo("America/Havana")).replace(tzinfo=None)
        datetimes = [(now + timedelta(hours=h)).isoformat() for h in range(hours)]
        preds = await get_production_service().predict(
            datetimes, config["location"]["lat"], config["location"]["lon"]
        )
        if not preds:
            return None
        scale = capacity / reference
        return [round(float(p["production_kw"]) * scale, 2) for p in preds]
    except Exception:
        return None


async def get_predictions_bundle(hours: int = 24) -> Dict[str, Any]:
    config = get_system_config()
    weather_data = await get_weather_with_fallback(
        config["location"]["lat"],
        config["location"]["lon"],
        config["solar"]["capacityKw"],
        config["location"]["name"],
    )
    override = await central_production_override(hours, config)
    predictions = generate_hourly_predictions(
        weather_data["forecast"], config, hours=hours, production_override=override
    )
    timeline = build_projected_solar_timeline(
        predictions, config["battery"], 75, _system_efficiency_pct(config)
    )
    battery_projection = generate_battery_projection(
        timeline,
        predictions,
        config["battery"]["capacityKwh"],
    )
    alerts = generate_alerts(
        predictions,
        battery_projection,
        weather_data["forecast"],
        config["solar"]["capacityKw"],
    )
    recommendations = generate_recommendations(predictions, battery_projection, config)
    return {
        "predictions": predictions,
        "alerts": alerts,
        "recommendations": recommendations,
        "battery": battery_projection,
        "timeline": timeline,
        "weather": weather_data,
        "timestamp": datetime.utcnow().isoformat(),
        "config": config,
    }
