"""
Aggregate configuration for the PV system combining DB data and defaults.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .panel_service import list_panels
from .battery_service import list_batteries
from .system_defaults import DEFAULT_SYSTEM_CONFIG


def _round(value: float) -> float:
    return round(value, 2)


_ZERO_SOLAR: Dict[str, Any] = {
    "capacityKw": 0.0,
    "panelRatedKw": None,
    "panelCount": 0,
    "strings": 0,
    "panelEfficiencyPercent": None,
    "panelAreaM2": None,
    "spec": None,
}

_ZERO_BATTERY: Dict[str, Any] = {
    "capacityKwh": 0.0,
    "moduleCapacityKwh": None,
    "moduleCount": 0,
    "maxDepthOfDischargePercent": None,
    "chargeRateKw": None,
    "dischargeRateKw": None,
    "efficiencyPercent": None,
    "spec": None,
}


def _aggregate_solar(panels: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not panels:
        return _ZERO_SOLAR

    total_capacity = 0.0
    total_count = 0
    total_strings = 0
    for panel in panels:
        quantity = float(panel.get("quantity") or 0)
        rated_power = float(panel.get("ratedPowerKw") or 0)
        total_capacity += quantity * rated_power
        total_count += int(quantity)
        total_strings += int(panel.get("quantity") or 0)

    primary = panels[0]
    return {
        "capacityKw": _round(total_capacity),
        "panelRatedKw": primary.get("ratedPowerKw"),
        "panelCount": total_count,
        "strings": total_strings,
        "panelEfficiencyPercent": primary.get("efficiencyPercent"),
        "panelAreaM2": primary.get("areaM2"),
        "spec": primary,
    }


def _aggregate_battery(batteries: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batteries:
        return _ZERO_BATTERY

    total_capacity = 0.0
    total_modules = 0
    for battery in batteries:
        quantity = float(battery.get("quantity") or 0)
        capacity = float(battery.get("capacityKwh") or 0)
        total_capacity += quantity * capacity
        total_modules += int(quantity)

    primary = batteries[0]
    return {
        "capacityKwh": _round(total_capacity),
        "moduleCapacityKwh": primary.get("capacityKwh"),
        "moduleCount": total_modules,
        "maxDepthOfDischargePercent": primary.get("maxDepthOfDischargePercent"),
        "chargeRateKw": primary.get("chargeRateKw"),
        "dischargeRateKw": primary.get("dischargeRateKw"),
        "efficiencyPercent": primary.get("efficiencyPercent"),
        "spec": primary,
    }


def get_system_config() -> Dict[str, Any]:
    from .location_config_service import get_location_config
    try:
        panels = list_panels()
        batteries = list_batteries()
    except Exception:  # pragma: no cover - DB failure fallback
        return {
            "location": DEFAULT_SYSTEM_CONFIG["location"],
            "solar": _ZERO_SOLAR,
            "battery": _ZERO_BATTERY,
        }

    return {
        "location": get_location_config(),
        "solar": _aggregate_solar(panels),
        "battery": _aggregate_battery(batteries),
    }
