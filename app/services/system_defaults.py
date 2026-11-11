"""
Default system configuration shared across services.
"""
from __future__ import annotations

DEFAULT_SYSTEM_CONFIG = {
    "location": {
        "lat": 23.1136,
        "lon": -82.3666,
        "name": "La Habana, Cuba",
    },
    "solar": {
        "capacityKw": 50.0,
        "panelRatedKw": 0.55,
        "panelCount": 10,
        "strings": 10,
        "panelEfficiencyPercent": 21.8,
        "panelAreaM2": 2.6,
        "spec": None,
    },
    "battery": {
        "capacityKwh": 100.0,
        "moduleCapacityKwh": 100.0,
        "moduleCount": 1,
        "maxDepthOfDischargePercent": 80.0,
        "efficiencyPercent": 92.0,
        "chargeRateKw": 25.0,
        "dischargeRateKw": 25.0,
        "spec": None,
    },
}

