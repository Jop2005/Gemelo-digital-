"""
Shared feature engineering for the solar production model.

CRITICAL: this module is the single source of truth for how raw weather
variables become model features. It is used BOTH when training the model
(notebook / training script) AND at inference time in the backend. Keeping
the transformation in one place guarantees there is no train/serve skew.

Input expected by ``build_features``:
    A pandas DataFrame indexed by a timezone-aware UTC ``DatetimeIndex`` with
    the columns that Open-Meteo provides:
        - temperature_2m        (°C)
        - relative_humidity_2m  (%)
        - wind_speed_10m        (m/s)
        - cloud_cover           (%)
        - shortwave_radiation   (W/m², GHI)

Output:
    A DataFrame (same index) with exactly the columns in ``FEATURE_COLUMNS``,
    in that order, ready to be fed to the model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# La Habana, Cuba
DEFAULT_LAT = 23.1136
DEFAULT_LON = -82.3666
DEFAULT_ALTITUDE_M = 50.0
LOCAL_TZ = "America/Havana"

# Order matters: the model is trained on this exact column order.
FEATURE_COLUMNS = [
    "shortwave_radiation",   # GHI from Open-Meteo (W/m²)
    "clearsky_ghi",          # theoretical clear-sky GHI for La Habana (pvlib)
    "clearsky_index",        # measured / clear-sky  (0 = overcast, 1 = clear)
    "solar_elevation",       # sun elevation angle (deg, clipped at 0)
    "effective_irradiance",  # radiation attenuated by cloud cover
    "cloud_cover",           # %
    "temperature_2m",        # °C
    "relative_humidity_2m",  # %
    "wind_speed_10m",        # m/s
    "temp_loss_factor",      # 1 - 0.004*(T-25)+  (cell-temperature derate proxy)
    "hour_sin", "hour_cos",  # local-time daily cycle
    "month_sin", "month_cos" # seasonal cycle
]

# Monotonic constraints for gradient boosting (aligned to FEATURE_COLUMNS):
# production must increase with irradiance/elevation and decrease with cloud cover.
MONOTONE_MAP = {
    "shortwave_radiation": 1,
    "clearsky_ghi": 1,
    "clearsky_index": 1,
    "solar_elevation": 1,
    "effective_irradiance": 1,
    "cloud_cover": -1,
}


def monotone_constraints() -> list[int]:
    """Return the monotone-constraint vector aligned to FEATURE_COLUMNS."""
    return [MONOTONE_MAP.get(col, 0) for col in FEATURE_COLUMNS]


def _clearsky(times: pd.DatetimeIndex, lat: float, lon: float, altitude: float):
    """Solar position + clear-sky GHI via pvlib, with graceful fallbacks."""
    import pvlib

    location = pvlib.location.Location(lat, lon, tz="UTC", altitude=altitude)
    solpos = location.get_solarposition(times)
    try:
        clearsky = location.get_clearsky(times, model="ineichen")
    except Exception:
        # Ineichen needs bundled turbidity tables; fall back if unavailable.
        clearsky = location.get_clearsky(times, model="simplified_solis")
    return solpos, clearsky


def build_features(
    df: pd.DataFrame,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    altitude: float = DEFAULT_ALTITUDE_M,
) -> pd.DataFrame:
    """
    Transform raw Open-Meteo weather into the model feature matrix.

    Args:
        df: DataFrame indexed by a tz-aware UTC DatetimeIndex, with the five
            Open-Meteo columns listed in the module docstring.

    Returns:
        DataFrame with columns == FEATURE_COLUMNS (same order), same index.
    """
    if df.index.tz is None:
        raise ValueError("build_features expects a timezone-aware (UTC) DatetimeIndex")

    times = df.index
    solpos, clearsky = _clearsky(times, lat, lon, altitude)

    out = pd.DataFrame(index=times)

    radiation = df["shortwave_radiation"].clip(lower=0)
    cs_ghi = clearsky["ghi"].clip(lower=0)
    cloud = df["cloud_cover"].clip(lower=0, upper=100)

    out["shortwave_radiation"] = radiation
    out["clearsky_ghi"] = cs_ghi
    # clearsky index: ratio of actual to clear-sky irradiance (sky transparency)
    ci = radiation / cs_ghi.replace(0, np.nan)
    out["clearsky_index"] = ci.clip(lower=0, upper=1.2).fillna(0.0)
    out["solar_elevation"] = solpos["elevation"].clip(lower=0)
    out["effective_irradiance"] = radiation * (1.0 - cloud / 100.0)
    out["cloud_cover"] = cloud
    out["temperature_2m"] = df["temperature_2m"]
    out["relative_humidity_2m"] = df["relative_humidity_2m"]
    out["wind_speed_10m"] = df["wind_speed_10m"]
    # crystalline-silicon temperature derate proxy: -0.4%/°C above 25°C
    out["temp_loss_factor"] = 1.0 - 0.004 * (df["temperature_2m"] - 25.0).clip(lower=0)

    # Local-time cyclic features (La Habana) — matches how the system is used.
    local = times.tz_convert(LOCAL_TZ)
    out["hour_sin"] = np.sin(2 * np.pi * local.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * local.hour / 24)
    out["month_sin"] = np.sin(2 * np.pi * local.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * local.month / 12)

    return out[FEATURE_COLUMNS]
