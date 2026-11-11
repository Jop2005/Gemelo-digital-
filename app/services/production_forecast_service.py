"""
Servicio central de predicción de producción solar.

ÚNICO punto donde se convierte un intervalo de tiempo en producción estimada.
Toda la app (resoluciones GraphQL ML, batería, dashboard, predicciones) debe
reutilizar esta clase en vez de calcular producción por su cuenta.

Pipeline (idéntico para cualquier motor):
    intervalo de tiempo
      -> clima horario (Open-Meteo)
      -> features compartidas (solar_features.build_features)
      -> MOTOR: factor de capacidad base, geometría de referencia 20°/sur   ← Strategy
      -> × factor de sombra (perfil 3D por hora)
      -> × factor de orientación/inclinación (pvlib, por hora)
      = producción (en capacidad de referencia; el escalado por kW lo hace el llamador)

Patrón Strategy: ``ProductionEngine`` define la interfaz del "motor" que produce
el factor de capacidad base. Se puede cambiar el modelo ML por una fórmula física
sin tocar el resto del pipeline:

    from app.services.production_forecast_service import get_production_service
    get_production_service().set_engine("physics")   # o "ml"
"""
from __future__ import annotations

import abc
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Asegurar que el ensamble esté en el path
_ENSAMBLE_ROOT = Path(__file__).parent.parent.parent / "ensamble"
if str(_ENSAMBLE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENSAMBLE_ROOT))

from .ml_model_service import ml_model_service
from .solar_features import (
    build_features,
    DEFAULT_LAT,
    DEFAULT_LON,
    DEFAULT_ALTITUDE_M,
)
from .shadow_profile_service import get_shadow_profile
from .orientation_factor_service import compute_orientation_factor_series
from .weather_cache_service import weather_cache
from .weather_source_service import get_active_weather_source
from .production_engine import ProductionEngine
from .ensemble_production_engine import EnsembleProductionEngine
LOCAL_TZ = ZoneInfo("America/Havana")

# Geometría de referencia del modelo/entrenamiento (debe coincidir con la usada
# por orientation_factor_service: PVGIS angle=20, aspect=0 -> 20° sur).
REF_TILT = 20.0
REF_AZIMUTH = 180.0
SYSTEM_LOSSES = 0.14  # mismas pérdidas de sistema que PVGIS/PVWatts


# --------------------------------------------------------------------------- #
# Strategy: motores de factor de capacidad
# --------------------------------------------------------------------------- #



class MLProductionEngine(ProductionEngine):
    """Motor por defecto: el modelo de machine learning entrenado."""

    name = "ml"

    def capacity_factor(
        self, features_df: pd.DataFrame, lat: float, lon: float
    ) -> np.ndarray:
        if not ml_model_service.model_loaded:
            raise RuntimeError(
                "ML model not loaded. Please ensure the model is trained and "
                "loaded at startup."
            )
        return np.asarray(ml_model_service.predict(features_df), dtype=float)


class PhysicsProductionEngine(ProductionEngine):
    """
    Motor alternativo: cadena física estándar (pvlib), sin entrenar nada.

    Reproduce la misma física que ``notebooks/modelos_train/experimento_fisica_ensemble.py``:
    GHI -> Erbs (directa/difusa) -> POA Hay-Davies (20°/sur) -> temperatura de
    celda (Faiman) -> potencia DC (PVWatts) - pérdidas de sistema.
    """

    name = "physics"

    def capacity_factor(
        self, features_df: pd.DataFrame, lat: float, lon: float
    ) -> np.ndarray:
        import pvlib

        times = features_df.index
        loc = pvlib.location.Location(lat, lon, tz="UTC", altitude=DEFAULT_ALTITUDE_M)
        solpos = loc.get_solarposition(times)
        zenith = solpos["apparent_zenith"]
        ghi = features_df["shortwave_radiation"].clip(lower=0)

        erbs = pvlib.irradiance.erbs(ghi, zenith, times)
        dni, dhi = erbs["dni"].fillna(0), erbs["dhi"].fillna(0)
        dni_extra = pvlib.irradiance.get_extra_radiation(times)

        poa = (
            pvlib.irradiance.get_total_irradiance(
                REF_TILT, REF_AZIMUTH,
                zenith, solpos["azimuth"],
                dni, ghi, dhi,
                dni_extra=dni_extra, model="haydavies",
            )["poa_global"]
            .clip(lower=0)
            .fillna(0)
        )
        tcell = pvlib.temperature.faiman(
            poa, features_df["temperature_2m"], features_df["wind_speed_10m"]
        )
        pdc = pvlib.pvsystem.pvwatts_dc(poa, tcell, pdc0=1.0, gamma_pdc=-0.004)
        cf = (pdc * (1 - SYSTEM_LOSSES)).clip(lower=0, upper=1).fillna(0)
        return cf.to_numpy(dtype=float)


    
_ENGINE_REGISTRY: Dict[str, type[ProductionEngine]] = {
    "ml": MLProductionEngine,
    "physics": PhysicsProductionEngine,
    # ========================================================================
    # MOTOR DEL ENSAMBLE DINÁMICO (añadido para la integración)
    # ========================================================================
    "ensemble": EnsembleProductionEngine,
    # ========================================================================
}


# --------------------------------------------------------------------------- #
# Servicio central
# --------------------------------------------------------------------------- #
class ProductionForecastService:
    """Predice producción solar para intervalos de tiempo, motor intercambiable."""

    def __init__(self, engine: Optional[ProductionEngine] = None):
        if engine is None:
            engine = self._engine_from_env()
        self._engine: ProductionEngine = engine

    @staticmethod
    def _engine_from_env() -> ProductionEngine:
        name = (os.environ.get("PRODUCTION_ENGINE") or "ensemble").strip().lower()
        return _ENGINE_REGISTRY.get(name, EnsembleProductionEngine)()

    # -- Strategy -- #
    @property
    def engine(self) -> ProductionEngine:
        return self._engine

    def set_engine(self, engine: "str | ProductionEngine") -> None:
        """Cambia el motor: acepta una instancia o un nombre ('ml' | 'physics')."""
        if isinstance(engine, ProductionEngine):
            self._engine = engine
            return
        key = str(engine).strip().lower()
        if key not in _ENGINE_REGISTRY:
            raise ValueError(
                f"Motor desconocido: {engine!r}. Opciones: {sorted(_ENGINE_REGISTRY)}"
            )
        self._engine = _ENGINE_REGISTRY[key]()

    @staticmethod
    def available_engines() -> List[str]:
        return sorted(_ENGINE_REGISTRY)

    # -- Datos -- #
    @staticmethod
    def _build_feature_frame(
        weather_data: Dict[str, Any],
        target_datetimes: List[datetime],
        lat: float,
        lon: float,
    ) -> pd.DataFrame:
        hourly = weather_data.get("hourly", {})
        api_times = pd.to_datetime(hourly.get("time", []), utc=True)
        weather_df = pd.DataFrame(
            {
                "temperature_2m": hourly["temperature_2m"],
                "relative_humidity_2m": hourly["relative_humidity_2m"],
                "wind_speed_10m": hourly["wind_speed_10m"],
                "cloud_cover": hourly["cloud_cover"],
                "shortwave_radiation": hourly["shortwave_radiation"],
            },
            index=api_times,
        )
        targets = pd.DatetimeIndex([pd.Timestamp(dt) for dt in target_datetimes])
        targets = (
            targets.tz_localize(LOCAL_TZ) if targets.tz is None else targets
        ).tz_convert("UTC").floor("h")
        weather_at_targets = weather_df.reindex(targets, method="nearest")
        return build_features(weather_at_targets, lat, lon)

    @staticmethod
    def _weather_source_warning() -> Optional[str]:
        try:
            active = get_active_weather_source()
        except Exception:
            return None
        if not active:
            return None
        name = (active.get("name") or "").lower()
        provider = (active.get("provider") or "").lower()
        if "open" not in name and "open" not in provider:
            return (
                f"La fuente configurada ('{active.get('name')}') se ignoró: el "
                "modelo de producción se entrenó con datos de Open-Meteo."
            )
        return None

    # -- Predicción -- #
    async def predict(
        self,
        datetimes: List[str],
        lat: float = DEFAULT_LAT,
        lon: float = DEFAULT_LON,
    ) -> List[Dict[str, Any]]:
        """
        Predice producción (capacidad de referencia) para una lista de datetimes ISO.

        El resultado es idéntico en forma al antiguo ``predict_solar_production``:
        lista de dicts con datetime, production_kw, weather, weather_source,
        weather_source_warning. La sombra y la orientación ya vienen aplicadas;
        el escalado por capacidad real lo hace el llamador.
        """
        try:
            target_datetimes = [
                datetime.fromisoformat(dt.replace("Z", "+00:00")) for dt in datetimes
            ]
        except Exception as e:
            raise ValueError(
                f"Invalid datetime format. Expected ISO (e.g. '2025-01-15T13:00:00'): {e}"
            )
        if not target_datetimes:
            return []

        min_date = (min(target_datetimes) - timedelta(days=1)).date()
        max_date = (max(target_datetimes) + timedelta(days=1)).date()

        # El clima horario viene del caché global: si el rango cae en la ventana
        # hoy+mañana se sirve de memoria; si no, hace una consulta puntual.
        try:
            weather_data = await weather_cache.get_hourly(lat, lon, min_date, max_date)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch weather data from Open-Meteo: {e}")

        features_df = self._build_feature_frame(weather_data, target_datetimes, lat, lon)

        # MOTOR (Strategy): factor de capacidad base en geometría de referencia.
        try:
            base_cf = self._engine.capacity_factor(features_df, lat, lon)
        except Exception as e:
            raise RuntimeError(f"Production engine '{self._engine.name}' failed: {e}")

        # Si el clima viene del mock (sin conexión), la predicción es de prueba.
        is_mock = bool(weather_data.get("_is_mock"))
        weather_source = "Datos de prueba (sin conexión)" if is_mock else "Open-Meteo"
        warning = self._weather_source_warning()
        if is_mock:
            warning = warning or (
                "Sin conexión con Open-Meteo: predicción calculada con datos simulados."
            )

        # Sombra (perfil 3D por hora, singleton en BD).
        shadow_slots: Dict[int, Dict[str, Any]] = {}
        try:
            profile = get_shadow_profile()
            if profile:
                shadow_slots = {s["hour"]: s for s in profile.get("slots", [])}
        except Exception:
            pass

        # Orientación/inclinación (opcional; 1.0 si los paneles no la declaran).
        orientation_factors = pd.Series(1.0, index=features_df.index)
        try:
            from .panel_service import list_panels

            panels = list_panels()
            if panels:
                orientation_factors = compute_orientation_factor_series(
                    features_df, panels, lat, lon
                )
        except Exception:
            pass

        results: List[Dict[str, Any]] = []
        for i, (dt, cf) in enumerate(zip(target_datetimes, base_cf)):
            local_hour = dt.astimezone(LOCAL_TZ).hour if dt.tzinfo else dt.hour

            shadow_factor = 1.0
            slot = shadow_slots.get(local_hour)
            if slot:
                if slot.get("prodOverride") is not None:
                    shadow_factor = slot["prodOverride"] / 100.0
                else:
                    shadow_factor = 1.0 - (slot["shadowPct"] / 100.0)

            orientation_factor = float(orientation_factors.iloc[i])

            results.append({
                "datetime": dt.isoformat(),
                "production_kw": round(float(cf) * shadow_factor * orientation_factor, 2),
                "weather": {
                    "temperature_2m": round(float(features_df.iloc[i]["temperature_2m"]), 1),
                    "relative_humidity_2m": round(float(features_df.iloc[i]["relative_humidity_2m"]), 1),
                    "wind_speed_10m": round(float(features_df.iloc[i]["wind_speed_10m"]), 1),
                    "cloud_cover": round(float(features_df.iloc[i]["cloud_cover"]), 1),
                    "shortwave_radiation": round(float(features_df.iloc[i]["shortwave_radiation"]), 1),
                },
                "weather_source": weather_source,
                "weather_source_warning": warning,
                "engine": self._engine.name,
            })

        return results


# Singleton compartido por toda la app.
production_forecast_service = ProductionForecastService()


def get_production_service() -> ProductionForecastService:
    return production_forecast_service
