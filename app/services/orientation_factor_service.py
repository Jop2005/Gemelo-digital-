"""
Corrección geométrica por inclinación y orientación del panel.

El modelo ML predice el factor de capacidad de un panel de REFERENCIA: 20° de
inclinación mirando al sur (la geometría con la que PVGIS generó la etiqueta de
entrenamiento). Un panel real con otra inclinación/orientación recibe más o menos
sol según la posición del astro a cada hora.

Este módulo calcula, para cada instante, el cociente

    factor(hora) = POA(inclinación_real, orientación_real)
                   ---------------------------------------
                   POA(20°, sur)                 ← referencia del modelo

usando la misma cadena física que ``experimento_fisica_ensemble.py``
(descomposición Erbs de la GHI + transposición Hay-Davies con pvlib). El factor
depende de la hora porque la posición del sol cambia a lo largo del día.

Es 100% OPCIONAL: la inclinación y la orientación son atributos opcionales del
panel. Si un panel no las trae, contribuye con factor 1.0 (la referencia) y el
resultado es idéntico al comportamiento anterior. Con varios paneles, el factor
del sistema es el promedio ponderado por la capacidad instalada de cada uno.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .solar_features import DEFAULT_LAT, DEFAULT_LON, DEFAULT_ALTITUDE_M

# Geometría de referencia del modelo (debe coincidir con la del entrenamiento:
# PVGIS angle=20, aspect=0 → 20° sur; aspect/azimut 180 = sur en pvlib).
REF_TILT = 20.0
REF_AZIMUTH = 180.0

# Orientación textual (frontend) → azimut pvlib (0=N, 90=E, 180=S, 270=O).
_ORIENTATION_AZIMUTH: Dict[str, float] = {
    "norte": 0.0, "noreste": 45.0, "este": 90.0, "sureste": 135.0,
    "sur": 180.0, "suroeste": 225.0, "oeste": 270.0, "noroeste": 315.0,
    # alias cortos y en inglés por robustez
    "n": 0.0, "ne": 45.0, "e": 90.0, "se": 135.0,
    "s": 180.0, "so": 225.0, "sw": 225.0, "o": 270.0, "w": 270.0, "no": 315.0, "nw": 315.0,
    "north": 0.0, "east": 90.0, "south": 180.0, "west": 270.0,
}


def azimuth_from_orientation(orientation: Optional[str]) -> Optional[float]:
    """Convierte la orientación textual a azimut pvlib, o None si no se reconoce."""
    if not orientation:
        return None
    return _ORIENTATION_AZIMUTH.get(str(orientation).strip().lower())


def _poa_global(solpos, ghi, dni, dhi, dni_extra, tilt: float, azimuth: float) -> pd.Series:
    """Irradiancia en el plano del panel (W/m²) para una geometría dada."""
    import pvlib

    return (
        pvlib.irradiance.get_total_irradiance(
            tilt, azimuth,
            solpos["apparent_zenith"], solpos["azimuth"],
            dni, ghi, dhi,
            dni_extra=dni_extra, model="haydavies",
        )["poa_global"]
        .clip(lower=0)
        .fillna(0)
    )


def compute_orientation_factor_series(
    weather_df: pd.DataFrame,
    panels: List[Dict[str, Any]],
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    altitude: float = DEFAULT_ALTITUDE_M,
) -> pd.Series:
    """
    Factor de corrección geométrica por hora, ponderado por capacidad de panel.

    Args:
        weather_df: DataFrame indexado por DatetimeIndex UTC con la columna
            ``shortwave_radiation`` (GHI en W/m²).
        panels: lista de paneles (dicts de ``panel_service``) con campos
            opcionales ``tiltDegrees`` y ``orientation``, y para el peso
            ``ratedPowerKw`` y ``quantity``.
        lat, lon, altitude: ubicación para la geometría solar (pvlib).

    Returns:
        Serie de factores (mismo índice que ``weather_df``). Vale 1.0 cuando no
        hay paneles con geometría declarada o cuando la referencia no recibe sol.
    """
    times = weather_df.index
    identity = pd.Series(1.0, index=times)

    # ¿Hay al menos un panel con inclinación Y orientación reconocible?
    oriented = [
        p for p in (panels or [])
        if p.get("tiltDegrees") is not None
        and azimuth_from_orientation(p.get("orientation")) is not None
    ]
    if not oriented:
        return identity

    try:
        import pvlib

        loc = pvlib.location.Location(lat, lon, tz="UTC", altitude=altitude)
        solpos = loc.get_solarposition(times)
        ghi = weather_df["shortwave_radiation"].clip(lower=0)
        zenith = solpos["apparent_zenith"]

        # GHI -> directa/difusa (Erbs) -> transposición al plano (Hay-Davies)
        erbs = pvlib.irradiance.erbs(ghi, zenith, times)
        dni, dhi = erbs["dni"].fillna(0), erbs["dhi"].fillna(0)
        dni_extra = pvlib.irradiance.get_extra_radiation(times)

        ref_poa = _poa_global(solpos, ghi, dni, dhi, dni_extra, REF_TILT, REF_AZIMUTH)
        # Evita dividir por cero (noche / sol bajo): esas horas quedan como factor 1.
        ref_safe = ref_poa.replace(0, np.nan)

        weighted = pd.Series(0.0, index=times)
        total_weight = 0.0
        for p in panels:
            cap = float(p.get("ratedPowerKw") or 0.0) * float(p.get("quantity") or 1)
            if cap <= 0:
                cap = float(p.get("quantity") or 1)  # peso de respaldo si falta kW

            tilt = p.get("tiltDegrees")
            az = azimuth_from_orientation(p.get("orientation"))
            if tilt is None or az is None:
                # Panel sin geometría declarada: aporta a la referencia (factor 1).
                ratio = identity
            else:
                poa = _poa_global(solpos, ghi, dni, dhi, dni_extra, float(tilt), float(az))
                # 1.5 de tope: una inclinación óptima supera algo a la referencia,
                # pero acota artefactos numéricos con sol muy bajo.
                ratio = (poa / ref_safe).clip(lower=0.0, upper=1.5).fillna(1.0)

            weighted = weighted + ratio * cap
            total_weight += cap

        if total_weight <= 0:
            return identity

        return (weighted / total_weight).fillna(1.0)
    except Exception:
        # Ante cualquier fallo (pvlib, datos), no corregir: devolver la identidad.
        return identity
