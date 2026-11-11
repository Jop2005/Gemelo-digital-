"""
Caché global de clima (singleton) para Open-Meteo.

Mantiene en memoria SOLO el clima de hoy + mañana y se reutiliza durante
``TTL_SECONDS`` (5 min). Pasado el TTL, la siguiente lectura refresca y
**sobrescribe** esos mismos valores: no se acumula histórico.

Guarda dos cosas, cada una con su propia frescura y lock de single-flight (de
modo que N llamadas concurrentes provoquen una sola petición a Open-Meteo):

* **Ventana horaria** (features del predictor de producción): hoy ± 1 día y
  mañana + 1 día, en UTC. El ±1 día es el margen técnico que el modelo ya usaba
  para el ``reindex(method="nearest")`` en los bordes del día; no es histórico.
* **Snapshot de display** (``current`` + ``daily`` 7 días, timezone local): una
  sola petición que alimenta a ``fetch_open_meteo_weather``.

Rangos de fecha fuera de la ventana hoy+mañana NO se cachean: el llamador recibe
una consulta puntual directa a Open-Meteo (igual que antes).
"""
from __future__ import annotations

import asyncio
import math
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict, Optional, Tuple

import httpx

OPENMETEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"
# Archivo histórico (reanálisis ERA5): cubre desde 1940 hasta ~5 días atrás, con
# las mismas variables horarias y sin huecos. Lo usamos para el modo "Histórico"
# (fechas que el endpoint /forecast ya no devuelve de forma fiable).
OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
# A partir de cuántos días atrás conviene pedir al archivo en lugar de /forecast.
ARCHIVE_CUTOFF_DAYS = 5
LOCAL_TZ = ZoneInfo("America/Havana")
LOCAL_UTC_OFFSET_H = 5  # La Habana = UTC-5 (constante en el mock, sin DST)
TTL_SECONDS = 300       # 5 minutos: refresco normal
MOCK_RETRY_SECONDS = 60  # cuando estamos en mock, reintentar la red antes

_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "cloud_cover",
    "shortwave_radiation",
]
_CURRENT_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "cloud_cover",
    "shortwave_radiation",
    "weather_code",
]
_DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "weather_code",
    "cloud_cover_mean",
    "shortwave_radiation_sum",
]


async def _fetch_hourly(
    lat: float, lon: float, start_date: date, end_date: date, *, archive: bool = False
) -> Dict[str, Any]:
    """
    Petición horaria (UTC) a Open-Meteo para un rango de fechas.

    Con ``archive=True`` se consulta el Archive API (reanálisis ERA5), válido para
    fechas históricas; con ``archive=False`` el endpoint /forecast (reciente+futuro).
    Ambos devuelven exactamente las mismas variables, así que el predictor no nota
    la diferencia.
    """
    base_url = OPENMETEO_ARCHIVE_URL if archive else OPENMETEO_BASE_URL
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(_HOURLY_VARS),
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "timezone": "UTC",
    }
    # 25 s: los rangos largos del Archive API (medio año horario) son pesados.
    async with httpx.AsyncClient(timeout=25) as client:
        response = await client.get(base_url, params=params)
    response.raise_for_status()
    return response.json()


async def _fetch_display(lat: float, lon: float) -> Dict[str, Any]:
    """Petición de display (``current`` + ``daily`` 7 días) en una sola llamada."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ",".join(_CURRENT_VARS),
        "daily": ",".join(_DAILY_VARS),
        "forecast_days": 7,
        "timezone": "auto",
    }
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get(OPENMETEO_BASE_URL, params=params)
    response.raise_for_status()
    return response.json()


def _key(lat: float, lon: float) -> Tuple[float, float]:
    return (round(lat, 3), round(lon, 3))


def _window_bounds() -> Tuple[date, date]:
    """Rango UTC que cubre hoy y mañana (con ±1 día de margen para el reindex)."""
    today = datetime.now(LOCAL_TZ).date()
    return today - timedelta(days=1), today + timedelta(days=2)


# --------------------------------------------------------------------------- #
# Datos de prueba (mock) deterministas — para la tesis cuando no hay conexión.
# No usan aleatoriedad: el mismo día producen siempre los mismos valores, de
# modo que las gráficas son reproducibles. Llevan la marca ``_is_mock``.
# --------------------------------------------------------------------------- #
def _bell_radiation(local_hour: int) -> float:
    """Curva campana de radiación (W/m²): 0 de noche, pico ~900 al mediodía."""
    if local_hour < 6 or local_hour > 19:
        return 0.0
    return 900.0 * math.exp(-((local_hour - 13) ** 2) / (2 * 9.0))


def _mock_hourly_raw(start_date: date, end_date: date) -> Dict[str, Any]:
    """Hourly con la forma de Open-Meteo (UTC) para un rango de fechas."""
    t = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    end = datetime(end_date.year, end_date.month, end_date.day, 23, tzinfo=timezone.utc)
    times, temps, hums, winds, clouds, rads = [], [], [], [], [], []
    while t <= end:
        local_hour = (t.hour - LOCAL_UTC_OFFSET_H) % 24
        rad = _bell_radiation(local_hour)
        times.append(t.strftime("%Y-%m-%dT%H:%M"))
        temps.append(round(22.0 + 8.0 * (rad / 900.0), 1))
        hums.append(70.0)
        winds.append(10.0)
        clouds.append(20.0)
        rads.append(round(rad, 1))
        t += timedelta(hours=1)
    return {
        "hourly": {
            "time": times,
            "temperature_2m": temps,
            "relative_humidity_2m": hums,
            "wind_speed_10m": winds,
            "cloud_cover": clouds,
            "shortwave_radiation": rads,
        },
        "_is_mock": True,
    }


def _mock_display_raw() -> Dict[str, Any]:
    """``current`` + ``daily`` (7 días) con la forma de Open-Meteo."""
    now = datetime.now(LOCAL_TZ)
    rad = _bell_radiation(now.hour)
    current = {
        "temperature_2m": round(22.0 + 8.0 * (rad / 900.0), 1),
        "relative_humidity_2m": 70.0,
        "wind_speed_10m": 10.0,
        "cloud_cover": 20.0,
        "shortwave_radiation": round(rad, 1),
        "weather_code": 1,
    }
    today = now.date()
    daily: Dict[str, Any] = {
        "time": [], "temperature_2m_max": [], "temperature_2m_min": [],
        "weather_code": [], "cloud_cover_mean": [], "shortwave_radiation_sum": [],
    }
    for i in range(7):
        d = today + timedelta(days=i)
        daily["time"].append(d.strftime("%Y-%m-%d"))
        daily["temperature_2m_max"].append(31.0)
        daily["temperature_2m_min"].append(23.0)
        daily["weather_code"].append(1)
        daily["cloud_cover_mean"].append(20.0)
        daily["shortwave_radiation_sum"].append(20.0)
    return {"current": current, "daily": daily, "_is_mock": True}


class WeatherCache:
    """Caché en memoria de hoy+mañana, refrescado cada ``TTL_SECONDS``."""

    def __init__(self) -> None:
        # Ventana horaria (predictor)
        self._hourly_lock = asyncio.Lock()
        self._hourly_raw: Optional[Dict[str, Any]] = None
        self._hourly_at: Optional[float] = None
        self._hourly_key: Optional[Tuple[float, float]] = None
        self._hourly_bounds: Optional[Tuple[date, date]] = None
        self._hourly_is_mock: bool = False
        # Snapshot de display
        self._display_lock = asyncio.Lock()
        self._display_raw: Optional[Dict[str, Any]] = None
        self._display_at: Optional[float] = None
        self._display_key: Optional[Tuple[float, float]] = None
        self._display_is_mock: bool = False

    @staticmethod
    def _ttl(is_mock: bool) -> float:
        # En mock reintentamos la red antes, para recuperarnos rápido al volver.
        return MOCK_RETRY_SECONDS if is_mock else TTL_SECONDS

    # ------------------------------------------------------------------ #
    # Ventana horaria (predictor)
    # ------------------------------------------------------------------ #
    def _hourly_fresh(self, key: Tuple[float, float], bounds: Tuple[date, date]) -> bool:
        return (
            self._hourly_raw is not None
            and self._hourly_key == key
            and self._hourly_bounds == bounds
            and self._hourly_at is not None
            and (time.monotonic() - self._hourly_at) < self._ttl(self._hourly_is_mock)
        )

    async def _ensure_window(self, lat: float, lon: float) -> None:
        key = _key(lat, lon)
        bounds = _window_bounds()
        if self._hourly_fresh(key, bounds):
            return
        async with self._hourly_lock:
            if self._hourly_fresh(key, bounds):  # otro coroutine lo refrescó
                return
            try:
                raw = await _fetch_hourly(lat, lon, bounds[0], bounds[1])
                is_mock = False
            except Exception:
                # Sin conexión: almacenamos datos de prueba deterministas y los
                # marcamos como mock para que el dashboard avise.
                raw = _mock_hourly_raw(bounds[0], bounds[1])
                is_mock = True
            self._hourly_raw = raw
            self._hourly_at = time.monotonic()
            self._hourly_key = key
            self._hourly_bounds = bounds
            self._hourly_is_mock = is_mock

    async def get_hourly(
        self, lat: float, lon: float, start_date: date, end_date: date
    ) -> Dict[str, Any]:
        """
        Devuelve datos horarios de Open-Meteo para ``[start_date, end_date]``.

        Si el rango cae dentro de la ventana hoy+mañana en caché, se sirve de
        memoria; si no, se hace una consulta puntual (no cacheada).
        """
        key = _key(lat, lon)
        bounds = _window_bounds()
        within = start_date >= bounds[0] and end_date <= bounds[1]
        if within:
            await self._ensure_window(lat, lon)
            if self._hourly_raw is not None and self._hourly_key == key:
                return self._hourly_raw
        # Fuera de ventana: consulta puntual (no cacheada). Para fechas claramente
        # pasadas usamos el Archive API (ERA5), que sí las devuelve; para el resto,
        # /forecast. Si el archivo aún no cubre una fecha muy reciente, reintentamos
        # con /forecast antes de caer a datos de prueba.
        today = datetime.now(LOCAL_TZ).date()
        use_archive = end_date <= today - timedelta(days=ARCHIVE_CUTOFF_DAYS)
        try:
            return await _fetch_hourly(lat, lon, start_date, end_date, archive=use_archive)
        except Exception:
            if use_archive:
                try:
                    return await _fetch_hourly(lat, lon, start_date, end_date, archive=False)
                except Exception:
                    pass
            return _mock_hourly_raw(start_date, end_date)

    # ------------------------------------------------------------------ #
    # Snapshot de display
    # ------------------------------------------------------------------ #
    def _display_fresh(self, key: Tuple[float, float]) -> bool:
        return (
            self._display_raw is not None
            and self._display_key == key
            and self._display_at is not None
            and (time.monotonic() - self._display_at) < self._ttl(self._display_is_mock)
        )

    async def get_display_raw(self, lat: float, lon: float) -> Dict[str, Any]:
        """JSON crudo de Open-Meteo (``current`` + ``daily``) para el display."""
        key = _key(lat, lon)
        if self._display_fresh(key):
            return self._display_raw  # type: ignore[return-value]
        async with self._display_lock:
            if self._display_fresh(key):
                return self._display_raw  # type: ignore[return-value]
            try:
                raw = await _fetch_display(lat, lon)
                is_mock = False
            except Exception:
                # Sin conexión: datos de prueba deterministas, marcados como mock.
                raw = _mock_display_raw()
                is_mock = True
            self._display_raw = raw
            self._display_at = time.monotonic()
            self._display_key = key
            self._display_is_mock = is_mock
            return raw

    def reset(self) -> None:
        """Vacía el caché (usado en tests para aislar cada caso)."""
        self._hourly_raw = self._hourly_at = self._hourly_key = self._hourly_bounds = None
        self._display_raw = self._display_at = self._display_key = None
        self._hourly_is_mock = self._display_is_mock = False

    # ------------------------------------------------------------------ #
    # Arranque / refresco proactivo
    # ------------------------------------------------------------------ #
    async def preload(self, lat: float, lon: float) -> None:
        """Calienta ambos cachés al arrancar (sin propagar errores de red)."""
        bounds = _window_bounds()
        try:
            await self.get_display_raw(lat, lon)
        except Exception:
            pass
        try:
            await self.get_hourly(lat, lon, bounds[0], bounds[1])
        except Exception:
            pass


# Singleton global compartido por toda la app.
weather_cache = WeatherCache()
