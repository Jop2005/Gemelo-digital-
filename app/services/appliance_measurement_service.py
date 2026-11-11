"""
Appliance measurement upload + per-hour-of-week consumption forecasting.

Accepts TSV/CSV files exported by power-quality analyzers (e.g. Hioki PW3360
format with columns: Date, Time, P(SUM), ...). Builds a 168-bin profile
(7 days x 24 hours) with the average real power in kW for each (weekday, hour)
slot. This profile is stored on the appliance document and used to forecast
hourly consumption for any future datetime.
"""
from __future__ import annotations

import csv
import io
import math
import statistics
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

DATE_FORMATS = (
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%Y-%m-%d",
    "%Y/%m/%d",
)
TIME_FORMATS = (
    "%H:%M:%S",
    "%H:%M",
)

# Column name variants we accept for power and timestamps
POWER_COLUMNS = ("P(SUM)", "P_SUM", "P SUM", "PSUM", "P", "kW", "KW", "Power")
DATE_COLUMNS = ("Date", "DATE", "Fecha", "FECHA")
TIME_COLUMNS = ("Time", "TIME", "Hora", "HORA")
DATETIME_COLUMNS = ("Datetime", "DateTime", "TIMESTAMP", "Timestamp")


def _detect_delimiter(sample: str) -> str:
    if "\t" in sample:
        return "\t"
    if ";" in sample and sample.count(";") > sample.count(","):
        return ";"
    return ","


def _parse_datetime(date_str: str, time_str: str) -> Optional[datetime]:
    date_str = (date_str or "").strip()
    time_str = (time_str or "").strip() or "00:00:00"
    if not date_str:
        return None
    for d_fmt in DATE_FORMATS:
        for t_fmt in TIME_FORMATS:
            try:
                return datetime.strptime(f"{date_str} {time_str}", f"{d_fmt} {t_fmt}")
            except ValueError:
                continue
    return None


def _parse_single_datetime(value: str) -> Optional[datetime]:
    value = (value or "").strip()
    if not value:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _resolve_column(fieldnames: List[str], candidates: Tuple[str, ...]) -> Optional[str]:
    for name in fieldnames:
        if name and name.strip() in candidates:
            return name
    lowered = {name.strip().lower(): name for name in fieldnames if name}
    for cand in candidates:
        key = cand.lower()
        if key in lowered:
            return lowered[key]
    return None


def parse_measurement_file(content: str) -> List[Tuple[datetime, float]]:
    """
    Parse a power-meter export file. Returns list of (datetime, power_kw) pairs.

    Handles two row formats:
      - Header row at line 0, units row at line 1 (Hioki style), data follows.
      - Plain CSV/TSV with a single header row.
    """
    if not content or not content.strip():
        raise ValueError("El archivo está vacío.")

    delimiter = _detect_delimiter(content[:2048])
    reader = csv.reader(io.StringIO(content), delimiter=delimiter)
    rows = [r for r in reader if any(cell.strip() for cell in r)]
    if len(rows) < 2:
        raise ValueError("El archivo no contiene filas de datos suficientes.")

    header = [c.strip() for c in rows[0]]
    # Hioki files have a second row with units (ACV, ACA, KW...). Skip it if
    # the first data row looks like units rather than numbers.
    data_start = 1
    if len(rows) > 2:
        candidate_units = [c.strip().upper() for c in rows[1]]
        unit_tokens = {"ACV", "ACA", "KW", "KVA", "KVAR", "KWH", "KVAH", "KVARH", "HZ", "DEGREE"}
        if any(tok in unit_tokens for tok in candidate_units):
            data_start = 2

    dt_col = _resolve_column(header, DATETIME_COLUMNS)
    date_col = _resolve_column(header, DATE_COLUMNS) if not dt_col else None
    time_col = _resolve_column(header, TIME_COLUMNS) if not dt_col else None
    power_col = _resolve_column(header, POWER_COLUMNS)

    if power_col is None:
        raise ValueError(
            "No se encontró columna de potencia. Se esperaba P(SUM), kW o similar."
        )
    if dt_col is None and (date_col is None or time_col is None):
        raise ValueError(
            "No se encontraron columnas de fecha/hora. Se esperaba Date+Time o Datetime."
        )

    idx_dt = header.index(dt_col) if dt_col else None
    idx_date = header.index(date_col) if date_col else None
    idx_time = header.index(time_col) if time_col else None
    idx_power = header.index(power_col)

    samples: List[Tuple[datetime, float]] = []
    for row in rows[data_start:]:
        if idx_power >= len(row):
            continue
        raw_power = row[idx_power].strip()
        if not raw_power:
            continue
        try:
            power_kw = float(raw_power)
        except ValueError:
            continue
        if idx_dt is not None:
            dt = _parse_single_datetime(row[idx_dt] if idx_dt < len(row) else "")
        else:
            d = row[idx_date] if idx_date is not None and idx_date < len(row) else ""
            t = row[idx_time] if idx_time is not None and idx_time < len(row) else ""
            dt = _parse_datetime(d, t)
        if dt is None:
            continue
        samples.append((dt, power_kw))

    if not samples:
        raise ValueError("No se pudo extraer ninguna muestra válida del archivo.")
    return samples


def build_hourly_profile(
    samples: List[Tuple[datetime, float]],
    uncovered_fill: str = "mean",
) -> Dict[str, Any]:
    """
    Build a 168-element profile (Monday=0 .. Sunday=6, hours 0..23) by
    averaging real power within each (weekday, hour) bin.

    `uncovered_fill` controls hours with no samples:
      - "mean": fall back to the global mean (profile always dense).
      - "zero": treat uncovered hours as 0 kW (equipo apagado).
    """
    if not samples:
        raise ValueError("No hay muestras para construir el perfil.")

    bins: List[List[float]] = [[] for _ in range(168)]
    for dt, power in samples:
        bin_idx = dt.weekday() * 24 + dt.hour
        bins[bin_idx].append(power)

    flat_powers = [p for _, p in samples]
    global_mean = sum(flat_powers) / len(flat_powers)
    fill_value = 0.0 if uncovered_fill == "zero" else global_mean

    profile: List[float] = []
    for slot in bins:
        if slot:
            profile.append(sum(slot) / len(slot))
        else:
            profile.append(fill_value)

    timestamps = [dt for dt, _ in samples]
    meta = {
        "samples": len(samples),
        "firstDate": min(timestamps).isoformat(),
        "lastDate": max(timestamps).isoformat(),
        "avgKw": round(global_mean, 4),
        "minKw": round(min(flat_powers), 4),
        "maxKw": round(max(flat_powers), 4),
        "stdKw": round(statistics.pstdev(flat_powers), 4) if len(flat_powers) > 1 else 0.0,
        "hoursCovered": sum(1 for slot in bins if slot),
    }
    return {"hourlyProfileKw": [round(v, 4) for v in profile], "meta": meta}


def forecast_kw(profile: List[float], dt: datetime) -> float:
    """Look up the average kW for the given datetime's (weekday, hour) slot."""
    if not profile or len(profile) != 168:
        return 0.0
    idx = dt.weekday() * 24 + dt.hour
    value = profile[idx]
    return value if (isinstance(value, (int, float)) and math.isfinite(value)) else 0.0


def forecast_series(profile: List[float], start: datetime, hours: int) -> List[Dict[str, Any]]:
    """Produce a list of {datetime, kW} forecasts for `hours` hours from start."""
    from datetime import timedelta

    out: List[Dict[str, Any]] = []
    for i in range(hours):
        dt = start + timedelta(hours=i)
        out.append({"datetime": dt.isoformat(), "kW": forecast_kw(profile, dt)})
    return out
