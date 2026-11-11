"""
Appliance measurement batches — stores raw Hioki readings and manages batch metadata.

Collections:
  mediciones_equipos  — raw per-minute readings: {applianceId, timestamp, powerKw, batchId}
  mediciones_lotes    — batch metadata with estimation snapshot at upload time

Design:
  - Upload accumulates data (upsert by applianceId+timestamp), never full-replace.
  - Each upload creates a new batch record with a snapshot of what was estimated
    for (a) this appliance and (b) all other appliances at upload time.
  - Deleting a batch removes its readings and re-rebuilds the 168-bin profile.
  - Daily report computes consumption from real readings where available,
    estimated from appliance config for the rest.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone, date
from typing import Any, Dict, List, Optional, Tuple

from bson import ObjectId

from app.database import get_database
from app.services.appliance_measurement_service import (
    build_hourly_profile,
    parse_measurement_file,
)

READINGS_COL = "mediciones_equipos"
BATCHES_COL = "mediciones_lotes"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _readings():
    return get_database()[READINGS_COL]


def _batches():
    return get_database()[BATCHES_COL]


def _ensure_indexes() -> None:
    r = _readings()
    r.create_index([("applianceId", 1), ("timestamp", 1)], unique=True)
    r.create_index("timestamp")
    r.create_index("batchId")
    b = _batches()
    b.create_index("batchId", unique=True)
    b.create_index("applianceId")


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _serialize_batch(doc: Dict[str, Any]) -> Dict[str, Any]:
    def _iso(v):
        return v.isoformat() if isinstance(v, datetime) else (str(v) if v else None)
    return {
        "batchId": doc.get("batchId", ""),
        "applianceId": doc.get("applianceId", ""),
        "applianceName": doc.get("applianceName", ""),
        "filename": doc.get("filename", ""),
        "uploadedAt": _iso(doc.get("uploadedAt")),
        "startDate": _iso(doc.get("startDate")),
        "endDate": _iso(doc.get("endDate")),
        "samples": doc.get("samples", 0),
        "kwhDayEstimatedThis": doc.get("kwhDayEstimatedThis", 0.0),
        "kwhDayEstimatedOthers": doc.get("kwhDayEstimatedOthers", 0.0),
    }


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload_batch(
    appliance_id: str,
    appliance_name: str,
    file_content: str,
    filename: str = "archivo.xls",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    kwh_day_estimated_this: float = 0.0,
    kwh_day_estimated_others: float = 0.0,
) -> Dict[str, Any]:
    """
    Parse a Hioki export file, filter by optional date range, upsert raw
    per-minute readings into mediciones_equipos, insert a batch metadata
    record, and rebuild the 168-bin hourly profile on the appliance document.

    Returns the batch metadata dict.
    """
    _ensure_indexes()

    # 1. Parse raw samples from file
    samples: List[Tuple[datetime, float]] = parse_measurement_file(file_content)
    if not samples:
        raise ValueError("El archivo no contiene lecturas válidas.")

    # 2. Apply optional date-range filter
    start_dt = _parse_iso(start_date)
    end_dt = _parse_iso(end_date)
    if start_dt or end_dt:
        filtered = []
        for dt, kw in samples:
            # Make dt timezone-aware for comparison if needed
            dt_cmp = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
            if start_dt:
                s_cmp = start_dt.replace(tzinfo=timezone.utc) if start_dt.tzinfo is None else start_dt
                if dt_cmp < s_cmp:
                    continue
            if end_dt:
                e_cmp = end_dt.replace(tzinfo=timezone.utc) if end_dt.tzinfo is None else end_dt
                if dt_cmp > e_cmp:
                    continue
            filtered.append((dt, kw))
        samples = filtered

    if not samples:
        raise ValueError("No hay lecturas dentro del rango de fechas especificado.")

    # 3. Upsert readings into mediciones_equipos
    batch_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    ops = []
    for dt, kw in samples:
        dt_utc = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
        ops.append({
            "filter": {"applianceId": appliance_id, "timestamp": dt_utc},
            "update": {
                "$set": {
                    "applianceId": appliance_id,
                    "timestamp": dt_utc,
                    "powerKw": round(kw, 4),
                    "batchId": batch_id,
                }
            },
            "upsert": True,
        })

    col = _readings()
    for op in ops:
        col.update_one(op["filter"], op["update"], upsert=True)

    # 4. Insert batch metadata
    timestamps = [dt for dt, _ in samples]
    batch_doc = {
        "batchId": batch_id,
        "applianceId": appliance_id,
        "applianceName": appliance_name,
        "filename": filename,
        "uploadedAt": now,
        "startDate": min(timestamps).replace(tzinfo=timezone.utc) if min(timestamps).tzinfo is None else min(timestamps),
        "endDate": max(timestamps).replace(tzinfo=timezone.utc) if max(timestamps).tzinfo is None else max(timestamps),
        "samples": len(samples),
        "kwhDayEstimatedThis": round(kwh_day_estimated_this, 4),
        "kwhDayEstimatedOthers": round(kwh_day_estimated_others, 4),
    }
    _batches().insert_one(batch_doc)

    # 5. Rebuild 168-bin profile from ALL accumulated readings for this appliance
    _rebuild_profile(appliance_id)

    return _serialize_batch(batch_doc)


# ---------------------------------------------------------------------------
# Delete batch
# ---------------------------------------------------------------------------

def delete_batch(batch_id: str) -> bool:
    batch = _batches().find_one({"batchId": batch_id})
    if not batch:
        return False

    appliance_id = batch["applianceId"]
    _readings().delete_many({"batchId": batch_id})
    _batches().delete_one({"batchId": batch_id})
    _rebuild_profile(appliance_id)
    return True


# ---------------------------------------------------------------------------
# List batches
# ---------------------------------------------------------------------------

def list_batches(appliance_id: str) -> List[Dict[str, Any]]:
    docs = _batches().find({"applianceId": appliance_id}).sort("uploadedAt", -1)
    return [_serialize_batch(d) for d in docs]


# ---------------------------------------------------------------------------
# Profile rebuild (internal)
# ---------------------------------------------------------------------------

def _rebuild_profile(appliance_id: str) -> None:
    """Rebuild and persist the 168-bin hourly profile from all stored readings."""
    from app.database import get_database as _db
    from datetime import datetime as _dt

    readings = list(
        _readings()
        .find({"applianceId": appliance_id}, {"timestamp": 1, "powerKw": 1, "_id": 0})
        .sort("timestamp", 1)
    )
    if not readings:
        # No readings left — clear the profile from the appliance document
        _db()["electrodomesticos"].update_one(
            {"_id": ObjectId(appliance_id)},
            {"$unset": {"hourlyProfileKw": "", "measurementMeta": ""},
             "$set": {"updatedAt": _dt.utcnow()}},
        )
        return

    appliance_doc = _db()["electrodomesticos"].find_one(
        {"_id": ObjectId(appliance_id)}, {"uncoveredHoursFill": 1}
    ) or {}
    uncovered_fill = appliance_doc.get("uncoveredHoursFill") or "mean"

    samples = [(r["timestamp"], r["powerKw"]) for r in readings]
    profile = build_hourly_profile(samples, uncovered_fill=uncovered_fill)
    meta = profile["meta"]
    avg_w = round(float(meta["avgKw"]) * 1000.0, 2)
    max_w = round(float(meta["maxKw"]) * 1000.0, 2)

    _db()["electrodomesticos"].update_one(
        {"_id": ObjectId(appliance_id)},
        {
            "$set": {
                "hourlyProfileKw": profile["hourlyProfileKw"],
                "measurementMeta": meta,
                "averagePowerW": avg_w,
                "maxPowerW": max_w,
                "measuredPowerW": avg_w,
                "updatedAt": _dt.utcnow(),
            }
        },
    )


# ---------------------------------------------------------------------------
# Appliance readings for a date range
# ---------------------------------------------------------------------------

def get_appliance_readings(
    appliance_id: str,
    start_date: str,
    end_date: str,
) -> List[Dict[str, Any]]:
    """Return minute-level readings for one appliance in a date range."""
    start_dt = _parse_iso(start_date)
    end_dt = _parse_iso(end_date)
    query: Dict[str, Any] = {"applianceId": appliance_id}
    ts_filter: Dict[str, Any] = {}
    if start_dt:
        ts_filter["$gte"] = start_dt.replace(tzinfo=timezone.utc) if start_dt.tzinfo is None else start_dt
    if end_dt:
        ts_filter["$lte"] = end_dt.replace(tzinfo=timezone.utc) if end_dt.tzinfo is None else end_dt
    if ts_filter:
        query["timestamp"] = ts_filter

    docs = _readings().find(query, {"_id": 0, "timestamp": 1, "powerKw": 1}).sort("timestamp", 1)
    return [
        {"timestamp": d["timestamp"].isoformat(), "powerKw": d["powerKw"]}
        for d in docs
    ]


# ---------------------------------------------------------------------------
# Daily report
# ---------------------------------------------------------------------------

def get_daily_report(date_str: str) -> Dict[str, Any]:
    """
    Build the daily consumption + production report for a given date (YYYY-MM-DD).

    Returns:
      date, productionKwh (from lecturas_historicas, may be None),
      measuredConsumptionKwh, estimatedConsumptionKwh, totalConsumptionKwh,
      appliances list with per-appliance breakdown and error analysis.
    """
    from app.services.lectura_service import get_day_production
    from app.database import get_database as _db

    try:
        day = date.fromisoformat(date_str)
    except ValueError:
        raise ValueError(f"Formato de fecha inválido: {date_str}. Use YYYY-MM-DD.")

    day_start = datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=timezone.utc)
    day_end = datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=timezone.utc)

    # 1. Production for the day (kWh = sum of kW readings × interval)
    production_kwh = get_day_production(date_str)

    # 2. Fetch all appliances
    appliance_docs = list(_db()["electrodomesticos"].find({}))

    # 3. For each appliance with Hioki data for this day, aggregate kWh
    #    (readings are 1-minute intervals → each reading ≈ powerKw × (1/60) kWh)
    appliance_results = []
    measured_total = 0.0
    estimated_total = 0.0

    for appl in appliance_docs:
        appl_id = str(appl["_id"])
        name = appl.get("name", "Sin nombre")
        quantity = appl.get("quantity", 1) or 1

        # Check for readings in this day
        readings = list(
            _readings()
            .find(
                {"applianceId": appl_id, "timestamp": {"$gte": day_start, "$lte": day_end}},
                {"_id": 0, "timestamp": 1, "powerKw": 1},
            )
            .sort("timestamp", 1)
        )

        if readings:
            # Real Hioki data — each reading is ~1 minute → kWh = kW × (1/60)
            real_kwh = sum(r["powerKw"] for r in readings) / 60.0 * quantity
            real_kwh = round(real_kwh, 4)

            # Find estimation snapshot for this day from batch metadata
            covering_batch = _batches().find_one(
                {
                    "applianceId": appl_id,
                    "startDate": {"$lte": day_end},
                    "endDate": {"$gte": day_start},
                },
                sort=[("uploadedAt", -1)],
            )
            kwh_estimated_this = covering_batch.get("kwhDayEstimatedThis", 0.0) if covering_batch else 0.0
            error_pct = None
            if kwh_estimated_this > 0 and real_kwh > 0:
                error_pct = round((kwh_estimated_this - real_kwh) / real_kwh * 100, 2)

            appliance_results.append({
                "applianceId": appl_id,
                "name": name,
                "mode": "medido",
                "kwhDay": real_kwh,
                "kwhDayEstimated": kwh_estimated_this if kwh_estimated_this > 0 else None,
                "errorPercent": error_pct,
                "readingCount": len(readings),
            })
            measured_total += real_kwh
        else:
            # No Hioki data — estimate from config
            avg_w = appl.get("averagePowerW") or 0.0
            active_h = appl.get("activeHours") or 0.0
            est_kwh = round(avg_w * active_h / 1000.0 * quantity, 4)

            appliance_results.append({
                "applianceId": appl_id,
                "name": name,
                "mode": "estimado",
                "kwhDay": est_kwh,
                "kwhDayEstimated": None,
                "errorPercent": None,
                "readingCount": 0,
            })
            estimated_total += est_kwh

    return {
        "date": date_str,
        "productionKwh": production_kwh,
        "measuredConsumptionKwh": round(measured_total, 4),
        "estimatedConsumptionKwh": round(estimated_total, 4),
        "totalConsumptionKwh": round(measured_total + estimated_total, 4),
        "hasRealData": measured_total > 0,
        "appliances": appliance_results,
    }


# ---------------------------------------------------------------------------
# Preview: detect date range from file without uploading
# ---------------------------------------------------------------------------

def preview_batch(file_content: str) -> Dict[str, Any]:
    """Parse a file and return detected date range and sample count without storing anything."""
    samples = parse_measurement_file(file_content)
    if not samples:
        return {"samples": 0, "startDate": None, "endDate": None}
    timestamps = [dt for dt, _ in samples]
    start = min(timestamps)
    end = max(timestamps)
    return {
        "samples": len(samples),
        "startDate": (start.replace(tzinfo=timezone.utc) if start.tzinfo is None else start).isoformat(),
        "endDate": (end.replace(tzinfo=timezone.utc) if end.tzinfo is None else end).isoformat(),
    }
