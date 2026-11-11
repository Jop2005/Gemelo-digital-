"""
Historical production readings — persists solar model snapshots for trend analysis.
Collection: lecturas_historicas

Stores only productionKw (from the solar model) every 5 minutes.
Consumption is computed on-demand from appliance data (medicion_service).
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from app.database import get_database

COLLECTION_NAME = "lecturas_historicas"


def _col():
    return get_database()[COLLECTION_NAME]


def _ensure_indexes() -> None:
    col = _col()
    col.create_index("timestamp")
    col.create_index([("timestamp", -1)])


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def save_reading(production_kw: float) -> str:
    """Persist a solar production snapshot. Returns the inserted id as string."""
    _ensure_indexes()
    now = datetime.now(timezone.utc)
    doc = {
        "timestamp": now,
        "productionKw": round(float(production_kw), 3),
    }
    result = _col().insert_one(doc)
    return str(result.inserted_id)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def get_readings(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 288,
) -> List[Dict[str, Any]]:
    """Query historical production readings with optional date range filter."""
    query: Dict[str, Any] = {}
    if start_date or end_date:
        ts_filter: Dict[str, Any] = {}
        if start_date:
            ts_filter["$gte"] = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        if end_date:
            ts_filter["$lte"] = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        query["timestamp"] = ts_filter

    cursor = _col().find(query).sort("timestamp", 1).limit(max(1, min(limit, 10_000)))
    return [_serialize(doc) for doc in cursor]


def get_daily_summaries(days: int = 30) -> List[Dict[str, Any]]:
    """Aggregate production readings into daily totals."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, min(days, 365)))

    pipeline = [
        {"$match": {"timestamp": {"$gte": cutoff}}},
        {
            "$group": {
                "_id": {
                    "year": {"$year": "$timestamp"},
                    "month": {"$month": "$timestamp"},
                    "day": {"$dayOfMonth": "$timestamp"},
                },
                "date": {"$first": "$timestamp"},
                # Readings every 5 min → kWh = kW × (5/60)
                "totalProductionKwh": {"$sum": {"$multiply": ["$productionKw", 0.08333]}},
                "maxProductionKw": {"$max": "$productionKw"},
                "count": {"$sum": 1},
            }
        },
        {"$sort": {"_id": 1}},
    ]
    results = list(_col().aggregate(pipeline))
    summaries = []
    for r in results:
        d = r["date"]
        date_str = d.strftime("%Y-%m-%d") if isinstance(d, datetime) else str(d)
        summaries.append({
            "date": date_str,
            "totalProductionKwh": round(r["totalProductionKwh"], 2),
            "maxProductionKw": round(r["maxProductionKw"], 2),
            "readingCount": r["count"],
        })
    return summaries


def get_day_production(date_str: str) -> Optional[float]:
    """
    Return total production kWh for a given date (YYYY-MM-DD).
    Returns None if no readings exist for that day.
    """
    try:
        day = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None

    day_start = day.replace(tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)

    pipeline = [
        {"$match": {"timestamp": {"$gte": day_start, "$lt": day_end}}},
        {"$group": {"_id": None, "total": {"$sum": {"$multiply": ["$productionKw", 0.08333]}}, "count": {"$sum": 1}}},
    ]
    results = list(_col().aggregate(pipeline))
    if not results or results[0]["count"] == 0:
        return None
    return round(results[0]["total"], 3)


def _serialize(doc: Dict[str, Any]) -> Dict[str, Any]:
    ts = doc.get("timestamp")
    return {
        "_id": str(doc["_id"]),
        "timestamp": ts.isoformat() if isinstance(ts, datetime) else str(ts),
        "productionKw": doc.get("productionKw", 0.0),
    }


# ---------------------------------------------------------------------------
# Seed (demo data — production only)
# ---------------------------------------------------------------------------

def seed_historical_data(days: int = 30) -> int:
    """
    Insert simulated 5-minute solar production readings for the past `days` days.
    Skips if data already exists for that period.
    """
    _ensure_indexes()
    col = _col()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    existing = col.count_documents({"timestamp": {"$gte": cutoff}})
    if existing > 0:
        return 0

    capacity_kw = 50.0
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(hours=days * 24)

    docs = []
    # 5-minute intervals
    total_intervals = days * 24 * 12
    for i in range(total_intervals):
        ts = start + timedelta(minutes=i * 5)
        hour = ts.hour
        day_of_year = ts.timetuple().tm_yday
        seasonal = 0.85 + 0.15 * math.sin(2 * math.pi * (day_of_year - 80) / 365)
        if 6 <= hour <= 19:
            solar_factor = math.exp(-0.5 * ((hour - 13) / 3.5) ** 2)
        else:
            solar_factor = 0.0
        cloud_factor = 1 - random.uniform(0, 60) * 0.006
        production = round(capacity_kw * solar_factor * seasonal * cloud_factor * random.uniform(0.85, 1.0), 3)
        docs.append({"timestamp": ts, "productionKw": production})

    if docs:
        col.insert_many(docs)
    return len(docs)
