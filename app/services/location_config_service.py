"""
Location configuration service.
Persists and retrieves the system's geographic location from MongoDB.
Collection: ubicacion_config (single-document, upserted).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.database import get_database
from app.services.system_defaults import DEFAULT_SYSTEM_CONFIG

COLLECTION_NAME = "ubicacion_config"
_DOC_KEY = "singleton"  # Always upsert the same document


def _col():
    return get_database()[COLLECTION_NAME]


def get_location_config() -> Dict[str, Any]:
    """
    Return the saved location, falling back to the default if none exists.
    """
    try:
        doc = _col().find_one({"_key": _DOC_KEY})
        if doc:
            return {
                "lat": float(doc["lat"]),
                "lon": float(doc["lon"]),
                "name": str(doc["name"]),
                "updatedAt": doc.get("updatedAt").isoformat() if doc.get("updatedAt") else None,
            }
    except Exception:
        pass
    return {**DEFAULT_SYSTEM_CONFIG["location"], "updatedAt": None}


def save_location_config(lat: float, lon: float, name: str) -> Dict[str, Any]:
    """
    Upsert the location configuration.
    Validates that lat ∈ [-90, 90] and lon ∈ [-180, 180].
    """
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Latitud inválida: {lat}. Debe estar entre -90 y 90.")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Longitud inválida: {lon}. Debe estar entre -180 y 180.")
    if not name or not name.strip():
        raise ValueError("El nombre de la ubicación no puede estar vacío.")

    now = datetime.now(timezone.utc)
    doc = {
        "_key": _DOC_KEY,
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "name": name.strip(),
        "updatedAt": now,
    }
    _col().update_one({"_key": _DOC_KEY}, {"$set": doc}, upsert=True)
    return {
        "lat": doc["lat"],
        "lon": doc["lon"],
        "name": doc["name"],
        "updatedAt": now.isoformat(),
    }
