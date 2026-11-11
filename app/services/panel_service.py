"""
Solar panel CRUD operations stored in MongoDB.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from bson import ObjectId
from pymongo.collection import Collection

from app.database import get_database

COLLECTION_NAME = "paneles"


def _collection() -> Collection:
    return get_database()[COLLECTION_NAME]


def _ensure_positive(value: Any, field: str) -> float:
    if value is None:
        raise ValueError(f"El campo {field} es obligatorio.")
    number = float(value)
    if not (number > 0):
        raise ValueError(f"El campo {field} debe ser un número mayor que cero.")
    return number


def _ensure_non_negative(value: Any, field: str) -> float:
    if value is None:
        raise ValueError(f"El campo {field} es obligatorio.")
    number = float(value)
    if number < 0:
        raise ValueError(f"El campo {field} debe ser un número mayor o igual que cero.")
    return number


def _map_panel(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "_id": str(doc["_id"]),
        "manufacturer": doc.get("manufacturer"),
        "model": doc.get("model"),
        "ratedPowerKw": doc.get("ratedPowerKw"),
        "quantity": doc.get("quantity"),
        "tiltDegrees": doc.get("tiltDegrees"),
        "orientation": doc.get("orientation"),
        "efficiencyPercent": doc.get("efficiencyPercent"),
        "areaM2": doc.get("areaM2"),
        "createdAt": doc.get("createdAt").isoformat() if doc.get("createdAt") else None,
        "updatedAt": doc.get("updatedAt").isoformat() if doc.get("updatedAt") else None,
    }


def list_panels() -> List[Dict[str, Any]]:
    cursor = _collection().find().sort("updatedAt", -1)
    return [_map_panel(doc) for doc in cursor]


def _object_id(panel_id: str) -> ObjectId:
    try:
        return ObjectId(panel_id)
    except Exception as exc:  # pragma: no cover - invalid ObjectId
        raise ValueError("Identificador de panel inválido.") from exc


def get_panel(panel_id: str) -> Optional[Dict[str, Any]]:
    doc = _collection().find_one({"_id": _object_id(panel_id)})
    return _map_panel(doc) if doc else None


def _ensure_text(value: Any, field: str) -> str:
    if value is None:
        raise ValueError(f"El campo {field} es obligatorio.")
    text = str(value).strip()
    if not text:
        raise ValueError(f"El campo {field} es obligatorio.")
    return text


def create_panel(payload: Dict[str, Any]) -> Dict[str, Any]:
    manufacturer = _ensure_text(payload.get("manufacturer"), "manufacturer")

    now = datetime.utcnow()
    document: Dict[str, Any] = {
        "manufacturer": manufacturer,
        "model": payload.get("model"),
        "ratedPowerKw": _ensure_positive(payload.get("ratedPowerKw"), "ratedPowerKw"),
        "quantity": int(_ensure_positive(payload.get("quantity"), "quantity")),
        "tiltDegrees": (
            _ensure_non_negative(payload.get("tiltDegrees"), "tiltDegrees")
            if payload.get("tiltDegrees") is not None
            else None
        ),
        "orientation": payload.get("orientation"),
        "efficiencyPercent": (
            _ensure_positive(payload.get("efficiencyPercent"), "efficiencyPercent")
            if payload.get("efficiencyPercent") is not None
            else None
        ),
        "areaM2": (
            _ensure_positive(payload.get("areaM2"), "areaM2")
            if payload.get("areaM2") is not None
            else None
        ),
        "createdAt": now,
        "updatedAt": now,
    }

    result = _collection().insert_one(document)
    document["_id"] = result.inserted_id
    return _map_panel(document)


def update_panel(panel_id: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    update: Dict[str, Any] = {}

    if "manufacturer" in payload:
        update["manufacturer"] = _ensure_text(payload.get("manufacturer"), "manufacturer")
    if "model" in payload:
        update["model"] = payload.get("model")
    if payload.get("ratedPowerKw") is not None:
        update["ratedPowerKw"] = _ensure_positive(payload["ratedPowerKw"], "ratedPowerKw")
    if payload.get("quantity") is not None:
        update["quantity"] = int(_ensure_positive(payload["quantity"], "quantity"))
    if "tiltDegrees" in payload and payload["tiltDegrees"] is not None:
        update["tiltDegrees"] = _ensure_non_negative(payload["tiltDegrees"], "tiltDegrees")
    if "orientation" in payload:
        update["orientation"] = payload.get("orientation")
    if "efficiencyPercent" in payload:
        if payload["efficiencyPercent"] is None:
            update["efficiencyPercent"] = None
        else:
            update["efficiencyPercent"] = _ensure_positive(
                payload["efficiencyPercent"], "efficiencyPercent"
            )
    if "areaM2" in payload:
        if payload["areaM2"] is None:
            update["areaM2"] = None
        else:
            update["areaM2"] = _ensure_positive(payload["areaM2"], "areaM2")

    if not update:
        return get_panel(panel_id)

    update["updatedAt"] = datetime.utcnow()
    result = _collection().find_one_and_update(
        {"_id": _object_id(panel_id)},
        {"$set": update},
        return_document=True,
    )
    return _map_panel(result) if result else None


def delete_panel(panel_id: str) -> bool:
    result = _collection().delete_one({"_id": _object_id(panel_id)})
    return result.deleted_count == 1
