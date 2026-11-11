"""
Battery bank CRUD helpers backed by MongoDB.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from bson import ObjectId
from pymongo.collection import Collection

from app.database import get_database

COLLECTION_NAME = "baterias"


def _collection() -> Collection:
    return get_database()[COLLECTION_NAME]


def _ensure_positive(value: Any, field: str) -> float:
    if value is None:
        raise ValueError(f"El campo {field} es obligatorio.")
    number = float(value)
    if number <= 0:
        raise ValueError(f"El campo {field} debe ser un número mayor que cero.")
    return number


def _ensure_text(value: Any, field: str) -> str:
    if value is None:
        raise ValueError(f"El campo {field} es obligatorio.")
    text = str(value).strip()
    if not text:
        raise ValueError(f"El campo {field} es obligatorio.")
    return text


def _object_id(battery_id: str) -> ObjectId:
    try:
        return ObjectId(battery_id)
    except Exception as exc:  # pragma: no cover
        raise ValueError("Identificador de batería inválido.") from exc


def _map_battery(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "_id": str(doc["_id"]),
        "manufacturer": doc.get("manufacturer"),
        "model": doc.get("model"),
        "capacityKwh": doc.get("capacityKwh"),
        "quantity": doc.get("quantity"),
        "maxDepthOfDischargePercent": doc.get("maxDepthOfDischargePercent"),
        "chargeRateKw": doc.get("chargeRateKw"),
        "dischargeRateKw": doc.get("dischargeRateKw"),
        "efficiencyPercent": doc.get("efficiencyPercent"),
        "createdAt": doc.get("createdAt").isoformat() if doc.get("createdAt") else None,
        "updatedAt": doc.get("updatedAt").isoformat() if doc.get("updatedAt") else None,
    }


def list_batteries() -> List[Dict[str, Any]]:
    cursor = _collection().find().sort("updatedAt", -1)
    return [_map_battery(doc) for doc in cursor]


def get_battery(battery_id: str) -> Optional[Dict[str, Any]]:
    doc = _collection().find_one({"_id": _object_id(battery_id)})
    return _map_battery(doc) if doc else None


_OPTIONAL_BATTERY_FIELDS = (
    "maxDepthOfDischargePercent",
    "chargeRateKw",
    "dischargeRateKw",
    "efficiencyPercent",
)


def _optional_positive(payload: Dict[str, Any], field: str) -> Optional[float]:
    if payload.get(field) is None:
        return None
    return _ensure_positive(payload.get(field), field)


def create_battery(payload: Dict[str, Any]) -> Dict[str, Any]:
    manufacturer = _ensure_text(payload.get("manufacturer"), "manufacturer")

    now = datetime.utcnow()
    document: Dict[str, Any] = {
        "manufacturer": manufacturer,
        "model": payload.get("model"),
        "capacityKwh": _ensure_positive(payload.get("capacityKwh"), "capacityKwh"),
        "quantity": int(_ensure_positive(payload.get("quantity"), "quantity")),
        "createdAt": now,
        "updatedAt": now,
    }
    for field in _OPTIONAL_BATTERY_FIELDS:
        document[field] = _optional_positive(payload, field)
    result = _collection().insert_one(document)
    document["_id"] = result.inserted_id
    return _map_battery(document)


def update_battery(battery_id: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    update: Dict[str, Any] = {}

    if "manufacturer" in payload:
        update["manufacturer"] = _ensure_text(payload.get("manufacturer"), "manufacturer")
    if "model" in payload:
        update["model"] = payload.get("model")

    numeric_fields = {
        "capacityKwh": _ensure_positive,
        "quantity": lambda value, field: int(_ensure_positive(value, field)),
    }

    for field, validator in numeric_fields.items():
        if field in payload and payload[field] is not None:
            update[field] = validator(payload[field], field)

    for field in _OPTIONAL_BATTERY_FIELDS:
        if field in payload:
            update[field] = (
                None
                if payload[field] is None
                else _ensure_positive(payload[field], field)
            )

    if not update:
        return get_battery(battery_id)

    update["updatedAt"] = datetime.utcnow()
    result = _collection().find_one_and_update(
        {"_id": _object_id(battery_id)},
        {"$set": update},
        return_document=True,
    )
    return _map_battery(result) if result else None


def delete_battery(battery_id: str) -> bool:
    result = _collection().delete_one({"_id": _object_id(battery_id)})
    return result.deleted_count == 1
