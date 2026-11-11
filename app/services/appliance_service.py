"""
Home appliance CRUD helpers backed by MongoDB.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from bson import ObjectId
from pymongo.collection import Collection

from app.database import get_database
from app.services.appliance_measurement_service import (
    build_hourly_profile,
    parse_measurement_file,
)

COLLECTION_NAME = "electrodomesticos"


def _collection() -> Collection:
    return get_database()[COLLECTION_NAME]


def _ensure_positive(value: Any, field: str) -> float:
    if value is None:
        raise ValueError(f"El campo {field} es obligatorio.")
    number = float(value)
    if number <= 0:
        raise ValueError(f"El campo {field} debe ser un número mayor que cero.")
    return number


def _ensure_non_negative(value: Any, field: str) -> float:
    if value is None:
        raise ValueError(f"El campo {field} es obligatorio.")
    number = float(value)
    if number < 0:
        raise ValueError(f"El campo {field} debe ser un número mayor o igual que cero.")
    return number


def _sanitize_hour_mask(raw: Any) -> List[int]:
    if not raw:
        return []
    if not isinstance(raw, (list, tuple, set)):
        raise ValueError("El campo activeHourMask debe ser una lista de horas.")
    hours = set()
    for value in raw:
        try:
            h = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= h <= 23:
            hours.add(h)
    return sorted(hours)


def _ensure_text(value: Any, field: str) -> str:
    if value is None:
        raise ValueError(f"El campo {field} es obligatorio.")
    text = str(value).strip()
    if not text:
        raise ValueError(f"El campo {field} es obligatorio.")
    return text


def _object_id(appliance_id: str) -> ObjectId:
    try:
        return ObjectId(appliance_id)
    except Exception as exc:  # pragma: no cover
        raise ValueError("Identificador de electrodoméstico inválido.") from exc


def _sanitize_modes(raw_modes: Any) -> List[Dict[str, Any]]:
    if not raw_modes:
        return []
    if not isinstance(raw_modes, list):
        raise ValueError("El campo modes debe ser una lista.")

    sanitized: List[Dict[str, Any]] = []
    for index, mode in enumerate(raw_modes):
        if not isinstance(mode, dict):
            raise ValueError(f"El modo #{index + 1} no tiene formato válido.")
        name = _ensure_text(mode.get("name"), f"modes[{index}].name")
        average_power = _ensure_positive(mode.get("averagePowerW"), f"modes[{index}].averagePowerW")
        max_power = mode.get("maxPowerW")
        sanitized.append(
            {
                "name": name,
                "averagePowerW": average_power,
                "maxPowerW": _ensure_positive(max_power, f"modes[{index}].maxPowerW")
                if max_power is not None
                else None,
            }
        )
    return sanitized


def _map_appliance(doc: Dict[str, Any]) -> Dict[str, Any]:
    always_on = doc.get("alwaysOn")
    return {
        "_id": str(doc["_id"]),
        "name": doc.get("name"),
        "category": doc.get("category"),
        "averagePowerW": doc.get("averagePowerW"),
        "maxPowerW": doc.get("maxPowerW"),
        "measuredPowerW": doc.get("measuredPowerW"),
        "quantity": doc.get("quantity"),
        "activeHours": doc.get("activeHours"),
        "selectedModeIndex": doc.get("selectedModeIndex"),
        "modes": doc.get("modes") or [],
        "alwaysOn": True if always_on is None else bool(always_on),
        "useMeasurements": bool(doc.get("useMeasurements", False)),
        "activeHourMask": list(doc.get("activeHourMask") or []),
        "uncoveredHoursFill": doc.get("uncoveredHoursFill") or "mean",
        "hourlyProfileKw": doc.get("hourlyProfileKw") or [],
        "measurementMeta": doc.get("measurementMeta"),
        "createdAt": doc.get("createdAt").isoformat() if doc.get("createdAt") else None,
        "updatedAt": doc.get("updatedAt").isoformat() if doc.get("updatedAt") else None,
    }


def list_appliances() -> List[Dict[str, Any]]:
    cursor = _collection().find().sort("updatedAt", -1)
    return [_map_appliance(doc) for doc in cursor]


def get_appliance(appliance_id: str) -> Optional[Dict[str, Any]]:
    doc = _collection().find_one({"_id": _object_id(appliance_id)})
    return _map_appliance(doc) if doc else None


def create_appliance(payload: Dict[str, Any]) -> Dict[str, Any]:
    name = _ensure_text(payload.get("name"), "name")
    now = datetime.utcnow()
    modes = _sanitize_modes(payload.get("modes"))

    avg_raw = payload.get("averagePowerW")
    max_raw = payload.get("maxPowerW")
    avg_w = float(avg_raw) if avg_raw is not None and float(avg_raw) > 0 else 0.0
    max_w = float(max_raw) if max_raw is not None and float(max_raw) > 0 else max(avg_w, 0.0)

    document: Dict[str, Any] = {
        "name": name,
        "category": payload.get("category"),
        "averagePowerW": avg_w,
        "maxPowerW": max_w,
        "measuredPowerW": (
            _ensure_non_negative(payload.get("measuredPowerW"), "measuredPowerW")
            if payload.get("measuredPowerW") is not None
            else None
        ),
        "quantity": int(_ensure_positive(payload.get("quantity"), "quantity")),
        "activeHours": (
            _ensure_non_negative(payload.get("activeHours"), "activeHours")
            if payload.get("activeHours") is not None
            else None
        ),
        "selectedModeIndex": int(payload["selectedModeIndex"])
        if payload.get("selectedModeIndex") is not None
        else None,
        "modes": modes,
        "alwaysOn": True if payload.get("alwaysOn") is None else bool(payload.get("alwaysOn")),
        "useMeasurements": bool(payload.get("useMeasurements", False)),
        "activeHourMask": _sanitize_hour_mask(payload.get("activeHourMask")),
        "uncoveredHoursFill": "zero" if payload.get("uncoveredHoursFill") == "zero" else "mean",
        "hourlyProfileKw": payload.get("hourlyProfileKw") or [],
        "measurementMeta": payload.get("measurementMeta"),
        "createdAt": now,
        "updatedAt": now,
    }

    if document["selectedModeIndex"] is not None and document["selectedModeIndex"] >= len(modes):
        raise ValueError("selectedModeIndex está fuera de rango para la lista de modos.")

    result = _collection().insert_one(document)
    document["_id"] = result.inserted_id
    return _map_appliance(document)


def update_appliance(appliance_id: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    update: Dict[str, Any] = {}

    if "name" in payload:
        update["name"] = _ensure_text(payload.get("name"), "name")
    if "category" in payload:
        update["category"] = payload.get("category")
    use_measurements = bool(payload.get("useMeasurements", False))
    if "averagePowerW" in payload and payload["averagePowerW"] is not None:
        val = float(payload["averagePowerW"])
        if not use_measurements and val <= 0:
            raise ValueError("El campo averagePowerW debe ser un número mayor que cero.")
        update["averagePowerW"] = max(val, 0.0)
    if "maxPowerW" in payload and payload["maxPowerW"] is not None:
        val = float(payload["maxPowerW"])
        if not use_measurements and val <= 0:
            raise ValueError("El campo maxPowerW debe ser un número mayor que cero.")
        update["maxPowerW"] = max(val, 0.0)
    if "measuredPowerW" in payload:
        value = payload["measuredPowerW"]
        update["measuredPowerW"] = (
            _ensure_non_negative(value, "measuredPowerW") if value is not None else None
        )
    if "quantity" in payload and payload["quantity"] is not None:
        update["quantity"] = int(_ensure_positive(payload["quantity"], "quantity"))
    if "activeHours" in payload:
        value = payload["activeHours"]
        update["activeHours"] = _ensure_non_negative(value, "activeHours") if value is not None else None
    if "selectedModeIndex" in payload:
        value = payload["selectedModeIndex"]
        update["selectedModeIndex"] = int(value) if value is not None else None
    if "modes" in payload:
        update["modes"] = _sanitize_modes(payload.get("modes"))
    if "alwaysOn" in payload:
        update["alwaysOn"] = bool(payload.get("alwaysOn"))
    if "useMeasurements" in payload:
        update["useMeasurements"] = bool(payload.get("useMeasurements"))
    if "activeHourMask" in payload:
        update["activeHourMask"] = _sanitize_hour_mask(payload.get("activeHourMask"))
    fill_changed = False
    if "uncoveredHoursFill" in payload:
        new_fill = "zero" if payload.get("uncoveredHoursFill") == "zero" else "mean"
        existing_doc = _collection().find_one(
            {"_id": _object_id(appliance_id)}, {"uncoveredHoursFill": 1}
        ) or {}
        fill_changed = (existing_doc.get("uncoveredHoursFill") or "mean") != new_fill
        update["uncoveredHoursFill"] = new_fill

    if not update:
        return get_appliance(appliance_id)

    modes_for_validation: Optional[List[Dict[str, Any]]] = None
    if "modes" in update:
        modes_for_validation = update["modes"]
    elif "selectedModeIndex" in update:
        existing = _collection().find_one({"_id": _object_id(appliance_id)})
        if not existing:
            return None
        modes_for_validation = existing.get("modes") or []

    if modes_for_validation is not None and update.get("selectedModeIndex") is not None:
        idx = update["selectedModeIndex"]
        if idx is not None and idx >= len(modes_for_validation):
            raise ValueError("selectedModeIndex está fuera de rango para la lista de modos.")

    update["updatedAt"] = datetime.utcnow()
    result = _collection().find_one_and_update(
        {"_id": _object_id(appliance_id)},
        {"$set": update},
        return_document=True,
    )
    # Si cambió el relleno de horas sin cobertura, reconstruir el perfil 168h
    # desde las lecturas almacenadas para que el cambio surta efecto ya.
    if result and fill_changed:
        try:
            from app.services.medicion_service import _rebuild_profile
            _rebuild_profile(appliance_id)
            result = _collection().find_one({"_id": _object_id(appliance_id)})
        except Exception:
            pass
    return _map_appliance(result) if result else None


def delete_appliance(appliance_id: str) -> bool:
    result = _collection().delete_one({"_id": _object_id(appliance_id)})
    return result.deleted_count == 1


def attach_measurement(appliance_id: str, file_content: str) -> Dict[str, Any]:
    """
    Parse a power-meter export file, build a 168-bin (weekday x hour) average
    consumption profile in kW, persist it onto the appliance document, and
    overwrite averagePowerW / maxPowerW / measuredPowerW with the file's
    derived values (kW -> W) so the manual inputs stay consistent with the
    uploaded measurements.
    """
    existing = _collection().find_one(
        {"_id": _object_id(appliance_id)}, {"uncoveredHoursFill": 1}
    ) or {}
    uncovered_fill = existing.get("uncoveredHoursFill") or "mean"

    samples = parse_measurement_file(file_content)
    profile = build_hourly_profile(samples, uncovered_fill=uncovered_fill)
    meta = profile["meta"]
    avg_w = round(float(meta["avgKw"]) * 1000.0, 2)
    max_w = round(float(meta["maxKw"]) * 1000.0, 2)

    update = {
        "hourlyProfileKw": profile["hourlyProfileKw"],
        "measurementMeta": meta,
        "averagePowerW": avg_w,
        "maxPowerW": max_w,
        "measuredPowerW": avg_w,
        "updatedAt": datetime.utcnow(),
    }
    result = _collection().find_one_and_update(
        {"_id": _object_id(appliance_id)},
        {"$set": update},
        return_document=True,
    )
    if not result:
        raise ValueError("Electrodoméstico no encontrado.")
    return _map_appliance(result)


def clear_measurement(appliance_id: str) -> Optional[Dict[str, Any]]:
    """Remove the uploaded consumption profile from an appliance."""
    result = _collection().find_one_and_update(
        {"_id": _object_id(appliance_id)},
        {
            "$set": {"updatedAt": datetime.utcnow()},
            "$unset": {"hourlyProfileKw": "", "measurementMeta": ""},
        },
        return_document=True,
    )
    return _map_appliance(result) if result else None
