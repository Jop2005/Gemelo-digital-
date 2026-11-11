"""
Configurable weather source registry and dynamic field mapping.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

import httpx
from bson import ObjectId
from pymongo.collection import Collection

from app.database import get_database

COLLECTION_NAME = "weather_sources"


def _collection() -> Collection:
    return get_database()[COLLECTION_NAME]


def _object_id(source_id: str) -> ObjectId:
    try:
        return ObjectId(source_id)
    except Exception as exc:  # pragma: no cover
        raise ValueError("Identificador de fuente meteorológica inválido.") from exc


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _json_safe(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return fallback
        try:
            parsed = json.loads(stripped)
            return parsed
        except json.JSONDecodeError:
            return fallback
    return fallback


def _to_number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, (int, float)):
        # epoch seconds/millis defensive parsing
        number = float(value)
        if number > 9_999_999_999:
            number = number / 1000
        try:
            return datetime.utcfromtimestamp(number).isoformat()
        except Exception:
            return datetime.utcnow().isoformat()

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return datetime.utcnow().isoformat()
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).isoformat()
        except ValueError:
            return text

    return datetime.utcnow().isoformat()


def _spanish_day_name(date: datetime) -> str:
    names = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    return names[date.weekday()].capitalize()


def _calculate_daily_production(radiation: float, capacity_kw: float, efficiency: float = 0.17) -> float:
    standard_radiation = 1000
    production_factor = max(0.0, radiation) / standard_radiation
    return round(max(0.0, capacity_kw * production_factor * efficiency * 24), 2)


def _guess_condition(cloud_cover: float) -> str:
    if cloud_cover < 20:
        return "sunny"
    if cloud_cover < 55:
        return "partly-cloudy"
    if cloud_cover < 80:
        return "cloudy"
    return "rainy"


def _serialize(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "_id": str(doc["_id"]),
        "name": doc.get("name"),
        "baseUrl": doc.get("baseUrl"),
        "authType": doc.get("authType", "none"),
        "authHeaderName": doc.get("authHeaderName"),
        "authQueryName": doc.get("authQueryName"),
        "authValue": doc.get("authValue"),
        "queryParams": doc.get("queryParams") or {},
        "fieldMapping": doc.get("fieldMapping") or {},
        "locationName": doc.get("locationName"),
        "enabled": bool(doc.get("enabled", True)),
        "isActive": bool(doc.get("isActive", False)),
        "createdAt": _to_iso(doc.get("createdAt")),
        "updatedAt": _to_iso(doc.get("updatedAt")),
    }


def list_weather_sources() -> List[Dict[str, Any]]:
    cursor = _collection().find().sort([("isActive", -1), ("updatedAt", -1)])
    return [_serialize(doc) for doc in cursor]


def get_weather_source(source_id: str) -> Optional[Dict[str, Any]]:
    doc = _collection().find_one({"_id": _object_id(source_id)})
    return _serialize(doc) if doc else None


def get_active_weather_source() -> Optional[Dict[str, Any]]:
    doc = _collection().find_one({"enabled": True, "isActive": True})
    if not doc:
        return None
    return _serialize(doc)


def _validate_source_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    name = _normalize_text(payload.get("name"))
    if not name:
        raise ValueError("El nombre de la fuente meteorológica es obligatorio.")

    auth_type = (_normalize_text(payload.get("authType")) or "none").lower()
    if auth_type not in {"none", "bearer", "api_key_header", "api_key_query", "mock"}:
        raise ValueError("Tipo de autenticación no soportado.")

    base_url = _normalize_text(payload.get("baseUrl"))
    if auth_type != "mock" and not base_url:
        raise ValueError("La URL base es obligatoria salvo que use modo mock.")

    return {
        "name": name,
        "baseUrl": base_url,
        "authType": auth_type,
        "authHeaderName": _normalize_text(payload.get("authHeaderName")),
        "authQueryName": _normalize_text(payload.get("authQueryName")),
        "authValue": _normalize_text(payload.get("authValue")),
        "queryParams": _json_safe(payload.get("queryParams"), {}),
        "fieldMapping": _json_safe(payload.get("fieldMapping"), {}),
        "locationName": _normalize_text(payload.get("locationName")),
        "enabled": bool(payload.get("enabled", True)),
        "isActive": bool(payload.get("isActive", False)),
    }


def save_weather_source(payload: Dict[str, Any], source_id: Optional[str] = None) -> Dict[str, Any]:
    sanitized = _validate_source_payload(payload)
    now = datetime.utcnow()

    if source_id:
        result = _collection().find_one_and_update(
            {"_id": _object_id(source_id)},
            {"$set": {**sanitized, "updatedAt": now}},
            return_document=True,
        )
        if not result:
            raise ValueError("Fuente meteorológica no encontrada.")
        source = _serialize(result)
    else:
        document = {
            **sanitized,
            "createdAt": now,
            "updatedAt": now,
        }
        inserted = _collection().insert_one(document)
        document["_id"] = inserted.inserted_id
        source = _serialize(document)

    if source.get("isActive"):
        set_active_weather_source(source["_id"])
        source = get_weather_source(source["_id"]) or source

    return source


def delete_weather_source(source_id: str) -> bool:
    result = _collection().delete_one({"_id": _object_id(source_id)})
    return result.deleted_count == 1


def set_active_weather_source(source_id: str) -> bool:
    source_obj_id = _object_id(source_id)
    source_doc = _collection().find_one({"_id": source_obj_id})
    if not source_doc:
        raise ValueError("Fuente meteorológica no encontrada.")

    _collection().update_many({}, {"$set": {"isActive": False, "updatedAt": datetime.utcnow()}})
    _collection().update_one(
        {"_id": source_obj_id},
        {"$set": {"isActive": True, "enabled": True, "updatedAt": datetime.utcnow()}},
    )
    return True


def _replace_templates(value: Any, context: Dict[str, Any]) -> Any:
    if isinstance(value, str):
        result = value
        for key, val in context.items():
            result = result.replace(f"{{{{{key}}}}}", str(val))
        return result
    return value


def _parse_path(path: str) -> List[Any]:
    tokens: List[Any] = []
    current = ""
    i = 0
    while i < len(path):
        ch = path[i]
        if ch == ".":
            if current:
                tokens.append(current)
                current = ""
            i += 1
            continue
        if ch == "[":
            if current:
                tokens.append(current)
                current = ""
            end = path.find("]", i)
            if end == -1:
                break
            raw_idx = path[i + 1 : end].strip()
            if raw_idx.isdigit():
                tokens.append(int(raw_idx))
            i = end + 1
            continue
        current += ch
        i += 1
    if current:
        tokens.append(current)
    return tokens


def _extract_path(data: Any, path: Optional[str]) -> Any:
    if not path:
        return None
    normalized = path.strip()
    if not normalized:
        return None
    if normalized.startswith("$."):
        normalized = normalized[2:]

    current: Any = data
    for token in _parse_path(normalized):
        if isinstance(token, int):
            if not isinstance(current, list) or token >= len(current):
                return None
            current = current[token]
            continue
        if not isinstance(current, dict):
            return None
        if token not in current:
            return None
        current = current[token]
    return current


def _flatten_leaf_fields(data: Any, prefix: str = "") -> Iterable[Dict[str, str]]:
    if isinstance(data, dict):
        for key, value in data.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            yield from _flatten_leaf_fields(value, next_prefix)
        return

    if isinstance(data, list):
        if not data:
            yield {
                "path": prefix,
                "valueType": "array",
                "sampleValue": "[]",
            }
            return

        yield {
            "path": prefix,
            "valueType": "array",
            "sampleValue": f"array[{len(data)}]",
        }
        for idx, value in enumerate(data[:2]):
            next_prefix = f"{prefix}[{idx}]"
            yield from _flatten_leaf_fields(value, next_prefix)
        return

    value_type = "null" if data is None else type(data).__name__
    preview = str(data)
    if len(preview) > 120:
        preview = preview[:117] + "..."
    yield {
        "path": prefix,
        "valueType": value_type,
        "sampleValue": preview,
    }


def _generate_mock_source_payload() -> Dict[str, Any]:
    now = datetime.utcnow()
    daily = []
    for day in range(7):
        date_obj = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=day)
        cloud = max(0, min(100, 18 + day * 9))
        daily.append(
            {
                "date_iso": date_obj.date().isoformat(),
                "temp": {"max_c": 31 - day * 0.4, "min_c": 22 - day * 0.25},
                "sky": {"cloud_pct": cloud, "condition_label": _guess_condition(cloud)},
                "solar": {"avg_wm2": max(180, 780 - day * 45)},
                "meta": {"confidence": round(94 - day * 2.7, 1), "source": "mock-v2"},
            }
        )

    return {
        "meta": {
            "provider": "Mock Climate Cloud",
            "generated_at": now.isoformat(),
            "station_id": "HAV-CU-001",
        },
        "measurements": {
            "current": {
                "temperature_c": 29.3,
                "humidity_pct": 73,
                "wind_kmh": 17.6,
                "cloud_pct": 34,
                "irradiance_wm2": 612,
                "summary": "Parcialmente nublado con claros",
            },
            "alerts": [
                {"code": "UV_MED", "description": "Indice UV moderado"},
                {"code": "BREEZE", "description": "Brisa sostenida"},
            ],
        },
        "forecast": {
            "daily": daily,
            "horizon_hours": 168,
            "engine": "mock-weather-engine-2026.1",
        },
        "extra": {
            "region": "La Habana",
            "country": "Cuba",
            "debug": {
                "response_ms": 48,
                "request_id": "mock-req-12345",
            },
        },
    }


def _build_request_context(lat: float, lon: float, timezone: str, location_name: str) -> Dict[str, Any]:
    return {
        "lat": lat,
        "lon": lon,
        "latitude": lat,
        "longitude": lon,
        "timezone": timezone,
        "locationName": location_name,
    }


async def fetch_source_payload(
    source: Dict[str, Any],
    lat: float,
    lon: float,
    location_name: str,
    use_mock: bool = False,
    timezone: str = "auto",
) -> Dict[str, Any]:
    auth_type = (source.get("authType") or "none").lower()
    if use_mock or auth_type == "mock":
        return _generate_mock_source_payload()

    base_url = _normalize_text(source.get("baseUrl"))
    if not base_url:
        raise ValueError("La URL base no está configurada para esta fuente.")

    context = _build_request_context(lat, lon, timezone, location_name)
    params_raw = _json_safe(source.get("queryParams"), {})
    params: Dict[str, Any] = {
        str(key): _replace_templates(value, context)
        for key, value in params_raw.items()
    }

    headers: Dict[str, str] = {"Accept": "application/json"}
    auth_value = _normalize_text(source.get("authValue"))

    if auth_type == "bearer" and auth_value:
        headers["Authorization"] = f"Bearer {auth_value}"
    elif auth_type == "api_key_header" and auth_value:
        header_name = _normalize_text(source.get("authHeaderName")) or "X-API-Key"
        headers[header_name] = auth_value
    elif auth_type == "api_key_query" and auth_value:
        query_name = _normalize_text(source.get("authQueryName")) or "api_key"
        params[query_name] = auth_value

    async with httpx.AsyncClient(timeout=8) as client:
        response = await client.get(base_url, params=params, headers=headers)

    response.raise_for_status()
    return response.json()


def _extract_from_forecast_item(item: Any, root_payload: Dict[str, Any], path: Optional[str]) -> Any:
    if not path:
        return None
    trimmed = path.strip()
    if trimmed.startswith("$."):
        return _extract_path(root_payload, trimmed[2:])
    return _extract_path(item, trimmed)


def map_payload_to_weather_data(
    source: Dict[str, Any],
    payload: Dict[str, Any],
    capacity_kw: float,
    default_location_name: str,
) -> Dict[str, Any]:
    mapping = _json_safe(source.get("fieldMapping"), {})

    required_paths = {
        "temperaturePath": "temperatura actual",
        "solarRadiationPath": "radiación solar actual",
        "cloudCoverPath": "nubosidad actual",
        "humidityPath": "humedad actual",
        "windSpeedPath": "viento actual",
        "forecastArrayPath": "lista de pronóstico diario",
        "forecastDatePath": "fecha del pronóstico",
        "forecastMaxTempPath": "máxima del pronóstico",
        "forecastMinTempPath": "mínima del pronóstico",
        "forecastSolarRadiationPath": "radiación del pronóstico",
        "forecastCloudCoverPath": "nubosidad del pronóstico",
    }

    missing = [label for key, label in required_paths.items() if not _normalize_text(mapping.get(key))]
    if missing:
        raise ValueError(
            "Faltan enlaces en el mapeo: " + ", ".join(missing) + "."
        )

    forecast_array = _extract_path(payload, mapping.get("forecastArrayPath"))
    if not isinstance(forecast_array, list) or not forecast_array:
        raise ValueError("El enlace del pronóstico diario no devuelve una lista válida.")

    forecast: List[Dict[str, Any]] = []
    for row in forecast_array[:7]:
        date_raw = _extract_from_forecast_item(row, payload, mapping.get("forecastDatePath"))
        max_temp = _to_number(_extract_from_forecast_item(row, payload, mapping.get("forecastMaxTempPath")))
        min_temp = _to_number(_extract_from_forecast_item(row, payload, mapping.get("forecastMinTempPath")))
        daily_radiation = _to_number(
            _extract_from_forecast_item(row, payload, mapping.get("forecastSolarRadiationPath"))
        )
        daily_cloud = _to_number(_extract_from_forecast_item(row, payload, mapping.get("forecastCloudCoverPath")))
        condition_raw = _extract_from_forecast_item(row, payload, mapping.get("forecastConditionPath"))
        condition = (
            str(condition_raw).strip() if condition_raw is not None and str(condition_raw).strip() else _guess_condition(daily_cloud)
        )

        date_iso = _to_iso(date_raw)
        try:
            date_obj = datetime.fromisoformat(date_iso.replace("Z", "+00:00"))
            day_of_week = _spanish_day_name(date_obj)
        except ValueError:
            day_of_week = "Sin día"

        forecast.append(
            {
                "date": date_iso,
                "dayOfWeek": day_of_week,
                "maxTemp": max_temp,
                "minTemp": min_temp,
                "solarRadiation": round(daily_radiation),
                "cloudCover": round(daily_cloud),
                "predictedProduction": _calculate_daily_production(daily_radiation, capacity_kw),
                "condition": condition,
            }
        )

    weather_data = {
        "temperature": _to_number(_extract_path(payload, mapping.get("temperaturePath"))),
        "solarRadiation": round(_to_number(_extract_path(payload, mapping.get("solarRadiationPath")))),
        "cloudCover": round(_to_number(_extract_path(payload, mapping.get("cloudCoverPath")))),
        "humidity": round(_to_number(_extract_path(payload, mapping.get("humidityPath")))),
        "windSpeed": _to_number(_extract_path(payload, mapping.get("windSpeedPath"))),
        "forecast": forecast,
        "provider": source.get("name") or "Fuente personalizada",
        "locationName": source.get("locationName") or default_location_name,
        "lastUpdated": datetime.utcnow().isoformat(),
        "description": _extract_path(payload, mapping.get("descriptionPath")),
    }

    return weather_data


async def test_weather_source(
    source_payload: Dict[str, Any],
    lat: float,
    lon: float,
    location_name: str,
    use_mock: bool = False,
) -> Dict[str, Any]:
    payload_for_validation = source_payload
    if use_mock:
        payload_for_validation = {
            **source_payload,
            "authType": "mock",
            "baseUrl": source_payload.get("baseUrl") or "mock://local",
        }

    source = _validate_source_payload(payload_for_validation)
    payload = await fetch_source_payload(source, lat, lon, location_name, use_mock=use_mock)
    candidates = list(_flatten_leaf_fields(payload))
    candidates_sorted = sorted(candidates, key=lambda item: item["path"])

    return {
        "success": True,
        "message": "Conexión exitosa. Seleccione los campos para enlazar.",
        "fields": candidates_sorted,
        "rawJson": json.dumps(payload, ensure_ascii=False, indent=2),
    }


async def get_active_weather_data(
    lat: float,
    lon: float,
    capacity_kw: float,
    location_name: str,
) -> Optional[Dict[str, Any]]:
    source = get_active_weather_source()
    if not source or not source.get("enabled"):
        return None

    payload = await fetch_source_payload(source, lat, lon, location_name, use_mock=False)
    return map_payload_to_weather_data(source, payload, capacity_kw, location_name)
