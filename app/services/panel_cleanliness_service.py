"""
Registro histórico de lecturas de limpieza de paneles.
Colección: lecturas_limpieza_paneles

Cada vez que un usuario sube una foto a "comprobar limpieza" (endpoint REST
/api/classify-panel), el resultado del clasificador se guarda aquí. Esto convierte
un análisis de "usar y tirar" en una serie temporal consultable: el dashboard lee
la última lectura para avisar del estado de los paneles.

Pensado para alimentarse hoy de forma manual (la foto la sube el usuario). El día
de mañana, una cámara fija que dispare contra el mismo endpoint poblaría esta
misma colección sin cambios (de ahí el campo `source`).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.database import get_database

COLLECTION_NAME = "lecturas_limpieza_paneles"

# Nota: el umbral de "sucio" (a partir de qué % de suciedad alertar) lo decide el
# cliente y es configurable desde el modal de limpieza (frontend lib/soiling.ts),
# para tener una única fuente de verdad. Aquí solo persistimos los porcentajes
# crudos del clasificador.


def _col():
    return get_database()[COLLECTION_NAME]


def _ensure_indexes() -> None:
    _col().create_index([("timestamp", -1)])


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def save_cleanliness_reading(
    clasificacion: str,
    porcentaje_limpio: float,
    porcentaje_sucio: float,
    source: str = "manual",
) -> Dict[str, Any]:
    """
    Persiste una lectura de limpieza. Devuelve el documento serializado.

    `source` distingue el origen: "manual" (foto subida por el usuario) o
    "camara" (captura automática), de cara a una futura cámara fija.
    """
    _ensure_indexes()
    doc = {
        "timestamp": datetime.now(timezone.utc),
        "clasificacion": clasificacion,
        "porcentajeLimpio": round(float(porcentaje_limpio), 2),
        "porcentajeSucio": round(float(porcentaje_sucio), 2),
        "source": source,
    }
    result = _col().insert_one(doc)
    doc["_id"] = result.inserted_id
    return _serialize(doc)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def get_latest_cleanliness() -> Optional[Dict[str, Any]]:
    """Devuelve la lectura de limpieza más reciente, o None si no hay ninguna."""
    # Desempate por _id (monotónico por inserción) cuando dos lecturas comparten
    # el mismo instante de timestamp.
    doc = _col().find_one(sort=[("timestamp", -1), ("_id", -1)])
    return _serialize(doc) if doc else None


def list_cleanliness_readings(limit: int = 50) -> List[Dict[str, Any]]:
    """Devuelve las últimas lecturas de limpieza, de la más reciente a la más antigua."""
    cursor = _col().find().sort([("timestamp", -1), ("_id", -1)]).limit(max(1, min(limit, 1000)))
    return [_serialize(doc) for doc in cursor]


def _serialize(doc: Dict[str, Any]) -> Dict[str, Any]:
    ts = doc.get("timestamp")
    return {
        "_id": str(doc["_id"]),
        "timestamp": ts.isoformat() if isinstance(ts, datetime) else str(ts),
        "clasificacion": doc.get("clasificacion", ""),
        "porcentajeLimpio": doc.get("porcentajeLimpio", 0.0),
        "porcentajeSucio": doc.get("porcentajeSucio", 0.0),
        "source": doc.get("source", "manual"),
    }
