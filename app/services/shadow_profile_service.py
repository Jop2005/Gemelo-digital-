"""
Shadow profile service.
Persists and retrieves the installation's hourly shadow profile from MongoDB.
Collection: shadow_profile (single-document, upserted via _key: "singleton").

A shadow profile contains one slot per solar hour of the day (typically 5–19h),
each recording the estimated fraction of the panel surface that is shaded and
an optional manual production-reduction override. It is measured once on a
representative clear day and reused for long-term production estimates.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.database import get_database

COLLECTION_NAME = "shadow_profile"
_DOC_KEY = "singleton"


def _col():
    return get_database()[COLLECTION_NAME]


def get_shadow_profile() -> Optional[Dict[str, Any]]:
    """
    Return the saved shadow profile, or None if none has been stored yet.
    """
    try:
        doc = _col().find_one({"_key": _DOC_KEY})
        if doc:
            return {
                "slots": doc.get("slots", []),
                "avgShadow": float(doc.get("avgShadow", 0)),
                "avgProd": float(doc.get("avgProd", 100)),
                "updatedAt": doc["updatedAt"].isoformat() if doc.get("updatedAt") else None,
            }
    except Exception:
        pass
    return None


def save_shadow_profile(
    slots: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Upsert the shadow profile.

    Each slot must have:
        hour (int 0–23), shadowPct (float 0–100), prodOverride (float|None)

    avgShadow and avgProd are derived here from the day-lit slots
    (those where shadowPct is provided; hours without sun are skipped by the caller).
    """
    if not slots:
        raise ValueError("El perfil debe contener al menos una franja horaria.")

    # Validate and normalise slots
    clean: List[Dict[str, Any]] = []
    for s in slots:
        hour = int(s["hour"])
        shadow = float(s.get("shadowPct", 0))
        override = s.get("prodOverride")
        if not (0 <= hour <= 23):
            raise ValueError(f"Hora inválida: {hour}")
        if not (0 <= shadow <= 100):
            raise ValueError(f"% sombra inválido ({shadow}) en hora {hour}")
        if override is not None:
            override = float(override)
            if not (0 <= override <= 100):
                raise ValueError(f"% producción inválido ({override}) en hora {hour}")
        clean.append({"hour": hour, "shadowPct": shadow, "prodOverride": override})

    # Derive summary stats from the slots marked as day-lit (non-null prodOverride OR any slot)
    avg_shadow = sum(s["shadowPct"] for s in clean) / len(clean)
    avg_prod = sum(
        s["prodOverride"] if s["prodOverride"] is not None else max(0.0, 100.0 - s["shadowPct"])
        for s in clean
    ) / len(clean)

    now = datetime.now(timezone.utc)
    doc = {
        "_key": _DOC_KEY,
        "slots": clean,
        "avgShadow": round(avg_shadow, 2),
        "avgProd": round(avg_prod, 2),
        "updatedAt": now,
    }
    _col().update_one({"_key": _DOC_KEY}, {"$set": doc}, upsert=True)

    return {
        "slots": clean,
        "avgShadow": doc["avgShadow"],
        "avgProd": doc["avgProd"],
        "updatedAt": now.isoformat(),
    }
