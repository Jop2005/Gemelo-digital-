"""
Session tracking — persists active JWT sessions so admins can see and revoke them.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from bson import ObjectId

from app.database import get_database

COLLECTION_NAME = "sessions"


def _collection():
    return get_database()[COLLECTION_NAME]


def _device_type(ua: str) -> str:
    ua_lower = ua.lower()
    if any(k in ua_lower for k in ("mobile", "android", "iphone", "ipad")):
        return "Móvil"
    return "Escritorio"


def _map_session(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "_id": str(doc["_id"]),
        "email": doc["email"],
        "ip": doc.get("ip") or "desconocida",
        "userAgent": doc.get("userAgent") or "",
        "deviceType": doc.get("deviceType") or "Escritorio",
        "jti": doc["jti"],
        "createdAt": doc["createdAt"].isoformat() if doc.get("createdAt") else None,
        "expiresAt": doc["expiresAt"].isoformat() if doc.get("expiresAt") else None,
        "isRevoked": doc.get("isRevoked", False),
    }


def create_session(
    email: str,
    ip: str,
    user_agent: str,
    jti: str,
    expires_at: datetime,
) -> Dict[str, Any]:
    now = datetime.utcnow()
    doc = {
        "email": email,
        "ip": ip,
        "userAgent": user_agent,
        "deviceType": _device_type(user_agent),
        "jti": jti,
        "createdAt": now,
        "expiresAt": expires_at,
        "isRevoked": False,
    }
    result = _collection().insert_one(doc)
    doc["_id"] = result.inserted_id
    return _map_session(doc)


def list_active_sessions() -> List[Dict[str, Any]]:
    """Returns non-revoked, non-expired sessions ordered newest first."""
    now = datetime.utcnow()
    cursor = _collection().find(
        {"isRevoked": False, "expiresAt": {"$gt": now}}
    ).sort("createdAt", -1)
    return [_map_session(doc) for doc in cursor]


def revoke_session(session_id: str) -> bool:
    result = _collection().update_one(
        {"_id": ObjectId(session_id)},
        {"$set": {"isRevoked": True}},
    )
    return result.modified_count > 0


def revoke_sessions_by_email(email: str) -> int:
    """Revoke all active sessions for a user (used when deleting an account)."""
    result = _collection().update_many(
        {"email": email, "isRevoked": False},
        {"$set": {"isRevoked": True}},
    )
    return result.modified_count


def is_token_revoked(jti: str) -> bool:
    doc = _collection().find_one({"jti": jti})
    if not doc:
        # JTI unknown (pre-session-tracking token) — let it through
        return False
    return doc.get("isRevoked", False)
