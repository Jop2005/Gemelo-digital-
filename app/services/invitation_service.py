"""
Service for managing invitation codes.
"""
from __future__ import annotations

import secrets
import string
from datetime import datetime
from typing import Any, Dict, List, Optional

from pymongo.collection import Collection

from app.database import get_database

COLLECTION_NAME = "invitation_codes"
CODE_LENGTH = 8


def _collection() -> Collection:
    return get_database()[COLLECTION_NAME]


def _generate_code(length: int = CODE_LENGTH) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def _map_invitation(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "_id": str(doc["_id"]),
        "code": doc["code"],
        "role": doc["role"],
        "isUsed": doc["isUsed"],
        "createdBy": doc.get("createdBy"),
        "usedBy": doc.get("usedBy"),
        "createdAt": doc["createdAt"].isoformat() if doc.get("createdAt") else None,
        "updatedAt": doc["updatedAt"].isoformat() if doc.get("updatedAt") else None,
    }


def create_invitation_code(role: str, created_by: str) -> Dict[str, Any]:
    """
    Generates a new unique invitation code.
    """
    if role not in ["admin", "user"]:
        raise ValueError("Rol inválido. Debe ser 'admin' o 'user'.")

    code = _generate_code()
    while _collection().find_one({"code": code}):
        code = _generate_code()

    now = datetime.utcnow()
    document = {
        "code": code,
        "role": role,
        "isUsed": False,
        "createdBy": created_by,
        "usedBy": None,
        "createdAt": now,
        "updatedAt": now,
    }
    
    result = _collection().insert_one(document)
    document["_id"] = result.inserted_id
    return _map_invitation(document)


def validate_invitation_code(code: str) -> str:
    """
    Validates an invitation code and returns the associated role.
    Raises ValueError if invalid or used.
    """
    invitation = _collection().find_one({"code": code})
    if not invitation:
        raise ValueError("Código de invitación inválido.")
    
    if invitation["isUsed"]:
        raise ValueError("Este código de invitación ya ha sido utilizado.")
        
    return invitation["role"]


def mark_invitation_code_as_used(code: str, used_by: str) -> None:
    """
    Marks an invitation code as used by a specific user.
    """
    now = datetime.utcnow()
    result = _collection().update_one(
        {"code": code, "isUsed": False},
        {
            "$set": {
                "isUsed": True,
                "usedBy": used_by,
                "updatedAt": now
            }
        }
    )
    
    if result.modified_count == 0:
        raise ValueError("No se pudo marcar el código como usado (puede que ya esté usado o no exista).")


def list_invitation_codes() -> List[Dict[str, Any]]:
    """
    Lists all invitation codes.
    """
    cursor = _collection().find().sort("createdAt", -1)
    return [_map_invitation(doc) for doc in cursor]
