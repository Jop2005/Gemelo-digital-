"""
JWT authentication utilities for GraphQL context injection.
"""
from __future__ import annotations

import uuid

import jwt
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from fastapi import Request

from app.config import settings

ALGORITHM = settings.JWT_ALGORITHM


def create_token(email: str, role: str, jti: Optional[str] = None) -> str:
    """Create a signed JWT token for the given user."""
    payload = {
        "sub": email,
        "role": role,
        "jti": jti or uuid.uuid4().hex,
        "exp": datetime.now(timezone.utc) + timedelta(days=settings.JWT_EXPIRE_DAYS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=ALGORITHM)


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode a JWT token. Returns the payload or None if invalid."""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


async def get_context(request: Request) -> Dict[str, Any]:
    """
    Strawberry context_getter: extracts JWT from Authorization header
    and injects current_user into the GraphQL context.
    """
    current_user: Optional[Dict[str, Any]] = None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        payload = verify_token(token)
        if payload and payload.get("jti"):
            from app.services.session_service import is_token_revoked
            if is_token_revoked(payload["jti"]):
                payload = None
        current_user = payload
    return {"request": request, "current_user": current_user}


def require_auth(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Raise if the request is not authenticated.
    Returns the current_user dict.
    """
    user = ctx.get("current_user")
    if not user:
        raise Exception("Autenticación requerida. Por favor inicia sesión.")
    return user


def require_admin(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Raise if the current user is not an admin.
    Returns the current_user dict.
    """
    user = require_auth(ctx)
    if user.get("role") != "admin":
        raise Exception("Acceso denegado. Se requieren privilegios de administrador.")
    return user
