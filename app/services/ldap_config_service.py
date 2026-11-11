"""
LDAP configuration service.

Persists the institutional LDAP connection settings in MongoDB so an
administrator can manage them from the admin panel (instead of environment
variables). Collection: ldap_config (single-document, upserted).

The bind password is stored as-is (the directory needs it to bind); it is never
returned to the client by the read path — see `get_ldap_config(include_secret=...)`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from app.database import get_database

COLLECTION_NAME = "ldap_config"
_DOC_KEY = "singleton"  # Always upsert the same document

# Masked placeholder returned to clients in place of the real bind password.
# If the client sends this value back on save, we keep the stored password.
SECRET_MASK = "********"

_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "serverUrl": "ldap://localhost:389",
    "baseDn": "dc=cujae,dc=edu,dc=cu",
    "bindDn": "",
    "bindPassword": "",
    "userSearchFilter": "(mail={email})",
    "emailAttr": "mail",
    "nameAttr": "cn",
    "useTls": False,
    "connectTimeout": 5,
}


def _col():
    return get_database()[COLLECTION_NAME]


def _serialize(doc: Dict[str, Any], include_secret: bool = False) -> Dict[str, Any]:
    """Map a stored document to the API shape, masking the bind password."""
    merged = {**_DEFAULTS, **{k: doc.get(k) for k in _DEFAULTS if doc.get(k) is not None}}
    raw_password = doc.get("bindPassword") or ""
    merged["bindPassword"] = raw_password if include_secret else (SECRET_MASK if raw_password else "")
    merged["hasBindPassword"] = bool(raw_password)
    merged["updatedAt"] = (
        doc["updatedAt"].isoformat() if doc.get("updatedAt") else None
    )
    return merged


def get_ldap_config(include_secret: bool = False) -> Dict[str, Any]:
    """
    Return the saved LDAP config, falling back to defaults if none exists.

    By default the bind password is masked. Internal callers that need to bind
    (ldap_service) must pass include_secret=True.
    """
    try:
        doc = _col().find_one({"_key": _DOC_KEY})
    except Exception:
        doc = None
    if not doc:
        base = {**_DEFAULTS, "hasBindPassword": False, "updatedAt": None}
        if not include_secret:
            base["bindPassword"] = ""
        return base
    return _serialize(doc, include_secret=include_secret)


def is_ldap_enabled() -> bool:
    """Cheap public check used to decide whether to expose the LDAP login tab."""
    cfg = get_ldap_config()
    return bool(cfg.get("enabled"))


def _coerce(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize an incoming config payload."""
    def _txt(key: str) -> str:
        value = payload.get(key)
        return value.strip() if isinstance(value, str) else (value or "")

    enabled = bool(payload.get("enabled", False))
    server_url = _txt("serverUrl")
    base_dn = _txt("baseDn")
    search_filter = _txt("userSearchFilter") or _DEFAULTS["userSearchFilter"]

    if "{email}" not in search_filter:
        raise ValueError("El filtro de búsqueda debe contener el marcador {email}.")

    if enabled and not server_url:
        raise ValueError("La URL del servidor LDAP es obligatoria cuando LDAP está habilitado.")
    if enabled and not base_dn:
        raise ValueError("El Base DN es obligatorio cuando LDAP está habilitado.")

    try:
        connect_timeout = int(payload.get("connectTimeout") or _DEFAULTS["connectTimeout"])
    except (TypeError, ValueError):
        connect_timeout = _DEFAULTS["connectTimeout"]
    connect_timeout = max(1, min(connect_timeout, 60))

    return {
        "enabled": enabled,
        "serverUrl": server_url or _DEFAULTS["serverUrl"],
        "baseDn": base_dn or _DEFAULTS["baseDn"],
        "bindDn": _txt("bindDn"),
        "userSearchFilter": search_filter,
        "emailAttr": _txt("emailAttr") or _DEFAULTS["emailAttr"],
        "nameAttr": _txt("nameAttr") or _DEFAULTS["nameAttr"],
        "useTls": bool(payload.get("useTls", False)),
        "connectTimeout": connect_timeout,
    }


def save_ldap_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Upsert the LDAP configuration.

    The bind password is only overwritten when the client sends a real value;
    sending the masked placeholder (or omitting it) keeps the stored secret.
    """
    doc = _coerce(payload)

    incoming_password = payload.get("bindPassword")
    if isinstance(incoming_password, str) and incoming_password and incoming_password != SECRET_MASK:
        doc["bindPassword"] = incoming_password
    # else: leave the stored password untouched.

    now = datetime.now(timezone.utc)
    doc["updatedAt"] = now
    doc["_key"] = _DOC_KEY

    _col().update_one({"_key": _DOC_KEY}, {"$set": doc}, upsert=True)
    return get_ldap_config()


def _merge_for_probe(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the effective config used by a connection test: the submitted form
    values layered over the stored config, resolving the masked password to the
    stored secret when the client did not type a new one.
    """
    stored = get_ldap_config(include_secret=True)
    merged = {**stored, **_coerce(payload)}
    incoming_password = payload.get("bindPassword")
    if isinstance(incoming_password, str) and incoming_password and incoming_password != SECRET_MASK:
        merged["bindPassword"] = incoming_password
    else:
        merged["bindPassword"] = stored.get("bindPassword", "")
    return merged


async def test_ldap_connection(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the supplied (or stored) configuration by attempting a bind,
    optionally verifying a sample (email, password) pair.

    Returns {"success": bool, "message": str, "sampleUser": Optional[str]}.
    Never raises — failures are reported in the result.
    """
    from app.services.ldap_service import probe_ldap

    merged = _merge_for_probe(payload)
    sample_email = (payload.get("sampleEmail") or "").strip()
    sample_password = payload.get("samplePassword") or ""
    return probe_ldap(merged, sample_email, sample_password)
