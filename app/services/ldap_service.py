"""
LDAP authentication service.

The connection settings live in MongoDB (managed from the admin panel via
ldap_config_service), not in environment variables. Authentication is a standard
search-bind flow against any RFC-4511 directory:

  1. Bind as a service account (or anonymously) to locate the user's DN by email
     using the configured search filter.
  2. Re-bind as that DN with the supplied password to verify credentials.
  3. Return the user's email and display name from LDAP attributes.

`authenticate_ldap` raises ValueError on any failure; the caller translates that
into the GraphQL error surface. Connections are opened through `_open_connection`,
a single seam that tests override with ldap3's in-memory MOCK_SYNC strategy so the
exact same code path can be exercised against a simulated directory.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

try:
    from ldap3 import Server, Connection, ALL, SUBTREE, Tls
    from ldap3.core.exceptions import LDAPException
    import ssl
    _LDAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LDAP_AVAILABLE = False


def _ensure_available() -> None:
    if not _LDAP_AVAILABLE:
        raise ValueError("Falta la dependencia 'ldap3' en el backend.")


def _build_server(config: Dict[str, Any]) -> "Server":
    tls = None
    if config.get("useTls"):
        tls = Tls(validate=ssl.CERT_NONE)
    return Server(
        config.get("serverUrl"),
        use_ssl=bool(config.get("useTls")),
        tls=tls,
        get_info=ALL,
        connect_timeout=int(config.get("connectTimeout") or 5),
    )


def _open_connection(server: "Server", user: Optional[str], password: Optional[str]) -> "Connection":
    """
    Open and bind a connection. This is the single seam tests override to inject
    an in-memory MOCK_SYNC directory. A bind failure raises LDAPException
    (because auto_bind=True), which callers translate into a ValueError.
    """
    return Connection(server, user=user, password=password, auto_bind=True)


def authenticate_ldap(email: str, password: str) -> Dict[str, str]:
    """
    Validate the (email, password) pair against the configured LDAP directory.

    Returns a dict with 'email' and 'name' on success. Raises ValueError on any
    kind of failure — never returns a falsy/empty result.
    """
    from app.services.ldap_config_service import get_ldap_config

    config = get_ldap_config(include_secret=True)
    if not config.get("enabled"):
        raise ValueError("La autenticación LDAP está deshabilitada en este servidor.")

    email_clean = (email or "").strip()
    if not email_clean or not password:
        raise ValueError("Correo y contraseña son obligatorios.")

    return _search_bind(config, email_clean, password)


def _search_bind(config: Dict[str, Any], email: str, password: str) -> Dict[str, str]:
    _ensure_available()

    server = _build_server(config)
    email_attr = config.get("emailAttr") or "mail"
    name_attr = config.get("nameAttr") or "cn"
    search_filter = (config.get("userSearchFilter") or "(mail={email})").replace("{email}", email)

    # Step 1: service bind (or anonymous) to locate the user DN.
    try:
        bind_dn = config.get("bindDn") or None
        search_conn = _open_connection(server, bind_dn, config.get("bindPassword") if bind_dn else None)
    except LDAPException as exc:
        raise ValueError(f"No se pudo contactar el servidor LDAP: {exc}") from exc

    try:
        search_conn.search(
            search_base=config.get("baseDn"),
            search_filter=search_filter,
            search_scope=SUBTREE,
            attributes=[email_attr, name_attr],
        )
        if not search_conn.entries:
            raise ValueError("Usuario LDAP no encontrado.")
        entry = search_conn.entries[0]
        user_dn = entry.entry_dn
        try:
            ldap_email = str(entry[email_attr].value)
        except Exception:
            ldap_email = email
        try:
            ldap_name = str(entry[name_attr].value)
        except Exception:
            ldap_name = ""
    finally:
        try:
            search_conn.unbind()
        except Exception:
            pass

    # Step 2: re-bind as the located DN with the supplied password.
    try:
        user_conn = _open_connection(server, user_dn, password)
    except LDAPException as exc:
        raise ValueError("Credenciales LDAP inválidas.") from exc

    try:
        user_conn.unbind()
    except Exception:
        pass

    return {"email": ldap_email.lower(), "name": ldap_name}


def probe_ldap(config: Dict[str, Any], sample_email: str = "", sample_password: str = "") -> Dict[str, Any]:
    """
    Attempt to reach/bind the directory (and optionally verify a sample login).
    Never raises: returns {"success", "message", "sampleUser"}.
    """
    try:
        _ensure_available()
        server = _build_server(config)
        bind_dn = config.get("bindDn") or None
        conn = _open_connection(server, bind_dn, config.get("bindPassword") if bind_dn else None)
        try:
            conn.unbind()
        except Exception:
            pass
    except ValueError as exc:
        return {"success": False, "message": str(exc), "sampleUser": None}
    except LDAPException as exc:
        return {"success": False, "message": f"No se pudo contactar/autenticar el servidor LDAP: {exc}", "sampleUser": None}
    except Exception as exc:  # pragma: no cover - defensive
        return {"success": False, "message": f"Error inesperado: {exc}", "sampleUser": None}

    # Reachability OK. Optionally verify the sample credentials end-to-end.
    if sample_email and sample_password:
        try:
            info = _search_bind(config, sample_email.strip(), sample_password)
            return {
                "success": True,
                "message": "Conexión LDAP correcta y credenciales de prueba válidas.",
                "sampleUser": info.get("name") or info.get("email"),
            }
        except ValueError as exc:
            return {
                "success": False,
                "message": f"Conexión LDAP correcta, pero la prueba de credenciales falló: {exc}",
                "sampleUser": None,
            }

    return {"success": True, "message": "Conexión LDAP establecida correctamente.", "sampleUser": None}
