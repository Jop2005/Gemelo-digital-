"""
Pruebas de integración para session_service.

Cubre el ciclo completo de sesiones JWT: crear, listar activas,
revocar por ID, revocar por email, detectar tokens revocados y
clasificación del tipo de dispositivo.
"""
import pytest
from datetime import datetime, timedelta
from bson import ObjectId

from app.services.session_service import (
    create_session,
    list_active_sessions,
    revoke_session,
    revoke_sessions_by_email,
    is_token_revoked,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_FUTURE = datetime.utcnow() + timedelta(days=7)
_PAST = datetime.utcnow() - timedelta(days=1)

_UA_DESKTOP = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
_UA_MOBILE = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15"


def _create(email: str = "admin@cujae.edu.cu",
            jti: str = "abc123",
            ua: str = _UA_DESKTOP,
            expires: datetime = None) -> dict:
    return create_session(
        email=email,
        ip="192.168.1.1",
        user_agent=ua,
        jti=jti,
        expires_at=expires or _FUTURE,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Creación
# ─────────────────────────────────────────────────────────────────────────────

class TestCrearSesion:

    def test_crear_retorna_documento_con_id(self, mongo_db):
        s = _create()
        assert "_id" in s
        assert isinstance(s["_id"], str)

    def test_crear_persiste_email(self, mongo_db):
        s = _create(email="operador@cujae.edu.cu")
        assert s["email"] == "operador@cujae.edu.cu"

    def test_crear_persiste_jti(self, mongo_db):
        s = _create(jti="token-xyz")
        assert s["jti"] == "token-xyz"

    def test_crear_no_revocada_por_defecto(self, mongo_db):
        s = _create()
        assert s["isRevoked"] is False

    def test_crear_detecta_dispositivo_escritorio(self, mongo_db):
        s = _create(ua=_UA_DESKTOP)
        assert s["deviceType"] == "Escritorio"

    def test_crear_detecta_dispositivo_movil_iphone(self, mongo_db):
        s = _create(ua=_UA_MOBILE)
        assert s["deviceType"] == "Móvil"

    def test_crear_detecta_dispositivo_movil_android(self, mongo_db):
        s = _create(ua="Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36")
        assert s["deviceType"] == "Móvil"

    def test_crear_con_ip_desconocida_usa_fallback(self, mongo_db):
        s = create_session("a@b.cu", "", _UA_DESKTOP, "jti-x", _FUTURE)
        assert s["ip"] == "desconocida"


# ─────────────────────────────────────────────────────────────────────────────
# Listado de sesiones activas
# ─────────────────────────────────────────────────────────────────────────────

class TestListarSesionesActivas:

    def test_listar_vacio_retorna_lista_vacia(self, mongo_db):
        assert list_active_sessions() == []

    def test_listar_devuelve_sesion_activa(self, mongo_db):
        _create(jti="jti-1")
        result = list_active_sessions()
        assert len(result) == 1

    def test_listar_excluye_sesiones_expiradas(self, mongo_db):
        _create(jti="active", expires=_FUTURE)
        _create(jti="expired", expires=_PAST)
        result = list_active_sessions()
        assert len(result) == 1
        assert result[0]["jti"] == "active"

    def test_listar_excluye_sesiones_revocadas(self, mongo_db):
        s = _create(jti="to-revoke")
        revoke_session(s["_id"])
        assert list_active_sessions() == []

    def test_listar_multiples_sesiones_activas(self, mongo_db):
        _create(jti="j1")
        _create(jti="j2")
        _create(jti="j3")
        assert len(list_active_sessions()) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Revocación por ID
# ─────────────────────────────────────────────────────────────────────────────

class TestRevocarSesion:

    def test_revocar_existente_retorna_true(self, mongo_db):
        s = _create()
        assert revoke_session(s["_id"]) is True

    def test_revocar_quita_de_activas(self, mongo_db):
        s = _create()
        revoke_session(s["_id"])
        assert list_active_sessions() == []

    def test_revocar_id_inexistente_retorna_false(self, mongo_db):
        assert revoke_session(str(ObjectId())) is False

    def test_revocar_una_no_afecta_otras(self, mongo_db):
        s1 = _create(jti="j1")
        _create(jti="j2")
        revoke_session(s1["_id"])
        actives = list_active_sessions()
        assert len(actives) == 1
        assert actives[0]["jti"] == "j2"


# ─────────────────────────────────────────────────────────────────────────────
# Revocación masiva por email
# ─────────────────────────────────────────────────────────────────────────────

class TestRevocarPorEmail:

    def test_revocar_por_email_retorna_cantidad_revocada(self, mongo_db):
        _create(email="a@b.cu", jti="j1")
        _create(email="a@b.cu", jti="j2")
        count = revoke_sessions_by_email("a@b.cu")
        assert count == 2

    def test_revocar_por_email_quita_todas_las_sesiones_del_usuario(self, mongo_db):
        _create(email="a@b.cu", jti="j1")
        _create(email="a@b.cu", jti="j2")
        revoke_sessions_by_email("a@b.cu")
        assert list_active_sessions() == []

    def test_revocar_por_email_no_afecta_otros_usuarios(self, mongo_db):
        _create(email="a@b.cu", jti="j1")
        _create(email="c@d.cu", jti="j2")
        revoke_sessions_by_email("a@b.cu")
        actives = list_active_sessions()
        assert len(actives) == 1
        assert actives[0]["email"] == "c@d.cu"

    def test_revocar_email_sin_sesiones_retorna_cero(self, mongo_db):
        count = revoke_sessions_by_email("nobody@x.cu")
        assert count == 0


# ─────────────────────────────────────────────────────────────────────────────
# Verificación de token revocado
# ─────────────────────────────────────────────────────────────────────────────

class TestIsTokenRevocado:

    def test_token_activo_no_esta_revocado(self, mongo_db):
        _create(jti="live-token")
        assert is_token_revoked("live-token") is False

    def test_token_revocado_retorna_true(self, mongo_db):
        s = _create(jti="dead-token")
        revoke_session(s["_id"])
        assert is_token_revoked("dead-token") is True

    def test_jti_desconocido_retorna_false(self, mongo_db):
        # JTI de tokens pre-tracking — se dejan pasar
        assert is_token_revoked("jti-que-no-existe") is False
