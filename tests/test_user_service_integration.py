"""
Pruebas de integración para user_service (requieren mongo_db).

Cubre: register_user, authenticate_user, change_password, list_users, delete_user.
"""
import pytest

from app.services.user_service import (
    register_user,
    authenticate_user,
    change_password,
    list_users,
    delete_user,
)
from app.services.invitation_service import create_invitation_code


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_code(mongo_db, role="user"):
    return create_invitation_code(role, "system")["code"]


def _register(mongo_db, email="op@test.cu", password="Segura123!", role="user"):
    code = _make_code(mongo_db, role)
    return register_user({"email": email, "password": password, "invitationCode": code})


# ─────────────────────────────────────────────────────────────────────────────
# Registro
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistrarUsuario:

    def test_registrar_retorna_usuario_con_id(self, mongo_db):
        user = _register(mongo_db)
        assert "_id" in user and len(user["_id"]) > 0

    def test_registrar_normaliza_email_a_minusculas(self, mongo_db):
        code = _make_code(mongo_db)
        user = register_user({"email": "OP@Test.Cu", "password": "Segura123!", "invitationCode": code})
        assert user["email"] == "op@test.cu"

    def test_registrar_persiste_rol_del_codigo(self, mongo_db):
        assert _register(mongo_db, role="user")["role"] == "user"

    def test_registrar_admin_via_codigo_admin(self, mongo_db):
        assert _register(mongo_db, role="admin")["role"] == "admin"

    def test_registrar_sin_email_lanza_error(self, mongo_db):
        code = _make_code(mongo_db)
        with pytest.raises(ValueError, match="[Cc]orreo"):
            register_user({"email": "", "password": "Segura123!", "invitationCode": code})

    def test_registrar_sin_codigo_lanza_error(self, mongo_db):
        with pytest.raises(ValueError, match="[Cc][oó]digo"):
            register_user({"email": "x@x.cu", "password": "Segura123!", "invitationCode": ""})

    def test_registrar_password_corta_lanza_error(self, mongo_db):
        code = _make_code(mongo_db)
        with pytest.raises(ValueError, match="contrase"):
            register_user({"email": "x@x.cu", "password": "abc", "invitationCode": code})

    def test_registrar_email_duplicado_lanza_error(self, mongo_db):
        _register(mongo_db, email="dup@test.cu")
        code2 = _make_code(mongo_db)
        with pytest.raises(ValueError, match="[Ee]xiste|[Dd]uplicado"):
            register_user({"email": "dup@test.cu", "password": "Segura123!", "invitationCode": code2})

    def test_codigo_invalido_lanza_error(self, mongo_db):
        with pytest.raises(ValueError):
            register_user({"email": "x@x.cu", "password": "Segura123!", "invitationCode": "XXXXXXXX"})

    def test_codigo_no_reutilizable(self, mongo_db):
        code = _make_code(mongo_db)
        register_user({"email": "a@x.cu", "password": "Segura123!", "invitationCode": code})
        with pytest.raises(ValueError):
            register_user({"email": "b@x.cu", "password": "Segura123!", "invitationCode": code})

    def test_registrar_persiste_nombre(self, mongo_db):
        code = _make_code(mongo_db)
        user = register_user({
            "email": "named@test.cu", "password": "Segura123!",
            "invitationCode": code, "name": "Fabian Test",
        })
        assert user.get("name") == "Fabian Test"

    def test_registrar_sin_nombre_no_falla(self, mongo_db):
        assert _register(mongo_db, email="noname@test.cu") is not None


# ─────────────────────────────────────────────────────────────────────────────
# Autenticación
# ─────────────────────────────────────────────────────────────────────────────

class TestAutenticarUsuario:

    def test_credenciales_correctas_retornan_usuario(self, mongo_db):
        _register(mongo_db, email="auth@test.cu", password="Segura123!")
        user = authenticate_user({"email": "auth@test.cu", "password": "Segura123!"})
        assert user["email"] == "auth@test.cu"

    def test_credenciales_correctas_retornan_rol(self, mongo_db):
        _register(mongo_db, email="auth2@test.cu", password="Segura123!", role="admin")
        user = authenticate_user({"email": "auth2@test.cu", "password": "Segura123!"})
        assert user["role"] == "admin"

    def test_email_insensible_a_mayusculas(self, mongo_db):
        _register(mongo_db, email="case@test.cu", password="Segura123!")
        user = authenticate_user({"email": "CASE@test.cu", "password": "Segura123!"})
        assert user["email"] == "case@test.cu"

    def test_password_incorrecta_lanza_error(self, mongo_db):
        _register(mongo_db, email="wrong@test.cu", password="Segura123!")
        with pytest.raises(ValueError, match="[Cc]redencial"):
            authenticate_user({"email": "wrong@test.cu", "password": "Incorrecta!"})

    def test_email_inexistente_lanza_error(self, mongo_db):
        with pytest.raises(ValueError, match="[Cc]redencial"):
            authenticate_user({"email": "noexiste@test.cu", "password": "Segura123!"})

    def test_sin_email_lanza_error(self, mongo_db):
        with pytest.raises(ValueError):
            authenticate_user({"email": "", "password": "Segura123!"})

    def test_sin_password_lanza_error(self, mongo_db):
        with pytest.raises(ValueError):
            authenticate_user({"email": "x@x.cu", "password": ""})


# ─────────────────────────────────────────────────────────────────────────────
# Cambio de contraseña
# ─────────────────────────────────────────────────────────────────────────────

class TestCambiarPassword:

    def test_cambio_exitoso_retorna_true(self, mongo_db):
        _register(mongo_db, email="chpass@test.cu", password="Antigua123!")
        assert change_password("chpass@test.cu", "Antigua123!", "Nueva4567!") is True

    def test_nueva_password_funciona_en_autenticacion(self, mongo_db):
        _register(mongo_db, email="chpass2@test.cu", password="Antigua123!")
        change_password("chpass2@test.cu", "Antigua123!", "Nueva4567!")
        user = authenticate_user({"email": "chpass2@test.cu", "password": "Nueva4567!"})
        assert user["email"] == "chpass2@test.cu"

    def test_password_anterior_ya_no_funciona(self, mongo_db):
        _register(mongo_db, email="chpass3@test.cu", password="Antigua123!")
        change_password("chpass3@test.cu", "Antigua123!", "Nueva4567!")
        with pytest.raises(ValueError):
            authenticate_user({"email": "chpass3@test.cu", "password": "Antigua123!"})

    def test_password_actual_incorrecta_lanza_error(self, mongo_db):
        _register(mongo_db, email="chpass4@test.cu", password="Antigua123!")
        with pytest.raises(ValueError, match="[Cc]orrecta|[Aa]ctual"):
            change_password("chpass4@test.cu", "Incorrecta!", "Nueva4567!")

    def test_nueva_password_corta_lanza_error(self, mongo_db):
        _register(mongo_db, email="chpass5@test.cu", password="Antigua123!")
        with pytest.raises(ValueError, match="contrase"):
            change_password("chpass5@test.cu", "Antigua123!", "corta")

    def test_usuario_inexistente_lanza_error(self, mongo_db):
        with pytest.raises(ValueError, match="[Nn]o encontrado"):
            change_password("noexiste@test.cu", "Antigua123!", "Nueva4567!")


# ─────────────────────────────────────────────────────────────────────────────
# Listado y eliminación
# ─────────────────────────────────────────────────────────────────────────────

class TestListarYEliminarUsuarios:

    def test_listar_vacio_retorna_lista_vacia(self, mongo_db):
        assert list_users() == []

    def test_listar_devuelve_usuario_registrado(self, mongo_db):
        _register(mongo_db, email="lst@test.cu")
        assert len(list_users()) == 1

    def test_listar_multiples_usuarios(self, mongo_db):
        _register(mongo_db, email="a@test.cu")
        _register(mongo_db, email="b@test.cu")
        _register(mongo_db, email="c@test.cu")
        assert len(list_users()) == 3

    def test_listar_estructura_correcta(self, mongo_db):
        _register(mongo_db, email="struct@test.cu")
        user = list_users()[0]
        for field in ["_id", "email", "role", "createdAt", "updatedAt"]:
            assert field in user

    def test_listar_no_expone_password_hash(self, mongo_db):
        _register(mongo_db, email="safe@test.cu")
        user = list_users()[0]
        assert "passwordHash" not in user

    def test_eliminar_existente_retorna_true(self, mongo_db):
        user = _register(mongo_db, email="del@test.cu")
        assert delete_user(user["_id"]) is True

    def test_eliminar_quita_de_listado(self, mongo_db):
        user = _register(mongo_db, email="del2@test.cu")
        delete_user(user["_id"])
        remaining = [u for u in list_users() if u["email"] == "del2@test.cu"]
        assert remaining == []

    def test_eliminar_inexistente_retorna_false(self, mongo_db):
        assert delete_user("000000000000000000000000") is False

    def test_eliminar_uno_no_afecta_otro(self, mongo_db):
        a = _register(mongo_db, email="a2@test.cu")
        _register(mongo_db, email="b2@test.cu")
        delete_user(a["_id"])
        assert len(list_users()) == 1
        assert list_users()[0]["email"] == "b2@test.cu"
