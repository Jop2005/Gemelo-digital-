"""
Pruebas de integración para invitation_service.

Cubre generación de códigos únicos, validación, uso único y listado.
"""
import pytest

from app.services.invitation_service import (
    create_invitation_code,
    validate_invitation_code,
    mark_invitation_code_as_used,
    list_invitation_codes,
)


# ─────────────────────────────────────────────────────────────────────────────
# Generación de códigos
# ─────────────────────────────────────────────────────────────────────────────

class TestCrearCodigoInvitacion:

    def test_crear_retorna_documento_con_id(self, mongo_db):
        code = create_invitation_code("admin", "creator@x.cu")
        assert "_id" in code
        assert isinstance(code["_id"], str)

    def test_crear_retorna_codigo_no_vacio(self, mongo_db):
        code = create_invitation_code("user", "creator@x.cu")
        assert len(code["code"]) > 0

    def test_crear_codigo_tiene_longitud_correcta(self, mongo_db):
        code = create_invitation_code("user", "creator@x.cu")
        # CODE_LENGTH = 8
        assert len(code["code"]) == 8

    def test_crear_con_rol_admin(self, mongo_db):
        code = create_invitation_code("admin", "creator@x.cu")
        assert code["role"] == "admin"

    def test_crear_con_rol_user(self, mongo_db):
        code = create_invitation_code("user", "creator@x.cu")
        assert code["role"] == "user"

    def test_crear_no_usado_por_defecto(self, mongo_db):
        code = create_invitation_code("user", "creator@x.cu")
        assert code["isUsed"] is False

    def test_crear_persiste_created_by(self, mongo_db):
        code = create_invitation_code("user", "admin@cujae.edu.cu")
        assert code["createdBy"] == "admin@cujae.edu.cu"

    def test_crear_used_by_es_none_por_defecto(self, mongo_db):
        code = create_invitation_code("user", "creator@x.cu")
        assert code["usedBy"] is None

    def test_crear_timestamps_presentes(self, mongo_db):
        code = create_invitation_code("user", "creator@x.cu")
        assert code["createdAt"] is not None
        assert code["updatedAt"] is not None

    def test_crear_rol_invalido_lanza_error(self, mongo_db):
        with pytest.raises(ValueError, match="[Rr]ol"):
            create_invitation_code("superadmin", "creator@x.cu")

    def test_crear_multiples_codigos_son_distintos(self, mongo_db):
        codes = [create_invitation_code("user", "c@x.cu")["code"] for _ in range(10)]
        assert len(set(codes)) == 10  # todos únicos


# ─────────────────────────────────────────────────────────────────────────────
# Validación
# ─────────────────────────────────────────────────────────────────────────────

class TestValidarCodigoInvitacion:

    def test_codigo_valido_retorna_rol(self, mongo_db):
        inv = create_invitation_code("admin", "c@x.cu")
        role = validate_invitation_code(inv["code"])
        assert role == "admin"

    def test_codigo_valido_user_retorna_user(self, mongo_db):
        inv = create_invitation_code("user", "c@x.cu")
        role = validate_invitation_code(inv["code"])
        assert role == "user"

    def test_codigo_inexistente_lanza_error(self, mongo_db):
        with pytest.raises(ValueError, match="[Ii]nválido"):
            validate_invitation_code("XXXXXXXX")

    def test_codigo_ya_usado_lanza_error(self, mongo_db):
        inv = create_invitation_code("user", "c@x.cu")
        mark_invitation_code_as_used(inv["code"], "operador@x.cu")
        with pytest.raises(ValueError, match="[Uu]tilizado|[Uu]sado"):
            validate_invitation_code(inv["code"])


# ─────────────────────────────────────────────────────────────────────────────
# Marcar como usado
# ─────────────────────────────────────────────────────────────────────────────

class TestMarcarCodigoUsado:

    def test_marcar_como_usado_funciona(self, mongo_db):
        inv = create_invitation_code("user", "c@x.cu")
        # No debe lanzar excepción
        mark_invitation_code_as_used(inv["code"], "operador@x.cu")

    def test_marcar_bloquea_validacion_posterior(self, mongo_db):
        inv = create_invitation_code("user", "c@x.cu")
        mark_invitation_code_as_used(inv["code"], "operador@x.cu")
        with pytest.raises(ValueError):
            validate_invitation_code(inv["code"])

    def test_marcar_codigo_ya_usado_lanza_error(self, mongo_db):
        inv = create_invitation_code("user", "c@x.cu")
        mark_invitation_code_as_used(inv["code"], "primero@x.cu")
        with pytest.raises(ValueError):
            mark_invitation_code_as_used(inv["code"], "segundo@x.cu")

    def test_marcar_codigo_inexistente_lanza_error(self, mongo_db):
        with pytest.raises(ValueError):
            mark_invitation_code_as_used("XXXXXXXX", "alguien@x.cu")


# ─────────────────────────────────────────────────────────────────────────────
# Listado
# ─────────────────────────────────────────────────────────────────────────────

class TestListarCodigos:

    def test_listar_vacio_retorna_lista_vacia(self, mongo_db):
        assert list_invitation_codes() == []

    def test_listar_devuelve_codigo_creado(self, mongo_db):
        create_invitation_code("user", "c@x.cu")
        assert len(list_invitation_codes()) == 1

    def test_listar_devuelve_todos_los_codigos(self, mongo_db):
        create_invitation_code("admin", "c@x.cu")
        create_invitation_code("user", "c@x.cu")
        create_invitation_code("user", "c@x.cu")
        assert len(list_invitation_codes()) == 3

    def test_listar_incluye_codigos_usados(self, mongo_db):
        inv = create_invitation_code("user", "c@x.cu")
        mark_invitation_code_as_used(inv["code"], "o@x.cu")
        result = list_invitation_codes()
        assert any(r["isUsed"] for r in result)

    def test_listar_estructura_correcta(self, mongo_db):
        create_invitation_code("user", "c@x.cu")
        item = list_invitation_codes()[0]
        for field in ["_id", "code", "role", "isUsed", "createdBy", "usedBy"]:
            assert field in item
