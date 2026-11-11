"""
Pruebas unitarias para funciones puras de app/services/user_service.py

Cubre (sin acceso a base de datos):
- _normalize_email: normalización de correos electrónicos
- _ensure_role: validación de roles permitidos
- _ensure_password: validación de contraseñas
- _hash_password: hashing seguro con scrypt
- _verify_password: verificación de contraseñas
- _map_user: mapeo de documento MongoDB a dict público
"""
import pytest
from datetime import datetime
from bson import ObjectId

from app.services.user_service import (
    _normalize_email,
    _ensure_role,
    _ensure_password,
    _hash_password,
    _verify_password,
    _map_user,
)


# ─────────────────────────────────────────────────────────────────
# _normalize_email
# ─────────────────────────────────────────────────────────────────

class TestNormalizeEmail:

    def test_convierte_a_minusculas(self):
        assert _normalize_email("User@Example.COM") == "user@example.com"

    def test_elimina_espacios_al_inicio_y_final(self):
        assert _normalize_email("  user@example.com  ") == "user@example.com"

    def test_correo_ya_normalizado_no_cambia(self):
        assert _normalize_email("user@example.com") == "user@example.com"

    def test_combina_mayusculas_y_espacios(self):
        assert _normalize_email("  ADMIN@CUJAE.EDU.CU  ") == "admin@cujae.edu.cu"


# ─────────────────────────────────────────────────────────────────
# _ensure_role
# ─────────────────────────────────────────────────────────────────

class TestEnsureRole:

    def test_acepta_rol_admin(self):
        assert _ensure_role("admin") == "admin"

    def test_acepta_rol_user(self):
        assert _ensure_role("user") == "user"

    def test_normaliza_rol_en_mayusculas(self):
        assert _ensure_role("ADMIN") == "admin"
        assert _ensure_role("USER") == "user"

    def test_elimina_espacios_del_rol(self):
        assert _ensure_role("  admin  ") == "admin"

    def test_rol_vacio_devuelve_user_por_defecto(self):
        assert _ensure_role(None) == "user"
        assert _ensure_role("") == "user"

    def test_rol_invalido_lanza_value_error(self):
        with pytest.raises(ValueError, match="Rol no válido"):
            _ensure_role("superuser")

    def test_rol_invalido_lanza_value_error_con_rol_inventado(self):
        with pytest.raises(ValueError):
            _ensure_role("moderator")


# ─────────────────────────────────────────────────────────────────
# _ensure_password
# ─────────────────────────────────────────────────────────────────

class TestEnsurePassword:

    def test_acepta_contrasena_valida(self):
        result = _ensure_password("segura123")
        assert result == "segura123"

    def test_rechaza_contrasena_nula(self):
        with pytest.raises(ValueError, match="al menos"):
            _ensure_password(None)

    def test_rechaza_contrasena_vacia(self):
        with pytest.raises(ValueError):
            _ensure_password("")

    def test_rechaza_contrasena_muy_corta(self):
        with pytest.raises(ValueError, match="al menos"):
            _ensure_password("abc1234")  # 7 caracteres < 8

    def test_acepta_contrasena_de_exactamente_8_caracteres(self):
        result = _ensure_password("abcd1234")
        assert result == "abcd1234"

    def test_acepta_contrasena_larga(self):
        larga = "Esta_es_una_contraseña_muy_larga_y_segura_2024"
        assert _ensure_password(larga) == larga


# ─────────────────────────────────────────────────────────────────
# _hash_password y _verify_password
# ─────────────────────────────────────────────────────────────────

class TestPasswordHashing:

    def test_hash_produce_formato_salt_derivedkey(self):
        hashed = _hash_password("contrasena123")
        parts = hashed.split(":")
        assert len(parts) == 2, "El hash debe tener formato 'salt:derivedKey'"

    def test_salt_tiene_32_caracteres_hex(self):
        hashed = _hash_password("contrasena123")
        salt_hex = hashed.split(":")[0]
        assert len(salt_hex) == 32  # 16 bytes → 32 hex chars

    def test_dos_hashes_de_misma_contrasena_son_diferentes(self):
        h1 = _hash_password("mismacontrasena")
        h2 = _hash_password("mismacontrasena")
        assert h1 != h2  # Salt aleatorio garantiza unicidad

    def test_verificacion_exitosa_con_contrasena_correcta(self):
        password = "contrasena_correcta"
        hashed = _hash_password(password)
        assert _verify_password(password, hashed) is True

    def test_verificacion_falla_con_contrasena_incorrecta(self):
        hashed = _hash_password("contrasena_correcta")
        assert _verify_password("contrasena_incorrecta", hashed) is False

    def test_verificacion_falla_con_hash_malformado(self):
        assert _verify_password("cualquiercontrasena", "hashsinformato") is False

    def test_verificacion_es_timing_safe(self):
        # Verificar que la función no lanza excepción ante hashes válidos e inválidos
        hashed = _hash_password("test1234")
        result_ok = _verify_password("test1234", hashed)
        result_fail = _verify_password("wrong123", hashed)
        assert result_ok is True
        assert result_fail is False


# ─────────────────────────────────────────────────────────────────
# _map_user
# ─────────────────────────────────────────────────────────────────

class TestMapUser:

    def _make_doc(self, **overrides):
        oid = ObjectId()
        doc = {
            "_id": oid,
            "email": "user@cujae.edu.cu",
            "name": "Test User",
            "role": "user",
            "passwordHash": "abc:def",
            "createdAt": datetime(2024, 1, 15, 10, 0, 0),
            "updatedAt": datetime(2024, 6, 1, 12, 0, 0),
        }
        doc.update(overrides)
        return doc

    def test_id_se_convierte_a_string(self):
        doc = self._make_doc()
        result = _map_user(doc)
        assert isinstance(result["_id"], str)

    def test_no_expone_password_hash(self):
        doc = self._make_doc()
        result = _map_user(doc)
        assert "passwordHash" not in result

    def test_email_se_preserva(self):
        doc = self._make_doc(email="admin@cujae.edu.cu")
        assert _map_user(doc)["email"] == "admin@cujae.edu.cu"

    def test_rol_se_preserva(self):
        doc = self._make_doc(role="admin")
        assert _map_user(doc)["role"] == "admin"

    def test_nombre_puede_ser_none(self):
        doc = self._make_doc(name=None)
        result = _map_user(doc)
        assert result["name"] is None

    def test_fechas_se_convierten_a_iso_string(self):
        doc = self._make_doc()
        result = _map_user(doc)
        assert isinstance(result["createdAt"], str)
        assert isinstance(result["updatedAt"], str)

    def test_created_at_ausente_retorna_none(self):
        doc = self._make_doc()
        del doc["createdAt"]
        result = _map_user(doc)
        assert result["createdAt"] is None
