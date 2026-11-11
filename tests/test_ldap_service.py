"""
Pruebas del flujo de autenticación LDAP.

En lugar de levantar un servidor LDAP real, se inyecta la estrategia en memoria
`MOCK_SYNC` de ldap3 a través del único punto de apertura de conexiones del
servicio (`ldap_service._open_connection`). Así se ejercita EXACTAMENTE el mismo
código de producción (search-bind, filtros, lectura de atributos) contra un
directorio simulado — demostrando que la integración funciona con cualquier LDAP
estándar (RFC 4511).
"""
import pytest
from ldap3 import Server, Connection, MOCK_SYNC
from ldap3.core.exceptions import LDAPBindError

import app.services.ldap_service as ldap_service
from app.services.ldap_config_service import (
    get_ldap_config,
    save_ldap_config,
    SECRET_MASK,
)
# Importado con alias para que pytest no lo recolecte como caso de prueba.
from app.services.ldap_config_service import test_ldap_connection as run_connection_test

# Directorio institucional simulado (DN -> atributos). userPassword es lo que la
# estrategia MOCK valida en el bind.
_DIRECTORY = {
    "cn=svc,dc=cujae,dc=edu,dc=cu": {
        "objectClass": ["inetOrgPerson"],
        "cn": "svc",
        "sn": "svc",
        "userPassword": "svcpass",
    },
    "uid=jdoe,ou=people,dc=cujae,dc=edu,dc=cu": {
        "objectClass": ["inetOrgPerson"],
        "cn": "Juan Doe",
        "sn": "Doe",
        "mail": "jdoe@cujae.edu.cu",
        "userPassword": "secret123",
    },
    "uid=mlopez,ou=people,dc=cujae,dc=edu,dc=cu": {
        "objectClass": ["inetOrgPerson"],
        "cn": "María López",
        "sn": "López",
        "mail": "mlopez@cujae.edu.cu",
        "userPassword": "claveSegura",
    },
}

_VALID_CONFIG = {
    "enabled": True,
    "serverUrl": "ldap://mock:389",
    "baseDn": "dc=cujae,dc=edu,dc=cu",
    "bindDn": "cn=svc,dc=cujae,dc=edu,dc=cu",
    "bindPassword": "svcpass",
    "userSearchFilter": "(mail={email})",
    "emailAttr": "mail",
    "nameAttr": "cn",
}


@pytest.fixture()
def mock_directory(monkeypatch):
    """
    Sustituye ldap_service._open_connection por una fábrica que devuelve
    conexiones MOCK_SYNC sembradas con el directorio simulado. Cada conexión
    valida la contraseña del bind contra userPassword, igual que un LDAP real.
    """
    def _fake_open(server, user, password):
        mock_server = Server("mock-cujae")
        conn = Connection(mock_server, user=user, password=password, client_strategy=MOCK_SYNC)
        for dn, attrs in _DIRECTORY.items():
            conn.strategy.add_entry(dn, dict(attrs))
        if not conn.bind():
            raise LDAPBindError("invalidCredentials")
        return conn

    monkeypatch.setattr(ldap_service, "_open_connection", _fake_open)
    return _fake_open


# ─────────────────────────────────────────────────────────────────────────────
# Autenticación contra el directorio simulado
# ─────────────────────────────────────────────────────────────────────────────

class TestAutenticacionLdap:

    def test_credenciales_validas_devuelven_email_y_nombre(self, mongo_db, mock_directory):
        save_ldap_config(_VALID_CONFIG)
        result = ldap_service.authenticate_ldap("jdoe@cujae.edu.cu", "secret123")
        assert result["email"] == "jdoe@cujae.edu.cu"
        assert result["name"] == "Juan Doe"

    def test_email_se_normaliza_a_minusculas(self, mongo_db, mock_directory):
        save_ldap_config(_VALID_CONFIG)
        result = ldap_service.authenticate_ldap("JDOE@cujae.edu.cu", "secret123")
        assert result["email"] == "jdoe@cujae.edu.cu"

    def test_password_incorrecta_lanza_credenciales_invalidas(self, mongo_db, mock_directory):
        save_ldap_config(_VALID_CONFIG)
        with pytest.raises(ValueError, match="[Ii]nválid"):
            ldap_service.authenticate_ldap("jdoe@cujae.edu.cu", "malísima")

    def test_usuario_inexistente_lanza_no_encontrado(self, mongo_db, mock_directory):
        save_ldap_config(_VALID_CONFIG)
        with pytest.raises(ValueError, match="no encontrado"):
            ldap_service.authenticate_ldap("fantasma@cujae.edu.cu", "x")

    def test_segundo_usuario_del_directorio(self, mongo_db, mock_directory):
        save_ldap_config(_VALID_CONFIG)
        result = ldap_service.authenticate_ldap("mlopez@cujae.edu.cu", "claveSegura")
        assert result["name"] == "María López"

    def test_password_vacia_se_rechaza_antes_de_conectar(self, mongo_db, mock_directory):
        save_ldap_config(_VALID_CONFIG)
        with pytest.raises(ValueError, match="obligatori"):
            ldap_service.authenticate_ldap("jdoe@cujae.edu.cu", "")

    def test_ldap_deshabilitado_lanza_error(self, mongo_db, mock_directory):
        save_ldap_config({**_VALID_CONFIG, "enabled": False})
        with pytest.raises(ValueError, match="deshabilitada"):
            ldap_service.authenticate_ldap("jdoe@cujae.edu.cu", "secret123")

    def test_filtro_por_uid_tambien_funciona(self, mongo_db, mock_directory):
        # Demuestra que el filtro es configurable (cualquier atributo del directorio).
        save_ldap_config({**_VALID_CONFIG, "userSearchFilter": "(cn={email})"})
        result = ldap_service.authenticate_ldap("Juan Doe", "secret123")
        assert result["email"] == "jdoe@cujae.edu.cu"


# ─────────────────────────────────────────────────────────────────────────────
# Botón "Probar conexión" (probe_ldap / test_ldap_connection)
# ─────────────────────────────────────────────────────────────────────────────

class TestProbarConexion:

    @pytest.mark.asyncio
    async def test_solo_bind_de_servicio_exitoso(self, mongo_db, mock_directory):
        result = await run_connection_test(_VALID_CONFIG)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_bind_de_servicio_con_password_incorrecta_falla(self, mongo_db, mock_directory):
        result = await run_connection_test({**_VALID_CONFIG, "bindPassword": "incorrecta"})
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_con_credenciales_de_muestra_validas(self, mongo_db, mock_directory):
        result = await run_connection_test({
            **_VALID_CONFIG,
            "sampleEmail": "jdoe@cujae.edu.cu",
            "samplePassword": "secret123",
        })
        assert result["success"] is True
        assert result["sampleUser"] == "Juan Doe"

    @pytest.mark.asyncio
    async def test_con_credenciales_de_muestra_invalidas(self, mongo_db, mock_directory):
        result = await run_connection_test({
            **_VALID_CONFIG,
            "sampleEmail": "jdoe@cujae.edu.cu",
            "samplePassword": "no-es",
        })
        assert result["success"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Servicio de configuración (enmascarado del secreto, persistencia)
# ─────────────────────────────────────────────────────────────────────────────

class TestConfiguracionLdap:

    def test_sin_datos_devuelve_defaults_deshabilitado(self, mongo_db):
        cfg = get_ldap_config()
        assert cfg["enabled"] is False
        assert cfg["bindPassword"] == ""
        assert cfg["hasBindPassword"] is False

    def test_password_se_enmascara_en_lectura(self, mongo_db):
        save_ldap_config(_VALID_CONFIG)
        cfg = get_ldap_config()
        assert cfg["bindPassword"] == SECRET_MASK
        assert cfg["hasBindPassword"] is True

    def test_include_secret_devuelve_password_real(self, mongo_db):
        save_ldap_config(_VALID_CONFIG)
        cfg = get_ldap_config(include_secret=True)
        assert cfg["bindPassword"] == "svcpass"

    def test_guardar_con_mascara_conserva_password_anterior(self, mongo_db):
        save_ldap_config(_VALID_CONFIG)
        # El cliente reenvía la máscara (no escribió una nueva contraseña).
        save_ldap_config({**_VALID_CONFIG, "bindPassword": SECRET_MASK, "baseDn": "dc=otro"})
        cfg = get_ldap_config(include_secret=True)
        assert cfg["bindPassword"] == "svcpass"
        assert cfg["baseDn"] == "dc=otro"

    def test_filtro_sin_marcador_email_lanza_error(self, mongo_db):
        with pytest.raises(ValueError, match="email"):
            save_ldap_config({**_VALID_CONFIG, "userSearchFilter": "(mail=fijo)"})

    def test_habilitado_sin_server_url_lanza_error(self, mongo_db):
        with pytest.raises(ValueError, match="servidor LDAP"):
            save_ldap_config({**_VALID_CONFIG, "serverUrl": ""})
