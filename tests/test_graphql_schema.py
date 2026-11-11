"""
Pruebas de integración para el esquema GraphQL (capa de resolvers).

Usa la ejecución síncrona de Strawberry directamente sobre el schema
(sin levantar FastAPI ni lifespan — sin modelos ML) con la BD
sustituida por mongomock. Verifica que los resolvers de query y
mutation funcionan correctamente end-to-end desde GraphQL hasta
la BD en memoria.

Nota: queries que dependen de servicios externos (clima, ML) no se
prueban aquí; el foco es el flujo CRUD y de autenticación.
"""
import asyncio
import pytest
import mongomock
import app.database as _db_module
from app.auth import create_token
from app.schema import schema


# ─────────────────────────────────────────────────────────────────────────────
# Cliente GraphQL síncrono
# ─────────────────────────────────────────────────────────────────────────────

class GQLClient:
    """Ejecuta consultas síncronas contra el schema Strawberry con contexto inyectado."""

    def __init__(self, context: dict):
        self._ctx = context

    def execute(self, query: str, variables: dict = None):
        result = schema.execute_sync(
            query,
            variable_values=variables or {},
            context_value=self._ctx,
        )
        data = result.data or {}
        errors = [{"message": str(e)} for e in (result.errors or [])]
        return {"data": data, "errors": errors}

    def execute_async(self, query: str, variables: dict = None):
        """Para resolvers async (ej: testWeatherSource) que no pueden usar execute_sync."""
        ctx = self._ctx

        async def _run():
            result = await schema.execute(
                query,
                variable_values=variables or {},
                context_value=ctx,
            )
            data = result.data or {}
            errors = [{"message": str(e)} for e in (result.errors or [])]
            return {"data": data, "errors": errors}

        return asyncio.run(_run())


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def db(monkeypatch):
    client = mongomock.MongoClient()
    test_db = client["gemelo_test"]
    monkeypatch.setattr(_db_module, "get_database", lambda: test_db)
    monkeypatch.setattr(_db_module, "_db", test_db)
    yield test_db
    client.close()


@pytest.fixture()
def anon(db):
    return GQLClient({"current_user": None, "request": None})


@pytest.fixture()
def admin(db):
    return GQLClient({"current_user": {"sub": "admin@test.cu", "role": "admin", "jti": "adm-jti"}, "request": None})


@pytest.fixture()
def user(db):
    return GQLClient({"current_user": {"sub": "op@test.cu", "role": "user", "jti": "op-jti"}, "request": None})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _has_errors(resp: dict) -> bool:
    return bool(resp.get("errors"))


def _data_is_null(resp: dict, key: str) -> bool:
    return resp["data"].get(key) is None


def _rejected(resp: dict, key: str) -> bool:
    return _has_errors(resp) or _data_is_null(resp, key)


# ─────────────────────────────────────────────────────────────────────────────
# Paneles — CRUD
# ─────────────────────────────────────────────────────────────────────────────

CREATE_PANEL = """
mutation($input: PanelInput!) {
  createPanel(input: $input) {
    _id
    manufacturer
    ratedPowerKw
    quantity
  }
}
"""

UPDATE_PANEL = """
mutation($id: String!, $input: PanelInput!) {
  updatePanel(id: $id, input: $input) {
    _id
    manufacturer
  }
}
"""

DELETE_PANEL = """
mutation($id: String!) {
  deletePanel(id: $id)
}
"""

LIST_PANELS = "query { panels { _id manufacturer } }"

_PANEL = {
    "manufacturer": "Longi Solar",
    "model": "Hi-MO 6",
    "ratedPowerKw": 0.54,
    "quantity": 93,
}


class TestGraphQLPaneles:

    def test_admin_puede_crear_panel(self, admin):
        r = admin.execute(CREATE_PANEL, {"input": _PANEL})
        assert not _has_errors(r)
        assert r["data"]["createPanel"]["manufacturer"] == "Longi Solar"
        assert r["data"]["createPanel"]["_id"] is not None

    def test_anonimo_no_puede_crear_panel(self, anon):
        r = anon.execute(CREATE_PANEL, {"input": _PANEL})
        assert _rejected(r, "createPanel")

    def test_user_no_puede_crear_panel(self, user):
        r = user.execute(CREATE_PANEL, {"input": _PANEL})
        assert _rejected(r, "createPanel")

    def test_listar_paneles_vacio(self, admin):
        r = admin.execute(LIST_PANELS)
        assert r["data"]["panels"] == []

    def test_listar_panel_creado(self, admin):
        admin.execute(CREATE_PANEL, {"input": _PANEL})
        r = admin.execute(LIST_PANELS)
        assert len(r["data"]["panels"]) == 1
        assert r["data"]["panels"][0]["manufacturer"] == "Longi Solar"

    def test_eliminar_panel_existente(self, admin):
        created = admin.execute(CREATE_PANEL, {"input": _PANEL})
        pid = created["data"]["createPanel"]["_id"]
        r = admin.execute(DELETE_PANEL, {"id": pid})
        assert r["data"]["deletePanel"] is True

    def test_eliminar_panel_quita_de_lista(self, admin):
        created = admin.execute(CREATE_PANEL, {"input": _PANEL})
        pid = created["data"]["createPanel"]["_id"]
        admin.execute(DELETE_PANEL, {"id": pid})
        r = admin.execute(LIST_PANELS)
        assert r["data"]["panels"] == []

    def test_actualizar_panel(self, admin):
        created = admin.execute(CREATE_PANEL, {"input": _PANEL})
        pid = created["data"]["createPanel"]["_id"]
        r = admin.execute(UPDATE_PANEL, {"id": pid, "input": {**_PANEL, "manufacturer": "Canadian Solar"}})
        assert r["data"]["updatePanel"]["manufacturer"] == "Canadian Solar"

    def test_listar_paneles_es_publico(self, anon):
        r = anon.execute(LIST_PANELS)
        assert not _has_errors(r)
        assert isinstance(r["data"]["panels"], list)


# ─────────────────────────────────────────────────────────────────────────────
# Baterías — CRUD
# ─────────────────────────────────────────────────────────────────────────────

CREATE_BATTERY = """
mutation($input: BatteryInput!) {
  createBattery(input: $input) {
    _id
    manufacturer
    capacityKwh
  }
}
"""

DELETE_BATTERY = "mutation($id: String!) { deleteBattery(id: $id) }"
LIST_BATTERIES = "query { batteries { _id manufacturer } }"

_BATTERY = {
    "manufacturer": "CATL",
    "model": "EnerOne",
    "capacityKwh": 100.0,
    "quantity": 1,
    "maxDepthOfDischargePercent": 90.0,
    "chargeRateKw": 25.0,
    "dischargeRateKw": 25.0,
}


class TestGraphQLBaterias:

    def test_admin_puede_crear_bateria(self, admin):
        r = admin.execute(CREATE_BATTERY, {"input": _BATTERY})
        assert not _has_errors(r)
        assert r["data"]["createBattery"]["manufacturer"] == "CATL"

    def test_listar_baterias_vacio(self, admin):
        r = admin.execute(LIST_BATTERIES)
        assert r["data"]["batteries"] == []

    def test_eliminar_bateria(self, admin):
        created = admin.execute(CREATE_BATTERY, {"input": _BATTERY})
        bid = created["data"]["createBattery"]["_id"]
        r = admin.execute(DELETE_BATTERY, {"id": bid})
        assert r["data"]["deleteBattery"] is True

    def test_anonimo_no_puede_crear_bateria(self, anon):
        r = anon.execute(CREATE_BATTERY, {"input": _BATTERY})
        assert _rejected(r, "createBattery")


# ─────────────────────────────────────────────────────────────────────────────
# Electrodomésticos — CRUD
# ─────────────────────────────────────────────────────────────────────────────

CREATE_APPLIANCE = """
mutation($input: ApplianceInput!) {
  createAppliance(input: $input) {
    _id
    name
    category
  }
}
"""

DELETE_APPLIANCE = "mutation($id: String!) { deleteAppliance(id: $id) }"
LIST_APPLIANCES = "query { appliances { _id name } }"

_APPLIANCE = {
    "name": "Aire acondicionado",
    "category": "Climatización",
    "averagePowerW": 1200.0,
    "maxPowerW": 1800.0,
    "quantity": 2,
}


class TestGraphQLElectrodomesticos:

    def test_admin_puede_crear_electrodomestico(self, admin):
        r = admin.execute(CREATE_APPLIANCE, {"input": _APPLIANCE})
        assert not _has_errors(r)
        assert r["data"]["createAppliance"]["name"] == "Aire acondicionado"

    def test_listar_electrodomesticos_vacio(self, admin):
        r = admin.execute(LIST_APPLIANCES)
        assert r["data"]["appliances"] == []

    def test_eliminar_electrodomestico(self, admin):
        created = admin.execute(CREATE_APPLIANCE, {"input": _APPLIANCE})
        aid = created["data"]["createAppliance"]["_id"]
        r = admin.execute(DELETE_APPLIANCE, {"id": aid})
        assert r["data"]["deleteAppliance"] is True

    def test_anonimo_no_puede_crear_electrodomestico(self, anon):
        r = anon.execute(CREATE_APPLIANCE, {"input": _APPLIANCE})
        assert _rejected(r, "createAppliance")


# ─────────────────────────────────────────────────────────────────────────────
# Inversores — CRUD
# ─────────────────────────────────────────────────────────────────────────────

CREATE_INVERTER = """
mutation($input: InverterInput!) {
  createInverter(input: $input) {
    _id
    manufacturer
    ratedPowerKw
  }
}
"""

DELETE_INVERTER = "mutation($id: String!) { deleteInverter(id: $id) }"
LIST_INVERTERS = "query { inverters { _id manufacturer } }"

_INVERTER = {
    "manufacturer": "SMA",
    "model": "Sunny Tripower",
    "ratedPowerKw": 50.0,
    "quantity": 1,
    "efficiencyPercent": 98.3,
}


class TestGraphQLInversores:

    def test_admin_puede_crear_inversor(self, admin):
        r = admin.execute(CREATE_INVERTER, {"input": _INVERTER})
        assert not _has_errors(r)
        assert r["data"]["createInverter"]["manufacturer"] == "SMA"

    def test_listar_inversores_vacio(self, admin):
        r = admin.execute(LIST_INVERTERS)
        assert r["data"]["inverters"] == []

    def test_eliminar_inversor(self, admin):
        created = admin.execute(CREATE_INVERTER, {"input": _INVERTER})
        iid = created["data"]["createInverter"]["_id"]
        r = admin.execute(DELETE_INVERTER, {"id": iid})
        assert r["data"]["deleteInverter"] is True

    def test_anonimo_no_puede_crear_inversor(self, anon):
        r = anon.execute(CREATE_INVERTER, {"input": _INVERTER})
        assert _rejected(r, "createInverter")


# ─────────────────────────────────────────────────────────────────────────────
# Autenticación via GraphQL
# ─────────────────────────────────────────────────────────────────────────────

REGISTER = """
mutation($input: RegisterInput!) {
  registerUser(input: $input) {
    user {
      email
      role
    }
    token
  }
}
"""

LOGIN = """
mutation($input: LoginInput!) {
  loginUser(input: $input) {
    user {
      email
      role
    }
    token
  }
}
"""


class TestGraphQLAuth:

    def _make_code(self, role: str = "user") -> str:
        from app.services.invitation_service import create_invitation_code
        return create_invitation_code(role, "system")["code"]

    def test_registrar_usuario_retorna_token(self, anon, db):
        code = self._make_code()
        r = anon.execute(REGISTER, {"input": {
            "email": "nuevo@test.cu",
            "password": "Password123!",
            "invitationCode": code,
            "name": "Nuevo",
        }})
        assert not _has_errors(r)
        assert r["data"]["registerUser"]["token"] is not None

    def test_registrar_usuario_retorna_email(self, anon, db):
        code = self._make_code()
        r = anon.execute(REGISTER, {"input": {
            "email": "nuevo2@test.cu",
            "password": "Password123!",
            "invitationCode": code,
        }})
        assert r["data"]["registerUser"]["user"]["email"] == "nuevo2@test.cu"

    def test_registrar_admin_con_codigo_admin(self, anon, db):
        code = self._make_code("admin")
        r = anon.execute(REGISTER, {"input": {
            "email": "adm@test.cu",
            "password": "Password123!",
            "invitationCode": code,
        }})
        assert r["data"]["registerUser"]["user"]["role"] == "admin"

    def test_login_credenciales_correctas(self, anon, db):
        code = self._make_code()
        anon.execute(REGISTER, {"input": {
            "email": "login@test.cu",
            "password": "Password123!",
            "invitationCode": code,
        }})
        r = anon.execute(LOGIN, {"input": {"email": "login@test.cu", "password": "Password123!"}})
        assert not _has_errors(r)
        assert r["data"]["loginUser"]["token"] is not None

    def test_login_password_incorrecta_da_error(self, anon, db):
        code = self._make_code()
        anon.execute(REGISTER, {"input": {
            "email": "login2@test.cu",
            "password": "Password123!",
            "invitationCode": code,
        }})
        r = anon.execute(LOGIN, {"input": {"email": "login2@test.cu", "password": "Wrong!"}})
        assert _rejected(r, "loginUser")

    def test_login_email_inexistente_da_error(self, anon, db):
        r = anon.execute(LOGIN, {"input": {"email": "noexiste@test.cu", "password": "pass"}})
        assert _rejected(r, "loginUser")

    def test_codigo_invalido_lanza_error(self, anon, db):
        r = anon.execute(REGISTER, {"input": {
            "email": "test@test.cu",
            "password": "Password123!",
            "invitationCode": "INVALID1",
        }})
        assert _rejected(r, "registerUser")

    def test_codigo_reutilizado_lanza_error(self, anon, db):
        code = self._make_code()
        anon.execute(REGISTER, {"input": {
            "email": "primero@test.cu",
            "password": "Password123!",
            "invitationCode": code,
        }})
        r = anon.execute(REGISTER, {"input": {
            "email": "segundo@test.cu",
            "password": "Password123!",
            "invitationCode": code,
        }})
        assert _rejected(r, "registerUser")


# ─────────────────────────────────────────────────────────────────────────────
# Ubicación
# ─────────────────────────────────────────────────────────────────────────────

QUERY_LOCATION = "query { locationConfig { lat lon name } }"

SAVE_LOCATION = """
mutation($input: LocationConfigInput!) {
  saveLocationConfig(input: $input) {
    lat
    lon
    name
  }
}
"""


class TestGraphQLUbicacion:

    def test_query_location_devuelve_fallback(self, anon):
        r = anon.execute(QUERY_LOCATION)
        assert not _has_errors(r)
        loc = r["data"]["locationConfig"]
        assert isinstance(loc["lat"], float)
        assert isinstance(loc["lon"], float)

    def test_admin_puede_guardar_ubicacion(self, admin):
        r = admin.execute(SAVE_LOCATION, {"input": {"lat": 23.1136, "lon": -82.3666, "name": "La Habana"}})
        assert not _has_errors(r)
        assert abs(r["data"]["saveLocationConfig"]["lat"] - 23.1136) < 0.001

    def test_guardar_y_recuperar_ubicacion(self, admin):
        admin.execute(SAVE_LOCATION, {"input": {"lat": 23.1136, "lon": -82.3666, "name": "CUJAE"}})
        r = admin.execute(QUERY_LOCATION)
        assert r["data"]["locationConfig"]["name"] == "CUJAE"

    def test_anonimo_no_puede_guardar_ubicacion(self, anon):
        r = anon.execute(SAVE_LOCATION, {"input": {"lat": 23.0, "lon": -82.0, "name": "X"}})
        assert _rejected(r, "saveLocationConfig")

    def test_user_no_puede_guardar_ubicacion(self, user):
        r = user.execute(SAVE_LOCATION, {"input": {"lat": 23.0, "lon": -82.0, "name": "X"}})
        assert _rejected(r, "saveLocationConfig")


# ─────────────────────────────────────────────────────────────────────────────
# Códigos de invitación
# ─────────────────────────────────────────────────────────────────────────────

GEN_CODE = """
mutation($role: String!, $createdBy: String!) {
  generateInvitationCode(role: $role, createdBy: $createdBy) {
    code
    role
    isUsed
  }
}
"""

LIST_CODES = "query { invitationCodes { code role isUsed } }"


class TestGraphQLCodigos:

    def test_admin_puede_generar_codigo(self, admin):
        r = admin.execute(GEN_CODE, {"role": "user", "createdBy": "admin@test.cu"})
        assert not _has_errors(r)
        assert len(r["data"]["generateInvitationCode"]["code"]) == 8

    def test_codigo_generado_no_esta_usado(self, admin):
        r = admin.execute(GEN_CODE, {"role": "user", "createdBy": "admin@test.cu"})
        assert r["data"]["generateInvitationCode"]["isUsed"] is False

    def test_codigo_con_rol_admin(self, admin):
        r = admin.execute(GEN_CODE, {"role": "admin", "createdBy": "admin@test.cu"})
        assert r["data"]["generateInvitationCode"]["role"] == "admin"

    def test_anonimo_no_puede_generar_codigo(self, anon):
        r = anon.execute(GEN_CODE, {"role": "user", "createdBy": "x"})
        assert _rejected(r, "generateInvitationCode")

    def test_admin_puede_listar_codigos(self, admin):
        admin.execute(GEN_CODE, {"role": "user", "createdBy": "admin@test.cu"})
        r = admin.execute(LIST_CODES)
        assert not _has_errors(r)
        assert len(r["data"]["invitationCodes"]) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Flujo end-to-end: registro → login → CRUD como admin
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphQLFlujoCompleto:
    """
    Simula el flujo real de un operador:
    1. Admin genera código de invitación
    2. Nuevo usuario se registra con ese código
    3. Usuario hace login
    4. Admin crea equipamiento
    5. Consulta es visible para todos
    """

    def test_flujo_registro_login_crud(self, anon, db):
        # 1. El admin (simulado) genera un código de invitación
        from app.services.invitation_service import create_invitation_code
        code = create_invitation_code("admin", "system")["code"]

        # 2. Se registra un nuevo admin
        r_reg = anon.execute(REGISTER, {"input": {
            "email": "nuevo_admin@test.cu",
            "password": "Admin1234!",
            "invitationCode": code,
        }})
        assert not _has_errors(r_reg)
        assert r_reg["data"]["registerUser"]["user"]["role"] == "admin"

        # 3. Hace login
        r_login = anon.execute(LOGIN, {"input": {
            "email": "nuevo_admin@test.cu",
            "password": "Admin1234!",
        }})
        assert r_login["data"]["loginUser"]["token"] is not None

        # 4. Crea un panel usando cliente admin (simula token validado)
        admin_ctx = GQLClient({"current_user": {"sub": "nuevo_admin@test.cu", "role": "admin", "jti": "e2e"}, "request": None})
        r_panel = admin_ctx.execute(CREATE_PANEL, {"input": _PANEL})
        assert not _has_errors(r_panel)
        assert r_panel["data"]["createPanel"]["manufacturer"] == "Longi Solar"

        # 5. La lista de paneles es visible para cualquiera
        r_list = anon.execute(LIST_PANELS)
        assert len(r_list["data"]["panels"]) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Fuentes meteorológicas — CRUD + activación + testeo con mock
# ─────────────────────────────────────────────────────────────────────────────

SAVE_WEATHER_SOURCE = """
mutation($input: WeatherSourceInput!, $id: String) {
  saveWeatherSource(input: $input, id: $id) {
    _id
    name
    baseUrl
    authType
    enabled
    isActive
    createdAt
    updatedAt
  }
}
"""

DELETE_WEATHER_SOURCE = """
mutation($id: String!) {
  deleteWeatherSource(id: $id)
}
"""

SET_ACTIVE_WEATHER_SOURCE = """
mutation($id: String!) {
  setActiveWeatherSource(id: $id)
}
"""

LIST_WEATHER_SOURCES = """
query {
  weatherSources {
    _id
    name
    authType
    enabled
    isActive
  }
}
"""

ACTIVE_WEATHER_SOURCE = """
query {
  activeWeatherSource {
    _id
    name
    isActive
  }
}
"""

TEST_WEATHER_SOURCE = """
mutation($input: WeatherSourceInput!, $useMock: Boolean) {
  testWeatherSource(input: $input, useMock: $useMock) {
    success
    message
    fields {
      path
      valueType
      sampleValue
    }
    rawJson
  }
}
"""

_OPEN_METEO_SOURCE = {
    "name": "Open-Meteo",
    "baseUrl": "https://api.open-meteo.com/v1/forecast",
    "authType": "none",
    "enabled": True,
    "isActive": False,
}

_MOCK_SOURCE = {
    "name": "Fuente Mock",
    "authType": "mock",
    "enabled": True,
    "isActive": False,
}


class TestGraphQLFuentesClima:

    def test_admin_puede_crear_fuente(self, admin):
        r = admin.execute(SAVE_WEATHER_SOURCE, {"input": _OPEN_METEO_SOURCE})
        assert not _has_errors(r)
        assert r["data"]["saveWeatherSource"]["name"] == "Open-Meteo"

    def test_anonimo_no_puede_crear_fuente(self, anon):
        r = anon.execute(SAVE_WEATHER_SOURCE, {"input": _OPEN_METEO_SOURCE})
        assert _has_errors(r) or _data_is_null(r, "saveWeatherSource")

    def test_user_no_puede_crear_fuente(self, user):
        r = user.execute(SAVE_WEATHER_SOURCE, {"input": _OPEN_METEO_SOURCE})
        assert _has_errors(r) or _data_is_null(r, "saveWeatherSource")

    def test_listar_fuentes_vacio(self, admin):
        r = admin.execute(LIST_WEATHER_SOURCES)
        assert not _has_errors(r)
        assert r["data"]["weatherSources"] == []

    def test_crear_y_listar_fuente(self, admin):
        admin.execute(SAVE_WEATHER_SOURCE, {"input": _OPEN_METEO_SOURCE})
        r = admin.execute(LIST_WEATHER_SOURCES)
        assert not _has_errors(r)
        assert len(r["data"]["weatherSources"]) == 1
        assert r["data"]["weatherSources"][0]["name"] == "Open-Meteo"

    def test_crear_multiples_y_listarlas(self, admin):
        admin.execute(SAVE_WEATHER_SOURCE, {"input": _OPEN_METEO_SOURCE})
        admin.execute(SAVE_WEATHER_SOURCE, {"input": {**_MOCK_SOURCE, "name": "Mock-A"}})
        admin.execute(SAVE_WEATHER_SOURCE, {"input": {**_MOCK_SOURCE, "name": "Mock-B"}})
        r = admin.execute(LIST_WEATHER_SOURCES)
        assert len(r["data"]["weatherSources"]) == 3

    def test_actualizar_fuente(self, admin):
        src = admin.execute(SAVE_WEATHER_SOURCE, {"input": _OPEN_METEO_SOURCE})
        src_id = src["data"]["saveWeatherSource"]["_id"]
        r = admin.execute(SAVE_WEATHER_SOURCE, {
            "input": {**_OPEN_METEO_SOURCE, "name": "Open-Meteo v2"},
            "id": src_id,
        })
        assert not _has_errors(r)
        assert r["data"]["saveWeatherSource"]["name"] == "Open-Meteo v2"

    def test_eliminar_fuente(self, admin):
        src = admin.execute(SAVE_WEATHER_SOURCE, {"input": _OPEN_METEO_SOURCE})
        src_id = src["data"]["saveWeatherSource"]["_id"]
        r = admin.execute(DELETE_WEATHER_SOURCE, {"id": src_id})
        assert not _has_errors(r)
        assert r["data"]["deleteWeatherSource"] is True

    def test_eliminar_quita_de_listado(self, admin):
        src = admin.execute(SAVE_WEATHER_SOURCE, {"input": _OPEN_METEO_SOURCE})
        src_id = src["data"]["saveWeatherSource"]["_id"]
        admin.execute(DELETE_WEATHER_SOURCE, {"id": src_id})
        r = admin.execute(LIST_WEATHER_SOURCES)
        assert r["data"]["weatherSources"] == []

    def test_activar_fuente(self, admin):
        src = admin.execute(SAVE_WEATHER_SOURCE, {"input": _OPEN_METEO_SOURCE})
        src_id = src["data"]["saveWeatherSource"]["_id"]
        r = admin.execute(SET_ACTIVE_WEATHER_SOURCE, {"id": src_id})
        assert not _has_errors(r)
        assert r["data"]["setActiveWeatherSource"] is True

    def test_fuente_activa_aparece_en_query(self, admin):
        src = admin.execute(SAVE_WEATHER_SOURCE, {"input": _OPEN_METEO_SOURCE})
        src_id = src["data"]["saveWeatherSource"]["_id"]
        admin.execute(SET_ACTIVE_WEATHER_SOURCE, {"id": src_id})
        r = admin.execute(ACTIVE_WEATHER_SOURCE)
        assert not _has_errors(r)
        assert r["data"]["activeWeatherSource"] is not None
        assert r["data"]["activeWeatherSource"]["isActive"] is True

    def test_sin_activa_devuelve_null(self, admin):
        admin.execute(SAVE_WEATHER_SOURCE, {"input": _OPEN_METEO_SOURCE})
        r = admin.execute(ACTIVE_WEATHER_SOURCE)
        assert not _has_errors(r)
        assert r["data"]["activeWeatherSource"] is None

    def test_activar_segunda_desactiva_primera(self, admin):
        a = admin.execute(SAVE_WEATHER_SOURCE, {"input": {**_OPEN_METEO_SOURCE, "name": "A"}})
        b = admin.execute(SAVE_WEATHER_SOURCE, {"input": {**_MOCK_SOURCE, "name": "B"}})
        id_a = a["data"]["saveWeatherSource"]["_id"]
        id_b = b["data"]["saveWeatherSource"]["_id"]
        admin.execute(SET_ACTIVE_WEATHER_SOURCE, {"id": id_a})
        admin.execute(SET_ACTIVE_WEATHER_SOURCE, {"id": id_b})
        r = admin.execute(ACTIVE_WEATHER_SOURCE)
        assert r["data"]["activeWeatherSource"]["_id"] == id_b

    def test_crear_fuente_sin_nombre_da_error(self, admin):
        r = admin.execute(SAVE_WEATHER_SOURCE, {"input": {**_OPEN_METEO_SOURCE, "name": ""}})
        assert _has_errors(r) or _data_is_null(r, "saveWeatherSource")

    def test_crear_fuente_auth_invalida_da_error(self, admin):
        r = admin.execute(SAVE_WEATHER_SOURCE, {"input": {**_OPEN_METEO_SOURCE, "authType": "oauth2"}})
        assert _has_errors(r) or _data_is_null(r, "saveWeatherSource")

    def test_testear_fuente_mock_exitoso(self, admin):
        r = admin.execute_async(TEST_WEATHER_SOURCE, {
            "input": _MOCK_SOURCE,
            "useMock": True,
        })
        assert not _has_errors(r)
        assert r["data"]["testWeatherSource"]["success"] is True

    def test_testear_fuente_mock_devuelve_fields(self, admin):
        r = admin.execute_async(TEST_WEATHER_SOURCE, {
            "input": _MOCK_SOURCE,
            "useMock": True,
        })
        assert len(r["data"]["testWeatherSource"]["fields"]) > 0

    def test_testear_fuente_mock_rawjson_no_vacio(self, admin):
        r = admin.execute_async(TEST_WEATHER_SOURCE, {
            "input": _MOCK_SOURCE,
            "useMock": True,
        })
        import json
        parsed = json.loads(r["data"]["testWeatherSource"]["rawJson"])
        assert isinstance(parsed, dict)

    def test_user_no_puede_testear_fuente(self, user):
        r = user.execute_async(TEST_WEATHER_SOURCE, {
            "input": _MOCK_SOURCE,
            "useMock": True,
        })
        assert _has_errors(r) or _data_is_null(r, "testWeatherSource")


# ─────────────────────────────────────────────────────────────────────────────
# Gestión de usuarios — users query, deleteUser, changePassword
# ─────────────────────────────────────────────────────────────────────────────

QUERY_USERS = """
query {
  users {
    _id
    email
    role
    createdAt
  }
}
"""

DELETE_USER = """
mutation($id: String!) {
  deleteUser(id: $id)
}
"""

CHANGE_PASSWORD = """
mutation($input: ChangePasswordInput!) {
  changePassword(input: $input)
}
"""

LOGIN = """
mutation($input: LoginInput!) {
  loginUser(input: $input) {
    user { email role }
    token
  }
}
"""


def _create_user_via_invitation(client: GQLClient, email: str, role: str = "user"):
    """Registra un usuario usando generateInvitationCode + registerUser."""
    # Generar código
    r_code = client.execute("""
        mutation($role: String!, $createdBy: String!) {
          generateInvitationCode(role: $role, createdBy: $createdBy) {
            code
          }
        }
    """, {"role": role, "createdBy": "admin@test.cu"})
    code = r_code["data"]["generateInvitationCode"]["code"]
    # Registrar usuario
    r_reg = client.execute("""
        mutation($input: RegisterInput!) {
          registerUser(input: $input) {
            user { _id email role }
            token
          }
        }
    """, {"input": {"email": email, "password": "Segura123!", "invitationCode": code}})
    return r_reg["data"]["registerUser"]["user"]


class TestGraphQLUsuarios:

    def test_admin_puede_listar_usuarios(self, admin, db):
        _create_user_via_invitation(admin, "listed@test.cu")
        r = admin.execute(QUERY_USERS)
        assert not _has_errors(r)
        assert len(r["data"]["users"]) >= 1

    def test_user_no_puede_listar_usuarios(self, user):
        r = user.execute(QUERY_USERS)
        assert _has_errors(r) or _data_is_null(r, "users")

    def test_anonimo_no_puede_listar_usuarios(self, anon):
        r = anon.execute(QUERY_USERS)
        assert _has_errors(r) or _data_is_null(r, "users")

    def test_admin_puede_eliminar_usuario(self, admin, db):
        target = _create_user_via_invitation(admin, "todelete@test.cu")
        r = admin.execute(DELETE_USER, {"id": target["_id"]})
        assert not _has_errors(r)
        assert r["data"]["deleteUser"] is True

    def test_eliminar_quita_de_lista(self, admin, db):
        target = _create_user_via_invitation(admin, "gone@test.cu")
        admin.execute(DELETE_USER, {"id": target["_id"]})
        r = admin.execute(QUERY_USERS)
        ids = [u["_id"] for u in r["data"]["users"]]
        assert target["_id"] not in ids

    def test_user_no_puede_eliminar_usuario(self, user, db):
        # Necesitamos un admin para crear el target
        from app.services.invitation_service import create_invitation_code
        from app.services.user_service import register_user
        code = create_invitation_code("user", "sys")["code"]
        target = register_user({"email": "target2@test.cu", "password": "Segura123!", "invitationCode": code})
        r = user.execute(DELETE_USER, {"id": target["_id"]})
        assert _has_errors(r) or r["data"].get("deleteUser") is None

    def test_admin_puede_cambiar_su_password(self, db):
        # Crear admin real con contraseña conocida
        from app.services.invitation_service import create_invitation_code
        from app.services.user_service import register_user
        code = create_invitation_code("admin", "sys")["code"]
        register_user({"email": "myadmin@test.cu", "password": "Antigua123!", "invitationCode": code})
        admin_ctx = GQLClient({"current_user": {"sub": "myadmin@test.cu", "role": "admin", "jti": "cp-jti"}, "request": None})
        r = admin_ctx.execute(CHANGE_PASSWORD, {
            "input": {"currentPassword": "Antigua123!", "newPassword": "NuevaSegura456!"}
        })
        assert not _has_errors(r)
        assert r["data"]["changePassword"] is True

    def test_cambio_password_incorrecta_da_error(self, db):
        from app.services.invitation_service import create_invitation_code
        from app.services.user_service import register_user
        code = create_invitation_code("user", "sys")["code"]
        register_user({"email": "badpass@test.cu", "password": "Antigua123!", "invitationCode": code})
        user_ctx = GQLClient({"current_user": {"sub": "badpass@test.cu", "role": "user", "jti": "bp-jti"}, "request": None})
        r = user_ctx.execute(CHANGE_PASSWORD, {
            "input": {"currentPassword": "Incorrecta!", "newPassword": "NuevaSegura456!"}
        })
        assert _has_errors(r)

    def test_estructura_usuario_en_lista(self, admin, db):
        _create_user_via_invitation(admin, "struct@test.cu")
        r = admin.execute(QUERY_USERS)
        user_item = r["data"]["users"][0]
        for field in ["_id", "email", "role"]:
            assert field in user_item


# ─────────────────────────────────────────────────────────────────────────────
# Sesiones activas — activeSessions query + revokeSession mutation
# ─────────────────────────────────────────────────────────────────────────────

QUERY_ACTIVE_SESSIONS = """
query {
  activeSessions {
    _id
    email
    deviceType
    createdAt
    expiresAt
  }
}
"""

REVOKE_SESSION = """
mutation($id: String!) {
  revokeSession(id: $id)
}
"""


class TestGraphQLSesiones:

    def _seed_session(self, email="op@test.cu"):
        from app.services.session_service import create_session
        from datetime import datetime, timedelta
        import uuid
        jti = uuid.uuid4().hex
        expires = datetime.utcnow() + timedelta(days=7)
        create_session(email, "127.0.0.1", "Mozilla/5.0", jti, expires)
        return jti

    def test_admin_puede_ver_sesiones_activas(self, admin, db):
        self._seed_session()
        r = admin.execute(QUERY_ACTIVE_SESSIONS)
        assert not _has_errors(r)
        assert isinstance(r["data"]["activeSessions"], list)

    def test_user_no_puede_ver_sesiones(self, user):
        r = user.execute(QUERY_ACTIVE_SESSIONS)
        assert _has_errors(r) or _data_is_null(r, "activeSessions")

    def test_sesion_creada_aparece_en_lista(self, admin, db):
        self._seed_session(email="sesion@test.cu")
        r = admin.execute(QUERY_ACTIVE_SESSIONS)
        emails = [s["email"] for s in r["data"]["activeSessions"]]
        assert "sesion@test.cu" in emails

    def test_estructura_sesion(self, admin, db):
        self._seed_session()
        r = admin.execute(QUERY_ACTIVE_SESSIONS)
        if r["data"]["activeSessions"]:
            s = r["data"]["activeSessions"][0]
            for field in ["_id", "email", "deviceType"]:
                assert field in s

    def test_admin_puede_revocar_sesion(self, admin, db):
        self._seed_session("rev@test.cu")
        r_list = admin.execute(QUERY_ACTIVE_SESSIONS)
        sessions = [s for s in r_list["data"]["activeSessions"] if s["email"] == "rev@test.cu"]
        assert sessions
        session_id = sessions[0]["_id"]
        r_rev = admin.execute(REVOKE_SESSION, {"id": session_id})
        assert not _has_errors(r_rev)
        assert r_rev["data"]["revokeSession"] is True

    def test_revocar_quita_sesion_de_lista(self, admin, db):
        self._seed_session("toremove@test.cu")
        r_list = admin.execute(QUERY_ACTIVE_SESSIONS)
        sessions = [s for s in r_list["data"]["activeSessions"] if s["email"] == "toremove@test.cu"]
        session_id = sessions[0]["_id"]
        admin.execute(REVOKE_SESSION, {"id": session_id})
        r_after = admin.execute(QUERY_ACTIVE_SESSIONS)
        remaining = [s for s in r_after["data"]["activeSessions"] if s["email"] == "toremove@test.cu"]
        assert remaining == []

    def test_user_no_puede_revocar_sesion(self, user, db):
        self._seed_session()
        r_list = GQLClient({"current_user": {"sub": "x", "role": "admin", "jti": "x"}, "request": None}).execute(QUERY_ACTIVE_SESSIONS)
        if r_list["data"].get("activeSessions"):
            sid = r_list["data"]["activeSessions"][0]["_id"]
            r = user.execute(REVOKE_SESSION, {"id": sid})
            assert _has_errors(r)


# ─────────────────────────────────────────────────────────────────────────────
# Perfil de sombras — saveShadowProfile mutation + shadowProfile query
# ─────────────────────────────────────────────────────────────────────────────

SAVE_SHADOW_PROFILE = """
mutation($slots: [ShadowSlotInput!]!) {
  saveShadowProfile(slots: $slots) {
    avgShadow
    avgProd
    updatedAt
    slots {
      hour
      shadowPct
    }
  }
}
"""

QUERY_SHADOW_PROFILE = """
query {
  shadowProfile {
    avgShadow
    avgProd
    slots {
      hour
      shadowPct
    }
  }
}
"""

_SHADOW_SLOTS = [
    {"hour": h, "shadowPct": float(h * 3 % 100), "prodOverride": None}
    for h in range(6, 20)
]


class TestGraphQLShadowProfile:

    def test_sin_perfil_query_retorna_null(self, admin):
        r = admin.execute(QUERY_SHADOW_PROFILE)
        assert not _has_errors(r)
        assert r["data"]["shadowProfile"] is None

    def test_admin_puede_guardar_perfil(self, admin):
        r = admin.execute(SAVE_SHADOW_PROFILE, {"slots": _SHADOW_SLOTS})
        assert not _has_errors(r)
        assert r["data"]["saveShadowProfile"]["avgShadow"] is not None

    def test_user_no_puede_guardar_perfil(self, user):
        r = user.execute(SAVE_SHADOW_PROFILE, {"slots": _SHADOW_SLOTS})
        assert _has_errors(r) or _data_is_null(r, "saveShadowProfile")

    def test_guardar_y_recuperar_perfil(self, admin):
        admin.execute(SAVE_SHADOW_PROFILE, {"slots": _SHADOW_SLOTS})
        r = admin.execute(QUERY_SHADOW_PROFILE)
        assert not _has_errors(r)
        profile = r["data"]["shadowProfile"]
        assert profile is not None
        assert len(profile["slots"]) == len(_SHADOW_SLOTS)

    def test_perfil_tiene_avg_shadow(self, admin):
        admin.execute(SAVE_SHADOW_PROFILE, {"slots": _SHADOW_SLOTS})
        r = admin.execute(QUERY_SHADOW_PROFILE)
        assert isinstance(r["data"]["shadowProfile"]["avgShadow"], float)

    def test_perfil_tiene_avg_prod(self, admin):
        admin.execute(SAVE_SHADOW_PROFILE, {"slots": _SHADOW_SLOTS})
        r = admin.execute(QUERY_SHADOW_PROFILE)
        assert isinstance(r["data"]["shadowProfile"]["avgProd"], float)

    def test_perfil_slots_tienen_hora_y_sombra(self, admin):
        admin.execute(SAVE_SHADOW_PROFILE, {"slots": _SHADOW_SLOTS})
        r = admin.execute(QUERY_SHADOW_PROFILE)
        slot = r["data"]["shadowProfile"]["slots"][0]
        assert "hour" in slot and "shadowPct" in slot

    def test_guardar_sobreescribe_perfil_anterior(self, admin):
        admin.execute(SAVE_SHADOW_PROFILE, {"slots": _SHADOW_SLOTS})
        new_slots = [{"hour": 12, "shadowPct": 99.0, "prodOverride": None}]
        admin.execute(SAVE_SHADOW_PROFILE, {"slots": new_slots})
        r = admin.execute(QUERY_SHADOW_PROFILE)
        assert len(r["data"]["shadowProfile"]["slots"]) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Medidas de electrodomésticos — upload, clear, appliancesConsumptionForecast
# ─────────────────────────────────────────────────────────────────────────────

UPLOAD_MEASUREMENT = """
mutation($id: String!, $fileContent: String!) {
  uploadApplianceMeasurement(id: $id, fileContent: $fileContent) {
    _id
    name
  }
}
"""

CLEAR_MEASUREMENT = """
mutation($id: String!) {
  clearApplianceMeasurement(id: $id) {
    _id
    name
  }
}
"""

APPLIANCES_FORECAST = """
query($hours: Int, $start: String) {
  appliancesConsumptionForecast(hours: $hours, start: $start) {
    totalConsumptionKw
    appliancesAlwaysOn
    appliancesWithProfile
    points {
      datetime
      consumptionKw
    }
  }
}
"""

_CSV_MEDIDA = """Date,Time,P(SUM)
2024-06-17,08:00:00,1.5
2024-06-17,09:00:00,2.3
2024-06-17,10:00:00,1.8
2024-06-17,11:00:00,2.0
2024-06-18,08:00:00,1.7
"""

_APPLIANCE_INPUT = {
    "name": "Aire Acondicionado Test",
    "averagePowerW": 1500.0,
    "maxPowerW": 2000.0,
    "quantity": 1,
}


class TestGraphQLMedidasElectrodomestico:

    def _create_appliance(self, admin):
        from app.services.appliance_service import create_appliance
        ap = create_appliance(_APPLIANCE_INPUT)
        return ap["_id"]

    def test_admin_puede_subir_medicion(self, admin, db):
        ap_id = self._create_appliance(admin)
        r = admin.execute(UPLOAD_MEASUREMENT, {"id": ap_id, "fileContent": _CSV_MEDIDA})
        assert not _has_errors(r)
        assert r["data"]["uploadApplianceMeasurement"]["_id"] == ap_id

    def test_user_no_puede_subir_medicion(self, user, db):
        from app.services.appliance_service import create_appliance
        ap = create_appliance(_APPLIANCE_INPUT)
        r = user.execute(UPLOAD_MEASUREMENT, {"id": ap["_id"], "fileContent": _CSV_MEDIDA})
        assert _has_errors(r) or _data_is_null(r, "uploadApplianceMeasurement")

    def test_admin_puede_limpiar_medicion(self, admin, db):
        ap_id = self._create_appliance(admin)
        admin.execute(UPLOAD_MEASUREMENT, {"id": ap_id, "fileContent": _CSV_MEDIDA})
        r = admin.execute(CLEAR_MEASUREMENT, {"id": ap_id})
        assert not _has_errors(r)
        assert r["data"]["clearApplianceMeasurement"]["_id"] == ap_id

    def test_forecast_sin_electrodomesticos_devuelve_0(self, admin, db):
        r = admin.execute(APPLIANCES_FORECAST, {"hours": 4})
        assert not _has_errors(r)
        assert r["data"]["appliancesConsumptionForecast"]["appliancesAlwaysOn"] == 0

    def test_forecast_con_electrodomestico_devuelve_puntos(self, admin, db):
        self._create_appliance(admin)
        r = admin.execute(APPLIANCES_FORECAST, {"hours": 6})
        assert not _has_errors(r)
        assert len(r["data"]["appliancesConsumptionForecast"]["points"]) == 6

    def test_forecast_puntos_tienen_datetime_y_kw(self, admin, db):
        self._create_appliance(admin)
        r = admin.execute(APPLIANCES_FORECAST, {"hours": 3})
        point = r["data"]["appliancesConsumptionForecast"]["points"][0]
        assert "datetime" in point
        assert "consumptionKw" in point

    def test_forecast_consumo_total_no_negativo(self, admin, db):
        self._create_appliance(admin)
        r = admin.execute(APPLIANCES_FORECAST, {"hours": 8})
        assert r["data"]["appliancesConsumptionForecast"]["totalConsumptionKw"] >= 0

    def test_forecast_con_perfil_subido(self, admin, db):
        ap_id = self._create_appliance(admin)
        admin.execute(UPLOAD_MEASUREMENT, {"id": ap_id, "fileContent": _CSV_MEDIDA})
        r = admin.execute(APPLIANCES_FORECAST, {"hours": 4})
        assert not _has_errors(r)
        assert r["data"]["appliancesConsumptionForecast"]["appliancesWithProfile"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Historial — historicalReadings + dailySummaries + seedHistoricalData
# ─────────────────────────────────────────────────────────────────────────────

# El historial almacena solo producción (productionKw); los resúmenes solo
# totales de producción. El seed inserta lecturas cada 5 min (24 × 12 por día).
INTERVALS_PER_DAY = 24 * 12

QUERY_HISTORICAL = """
query($limit: Int) {
  historicalReadings(limit: $limit) {
    _id
    timestamp
    productionKw
  }
}
"""

QUERY_DAILY_SUMMARIES = """
query($days: Int) {
  dailySummaries(days: $days) {
    date
    totalProductionKwh
    maxProductionKw
    readingCount
  }
}
"""

SEED_HISTORICAL = """
mutation($days: Int) {
  seedHistoricalData(days: $days)
}
"""


class TestGraphQLHistorial:

    def test_historico_vacio_retorna_lista_vacia(self, admin):
        r = admin.execute(QUERY_HISTORICAL, {})
        assert not _has_errors(r)
        assert r["data"]["historicalReadings"] == []

    def test_seed_inserta_datos(self, admin):
        r = admin.execute(SEED_HISTORICAL, {"days": 2})
        assert not _has_errors(r)
        assert r["data"]["seedHistoricalData"] == 2 * INTERVALS_PER_DAY

    def test_historico_con_datos_retorna_lecturas(self, admin):
        admin.execute(SEED_HISTORICAL, {"days": 2})
        r = admin.execute(QUERY_HISTORICAL, {"limit": 10})
        assert not _has_errors(r)
        assert len(r["data"]["historicalReadings"]) == 10

    def test_historico_estructura_lectura(self, admin):
        admin.execute(SEED_HISTORICAL, {"days": 1})
        r = admin.execute(QUERY_HISTORICAL, {"limit": 1})
        reading = r["data"]["historicalReadings"][0]
        for field in ["_id", "timestamp", "productionKw"]:
            assert field in reading

    def test_historico_produccion_no_negativa(self, admin):
        admin.execute(SEED_HISTORICAL, {"days": 2})
        r = admin.execute(QUERY_HISTORICAL, {"limit": 48})
        for reading in r["data"]["historicalReadings"]:
            assert reading["productionKw"] >= 0

    def test_resumenes_vacio_retorna_lista_vacia(self, admin):
        r = admin.execute(QUERY_DAILY_SUMMARIES, {"days": 7})
        assert not _has_errors(r)
        assert r["data"]["dailySummaries"] == []

    def test_resumenes_con_datos_retorna_entradas(self, admin):
        admin.execute(SEED_HISTORICAL, {"days": 3})
        r = admin.execute(QUERY_DAILY_SUMMARIES, {"days": 5})
        assert not _has_errors(r)
        assert len(r["data"]["dailySummaries"]) > 0

    def test_resumenes_estructura(self, admin):
        admin.execute(SEED_HISTORICAL, {"days": 2})
        r = admin.execute(QUERY_DAILY_SUMMARIES, {"days": 5})
        summary = r["data"]["dailySummaries"][0]
        for field in ["date", "totalProductionKwh", "maxProductionKw", "readingCount"]:
            assert field in summary

    def test_resumenes_dias_parametro(self, admin):
        admin.execute(SEED_HISTORICAL, {"days": 7})
        s7 = admin.execute(QUERY_DAILY_SUMMARIES, {"days": 7})
        s1 = admin.execute(QUERY_DAILY_SUMMARIES, {"days": 1})
        assert len(s7["data"]["dailySummaries"]) >= len(s1["data"]["dailySummaries"])

    def test_user_puede_ver_historial(self, user):
        r = user.execute(QUERY_HISTORICAL, {})
        assert not _has_errors(r)

    def test_user_no_puede_hacer_seed(self, user):
        r = user.execute(SEED_HISTORICAL, {"days": 1})
        assert _has_errors(r) or r["data"].get("seedHistoricalData") is None
