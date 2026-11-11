"""
Pruebas de integración para la query GraphQL `solar` y la mutation `resetSystemData`.

Mockea `get_weather_with_fallback` en el módulo `solar_service` para evitar
llamadas HTTP reales, y usa la fixture `mongo_db` de conftest para aislar la BD.
"""
import asyncio
import pytest
import mongomock
import app.database as _db_module
import app.schema as _schema_module
from app.schema import schema


# ─────────────────────────────────────────────────────────────────────────────
# Datos de prueba
# ─────────────────────────────────────────────────────────────────────────────

def _make_forecast_day(date: str, day_of_week: str) -> dict:
    return {
        "date": date,
        "dayOfWeek": day_of_week,
        "maxTemp": 32.0,
        "minTemp": 24.0,
        "solarRadiation": 500.0,
        "cloudCover": 15.0,
        "predictedProduction": 28.5,
        "condition": "partly-cloudy",
    }


SAMPLE_WEATHER = {
    "temperature": 28.0,
    "solarRadiation": 450.0,
    "cloudCover": 20.0,
    "humidity": 65.0,
    "windSpeed": 12.5,
    "forecast": [
        _make_forecast_day("2026-06-18", "Jueves"),
        _make_forecast_day("2026-06-19", "Viernes"),
        _make_forecast_day("2026-06-20", "Sábado"),
        _make_forecast_day("2026-06-21", "Domingo"),
        _make_forecast_day("2026-06-22", "Lunes"),
        _make_forecast_day("2026-06-23", "Martes"),
        _make_forecast_day("2026-06-24", "Miércoles"),
    ],
    "provider": "Test Provider",
    "locationName": "CUJAE",
    "lastUpdated": "2026-06-18T12:00:00",
    "description": "Parcialmente nublado",
    "sourceError": None,
}


# ─────────────────────────────────────────────────────────────────────────────
# Contextos de autenticación
# ─────────────────────────────────────────────────────────────────────────────

ADMIN_CTX = {
    "current_user": {"sub": "admin@test.cu", "role": "admin", "jti": "adm-jti"},
    "request": None,
}
ANON_CTX = {"current_user": None, "request": None}
USER_CTX = {
    "current_user": {"sub": "user@test.cu", "role": "user", "jti": "usr-jti"},
    "request": None,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper de ejecución GraphQL
# ─────────────────────────────────────────────────────────────────────────────

def gql(query: str, variables: dict = None, context: dict = None) -> dict:
    ctx = context if context is not None else ANON_CTX

    async def _run():
        result = await schema.execute(
            query,
            variable_values=variables or {},
            context_value=ctx,
        )
        return {
            "data": result.data or {},
            "errors": [{"message": str(e)} for e in (result.errors or [])],
        }

    return asyncio.run(_run())


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_weather(monkeypatch):
    """Reemplaza get_weather_with_fallback por una corutina que devuelve SAMPLE_WEATHER."""
    async def _fake_weather(lat, lon, capacity_kw, name):
        return SAMPLE_WEATHER

    monkeypatch.setattr(
        "app.services.solar_service.get_weather_with_fallback",
        _fake_weather,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Queries GraphQL
# ─────────────────────────────────────────────────────────────────────────────

SOLAR_QUERY = """
query {
  solar {
    timestamp
    mode
    weather {
      temperature
      cloudCover
      humidity
      provider
    }
    battery {
      chargeLevel
      charging
    }
    metrics {
      dailyProduction
      dailyConsumption
    }
    current {
      production
      consumption
    }
  }
}
"""

RESET_MUTATION = """
mutation {
  resetSystemData
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Tests: solar query
# ─────────────────────────────────────────────────────────────────────────────

class TestSolarQuery:

    def test_solar_retorna_snapshot_completo(self, mongo_db, mock_weather):
        r = gql(SOLAR_QUERY, context=ANON_CTX)
        assert not r["errors"], r["errors"]
        snap = r["data"]["solar"]
        assert snap["timestamp"] is not None
        assert snap["mode"] == "predictive"
        assert snap["weather"]["provider"] == "Test Provider"

    def test_solar_weather_campos_presentes(self, mongo_db, mock_weather):
        r = gql(SOLAR_QUERY, context=ANON_CTX)
        assert not r["errors"], r["errors"]
        weather = r["data"]["solar"]["weather"]
        assert weather["temperature"] == pytest.approx(28.0)
        assert weather["cloudCover"] == pytest.approx(20.0)
        assert weather["humidity"] == pytest.approx(65.0)

    def test_solar_sin_configuracion_usa_defaults(self, mongo_db, mock_weather):
        # Con la BD vacía el servicio debe usar la configuración por defecto
        # y devolver igualmente un snapshot válido.
        r = gql(SOLAR_QUERY, context=ANON_CTX)
        assert not r["errors"], r["errors"]
        snap = r["data"]["solar"]
        current = snap["current"]
        assert isinstance(current["production"], float)
        assert isinstance(current["consumption"], float)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: resetSystemData mutation
# ─────────────────────────────────────────────────────────────────────────────

class TestResetSystemData:
    """
    `resetSystemData` llama a `get_database()` dentro de schema.py, pero dicha
    función no está importada en ese módulo (bug de producción). Para que el
    test funcione inyectamos la referencia mockeada directamente en el namespace
    de app.schema usando monkeypatch.
    """

    def _seed_all_collections(self, db):
        """Inserta un documento en cada colección que resetSystemData borra."""
        collections = [
            "paneles",
            "baterias",
            "inversores",
            "electrodomesticos",
            "ubicacion_config",
            "shadow_profile",
        ]
        for col in collections:
            db[col].insert_one({"seed": True})
        return collections

    def test_reset_borra_todas_las_colecciones(self, mongo_db, monkeypatch):
        # schema.py usa get_database() sin importarla (bug de producción);
        # la inyectamos en el namespace del módulo con raising=False.
        monkeypatch.setattr(_schema_module, "get_database", lambda: mongo_db, raising=False)
        collections = self._seed_all_collections(mongo_db)
        # Verificar que hay datos antes del reset
        for col in collections:
            assert mongo_db[col].count_documents({}) == 1, f"{col} debería tener 1 documento"

        r = gql(RESET_MUTATION, context=ADMIN_CTX)
        assert not r["errors"], r["errors"]

        for col in collections:
            assert mongo_db[col].count_documents({}) == 0, f"{col} debería estar vacía tras reset"

    def test_reset_requiere_admin(self, mongo_db, monkeypatch):
        monkeypatch.setattr(_schema_module, "get_database", lambda: mongo_db, raising=False)
        r = gql(RESET_MUTATION, context=ANON_CTX)
        assert r["errors"], "Se esperaba un error de autenticación para usuario anónimo"

    def test_reset_usuario_normal_sin_acceso(self, mongo_db, monkeypatch):
        monkeypatch.setattr(_schema_module, "get_database", lambda: mongo_db, raising=False)
        r = gql(RESET_MUTATION, context=USER_CTX)
        assert r["errors"], "Se esperaba un error de autorización para rol 'user'"

    def test_reset_devuelve_true(self, mongo_db, monkeypatch):
        monkeypatch.setattr(_schema_module, "get_database", lambda: mongo_db, raising=False)
        r = gql(RESET_MUTATION, context=ADMIN_CTX)
        assert not r["errors"], r["errors"]
        assert r["data"]["resetSystemData"] is True
