"""
Pruebas de integración para los resolvers GraphQL de predicción solar ML.

Estrategia de mock
------------------
Los servicios ML hacen llamadas de red (Open-Meteo) y cargan archivos joblib,
por lo que se reemplazan con funciones falsas usando ``monkeypatch``:

- ``app.schema.predict_solar_production``   → mock async para ``mlPredict``
- ``app.schema.predict_next_hours``         → mock async para ``mlPredictNextHours``
- ``app.schema.predict_for_date_range``     → mock async para ``mlPredictDateRange``
- ``app.services.ml_prediction_service.predict_for_specific_hours``
                                             → mock async para ``mlPredictForHours``
  (usa import local dentro del resolver, por eso se parchea en el módulo fuente)
- ``app.services.ml_model_service.ml_model_service.get_model_info``
                                             → lambda síncrona para ``mlModelInfo``

La base de datos se sustituye con mongomock a través de la fixture ``mongo_db``
definida en conftest.py. Con la BD vacía, ``get_system_config()`` devuelve los
defaults del sistema (capacityKw del DEFAULT_SYSTEM_CONFIG), y
``_scale_ml_predictions`` escala las predicciones cuando
``get_reference_capacity_kw()`` devuelve un valor distinto de None.
Para aislar los tests del escalado se hace que el servicio de modelo indique
``reference_capacity_kw = None`` salvo en los casos en que se prueba el campo
de producción con escala controlada.

Todas las ejecuciones GraphQL usan ``asyncio.run()`` directamente sobre
``schema.execute()`` para reproducir exactamente cómo trabaja el código de
producción con resolvers async.
"""
import asyncio
from typing import List, Dict, Any

import pytest

import app.database as _db_module
import mongomock
from app.schema import schema


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gql(query: str, variables: dict = None) -> dict:
    """Ejecuta una query GraphQL async y devuelve {data, errors}."""
    async def _run():
        result = await schema.execute(
            query,
            variable_values=variables or {},
            context_value={},
        )
        return {
            "data": result.data or {},
            "errors": [{"message": str(e)} for e in (result.errors or [])],
        }
    return asyncio.run(_run())


def _make_prediction(dt: str, production_kw: float = 5.0) -> Dict[str, Any]:
    """Construye un dict de predicción con la firma esperada por el resolver."""
    return {
        "datetime": dt,
        "production_kw": production_kw,
        "weather": {
            "temperature_2m": 25.0,
            "relative_humidity_2m": 70.0,
            "wind_speed_10m": 5.0,
            "cloud_cover": 10.0,
            "shortwave_radiation": 300.0,
        },
        "weather_source": "Open-Meteo",
        "weather_source_warning": None,
    }


def _make_predictions(datetimes: List[str], production_kw: float = 5.0) -> List[Dict[str, Any]]:
    return [_make_prediction(dt, production_kw) for dt in datetimes]


def _async_returning(value):
    """Devuelve una corrutina que siempre resuelve a ``value``."""
    async def _mock(*args, **kwargs):
        return value
    return _mock


def _async_raising(exc):
    """Devuelve una corrutina que siempre lanza ``exc``."""
    async def _mock(*args, **kwargs):
        raise exc
    return _mock


# ─────────────────────────────────────────────────────────────────────────────
# Fixture compartida: BD en memoria + modelo sin referencia de capacidad
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def mongo_db(monkeypatch):
    client = mongomock.MongoClient()
    test_db = client["gemelo_test"]
    monkeypatch.setattr(_db_module, "get_database", lambda: test_db)
    monkeypatch.setattr(_db_module, "_db", test_db)
    yield test_db
    client.close()


@pytest.fixture()
def no_scaling(monkeypatch):
    """
    Hace que ml_model_service.get_reference_capacity_kw() devuelva None
    para que _scale_ml_predictions pase las predicciones sin modificar.
    Esto simplifica las aserciones sobre production_kw en la mayoría de tests.
    """
    from app.services.ml_model_service import ml_model_service
    monkeypatch.setattr(ml_model_service, "get_reference_capacity_kw", lambda: None)


# ─────────────────────────────────────────────────────────────────────────────
# Queries GraphQL reutilizables
# ─────────────────────────────────────────────────────────────────────────────

_Q_MODEL_INFO = """
query {
  mlModelInfo {
    loaded
    modelName
    testRmse
    testR2
    testMae
    features
    trainingDate
    requiresScaling
    referenceCapacityKw
    message
  }
}
"""

_Q_ML_PREDICT = """
query($datetimes: [String!]!, $lat: Float, $lon: Float) {
  mlPredict(datetimes: $datetimes, lat: $lat, lon: $lon) {
    datetime
    productionKw
    weather {
      temperature2m
      relativeHumidity2m
      windSpeed10m
      cloudCover
      shortwaveRadiation
    }
    weatherSource
  }
}
"""

_Q_ML_PREDICT_NEXT_HOURS = """
query($hours: Int, $lat: Float, $lon: Float) {
  mlPredictNextHours(hours: $hours, lat: $lat, lon: $lon) {
    datetime
    productionKw
    weather {
      temperature2m
    }
  }
}
"""

_Q_ML_PREDICT_DATE_RANGE = """
query($startDate: String!, $endDate: String!, $lat: Float, $lon: Float) {
  mlPredictDateRange(startDate: $startDate, endDate: $endDate, lat: $lat, lon: $lon) {
    datetime
    productionKw
    weather {
      temperature2m
    }
  }
}
"""

_Q_ML_PREDICT_FOR_HOURS = """
query($date: String!, $hours: [Int!]!, $lat: Float, $lon: Float) {
  mlPredictForHours(date: $date, hours: $hours, lat: $lat, lon: $lon) {
    datetime
    productionKw
    weather {
      temperature2m
    }
  }
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# TestMlModelInfo
# ─────────────────────────────────────────────────────────────────────────────

class TestMlModelInfo:

    def test_modelo_no_cargado(self, mongo_db, monkeypatch):
        monkeypatch.setattr(
            "app.services.ml_model_service.ml_model_service.get_model_info",
            lambda: {"loaded": False, "message": "Model not loaded"},
        )
        r = _gql(_Q_MODEL_INFO)
        assert not r["errors"]
        info = r["data"]["mlModelInfo"]
        assert info["loaded"] is False
        assert info["message"] == "Model not loaded"

    def test_modelo_cargado(self, mongo_db, monkeypatch):
        monkeypatch.setattr(
            "app.services.ml_model_service.ml_model_service.get_model_info",
            lambda: {
                "loaded": True,
                "model_name": "random_forest",
                "test_rmse": 0.42,
                "test_r2": 0.95,
                "test_mae": 0.31,
                "features": ["temperature_2m", "cloud_cover", "shortwave_radiation"],
                "training_date": "2025-01-15",
                "requires_scaling": False,
                "reference_capacity_kw": 50.0,
            },
        )
        r = _gql(_Q_MODEL_INFO)
        assert not r["errors"]
        info = r["data"]["mlModelInfo"]
        assert info["loaded"] is True
        assert info["modelName"] == "random_forest"
        assert abs(info["testRmse"] - 0.42) < 1e-6
        assert abs(info["testR2"] - 0.95) < 1e-6
        assert abs(info["testMae"] - 0.31) < 1e-6
        assert "temperature_2m" in info["features"]
        assert info["trainingDate"] == "2025-01-15"
        assert info["requiresScaling"] is False
        assert abs(info["referenceCapacityKw"] - 50.0) < 1e-6
        assert info["message"] is None


# ─────────────────────────────────────────────────────────────────────────────
# TestMlPredict
# ─────────────────────────────────────────────────────────────────────────────

class TestMlPredict:

    def test_retorna_predicciones_para_datetimes_dados(self, mongo_db, no_scaling, monkeypatch):
        dts = ["2026-06-18T12:00:00", "2026-06-18T13:00:00"]
        monkeypatch.setattr(
            "app.schema.predict_solar_production",
            _async_returning(_make_predictions(dts)),
        )
        r = _gql(_Q_ML_PREDICT, {
            "datetimes": dts,
            "lat": 23.11,
            "lon": -82.37,
        })
        assert not r["errors"]
        preds = r["data"]["mlPredict"]
        assert len(preds) == 2

    def test_campos_de_prediccion_correctos(self, mongo_db, no_scaling, monkeypatch):
        dt = "2026-06-18T14:00:00"
        monkeypatch.setattr(
            "app.schema.predict_solar_production",
            _async_returning(_make_predictions([dt], production_kw=7.5)),
        )
        r = _gql(_Q_ML_PREDICT, {
            "datetimes": [dt],
            "lat": 23.11,
            "lon": -82.37,
        })
        assert not r["errors"]
        pred = r["data"]["mlPredict"][0]
        assert pred["datetime"] == dt
        assert abs(pred["productionKw"] - 7.5) < 1e-6
        assert pred["weatherSource"] == "Open-Meteo"
        weather = pred["weather"]
        assert abs(weather["temperature2m"] - 25.0) < 1e-6
        assert abs(weather["relativeHumidity2m"] - 70.0) < 1e-6
        assert abs(weather["windSpeed10m"] - 5.0) < 1e-6
        assert abs(weather["cloudCover"] - 10.0) < 1e-6
        assert abs(weather["shortwaveRadiation"] - 300.0) < 1e-6

    def test_lista_vacia_sin_datetimes(self, mongo_db, no_scaling, monkeypatch):
        monkeypatch.setattr(
            "app.schema.predict_solar_production",
            _async_returning([]),
        )
        r = _gql(_Q_ML_PREDICT, {
            "datetimes": [],
            "lat": 23.11,
            "lon": -82.37,
        })
        assert not r["errors"]
        assert r["data"]["mlPredict"] == []

    def test_modelo_no_cargado_devuelve_error(self, mongo_db, no_scaling, monkeypatch):
        monkeypatch.setattr(
            "app.schema.predict_solar_production",
            _async_raising(RuntimeError("Model not loaded")),
        )
        r = _gql(_Q_ML_PREDICT, {
            "datetimes": ["2026-06-18T12:00:00"],
            "lat": 23.11,
            "lon": -82.37,
        })
        assert r["errors"]


# ─────────────────────────────────────────────────────────────────────────────
# TestMlPredictNextHours
# ─────────────────────────────────────────────────────────────────────────────

class TestMlPredictNextHours:

    def test_retorna_n_predicciones(self, mongo_db, no_scaling, monkeypatch):
        dts = [f"2026-06-18T{h:02d}:00:00" for h in range(3)]
        monkeypatch.setattr(
            "app.schema.predict_next_hours",
            _async_returning(_make_predictions(dts)),
        )
        r = _gql(_Q_ML_PREDICT_NEXT_HOURS, {
            "hours": 3,
            "lat": 23.11,
            "lon": -82.37,
        })
        assert not r["errors"]
        assert len(r["data"]["mlPredictNextHours"]) == 3

    def test_valor_por_defecto_24_horas(self, mongo_db, no_scaling, monkeypatch):
        dts = [f"2026-06-18T{h:02d}:00:00" for h in range(24)]
        monkeypatch.setattr(
            "app.schema.predict_next_hours",
            _async_returning(_make_predictions(dts)),
        )
        r = _gql(_Q_ML_PREDICT_NEXT_HOURS, {
            "lat": 23.11,
            "lon": -82.37,
        })
        assert not r["errors"]
        assert len(r["data"]["mlPredictNextHours"]) == 24


# ─────────────────────────────────────────────────────────────────────────────
# TestMlPredictDateRange
# ─────────────────────────────────────────────────────────────────────────────

class TestMlPredictDateRange:

    def test_retorna_predicciones_para_rango(self, mongo_db, no_scaling, monkeypatch):
        dts = [f"2026-06-18T{h:02d}:00:00" for h in range(6)]
        monkeypatch.setattr(
            "app.schema.predict_for_date_range",
            _async_returning(_make_predictions(dts)),
        )
        r = _gql(_Q_ML_PREDICT_DATE_RANGE, {
            "startDate": "2026-06-18",
            "endDate": "2026-06-18",
            "lat": 23.11,
            "lon": -82.37,
        })
        assert not r["errors"]
        preds = r["data"]["mlPredictDateRange"]
        assert len(preds) == 6
        assert preds[0]["datetime"] == "2026-06-18T00:00:00"


# ─────────────────────────────────────────────────────────────────────────────
# TestMlPredictForHours
# ─────────────────────────────────────────────────────────────────────────────

class TestMlPredictForHours:

    def test_retorna_predicciones_para_horas_especificas(self, mongo_db, no_scaling, monkeypatch):
        hours = [8, 12, 16]
        dts = [f"2026-06-18T{h:02d}:00:00" for h in hours]
        monkeypatch.setattr(
            "app.services.ml_prediction_service.predict_for_specific_hours",
            _async_returning(_make_predictions(dts)),
        )
        r = _gql(_Q_ML_PREDICT_FOR_HOURS, {
            "date": "2026-06-18",
            "hours": hours,
            "lat": 23.11,
            "lon": -82.37,
        })
        assert not r["errors"]
        preds = r["data"]["mlPredictForHours"]
        assert len(preds) == 3
        returned_datetimes = [p["datetime"] for p in preds]
        assert "2026-06-18T08:00:00" in returned_datetimes
        assert "2026-06-18T12:00:00" in returned_datetimes
        assert "2026-06-18T16:00:00" in returned_datetimes
