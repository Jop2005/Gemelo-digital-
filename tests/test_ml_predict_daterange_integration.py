"""
Prueba de INTEGRACIÓN REAL de la query GraphQL ``mlPredictDateRange``.

A diferencia de ``test_ml_production.py`` (que mockea el modelo y el clima),
esta suite ejercita el pipeline COMPLETO de verdad:

    mlPredictDateRange  →  ProductionForecastService  →  modelo Random Forest
    (havana_v1, 33 MB)  +  Open-Meteo en vivo (vía weather_cache)  +  sombra/orientación

Objetivo
--------
Validar AHORA — "por si acaso", antes de migrar la pantalla de Predicción a esta
query — que el backend ya devuelve, vía GraphQL, una predicción de producción
solar utilizable por el frontend SIN cambiar la pantalla:

  - El modelo ML real carga desde disco.
  - La query responde sin errores para un rango de días.
  - Las filas horarias (``productionKw``) se pueden agregar a kWh/día, que es
    exactamente la forma que pinta ``EstadisticasPanel`` (un punto por día).
  - La producción es físicamente sensata: nunca negativa, ~0 de madrugada y
    mayor al mediodía.
  - ``weatherSource`` es "Open-Meteo" (el modelo está entrenado contra esa fuente).

Robustez / CI
-------------
Es un test de integración con dependencias externas (archivo del modelo +
red hacia Open-Meteo). Para no romper en entornos sin modelo o sin internet:

  - Si ``models/solar_production_havana_v1.pkl`` no existe → ``skip``.
  - Si Open-Meteo no es alcanzable (la query falla por red) → ``skip``.

Se ejecuta igual que el resto de la suite::

    ./venv/bin/python -m pytest tests/test_ml_predict_daterange_integration.py -v

Para excluirlo en CI offline basta con deseleccionar la marca ``integration``::

    ./venv/bin/python -m pytest -m "not integration"
"""
import asyncio
import os
import datetime as dt
from typing import Dict, List, Any

import pytest

from app.schema import schema
from app.services.ml_model_service import ml_model_service


pytestmark = pytest.mark.integration


_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "solar_production_havana_v1.pkl",
)

# La Habana (mismas coordenadas que el sistema bajo estudio).
_LAT, _LON = 23.1136, -82.3666


_Q_DATE_RANGE = """
query($startDate: String!, $endDate: String!, $lat: Float, $lon: Float) {
  mlPredictDateRange(startDate: $startDate, endDate: $endDate, lat: $lat, lon: $lon) {
    datetime
    productionKw
    weather {
      shortwaveRadiation
      cloudCover
      temperature2m
    }
    weatherSource
  }
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gql(query: str, variables: dict | None = None) -> dict:
    """Ejecuta una query async contra el schema real y normaliza la respuesta."""
    async def _run():
        result = await schema.execute(
            query,
            variable_values=variables or {},
            context_value={},  # las queries ml* son públicas, no requieren auth
        )
        return {
            "data": result.data or {},
            "errors": [str(e) for e in (result.errors or [])],
        }

    return asyncio.run(_run())


def _looks_like_network_error(errors: List[str]) -> bool:
    needles = ("open-meteo", "timeout", "connection", "network", "getaddrinfo",
               "temporarily", "resolve", "ssl", "httpx", "connecterror")
    blob = " ".join(errors).lower()
    return any(n in blob for n in needles)


def _to_daily_kwh(preds: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Agrega las predicciones horarias (kW, paso de 1 h) a kWh por día.

    Con paso horario, kW·1h == kWh, así que basta sumar ``productionKw`` por
    fecha. Esta es la MISMA transformación que necesitaría el frontend para
    alimentar el gráfico de barras de la vista de Predicción sin cambiar nada
    del layout (un punto = un día).
    """
    daily: Dict[str, float] = {}
    for p in preds:
        day = p["datetime"][:10]
        daily[day] = daily.get(day, 0.0) + p["productionKw"]
    return daily


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def real_model():
    """Carga el modelo real una sola vez para todo el módulo (o salta)."""
    if not os.path.exists(_MODEL_PATH):
        pytest.skip(f"Modelo no encontrado en {_MODEL_PATH}; se omite la integración.")
    ml_model_service.load_model(model_name="havana_v1")
    info = ml_model_service.get_model_info()
    if not info.get("loaded"):
        pytest.skip("El modelo ML real no quedó cargado; se omite la integración.")
    return info


@pytest.fixture(scope="module")
def date_range():
    """Rango [hoy, hoy+2] — siempre dentro de la ventana de pronóstico de Open-Meteo."""
    today = dt.date.today()
    return today.isoformat(), (today + dt.timedelta(days=2)).isoformat()


@pytest.fixture(scope="module")
def live_predictions(real_model, date_range):
    """
    Ejecuta la query real UNA vez y reparte el resultado a los tests.
    Salta toda la suite si Open-Meteo no es alcanzable.
    """
    start, end = date_range
    r = _gql(_Q_DATE_RANGE, {"startDate": start, "endDate": end, "lat": _LAT, "lon": _LON})
    if r["errors"]:
        if _looks_like_network_error(r["errors"]):
            pytest.skip(f"Open-Meteo no alcanzable; se omite la integración: {r['errors']}")
        pytest.fail(f"mlPredictDateRange devolvió errores inesperados: {r['errors']}")
    preds = r["data"]["mlPredictDateRange"]
    if not preds:
        pytest.skip("Open-Meteo no devolvió datos para el rango; se omite la integración.")
    return preds


@pytest.fixture(scope="module")
def historical_predictions(real_model):
    """
    Predicción para un rango de hace ~1 año: fuerza el camino del Archive API
    (ERA5), que es el que alimenta el modo 'Histórico' de la pantalla.
    """
    end = dt.date.today() - dt.timedelta(days=365)
    start = end - dt.timedelta(days=2)
    r = _gql(
        _Q_DATE_RANGE,
        {"startDate": start.isoformat(), "endDate": end.isoformat(), "lat": _LAT, "lon": _LON},
    )
    if r["errors"]:
        if _looks_like_network_error(r["errors"]):
            pytest.skip(f"Open-Meteo Archive no alcanzable; se omite: {r['errors']}")
        pytest.fail(f"mlPredictDateRange (histórico) devolvió errores: {r['errors']}")
    preds = r["data"]["mlPredictDateRange"]
    if not preds:
        pytest.skip("El archivo no devolvió datos para el rango histórico; se omite.")
    return preds


# ─────────────────────────────────────────────────────────────────────────────
# TestMlPredictDateRangeIntegration
# ─────────────────────────────────────────────────────────────────────────────

class TestMlPredictDateRangeIntegration:
    """El pipeline real (modelo + Open-Meteo) responde y es utilizable por la pantalla."""

    def test_modelo_real_carga_con_metricas(self, real_model):
        assert real_model["loaded"] is True
        assert real_model["model_name"]
        # El modelo desplegado documenta R² ≈ 0.79 en test.
        assert real_model["test_r2"] is not None
        assert 0.0 < real_model["test_r2"] <= 1.0
        # 14 features según el contrato de solar_features.build_features.
        assert len(real_model["features"]) == 14

    def test_devuelve_filas_horarias(self, live_predictions):
        # ~24 filas por día en un rango de 2-3 días.
        assert len(live_predictions) >= 24
        row = live_predictions[0]
        assert "datetime" in row
        assert isinstance(row["productionKw"], (int, float))
        assert row["weather"]["shortwaveRadiation"] is not None

    def test_fuente_es_open_meteo(self, live_predictions):
        # El modelo está entrenado contra Open-Meteo; la fuente debe coincidir.
        assert all(p["weatherSource"] == "Open-Meteo" for p in live_predictions)

    def test_produccion_nunca_negativa(self, live_predictions):
        assert all(p["productionKw"] >= 0 for p in live_predictions), \
            "El modelo nunca debe predecir producción negativa (clip a 0)."

    def test_curva_diurna_coherente(self, live_predictions):
        """De madrugada ~0; al mediodía claramente mayor. Sanity físico."""
        def hour(p: Dict[str, Any]) -> int:
            return int(p["datetime"][11:13])

        night = [p["productionKw"] for p in live_predictions if hour(p) in (0, 1, 2, 3, 4, 23)]
        midday = [p["productionKw"] for p in live_predictions if hour(p) in (11, 12, 13, 14)]

        assert night, "Faltan horas nocturnas en la muestra."
        assert midday, "Faltan horas de mediodía en la muestra."
        assert max(night) < 0.5, "La producción nocturna debería ser ~0."
        assert max(midday) > max(night), "El mediodía debería superar a la noche."

    def test_agregacion_diaria_es_la_forma_de_la_pantalla(self, live_predictions):
        """
        Comprueba que las filas horarias se agregan a kWh/día — la forma exacta
        que ``EstadisticasPanel`` pinta en la vista de Predicción. Esto demuestra
        que se puede alimentar la pantalla desde esta query SIN tocar el layout.
        """
        daily = _to_daily_kwh(live_predictions)
        assert len(daily) >= 1
        # Todos los días no negativos y al menos un día con producción real.
        assert all(v >= 0 for v in daily.values())
        assert any(v > 1.0 for v in daily.values()), \
            "Se esperaba al menos un día con producción significativa (>1 kWh)."

        # La estructura que consumiría el gráfico: [{label, Producción}, ...]
        chart = [
            {"label": day, "Producción": round(kwh, 1)}
            for day, kwh in sorted(daily.items())
        ]
        assert all("label" in pt and "Producción" in pt for pt in chart)


# ─────────────────────────────────────────────────────────────────────────────
# TestMlPredictDateRangeHistorical — modo 'Histórico' vía Archive API (ERA5)
# ─────────────────────────────────────────────────────────────────────────────

class TestMlPredictDateRangeHistorical:
    """
    Para fechas antiguas el backend enruta al Archive API de Open-Meteo. Estos
    tests verifican que ese camino devuelve datos REALES (no el mock determinista)
    y utilizables para reconstruir el histórico con el modelo.
    """

    def test_devuelve_datos_para_fecha_antigua(self, historical_predictions):
        assert len(historical_predictions) >= 24
        assert all(p["productionKw"] >= 0 for p in historical_predictions)

    def test_no_es_el_mock_determinista(self, historical_predictions):
        """
        El mock interno produce siempre nubosidad constante (20%) y pico de
        radiación exacto de 900 W/m². Datos reales del archivo varían: si vemos
        variedad de nubosidad y un pico distinto de 900, vienen de Open-Meteo.
        """
        clouds = {round(p["weather"]["shortwaveRadiation"]) for p in historical_predictions}
        cloud_cover = {round(p["weather"]["cloudCover"]) for p in historical_predictions}
        rad_max = max(p["weather"]["shortwaveRadiation"] for p in historical_predictions)
        assert len(cloud_cover) > 2, "Nubosidad constante ⇒ probablemente datos mock."
        assert rad_max != 900.0, "Pico de 900 exacto ⇒ probablemente datos mock."
        assert len(clouds) > 2

    def test_agregacion_diaria_historica(self, historical_predictions):
        daily = _to_daily_kwh(historical_predictions)
        assert len(daily) >= 1
        assert all(v >= 0 for v in daily.values())
        assert any(v > 1.0 for v in daily.values()), \
            "Se esperaba al menos un día histórico con producción significativa."


# ─────────────────────────────────────────────────────────────────────────────
# TestMonthlyProduction — pastel anual (histórico pasado + climatología futura)
# ─────────────────────────────────────────────────────────────────────────────

_Q_MONTHLY = """
query {
  monthlyProduction {
    month
    monthName
    productionKwh
    source
  }
}
"""


@pytest.fixture(scope="module")
def monthly(real_model):
    r = _gql(_Q_MONTHLY)
    if r["errors"]:
        if _looks_like_network_error(r["errors"]):
            pytest.skip(f"Open-Meteo no alcanzable; se omite: {r['errors']}")
        pytest.fail(f"monthlyProduction devolvió errores: {r['errors']}")
    return r["data"]["monthlyProduction"]


class TestMonthlyProduction:
    """El pastel anual mezcla histórico (pasado) y predicción (futuro) por mes."""

    def test_devuelve_doce_meses(self, monthly):
        assert len(monthly) == 12
        assert [m["month"] for m in monthly] == list(range(1, 13))

    def test_produccion_no_negativa_y_con_datos(self, monthly):
        assert all(m["productionKwh"] >= 0 for m in monthly)
        assert sum(m["productionKwh"] for m in monthly) > 0

    def test_origen_coherente_con_el_mes_actual(self, monthly):
        """Meses anteriores → histórico; el actual → mixto; los futuros → predicción."""
        current = dt.date.today().month
        by_month = {m["month"]: m["source"] for m in monthly}
        assert all(s in {"historico", "prediccion", "mixto"} for s in by_month.values())
        if current > 1:
            assert by_month[current - 1] == "historico"
        if current < 12:
            assert by_month[current + 1] == "prediccion"
        assert by_month[current] == "mixto"
