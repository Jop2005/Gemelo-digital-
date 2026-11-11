"""
Pruebas unitarias para funciones puras de app/services/prediction_service.py

Cubre:
- predict_production: cálculo de producción solar por hora
- _get_hour_efficiency_factor: factor de eficiencia según hora del día
- _model_confidence_pct: confianza = R² real del modelo
- _estimate_hourly_temperature: temperatura estimada por hora
- _predict_consumption: consumo estimado por hora
"""
import math
import pytest
from app.services.prediction_service import (
    predict_production,
    _get_hour_efficiency_factor,
    _model_confidence_pct,
    _estimate_hourly_temperature,
    _predict_consumption,
)

# ─── Contexto solar de referencia ─────────────────────────────────
CONTEXT_50KW = {
    "capacityKw": 50.0,
    "panelEfficiency": 0.20,
    "arrayAreaM2": 250.0,
}


# ─────────────────────────────────────────────────────────────────
# _get_hour_efficiency_factor
# ─────────────────────────────────────────────────────────────────

class TestHourEfficiencyFactor:

    def test_horas_pico_12_14_retornan_1(self):
        for hour in (12, 13, 14):
            assert _get_hour_efficiency_factor(hour) == 1.0, f"hora {hour}"

    def test_horas_11_15_retornan_095(self):
        for hour in (11, 15):
            assert _get_hour_efficiency_factor(hour) == 0.95, f"hora {hour}"

    def test_horas_10_16_retornan_085(self):
        for hour in (10, 16):
            assert _get_hour_efficiency_factor(hour) == 0.85, f"hora {hour}"

    def test_horas_8_17_retornan_070(self):
        for hour in (8, 17):
            assert _get_hour_efficiency_factor(hour) == 0.70, f"hora {hour}"

    def test_horas_7_18_retornan_050(self):
        for hour in (7, 18):
            assert _get_hour_efficiency_factor(hour) == 0.50, f"hora {hour}"

    def test_horas_6_19_retornan_030(self):
        for hour in (6, 19):
            assert _get_hour_efficiency_factor(hour) == 0.30, f"hora {hour}"

    def test_noche_retorna_0(self):
        for hour in (0, 1, 2, 3, 4, 5, 20, 21, 22, 23):
            assert _get_hour_efficiency_factor(hour) == 0.0, f"hora {hour}"

    def test_factor_decrece_al_alejarse_del_mediodia(self):
        assert _get_hour_efficiency_factor(13) >= _get_hour_efficiency_factor(11)
        assert _get_hour_efficiency_factor(11) >= _get_hour_efficiency_factor(10)
        assert _get_hour_efficiency_factor(10) >= _get_hour_efficiency_factor(8)


# ─────────────────────────────────────────────────────────────────
# predict_production
# ─────────────────────────────────────────────────────────────────

class TestPredictProduction:

    def test_retorna_cero_antes_de_las_6h(self):
        assert predict_production(800, 25, 10, 5, CONTEXT_50KW) == 0.0

    def test_retorna_cero_despues_de_las_20h(self):
        assert predict_production(800, 25, 10, 21, CONTEXT_50KW) == 0.0

    def test_produce_en_hora_pico(self):
        prod = predict_production(800, 25, 0, 13, CONTEXT_50KW)
        assert prod > 0

    def test_nubosidad_reduce_produccion(self):
        clear = predict_production(800, 25, 0, 13, CONTEXT_50KW)
        cloudy = predict_production(800, 25, 80, 13, CONTEXT_50KW)
        assert cloudy < clear

    def test_no_supera_capacidad_maxima(self):
        # Radiación muy alta pero el sistema tiene cap=50 kW
        prod = predict_production(5000, 25, 0, 13, CONTEXT_50KW)
        assert prod <= 50.0

    def test_produccion_mediodía_mayor_que_mañana(self):
        noon = predict_production(600, 25, 10, 13, CONTEXT_50KW)
        morning = predict_production(600, 25, 10, 8, CONTEXT_50KW)
        assert noon > morning

    def test_resultado_es_no_negativo(self):
        assert predict_production(0, 50, 100, 13, CONTEXT_50KW) >= 0.0

    def test_resultado_redondeado_a_2_decimales(self):
        prod = predict_production(800, 25, 30, 12, CONTEXT_50KW)
        assert round(prod, 2) == prod


# ─────────────────────────────────────────────────────────────────
# _model_confidence_pct  (confianza = R² real del modelo, no una fórmula)
# ─────────────────────────────────────────────────────────────────

class TestModelConfidence:

    def test_confianza_es_el_r2_del_modelo(self, monkeypatch):
        monkeypatch.setattr(
            "app.services.prediction_service.ml_model_service.get_test_r2",
            lambda: 0.789,
        )
        assert _model_confidence_pct() == 79  # round(0.789 * 100)

    def test_cero_si_no_hay_modelo(self, monkeypatch):
        monkeypatch.setattr(
            "app.services.prediction_service.ml_model_service.get_test_r2",
            lambda: None,
        )
        assert _model_confidence_pct() == 0

    def test_se_acota_entre_0_y_100(self, monkeypatch):
        monkeypatch.setattr(
            "app.services.prediction_service.ml_model_service.get_test_r2",
            lambda: 1.5,
        )
        assert _model_confidence_pct() == 100
        monkeypatch.setattr(
            "app.services.prediction_service.ml_model_service.get_test_r2",
            lambda: -0.3,
        )
        assert _model_confidence_pct() == 0


# ─────────────────────────────────────────────────────────────────
# _estimate_hourly_temperature
# ─────────────────────────────────────────────────────────────────

class TestEstimateHourlyTemperature:

    def test_temperatura_promedio_entre_min_y_max(self):
        temp = _estimate_hourly_temperature(12, 32, 22)
        # Debe estar entre min y max
        assert 22 <= temp <= 32

    def test_temperatura_es_float_redondeado(self):
        temp = _estimate_hourly_temperature(13, 30, 20)
        assert isinstance(temp, float)
        assert round(temp, 1) == temp


# ─────────────────────────────────────────────────────────────────
# _predict_consumption
# ─────────────────────────────────────────────────────────────────

class TestPredictConsumption:

    def test_consumo_pico_en_mañana_7_9(self):
        for hour in (7, 8, 9):
            assert _predict_consumption(hour) == pytest.approx(35 * 1.3), f"hora {hour}"

    def test_consumo_pico_en_tarde_18_22(self):
        for hour in (18, 19, 20, 21, 22):
            assert _predict_consumption(hour) == pytest.approx(35 * 1.3), f"hora {hour}"

    def test_consumo_normal_en_horario_diurno(self):
        for hour in (6, 10, 11, 12, 13, 14, 15, 16, 17):
            assert _predict_consumption(hour) == 35, f"hora {hour}"

    def test_consumo_nocturno_menor_que_diurno(self):
        night = _predict_consumption(2)
        day = _predict_consumption(12)
        assert night < day

    def test_consumo_positivo_siempre(self):
        for hour in range(24):
            assert _predict_consumption(hour) > 0
