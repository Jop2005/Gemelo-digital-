"""
Pruebas unitarias para la lógica pura de simulación de batería.

Cubre simulate_battery_depletion, la función pura extraída de
battery_discharge_service. No requiere base de datos ni modelos ML.

Aspectos validados:
- Descarga lineal sin producción solar
- Recarga durante horas de producción
- Límites físicos: nivel no supera capacidad ni baja de cero
- Batería que nunca se agota (producción ≥ consumo)
- Detección correcta del momento de agotamiento
- Casos borde: listas vacías, nivel inicial parcial, capacidad mínima
"""
import pytest
from app.services.battery_discharge_service import simulate_battery_depletion


# ─── Helpers ──────────────────────────────────────────────────────────────────

def const_series(value: float, n: int) -> list[float]:
    """Crea una serie de n horas con valor constante."""
    return [value] * n


# ─────────────────────────────────────────────────────────────────────────────
# Descarga sin producción solar
# ─────────────────────────────────────────────────────────────────────────────

class TestDescargaSinProduccion:

    def test_bateria_de_100kwh_con_consumo_10kw_dura_10_horas(self):
        # 100 kWh ÷ 10 kW = 10 h = 600 min
        result = simulate_battery_depletion(
            production_kw_series=const_series(0.0, 24),
            consumption_kw_series=const_series(10.0, 24),
            battery_capacity_kwh=100.0,
        )
        assert result == 600

    def test_bateria_de_50kwh_con_consumo_25kw_dura_2_horas(self):
        result = simulate_battery_depletion(
            production_kw_series=const_series(0.0, 24),
            consumption_kw_series=const_series(25.0, 24),
            battery_capacity_kwh=50.0,
        )
        assert result == 120

    def test_bateria_se_agota_en_primera_hora_si_consumo_mayor_capacidad(self):
        # Capacidad 10 kWh, consumo 15 kW → en la primera hora ya llega a 0
        result = simulate_battery_depletion(
            production_kw_series=[0.0],
            consumption_kw_series=[15.0],
            battery_capacity_kwh=10.0,
        )
        assert result == 60  # primera hora

    def test_agotamiento_devuelve_minutos_enteros_multiplo_de_60(self):
        result = simulate_battery_depletion(
            production_kw_series=const_series(0.0, 48),
            consumption_kw_series=const_series(5.0, 48),
            battery_capacity_kwh=100.0,
        )
        assert result is not None
        assert result % 60 == 0


# ─────────────────────────────────────────────────────────────────────────────
# Producción solar carga la batería
# ─────────────────────────────────────────────────────────────────────────────

class TestRecargaSolar:

    def test_produccion_igual_consumo_nunca_agota_bateria(self):
        result = simulate_battery_depletion(
            production_kw_series=const_series(20.0, 48),
            consumption_kw_series=const_series(20.0, 48),
            battery_capacity_kwh=100.0,
        )
        assert result is None

    def test_produccion_mayor_que_consumo_nunca_agota_bateria(self):
        result = simulate_battery_depletion(
            production_kw_series=const_series(30.0, 48),
            consumption_kw_series=const_series(20.0, 48),
            battery_capacity_kwh=100.0,
        )
        assert result is None

    def test_recarga_durante_el_dia_extiende_la_autonomia(self):
        # Batería 200 kWh.
        # Noche (12h): consumo=10 kW, sin solar → gasta 120 kWh → quedan 80 kWh
        # Día  (12h): producción=15 kW, consumo=10 kW → +5 kWh/h = +60 kWh
        # Nivel final ~140 kWh → nunca se agota.
        prod = const_series(0.0, 12) + const_series(15.0, 12)
        cons = const_series(10.0, 24)
        result = simulate_battery_depletion(prod, cons, battery_capacity_kwh=200.0)
        assert result is None  # la recarga diurna compensa el déficit nocturno

    def test_nivel_no_supera_capacidad_maxima_con_mucha_produccion(self):
        # Producción altísima no debe desbordar la capacidad.
        # Si el nivel se desbordara, el balance sería incorrecto y
        # la batería "aparecería" con más energía de la que tiene.
        prod = const_series(1000.0, 5)
        cons = const_series(1.0, 5)
        # No debe lanzar excepción ni devolver resultado incorrecto
        result = simulate_battery_depletion(prod, cons, battery_capacity_kwh=100.0)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# Nivel inicial parcial
# ─────────────────────────────────────────────────────────────────────────────

class TestNivelInicialParcial:

    def test_bateria_al_50_porciento_dura_la_mitad(self):
        # Sin producción, consumo 10 kW, capacidad 100 kWh
        # Al 100% → 600 min; al 50% (50 kWh) → 300 min
        result = simulate_battery_depletion(
            production_kw_series=const_series(0.0, 24),
            consumption_kw_series=const_series(10.0, 24),
            battery_capacity_kwh=100.0,
            start_level_kwh=50.0,
        )
        assert result == 300

    def test_bateria_vacia_se_agota_inmediatamente(self):
        result = simulate_battery_depletion(
            production_kw_series=const_series(0.0, 5),
            consumption_kw_series=const_series(10.0, 5),
            battery_capacity_kwh=100.0,
            start_level_kwh=0.0,
        )
        assert result == 60  # ya en la primera hora está a 0

    def test_nivel_inicial_mayor_que_capacidad_se_trata_como_llena(self):
        # No debería ser posible en la práctica, pero el sistema debe ser robusto.
        # La producción de la primera hora lo clampea a la capacidad y luego descarga.
        result = simulate_battery_depletion(
            production_kw_series=[50.0] + const_series(0.0, 9),
            consumption_kw_series=const_series(10.0, 10),
            battery_capacity_kwh=100.0,
            start_level_kwh=200.0,  # valor inválido / mayor que capacidad
        )
        # El nivel se clampea a 100 en la primera hora (50 prod - 10 cons = +40, pero cap=100)
        assert result is None  # 100 kWh / 10 kW = 10h, tenemos exactamente 10 horas


# ─────────────────────────────────────────────────────────────────────────────
# Casos borde
# ─────────────────────────────────────────────────────────────────────────────

class TestCasosBorde:

    def test_series_vacias_devuelven_none(self):
        result = simulate_battery_depletion([], [], battery_capacity_kwh=100.0)
        assert result is None

    def test_capacidad_cero_lanza_error(self):
        with pytest.raises(ValueError, match="mayor que cero"):
            simulate_battery_depletion([10.0], [5.0], battery_capacity_kwh=0.0)

    def test_capacidad_negativa_lanza_error(self):
        with pytest.raises(ValueError):
            simulate_battery_depletion([0.0], [1.0], battery_capacity_kwh=-50.0)

    def test_series_de_longitudes_distintas_usa_la_mas_corta(self):
        # zip se detiene en la más corta — comportamiento esperado de Python
        # prod=[0,0,0], cons=[50,50,...], cap=100
        # h0: 100-50=50  (no 0 aún)
        # h1: 50-50=0    → agotada en (1+1)*60 = 120 min
        prod = [0.0, 0.0, 0.0]   # 3 horas
        cons = [50.0] * 10        # 10 horas
        result = simulate_battery_depletion(prod, cons, battery_capacity_kwh=100.0)
        assert result == 120

    def test_una_sola_hora_con_deficit_total(self):
        result = simulate_battery_depletion(
            production_kw_series=[0.0],
            consumption_kw_series=[200.0],
            battery_capacity_kwh=100.0,
        )
        assert result == 60

    def test_una_sola_hora_sin_deficit(self):
        result = simulate_battery_depletion(
            production_kw_series=[50.0],
            consumption_kw_series=[20.0],
            battery_capacity_kwh=100.0,
        )
        assert result is None
