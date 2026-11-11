"""
Pruebas unitarias para app/services/analytics.py

Cubre:
- calculate_system_metrics: balance energético, totales diarios, CO₂ evitado
- calculate_energy_flow: todos los escenarios de distribución de energía
"""
import pytest
from app.services.analytics import calculate_system_metrics, calculate_energy_flow


# ─────────────────────────────────────────────────────────────────
# Fixtures y helpers
# ─────────────────────────────────────────────────────────────────

def make_current(production: float, consumption: float, efficiency: float = 80.0) -> dict:
    return {"production": production, "consumption": consumption, "efficiency": efficiency}


def make_history(*pairs: tuple) -> list:
    return [{"production": p, "consumption": c} for p, c in pairs]


# ─────────────────────────────────────────────────────────────────
# calculate_system_metrics
# ─────────────────────────────────────────────────────────────────

class TestCalculateSystemMetrics:

    def test_balance_positivo_cuando_produccion_supera_consumo(self):
        result = calculate_system_metrics(make_current(30, 20), [])
        assert result["energyBalance"] == 10.0

    def test_balance_negativo_cuando_consumo_supera_produccion(self):
        result = calculate_system_metrics(make_current(10, 25, 70), [])
        assert result["energyBalance"] == -15.0

    def test_suma_produccion_diaria_del_historial(self):
        history = make_history((10, 8), (20, 15), (15, 12))
        result = calculate_system_metrics(make_current(20, 15), history)
        assert result["dailyProduction"] == 45.0

    def test_suma_consumo_diario_del_historial(self):
        history = make_history((10, 8), (20, 15), (15, 12))
        result = calculate_system_metrics(make_current(20, 15), history)
        assert result["dailyConsumption"] == 35.0

    def test_co2_evitado_usa_el_factor_de_red(self):
        from app.services.analytics import grid_co2_factor
        history = make_history((100, 80))
        result = calculate_system_metrics(make_current(0, 0, 0), history)
        assert result["co2Avoided"] == round(100 * grid_co2_factor(), 2)

    def test_co2_es_cero_sin_historial(self):
        result = calculate_system_metrics(make_current(20, 15), [])
        assert result["co2Avoided"] == 0.0

    def test_redondea_a_dos_decimales(self):
        result = calculate_system_metrics(make_current(10.333, 7.666), [])
        assert result["currentProduction"] == 10.33
        assert result["currentConsumption"] == 7.67
        # 10.333 - 7.666 = 2.667 → 2.67
        assert result["energyBalance"] == 2.67

    def test_eficiencia_se_propaga_sin_cambio(self):
        result = calculate_system_metrics(make_current(20, 15, 92.5), [])
        assert result["systemEfficiency"] == 92.5

    def test_balance_cero_cuando_produccion_igual_consumo(self):
        result = calculate_system_metrics(make_current(25, 25), [])
        assert result["energyBalance"] == 0.0

    def test_historial_vacio_da_totales_diarios_cero(self):
        result = calculate_system_metrics(make_current(20, 15), [])
        assert result["dailyProduction"] == 0.0
        assert result["dailyConsumption"] == 0.0


# ─────────────────────────────────────────────────────────────────
# calculate_energy_flow
# ─────────────────────────────────────────────────────────────────

class TestCalculateEnergyFlow:

    def test_todo_va_a_carga_cuando_produccion_igual_consumo(self):
        flow = calculate_energy_flow(20, 20, False, 0)
        assert flow["solarToLoad"] == 20.0
        assert flow["solarToBattery"] == 0.0
        assert flow["solarToGrid"] == 0.0
        assert flow["batteryToLoad"] == 0.0
        assert flow["gridToLoad"] == 0.0

    def test_carga_bateria_con_excedente(self):
        # Surplus=10, batteryPowerFlow=8 → min(10,8)=8
        flow = calculate_energy_flow(30, 20, True, 8)
        assert flow["solarToLoad"] == 20.0
        assert flow["solarToBattery"] == 8.0
        assert flow["solarToGrid"] == 2.0

    def test_exporta_a_red_cuando_bateria_no_carga(self):
        flow = calculate_energy_flow(30, 20, False, 0)
        assert flow["solarToGrid"] == 10.0
        assert flow["solarToBattery"] == 0.0

    def test_descarga_bateria_para_cubrir_deficit_parcial(self):
        # Deficit=15, batteryPowerFlow=-12 → batteryToLoad=min(15,12)=12, gridToLoad=3
        flow = calculate_energy_flow(10, 25, False, -12)
        assert flow["solarToLoad"] == 10.0
        assert flow["batteryToLoad"] == 12.0
        assert flow["gridToLoad"] == 3.0

    def test_importa_todo_de_red_si_bateria_esta_cargando(self):
        flow = calculate_energy_flow(10, 25, True, 5)
        assert flow["gridToLoad"] == 15.0
        assert flow["batteryToLoad"] == 0.0

    def test_dependencia_total_de_red_con_produccion_cero(self):
        flow = calculate_energy_flow(0, 30, False, 0)
        assert flow["solarToLoad"] == 0.0
        assert flow["gridToLoad"] == 30.0

    def test_todos_los_flujos_son_no_negativos(self):
        flow = calculate_energy_flow(15, 40, False, -20)
        for key, value in flow.items():
            assert value >= 0.0, f"{key} debe ser >= 0, obtenido {value}"

    def test_redondea_a_dos_decimales(self):
        flow = calculate_energy_flow(10.333, 7.666, False, 0)
        for value in flow.values():
            assert round(value, 2) == value
