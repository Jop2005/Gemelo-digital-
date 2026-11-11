"""
Pruebas para appliance_measurement_service.

Cubre: parsing CSV/TSV, construcción de perfil 168-slot, pronóstico kW,
integración con appliance_service (attach_measurement / clear_measurement).
"""
import pytest
from datetime import datetime

from app.services.appliance_measurement_service import (
    parse_measurement_file,
    build_hourly_profile,
    forecast_kw,
    forecast_series,
)
from app.services.appliance_service import (
    create_appliance,
    list_appliances,
    attach_measurement,
    clear_measurement,
)


# ─────────────────────────────────────────────────────────────────────────────
# CSV de ejemplo (formato simple)
# ─────────────────────────────────────────────────────────────────────────────

_CSV_SIMPLE = """Date,Time,P(SUM)
2024-06-17,08:00:00,1.5
2024-06-17,09:00:00,2.3
2024-06-17,10:00:00,1.8
2024-06-17,11:00:00,2.0
2024-06-17,12:00:00,1.9
"""

_CSV_TSV = "Date\tTime\tP(SUM)\n2024-06-17\t08:00:00\t1.5\n2024-06-17\t09:00:00\t2.1\n"

_CSV_NO_POWER_COL = """Date,Time,Voltage
2024-06-17,08:00:00,220
"""

_CSV_NO_DATE = """P(SUM)
1.5
2.3
"""

_CSV_EMPTY = "Date,Time,P(SUM)\n"

_CSV_DATETIME_COL = """Datetime,P(SUM)
2024-06-17T08:00:00,1.5
2024-06-17T09:00:00,2.3
2024-06-17T10:00:00,1.8
"""


# ─────────────────────────────────────────────────────────────────────────────
# parse_measurement_file
# ─────────────────────────────────────────────────────────────────────────────

class TestParseMeasurementFile:

    def test_csv_simple_retorna_muestras(self):
        samples = parse_measurement_file(_CSV_SIMPLE)
        assert len(samples) == 5

    def test_csv_simple_valores_potencia_correctos(self):
        samples = parse_measurement_file(_CSV_SIMPLE)
        assert samples[0][1] == pytest.approx(1.5)
        assert samples[1][1] == pytest.approx(2.3)

    def test_csv_tsv_separado_por_tab(self):
        samples = parse_measurement_file(_CSV_TSV)
        assert len(samples) == 2

    def test_csv_con_columna_datetime(self):
        samples = parse_measurement_file(_CSV_DATETIME_COL)
        assert len(samples) == 3

    def test_muestras_son_tuplas_datetime_float(self):
        samples = parse_measurement_file(_CSV_SIMPLE)
        dt, power = samples[0]
        assert isinstance(dt, datetime)
        assert isinstance(power, float)

    def test_sin_columna_potencia_lanza_error(self):
        with pytest.raises(ValueError, match="[Pp]otencia|P.SUM"):
            parse_measurement_file(_CSV_NO_POWER_COL)

    def test_sin_columna_fecha_lanza_error(self):
        with pytest.raises(ValueError):
            parse_measurement_file(_CSV_NO_DATE)

    def test_archivo_vacio_lanza_error(self):
        with pytest.raises(ValueError, match="[Vv]acío|vacío|empty"):
            parse_measurement_file("")

    def test_archivo_solo_header_sin_datos_lanza_error(self):
        with pytest.raises(ValueError):
            parse_measurement_file(_CSV_EMPTY)

    def test_hora_8_lunes_extraida_correctamente(self):
        samples = parse_measurement_file(_CSV_SIMPLE)
        dt, _ = samples[0]
        assert dt.hour == 8
        assert dt.weekday() == 0  # Monday


# ─────────────────────────────────────────────────────────────────────────────
# build_hourly_profile
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildHourlyProfile:

    def _samples(self):
        return parse_measurement_file(_CSV_SIMPLE)

    def test_perfil_tiene_168_slots(self):
        samples = self._samples()
        result = build_hourly_profile(samples)
        assert len(result["hourlyProfileKw"]) == 168

    def test_perfil_slot_lunes_8_tiene_valor_correcto(self):
        samples = self._samples()
        result = build_hourly_profile(samples)
        # Monday=0, hour=8 → índice 0*24 + 8 = 8
        assert result["hourlyProfileKw"][8] == pytest.approx(1.5, abs=0.01)

    def test_perfil_meta_contiene_campos(self):
        samples = self._samples()
        result = build_hourly_profile(samples)
        for field in ["samples", "firstDate", "lastDate", "avgKw", "minKw", "maxKw"]:
            assert field in result["meta"]

    def test_perfil_meta_samples_correcto(self):
        samples = self._samples()
        result = build_hourly_profile(samples)
        assert result["meta"]["samples"] == 5

    def test_slots_sin_datos_usan_media_global(self):
        samples = self._samples()
        result = build_hourly_profile(samples)
        avg = result["meta"]["avgKw"]
        # slot del martes a las 8 (no hay datos) debe ser la media global
        assert result["hourlyProfileKw"][1 * 24 + 8] == pytest.approx(avg, abs=0.01)

    def test_lista_vacia_lanza_error(self):
        with pytest.raises(ValueError, match="[Mm]uestras|[Ss]amples"):
            build_hourly_profile([])


# ─────────────────────────────────────────────────────────────────────────────
# forecast_kw
# ─────────────────────────────────────────────────────────────────────────────

class TestForecastKw:

    def _profile(self):
        samples = parse_measurement_file(_CSV_SIMPLE)
        return build_hourly_profile(samples)["hourlyProfileKw"]

    def test_forecast_lunes_8_valor_correcto(self):
        profile = self._profile()
        dt = datetime(2024, 6, 17, 8)  # Monday
        assert forecast_kw(profile, dt) == pytest.approx(1.5, abs=0.01)

    def test_forecast_perfil_vacio_retorna_0(self):
        assert forecast_kw([], datetime(2024, 6, 17, 8)) == 0.0

    def test_forecast_perfil_incorrecto_retorna_0(self):
        assert forecast_kw([1.0] * 10, datetime(2024, 6, 17, 8)) == 0.0

    def test_forecast_no_negativo(self):
        profile = self._profile()
        for day in range(7):
            for hour in range(24):
                dt = datetime(2024, 6, 17 + day, hour)
                assert forecast_kw(profile, dt) >= 0


# ─────────────────────────────────────────────────────────────────────────────
# forecast_series
# ─────────────────────────────────────────────────────────────────────────────

class TestForecastSeries:

    def _profile(self):
        samples = parse_measurement_file(_CSV_SIMPLE)
        return build_hourly_profile(samples)["hourlyProfileKw"]

    def test_series_longitud_correcta(self):
        profile = self._profile()
        result = forecast_series(profile, datetime(2024, 6, 17, 8), 6)
        assert len(result) == 6

    def test_series_estructura_por_elemento(self):
        profile = self._profile()
        result = forecast_series(profile, datetime(2024, 6, 17, 8), 3)
        for item in result:
            assert "datetime" in item
            assert "kW" in item

    def test_series_24_horas(self):
        profile = self._profile()
        result = forecast_series(profile, datetime(2024, 6, 17, 0), 24)
        assert len(result) == 24


# ─────────────────────────────────────────────────────────────────────────────
# attach_measurement y clear_measurement (integración con appliance_service)
# ─────────────────────────────────────────────────────────────────────────────

def _base_appliance():
    return {
        "name": "Aire Acondicionado",
        "averagePowerW": 1500,
        "maxPowerW": 2000,
        "quantity": 1,
    }


class TestAttachClearMeasurement:

    def test_attach_measurement_agrega_perfil(self, mongo_db):
        ap = create_appliance(_base_appliance())
        updated = attach_measurement(ap["_id"], _CSV_SIMPLE)
        assert updated.get("hourlyProfileKw") is not None
        assert len(updated["hourlyProfileKw"]) == 168

    def test_attach_measurement_agrega_meta(self, mongo_db):
        ap = create_appliance(_base_appliance())
        updated = attach_measurement(ap["_id"], _CSV_SIMPLE)
        assert updated.get("measurementMeta") is not None

    def test_attach_measurement_visible_en_list(self, mongo_db):
        ap = create_appliance(_base_appliance())
        attach_measurement(ap["_id"], _CSV_SIMPLE)
        apps = list_appliances()
        found = next((a for a in apps if a["_id"] == ap["_id"]), None)
        assert found is not None
        assert len(found.get("hourlyProfileKw", [])) == 168

    def test_clear_measurement_elimina_perfil(self, mongo_db):
        ap = create_appliance(_base_appliance())
        attach_measurement(ap["_id"], _CSV_SIMPLE)
        cleared = clear_measurement(ap["_id"])
        assert cleared.get("hourlyProfileKw") is None or cleared.get("hourlyProfileKw") == []

    def test_clear_measurement_elimina_meta(self, mongo_db):
        ap = create_appliance(_base_appliance())
        attach_measurement(ap["_id"], _CSV_SIMPLE)
        cleared = clear_measurement(ap["_id"])
        assert cleared.get("measurementMeta") is None

    def test_clear_measurement_id_inexistente_retorna_none(self, mongo_db):
        result = clear_measurement("000000000000000000000000")
        assert result is None

    def test_attach_csv_invalido_lanza_error(self, mongo_db):
        ap = create_appliance(_base_appliance())
        with pytest.raises(ValueError):
            attach_measurement(ap["_id"], "sin,cabeceras\nvalidas,aqui")

    def test_attach_y_clear_deja_appliance_sin_perfil(self, mongo_db):
        ap = create_appliance(_base_appliance())
        attach_measurement(ap["_id"], _CSV_SIMPLE)
        cleared = clear_measurement(ap["_id"])
        profile = cleared.get("hourlyProfileKw")
        assert profile is None or profile == []
