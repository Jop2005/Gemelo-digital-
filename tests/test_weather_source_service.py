"""
Pruebas de integración para weather_source_service.

Cubre: CRUD de fuentes meteorológicas, validaciones, modo mock,
extracción de campos (path parser), mapeo de payload y activación.
"""
import pytest

from app.services.weather_source_service import (
    list_weather_sources,
    get_weather_source,
    get_active_weather_source,
    save_weather_source,
    delete_weather_source,
    set_active_weather_source,
    map_payload_to_weather_data,
    _extract_path,
    _parse_path,
    _flatten_leaf_fields,
    _generate_mock_source_payload,
    test_weather_source as _test_weather_source,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers compartidos
# ─────────────────────────────────────────────────────────────────────────────

def _source_payload(**overrides):
    base = {
        "name": "Open-Meteo Test",
        "baseUrl": "https://api.open-meteo.com/v1/forecast",
        "authType": "none",
        "enabled": True,
        "isActive": False,
    }
    base.update(overrides)
    return base


def _mock_source_payload(**overrides):
    base = {
        "name": "Fuente Mock",
        "authType": "mock",
        "enabled": True,
        "isActive": False,
    }
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Listado y lectura
# ─────────────────────────────────────────────────────────────────────────────

class TestListarFuentesMeteoro:

    def test_listar_vacio_retorna_lista_vacia(self, mongo_db):
        assert list_weather_sources() == []

    def test_listar_devuelve_fuente_creada(self, mongo_db):
        save_weather_source(_source_payload())
        assert len(list_weather_sources()) == 1

    def test_listar_multiples_fuentes(self, mongo_db):
        save_weather_source(_source_payload(name="API-A"))
        save_weather_source(_source_payload(name="API-B"))
        save_weather_source(_mock_source_payload(name="Mock-C"))
        assert len(list_weather_sources()) == 3

    def test_fuente_activa_aparece_primero(self, mongo_db):
        a = save_weather_source(_source_payload(name="Inactiva"))
        save_weather_source(_source_payload(name="Activa", isActive=True))
        sources = list_weather_sources()
        assert sources[0]["name"] == "Activa"

    def test_estructura_documento_correcta(self, mongo_db):
        save_weather_source(_source_payload())
        doc = list_weather_sources()[0]
        for field in ["_id", "name", "baseUrl", "authType", "enabled", "isActive", "createdAt", "updatedAt"]:
            assert field in doc


class TestObtenerFuente:

    def test_obtener_por_id_existente(self, mongo_db):
        created = save_weather_source(_source_payload(name="Mi fuente"))
        result = get_weather_source(created["_id"])
        assert result is not None
        assert result["name"] == "Mi fuente"

    def test_obtener_por_id_inexistente_retorna_none(self, mongo_db):
        result = get_weather_source("000000000000000000000000")
        assert result is None

    def test_obtener_activa_sin_ninguna_activa(self, mongo_db):
        save_weather_source(_source_payload())
        assert get_active_weather_source() is None

    def test_obtener_activa_retorna_fuente_activa(self, mongo_db):
        save_weather_source(_source_payload(name="Activa", isActive=True))
        active = get_active_weather_source()
        assert active is not None
        assert active["name"] == "Activa"

    def test_obtener_activa_solo_si_enabled(self, mongo_db):
        # Crear fuente inactiva y deshabilitada — no debe aparecer como activa
        save_weather_source(_source_payload(name="Inactiva y deshabilitada", enabled=False, isActive=False))
        assert get_active_weather_source() is None


# ─────────────────────────────────────────────────────────────────────────────
# Creación y validaciones
# ─────────────────────────────────────────────────────────────────────────────

class TestCrearFuente:

    def test_crear_retorna_id(self, mongo_db):
        src = save_weather_source(_source_payload())
        assert "_id" in src and len(src["_id"]) > 0

    def test_crear_persiste_nombre(self, mongo_db):
        src = save_weather_source(_source_payload(name="MeteoFuente"))
        assert src["name"] == "MeteoFuente"

    def test_crear_persiste_url_base(self, mongo_db):
        src = save_weather_source(_source_payload(baseUrl="https://api.ejemplo.cu/weather"))
        assert src["baseUrl"] == "https://api.ejemplo.cu/weather"

    def test_crear_auth_type_none_por_defecto(self, mongo_db):
        src = save_weather_source(_source_payload())
        assert src["authType"] == "none"

    def test_crear_enabled_true_por_defecto(self, mongo_db):
        src = save_weather_source(_source_payload())
        assert src["enabled"] is True

    def test_crear_sin_nombre_lanza_error(self, mongo_db):
        with pytest.raises(ValueError, match="[Nn]ombre"):
            save_weather_source({"name": "", "authType": "none"})

    def test_crear_auth_type_invalido_lanza_error(self, mongo_db):
        with pytest.raises(ValueError, match="[Aa]utenticaci"):
            save_weather_source(_source_payload(authType="oauth2"))

    def test_crear_sin_url_y_sin_mock_lanza_error(self, mongo_db):
        with pytest.raises(ValueError, match="[Uu][Rr][Ll]|[Uu]rl"):
            save_weather_source({"name": "Sin URL", "authType": "none", "baseUrl": None})

    def test_crear_mock_sin_url_es_valido(self, mongo_db):
        src = save_weather_source(_mock_source_payload())
        assert src is not None

    def test_crear_bearer_auth(self, mongo_db):
        src = save_weather_source(_source_payload(authType="bearer", authValue="mi-token-secreto"))
        assert src["authType"] == "bearer"
        assert src["authValue"] == "mi-token-secreto"

    def test_crear_api_key_header(self, mongo_db):
        src = save_weather_source(_source_payload(
            authType="api_key_header", authHeaderName="X-API-Key", authValue="abc123"
        ))
        assert src["authHeaderName"] == "X-API-Key"

    def test_crear_api_key_query(self, mongo_db):
        src = save_weather_source(_source_payload(
            authType="api_key_query", authQueryName="apikey", authValue="xyz"
        ))
        assert src["authQueryName"] == "apikey"


# ─────────────────────────────────────────────────────────────────────────────
# Actualización
# ─────────────────────────────────────────────────────────────────────────────

class TestActualizarFuente:

    def test_actualizar_nombre(self, mongo_db):
        src = save_weather_source(_source_payload(name="Original"))
        updated = save_weather_source(_source_payload(name="Actualizado"), src["_id"])
        assert updated["name"] == "Actualizado"

    def test_actualizar_url(self, mongo_db):
        src = save_weather_source(_source_payload())
        updated = save_weather_source(_source_payload(baseUrl="https://nueva-api.cu/v2"), src["_id"])
        assert "nueva-api.cu" in updated["baseUrl"]

    def test_actualizar_id_inexistente_lanza_error(self, mongo_db):
        with pytest.raises(ValueError, match="[Nn]o encontrada"):
            save_weather_source(_source_payload(), "000000000000000000000000")

    def test_actualizar_actualiza_updatedAt(self, mongo_db):
        src = save_weather_source(_source_payload(name="V1"))
        first_ts = src["updatedAt"]
        import time; time.sleep(0.01)
        updated = save_weather_source(_source_payload(name="V2"), src["_id"])
        assert updated["updatedAt"] >= first_ts


# ─────────────────────────────────────────────────────────────────────────────
# Eliminación
# ─────────────────────────────────────────────────────────────────────────────

class TestEliminarFuente:

    def test_eliminar_existente_retorna_true(self, mongo_db):
        src = save_weather_source(_source_payload())
        assert delete_weather_source(src["_id"]) is True

    def test_eliminar_quita_de_listado(self, mongo_db):
        src = save_weather_source(_source_payload())
        delete_weather_source(src["_id"])
        assert list_weather_sources() == []

    def test_eliminar_inexistente_retorna_false(self, mongo_db):
        assert delete_weather_source("000000000000000000000000") is False

    def test_eliminar_una_no_afecta_otra(self, mongo_db):
        a = save_weather_source(_source_payload(name="A"))
        save_weather_source(_source_payload(name="B"))
        delete_weather_source(a["_id"])
        remaining = list_weather_sources()
        assert len(remaining) == 1
        assert remaining[0]["name"] == "B"


# ─────────────────────────────────────────────────────────────────────────────
# Activación de fuente
# ─────────────────────────────────────────────────────────────────────────────

class TestActivarFuente:

    def test_activar_marca_como_activa(self, mongo_db):
        src = save_weather_source(_source_payload())
        set_active_weather_source(src["_id"])
        active = get_active_weather_source()
        assert active is not None
        assert active["_id"] == src["_id"]

    def test_activar_desactiva_las_demas(self, mongo_db):
        a = save_weather_source(_source_payload(name="A", isActive=True))
        b = save_weather_source(_source_payload(name="B"))
        # "A" estaba activa; ahora activamos "B"
        set_active_weather_source(b["_id"])
        active = get_active_weather_source()
        assert active["_id"] == b["_id"]
        # "A" ya no debe estar activa
        doc_a = get_weather_source(a["_id"])
        assert doc_a["isActive"] is False

    def test_activar_id_inexistente_lanza_error(self, mongo_db):
        with pytest.raises(ValueError, match="[Nn]o encontrada"):
            set_active_weather_source("000000000000000000000000")

    def test_crear_con_isActive_true_llama_activacion(self, mongo_db):
        a = save_weather_source(_source_payload(name="Primero", isActive=True))
        b = save_weather_source(_source_payload(name="Segundo", isActive=True))
        active = get_active_weather_source()
        assert active["_id"] == b["_id"]


# ─────────────────────────────────────────────────────────────────────────────
# _parse_path y _extract_path
# ─────────────────────────────────────────────────────────────────────────────

class TestParsePath:

    def test_clave_simple(self):
        assert _parse_path("temperature") == ["temperature"]

    def test_clave_anidada(self):
        assert _parse_path("measurements.current.temperature_c") == [
            "measurements", "current", "temperature_c"
        ]

    def test_indice_de_array(self):
        tokens = _parse_path("forecast.daily[0].temp")
        assert tokens == ["forecast", "daily", 0, "temp"]

    def test_prefijo_dollar(self):
        # _extract_path maneja el prefijo $., parse_path no tiene que incluirlo
        data = {"a": {"b": 42}}
        result = _extract_path(data, "$.a.b")
        assert result == 42

    def test_clave_inexistente_retorna_none(self):
        assert _extract_path({"a": 1}, "b.c") is None

    def test_path_none_retorna_none(self):
        assert _extract_path({"a": 1}, None) is None

    def test_path_vacio_retorna_none(self):
        assert _extract_path({"a": 1}, "") is None

    def test_indice_fuera_de_rango_retorna_none(self):
        assert _extract_path({"arr": [1, 2]}, "arr[5]") is None

    def test_extraer_valor_numerico(self):
        data = {"measurements": {"current": {"temperature_c": 29.3}}}
        assert _extract_path(data, "measurements.current.temperature_c") == 29.3

    def test_extraer_array(self):
        data = {"forecast": {"daily": [{"temp": 30}, {"temp": 28}]}}
        result = _extract_path(data, "forecast.daily")
        assert isinstance(result, list)
        assert len(result) == 2


# ─────────────────────────────────────────────────────────────────────────────
# _flatten_leaf_fields
# ─────────────────────────────────────────────────────────────────────────────

class TestFlattenLeafFields:

    def test_dict_plano(self):
        fields = list(_flatten_leaf_fields({"a": 1, "b": "x"}))
        paths = [f["path"] for f in fields]
        assert "a" in paths
        assert "b" in paths

    def test_dict_anidado(self):
        fields = list(_flatten_leaf_fields({"meta": {"provider": "mock", "version": 2}}))
        paths = [f["path"] for f in fields]
        assert "meta.provider" in paths
        assert "meta.version" in paths

    def test_array_genera_entrada(self):
        fields = list(_flatten_leaf_fields({"forecast": [{"temp": 30}, {"temp": 28}]}))
        paths = [f["path"] for f in fields]
        assert any("forecast" in p for p in paths)

    def test_valor_none(self):
        fields = list(_flatten_leaf_fields({"val": None}))
        assert fields[0]["valueType"] == "null"

    def test_preview_largo_truncado(self):
        long_string = "x" * 200
        fields = list(_flatten_leaf_fields({"long": long_string}))
        assert len(fields[0]["sampleValue"]) <= 123


# ─────────────────────────────────────────────────────────────────────────────
# map_payload_to_weather_data
# ─────────────────────────────────────────────────────────────────────────────

def _mock_field_mapping():
    return {
        "temperaturePath": "measurements.current.temperature_c",
        "solarRadiationPath": "measurements.current.irradiance_wm2",
        "cloudCoverPath": "measurements.current.cloud_pct",
        "humidityPath": "measurements.current.humidity_pct",
        "windSpeedPath": "measurements.current.wind_kmh",
        "forecastArrayPath": "forecast.daily",
        "forecastDatePath": "date_iso",
        "forecastMaxTempPath": "temp.max_c",
        "forecastMinTempPath": "temp.min_c",
        "forecastSolarRadiationPath": "solar.avg_wm2",
        "forecastCloudCoverPath": "sky.cloud_pct",
    }


class TestMapPayloadToWeatherData:

    def test_mapeo_con_mock_payload(self):
        source = {
            "name": "Mock",
            "fieldMapping": _mock_field_mapping(),
            "locationName": "La Habana",
        }
        payload = _generate_mock_source_payload()
        result = map_payload_to_weather_data(source, payload, capacity_kw=50.0, default_location_name="CUJAE")
        assert result["temperature"] == pytest.approx(29.3)
        assert result["humidity"] == 73
        assert result["cloudCover"] == 34

    def test_mapeo_incluye_pronostico_7_dias(self):
        source = {
            "name": "Mock",
            "fieldMapping": _mock_field_mapping(),
            "locationName": "La Habana",
        }
        payload = _generate_mock_source_payload()
        result = map_payload_to_weather_data(source, payload, capacity_kw=50.0, default_location_name="CUJAE")
        assert len(result["forecast"]) == 7

    def test_cada_dia_pronostico_tiene_campos_requeridos(self):
        source = {
            "name": "Mock",
            "fieldMapping": _mock_field_mapping(),
        }
        payload = _generate_mock_source_payload()
        result = map_payload_to_weather_data(source, payload, capacity_kw=50.0, default_location_name="CUJAE")
        for day in result["forecast"]:
            for field in ["date", "dayOfWeek", "maxTemp", "minTemp", "solarRadiation", "cloudCover", "predictedProduction", "condition"]:
                assert field in day

    def test_mapeo_sin_paths_requeridos_lanza_error(self):
        source = {
            "name": "Incompleto",
            "fieldMapping": {},  # sin mapeos
        }
        payload = _generate_mock_source_payload()
        with pytest.raises(ValueError, match="[Ff]altan|[Mm]apeo"):
            map_payload_to_weather_data(source, payload, capacity_kw=50.0, default_location_name="CUJAE")

    def test_produced_kw_positivo(self):
        source = {
            "name": "Mock",
            "fieldMapping": _mock_field_mapping(),
        }
        payload = _generate_mock_source_payload()
        result = map_payload_to_weather_data(source, payload, capacity_kw=50.0, default_location_name="CUJAE")
        for day in result["forecast"]:
            assert day["predictedProduction"] >= 0

    def test_provider_en_resultado(self):
        source = {"name": "Fuente Personalizada", "fieldMapping": _mock_field_mapping()}
        payload = _generate_mock_source_payload()
        result = map_payload_to_weather_data(source, payload, 50.0, "CUJAE")
        assert result["provider"] == "Fuente Personalizada"

    def test_forecast_array_invalido_lanza_error(self):
        source = {
            "name": "Roto",
            "fieldMapping": {
                **_mock_field_mapping(),
                "forecastArrayPath": "no.existe",  # path que no devuelve lista
            },
        }
        payload = _generate_mock_source_payload()
        with pytest.raises(ValueError, match="[Pp]ronóstico|[Ll]ista"):
            map_payload_to_weather_data(source, payload, 50.0, "CUJAE")


# ─────────────────────────────────────────────────────────────────────────────
# test_weather_source (modo mock — sin red)
# ─────────────────────────────────────────────────────────────────────────────

class TestTestWeatherSource:

    @pytest.mark.asyncio
    async def test_modo_mock_retorna_success(self):
        payload = _mock_source_payload()
        result = await _test_weather_source(payload, lat=23.1136, lon=-82.3666, location_name="CUJAE", use_mock=True)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_modo_mock_retorna_fields(self):
        result = await _test_weather_source(
            _mock_source_payload(), lat=23.1136, lon=-82.3666, location_name="CUJAE", use_mock=True
        )
        assert len(result["fields"]) > 0

    @pytest.mark.asyncio
    async def test_modo_mock_rawjson_es_json_valido(self):
        import json
        result = await _test_weather_source(
            _mock_source_payload(), lat=23.1136, lon=-82.3666, location_name="CUJAE", use_mock=True
        )
        parsed = json.loads(result["rawJson"])
        assert isinstance(parsed, dict)

    @pytest.mark.asyncio
    async def test_modo_mock_incluye_temperatura_en_fields(self):
        result = await _test_weather_source(
            _mock_source_payload(), lat=23.1136, lon=-82.3666, location_name="CUJAE", use_mock=True
        )
        paths = [f["path"] for f in result["fields"]]
        # El payload mock tiene measurements.current.temperature_c
        assert any("temperature" in p for p in paths)

    @pytest.mark.asyncio
    async def test_fuente_invalida_sin_nombre_lanza_error(self):
        with pytest.raises(ValueError, match="[Nn]ombre"):
            await _test_weather_source({"name": ""}, 23.1, -82.3, "CUJAE", use_mock=True)
