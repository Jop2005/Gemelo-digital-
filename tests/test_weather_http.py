"""
Tests de la capa HTTP del sistema de fuentes meteorológicas.

Usa `respx` para interceptar las llamadas httpx sin salir a la red.
Cubre exactamente lo que faltaba antes: que los headers de autenticación
se construyen bien, que los templates {{lat}}/{{lon}} se sustituyen,
que los errores HTTP se propagan, que el pipeline completo
(DB → fetch → map → WeatherDataType) funciona con una API propia, y
que el mecanismo de fallback (custom → Open-Meteo → generado) actúa
correctamente en cada combinación de fallo.

Lo que sigue siendo MANUAL (fuera del alcance de cualquier mock):
  - La UI de descubrimiento de campos y el mapeador visual del frontend.
  - Certificados SSL / TLS inválidos de APIs de terceros.
  - Rate-limiting real de APIs externas.
  - Expiración de tokens de autenticación en mitad de una sesión.
  - Validación de calidad de datos reales (temperaturas coherentes, etc.).
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Dict

import httpx
import pytest
import respx

import app.database as _db_module
import mongomock

from app.services.weather_source_service import (
    fetch_source_payload,
    get_active_weather_data,
    map_payload_to_weather_data,
    save_weather_source,
    set_active_weather_source,
    get_active_weather_source,
)
from app.services.weather_service import get_weather_with_fallback
from app.schema import schema


# ─────────────────────────────────────────────────────────────────────────────
# Datos de ejemplo: payload de una API propia y su fieldMapping
# ─────────────────────────────────────────────────────────────────────────────

_CUSTOM_API_URL = "https://api.clima-propia.cu/v1/actual"

_CUSTOM_PAYLOAD: Dict[str, Any] = {
    "datos": {
        "temp_c": 29.3,
        "irradiancia_wm2": 612,
        "nubosidad_pct": 34,
        "humedad_pct": 73,
        "viento_kmh": 17.6,
    },
    "pronostico": [
        {
            "fecha": f"2024-06-{17 + i:02d}",
            "max_c": round(32.0 - i * 0.5, 1),
            "min_c": round(24.0 - i * 0.3, 1),
            "irr_media": 650 - i * 20,
            "nubes_pct": 30 + i * 5,
        }
        for i in range(7)
    ],
}

_CUSTOM_MAPPING: Dict[str, str] = {
    "temperaturePath":          "datos.temp_c",
    "solarRadiationPath":       "datos.irradiancia_wm2",
    "cloudCoverPath":           "datos.nubosidad_pct",
    "humidityPath":             "datos.humedad_pct",
    "windSpeedPath":            "datos.viento_kmh",
    "forecastArrayPath":        "pronostico",
    "forecastDatePath":         "fecha",
    "forecastMaxTempPath":      "max_c",
    "forecastMinTempPath":      "min_c",
    "forecastSolarRadiationPath": "irr_media",
    "forecastCloudCoverPath":   "nubes_pct",
}

_OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"

# Respuesta mínima que satisface el parser de Open-Meteo
_OPENMETEO_PAYLOAD: Dict[str, Any] = {
    "current": {
        "temperature_2m": 27.5,
        "relative_humidity_2m": 68,
        "wind_speed_10m": 14.2,
        "weather_code": 2,
        "shortwave_radiation": 480.0,
        "cloud_cover": 45,
    },
    "daily": {
        "time": [f"2024-06-{17 + i:02d}" for i in range(7)],
        "temperature_2m_max": [31, 30, 29, 28, 30, 31, 32],
        "temperature_2m_min": [24, 23, 23, 22, 23, 24, 25],
        "shortwave_radiation_sum": [22, 20, 18, 17, 21, 23, 24],
        "cloud_cover_mean": [30, 40, 50, 60, 35, 25, 20],
        "weather_code": [1, 2, 3, 4, 2, 1, 0],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: fuente personalizada lista para usar
# ─────────────────────────────────────────────────────────────────────────────

def _custom_source(**overrides):
    base = {
        "name": "API Propia Cuba",
        "baseUrl": _CUSTOM_API_URL,
        "authType": "none",
        "fieldMapping": _CUSTOM_MAPPING,
        "enabled": True,
        "isActive": False,
    }
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# 1. CAPA HTTP — fetch_source_payload
#    Prueba que los headers, query params y autenticación se construyen bien.
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchSourcePayloadHTTP:
    """
    Prueba la función que hace el GET real a la API externa.
    No sale a la red: respx intercepta httpx a nivel de transport.
    """

    @pytest.mark.asyncio
    async def test_auth_none_realiza_get_sin_headers_extra(self):
        source = {"name": "Test", "baseUrl": _CUSTOM_API_URL, "authType": "none",
                  "queryParams": {}, "fieldMapping": {}}
        with respx.mock:
            route = respx.get(_CUSTOM_API_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )
            result = await fetch_source_payload(source, lat=23.1, lon=-82.3, location_name="CUJAE")

        assert route.called
        req = route.calls[0].request
        assert "authorization" not in req.headers
        assert result["datos"]["temp_c"] == 29.3

    @pytest.mark.asyncio
    async def test_auth_bearer_añade_header_authorization(self):
        source = {"name": "Test", "baseUrl": _CUSTOM_API_URL, "authType": "bearer",
                  "authValue": "mi-token-secreto", "queryParams": {}, "fieldMapping": {}}
        with respx.mock:
            route = respx.get(_CUSTOM_API_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )
            await fetch_source_payload(source, lat=23.1, lon=-82.3, location_name="CUJAE")

        req = route.calls[0].request
        assert req.headers["authorization"] == "Bearer mi-token-secreto"

    @pytest.mark.asyncio
    async def test_auth_api_key_header_añade_header_personalizado(self):
        source = {
            "name": "Test", "baseUrl": _CUSTOM_API_URL,
            "authType": "api_key_header", "authHeaderName": "X-Climate-Key",
            "authValue": "clave-abc123", "queryParams": {}, "fieldMapping": {},
        }
        with respx.mock:
            route = respx.get(_CUSTOM_API_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )
            await fetch_source_payload(source, lat=23.1, lon=-82.3, location_name="CUJAE")

        req = route.calls[0].request
        assert req.headers["x-climate-key"] == "clave-abc123"

    @pytest.mark.asyncio
    async def test_auth_api_key_query_añade_param_en_url(self):
        source = {
            "name": "Test", "baseUrl": _CUSTOM_API_URL,
            "authType": "api_key_query", "authQueryName": "apikey",
            "authValue": "xyz987", "queryParams": {}, "fieldMapping": {},
        }
        with respx.mock:
            route = respx.get(_CUSTOM_API_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )
            await fetch_source_payload(source, lat=23.1, lon=-82.3, location_name="CUJAE")

        req = route.calls[0].request
        assert "apikey=xyz987" in str(req.url)

    @pytest.mark.asyncio
    async def test_template_lat_lon_se_sustituyen_en_query_params(self):
        source = {
            "name": "Test", "baseUrl": _CUSTOM_API_URL, "authType": "none",
            "queryParams": {"latitude": "{{lat}}", "longitude": "{{lon}}"},
            "fieldMapping": {},
        }
        with respx.mock:
            route = respx.get(_CUSTOM_API_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )
            await fetch_source_payload(source, lat=23.1136, lon=-82.3666, location_name="CUJAE")

        params = dict(route.calls[0].request.url.params)
        assert params["latitude"] == "23.1136"
        assert params["longitude"] == "-82.3666"

    @pytest.mark.asyncio
    async def test_template_location_name_se_sustituye(self):
        source = {
            "name": "Test", "baseUrl": _CUSTOM_API_URL, "authType": "none",
            "queryParams": {"city": "{{locationName}}"},
            "fieldMapping": {},
        }
        with respx.mock:
            route = respx.get(_CUSTOM_API_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )
            await fetch_source_payload(source, lat=23.1, lon=-82.3, location_name="La Habana")

        params = dict(route.calls[0].request.url.params)
        assert params["city"] == "La Habana"

    @pytest.mark.asyncio
    async def test_http_401_lanza_excepcion(self):
        source = {"name": "Test", "baseUrl": _CUSTOM_API_URL, "authType": "none",
                  "queryParams": {}, "fieldMapping": {}}
        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(return_value=httpx.Response(401))
            with pytest.raises(Exception):
                await fetch_source_payload(source, lat=23.1, lon=-82.3, location_name="CUJAE")

    @pytest.mark.asyncio
    async def test_http_500_lanza_excepcion(self):
        source = {"name": "Test", "baseUrl": _CUSTOM_API_URL, "authType": "none",
                  "queryParams": {}, "fieldMapping": {}}
        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(return_value=httpx.Response(500))
            with pytest.raises(Exception):
                await fetch_source_payload(source, lat=23.1, lon=-82.3, location_name="CUJAE")

    @pytest.mark.asyncio
    async def test_http_404_lanza_excepcion(self):
        source = {"name": "Test", "baseUrl": _CUSTOM_API_URL, "authType": "none",
                  "queryParams": {}, "fieldMapping": {}}
        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(return_value=httpx.Response(404))
            with pytest.raises(Exception):
                await fetch_source_payload(source, lat=23.1, lon=-82.3, location_name="CUJAE")

    @pytest.mark.asyncio
    async def test_timeout_lanza_excepcion(self):
        source = {"name": "Test", "baseUrl": _CUSTOM_API_URL, "authType": "none",
                  "queryParams": {}, "fieldMapping": {}}
        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(
                side_effect=httpx.TimeoutException("timeout simulado")
            )
            with pytest.raises(Exception):
                await fetch_source_payload(source, lat=23.1, lon=-82.3, location_name="CUJAE")

    @pytest.mark.asyncio
    async def test_url_inalcanzable_lanza_excepcion(self):
        """Con respx.mock activo, URLs no mocked lanzan ConnectError."""
        source = {"name": "Test", "baseUrl": _CUSTOM_API_URL, "authType": "none",
                  "queryParams": {}, "fieldMapping": {}}
        with respx.mock:
            # No mockeamos _CUSTOM_API_URL → ConnectError
            with pytest.raises(Exception):
                await fetch_source_payload(source, lat=23.1, lon=-82.3, location_name="CUJAE")

    @pytest.mark.asyncio
    async def test_respuesta_json_correcta_retorna_dict(self):
        source = {"name": "Test", "baseUrl": _CUSTOM_API_URL, "authType": "none",
                  "queryParams": {}, "fieldMapping": {}}
        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )
            result = await fetch_source_payload(source, lat=23.1, lon=-82.3, location_name="CUJAE")

        assert isinstance(result, dict)
        assert "datos" in result
        assert "pronostico" in result

    @pytest.mark.asyncio
    async def test_api_key_header_nombre_defecto_cuando_no_configurado(self):
        """Si authHeaderName está vacío, usa 'X-API-Key' por defecto."""
        source = {
            "name": "Test", "baseUrl": _CUSTOM_API_URL,
            "authType": "api_key_header", "authHeaderName": None,
            "authValue": "clave-xyz", "queryParams": {}, "fieldMapping": {},
        }
        with respx.mock:
            route = respx.get(_CUSTOM_API_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )
            await fetch_source_payload(source, lat=23.1, lon=-82.3, location_name="CUJAE")

        req = route.calls[0].request
        assert "x-api-key" in req.headers
        assert req.headers["x-api-key"] == "clave-xyz"

    @pytest.mark.asyncio
    async def test_api_key_query_nombre_defecto_cuando_no_configurado(self):
        """Si authQueryName está vacío, usa 'api_key' por defecto."""
        source = {
            "name": "Test", "baseUrl": _CUSTOM_API_URL,
            "authType": "api_key_query", "authQueryName": None,
            "authValue": "clave-abc", "queryParams": {}, "fieldMapping": {},
        }
        with respx.mock:
            route = respx.get(_CUSTOM_API_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )
            await fetch_source_payload(source, lat=23.1, lon=-82.3, location_name="CUJAE")

        params_str = str(route.calls[0].request.url)
        assert "api_key=clave-abc" in params_str


# ─────────────────────────────────────────────────────────────────────────────
# 2. PIPELINE COMPLETO — DB + fetch + map
#    Prueba que con una fuente activa en la BD, el sistema la usa y
#    produce un WeatherDataType coherente.
# ─────────────────────────────────────────────────────────────────────────────

class TestGetActiveWeatherDataHTTP:
    """
    Prueba get_active_weather_data: lee la fuente activa de MongoDB,
    hace el fetch (mocked), mapea el payload al formato interno.
    """

    @pytest.mark.asyncio
    async def test_sin_fuente_activa_retorna_none(self, mongo_db):
        save_weather_source(_custom_source(name="Inactiva"))
        result = await get_active_weather_data(23.1, -82.3, 50.0, "CUJAE")
        assert result is None

    @pytest.mark.asyncio
    async def test_fuente_activa_retorna_datos_del_endpoint(self, mongo_db):
        src = save_weather_source(_custom_source())
        set_active_weather_source(src["_id"])

        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )
            result = await get_active_weather_data(23.1136, -82.3666, 50.0, "CUJAE")

        assert result is not None

    @pytest.mark.asyncio
    async def test_temperatura_extraida_de_api_propia(self, mongo_db):
        src = save_weather_source(_custom_source())
        set_active_weather_source(src["_id"])

        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )
            result = await get_active_weather_data(23.1136, -82.3666, 50.0, "CUJAE")

        assert abs(result["temperature"] - 29.3) < 0.01

    @pytest.mark.asyncio
    async def test_humedad_extraida_correctamente(self, mongo_db):
        src = save_weather_source(_custom_source())
        set_active_weather_source(src["_id"])

        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )
            result = await get_active_weather_data(23.1136, -82.3666, 50.0, "CUJAE")

        assert result["humidity"] == 73

    @pytest.mark.asyncio
    async def test_radiacion_solar_extraida_correctamente(self, mongo_db):
        src = save_weather_source(_custom_source())
        set_active_weather_source(src["_id"])

        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )
            result = await get_active_weather_data(23.1136, -82.3666, 50.0, "CUJAE")

        assert result["solarRadiation"] == 612

    @pytest.mark.asyncio
    async def test_proveedor_es_nombre_de_la_fuente_propia(self, mongo_db):
        """El provider debe ser el nombre de la fuente configurada, no 'Open-Meteo'."""
        src = save_weather_source(_custom_source(name="API Cuba Clima"))
        set_active_weather_source(src["_id"])

        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )
            result = await get_active_weather_data(23.1136, -82.3666, 50.0, "CUJAE")

        assert result["provider"] == "API Cuba Clima"
        assert "Open-Meteo" not in result["provider"]

    @pytest.mark.asyncio
    async def test_pronostico_7_dias_generado(self, mongo_db):
        src = save_weather_source(_custom_source())
        set_active_weather_source(src["_id"])

        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )
            result = await get_active_weather_data(23.1136, -82.3666, 50.0, "CUJAE")

        assert len(result["forecast"]) == 7

    @pytest.mark.asyncio
    async def test_pronostico_tiene_produccion_estimada(self, mongo_db):
        """Cada día del pronóstico debe tener predictedProduction calculada."""
        src = save_weather_source(_custom_source())
        set_active_weather_source(src["_id"])

        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )
            result = await get_active_weather_data(23.1136, -82.3666, 50.0, "CUJAE")

        for day in result["forecast"]:
            assert "predictedProduction" in day
            assert day["predictedProduction"] >= 0

    @pytest.mark.asyncio
    async def test_fuente_deshabilitada_retorna_none(self, mongo_db):
        """Una fuente con enabled=False no debe usarse aunque esté marcada activa."""
        src = save_weather_source(_custom_source(enabled=False))
        # Forzamos isActive sin pasar por set_active (que pone enabled=True)
        from app.database import get_database
        from bson import ObjectId
        get_database()["weather_sources"].update_one(
            {"_id": ObjectId(src["_id"])},
            {"$set": {"isActive": True, "enabled": False}},
        )
        result = await get_active_weather_data(23.1, -82.3, 50.0, "CUJAE")
        assert result is None

    @pytest.mark.asyncio
    async def test_cambio_de_fuente_activa_usa_nueva_url(self, mongo_db):
        """Cuando se cambia la fuente activa, el sistema llama a la nueva URL."""
        OTRA_URL = "https://api.otro-servicio.cu/datos"
        src_a = save_weather_source(_custom_source(name="Fuente A"))
        src_b = save_weather_source({
            "name": "Fuente B",
            "baseUrl": OTRA_URL,
            "authType": "none",
            "fieldMapping": _CUSTOM_MAPPING,
            "enabled": True,
            "isActive": False,
        })
        set_active_weather_source(src_a["_id"])
        set_active_weather_source(src_b["_id"])  # cambiamos a B

        with respx.mock:
            route_b = respx.get(OTRA_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )
            result = await get_active_weather_data(23.1, -82.3, 50.0, "CUJAE")

        assert route_b.called
        assert result["provider"] == "Fuente B"

    @pytest.mark.asyncio
    async def test_error_http_propaga_excepcion(self, mongo_db):
        src = save_weather_source(_custom_source())
        set_active_weather_source(src["_id"])

        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(return_value=httpx.Response(503))
            with pytest.raises(Exception):
                await get_active_weather_data(23.1, -82.3, 50.0, "CUJAE")


# ─────────────────────────────────────────────────────────────────────────────
# 3. CADENA DE FALLBACK — get_weather_with_fallback
#    Custom API → Open-Meteo → Datos generados
# ─────────────────────────────────────────────────────────────────────────────

class TestGetWeatherWithFallbackHTTP:
    """
    Prueba el mecanismo de fallback completo.
    El sistema tiene tres niveles:
      1. Fuente personalizada activa (si hay una configurada y funciona)
      2. Open-Meteo (si la fuente personalizada falla o no hay ninguna)
      3. Datos generados internamente (si Open-Meteo también falla)
    """

    @pytest.mark.asyncio
    async def test_fuente_propia_activa_es_usada_primero(self, mongo_db):
        src = save_weather_source(_custom_source(name="Mi API"))
        set_active_weather_source(src["_id"])

        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )
            result = await get_weather_with_fallback(23.1136, -82.3666, 50.0, "CUJAE")

        assert result["provider"] == "Mi API"

    @pytest.mark.asyncio
    async def test_sin_fuente_activa_usa_openmeteo(self, mongo_db):
        """Sin fuente configurada, el sistema hace fallback a Open-Meteo."""
        with respx.mock:
            respx.get(_OPENMETEO_URL).mock(
                return_value=httpx.Response(200, json=_OPENMETEO_PAYLOAD)
            )
            result = await get_weather_with_fallback(23.1136, -82.3666, 50.0, "CUJAE")

        # Open-Meteo es la fuente cuando no hay custom source activa
        assert result is not None
        assert result.get("temperature") is not None

    @pytest.mark.asyncio
    async def test_fuente_propia_falla_usa_openmeteo(self, mongo_db):
        """Si la API propia devuelve error, el sistema recurre a Open-Meteo."""
        src = save_weather_source(_custom_source())
        set_active_weather_source(src["_id"])

        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(return_value=httpx.Response(503))
            respx.get(_OPENMETEO_URL).mock(
                return_value=httpx.Response(200, json=_OPENMETEO_PAYLOAD)
            )
            result = await get_weather_with_fallback(23.1136, -82.3666, 50.0, "CUJAE")

        assert result is not None
        assert result.get("temperature") is not None

    @pytest.mark.asyncio
    async def test_ambas_fuentes_fallan_retorna_datos_generados(self, mongo_db):
        """
        Si tanto la fuente propia como Open-Meteo fallan, el sistema
        genera datos de simulación internos en lugar de lanzar excepción.
        """
        src = save_weather_source(_custom_source())
        set_active_weather_source(src["_id"])

        with respx.mock:
            # Ninguna URL mocked → ambas lanzarán ConnectError
            result = await get_weather_with_fallback(23.1136, -82.3666, 50.0, "CUJAE")

        assert result is not None
        # El proveedor indica que son datos simulados
        assert "simulad" in result.get("provider", "").lower() or result.get("temperature") is not None

    @pytest.mark.asyncio
    async def test_timeout_api_propia_no_bloquea_sistema(self, mongo_db):
        """Un timeout en la fuente propia no debe bloquear: cae a Open-Meteo."""
        src = save_weather_source(_custom_source())
        set_active_weather_source(src["_id"])

        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(
                side_effect=httpx.TimeoutException("timeout")
            )
            respx.get(_OPENMETEO_URL).mock(
                return_value=httpx.Response(200, json=_OPENMETEO_PAYLOAD)
            )
            result = await get_weather_with_fallback(23.1136, -82.3666, 50.0, "CUJAE")

        assert result is not None

    @pytest.mark.asyncio
    async def test_resultado_siempre_tiene_temperatura(self, mongo_db):
        """En cualquier escenario de fallback, debe haber temperatura."""
        # Sin fuentes activas, con Open-Meteo también forzado a fallar
        with respx.mock:
            # Nada mocked: Open-Meteo falla → datos generados
            result = await get_weather_with_fallback(23.1, -82.3, 50.0, "CUJAE")

        assert "temperature" in result
        assert result["temperature"] is not None

    @pytest.mark.asyncio
    async def test_resultado_tiene_pronostico(self, mongo_db):
        """Cualquier nivel de la cadena de fallback produce un pronóstico."""
        with respx.mock:
            result = await get_weather_with_fallback(23.1, -82.3, 50.0, "CUJAE")

        assert "forecast" in result
        assert isinstance(result["forecast"], list)

    @pytest.mark.asyncio
    async def test_fuente_propia_exitosa_no_llama_openmeteo(self, mongo_db):
        """Cuando la fuente propia funciona, Open-Meteo NO debe ser llamada."""
        src = save_weather_source(_custom_source())
        set_active_weather_source(src["_id"])

        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )
            route_openmeteo = respx.get(_OPENMETEO_URL).mock(
                return_value=httpx.Response(200, json=_OPENMETEO_PAYLOAD)
            )
            await get_weather_with_fallback(23.1136, -82.3666, 50.0, "CUJAE")

        assert not route_openmeteo.called


# ─────────────────────────────────────────────────────────────────────────────
# 4. CAPA GRAPHQL — query `weather`
#    Prueba el resolver end-to-end: desde la query GQL hasta el HTTP mock.
# ─────────────────────────────────────────────────────────────────────────────

WEATHER_QUERY = """
query {
  weather {
    temperature
    humidity
    solarRadiation
    cloudCover
    windSpeed
    provider
    locationName
    lastUpdated
    forecast {
      date
      maxTemp
      minTemp
      solarRadiation
      condition
      predictedProduction
    }
  }
}
"""


def _make_gql_ctx():
    return {"current_user": None, "request": None}


def _run_async(coro):
    return asyncio.run(coro)


class TestGraphQLWeatherQueryHTTP:
    """
    Prueba la query `weather` de GraphQL de extremo a extremo:
    GQL resolver → get_weather_with_fallback → fetch HTTP (mocked) → WeatherDataType.
    """

    def test_weather_query_con_fuente_propia_activa(self, mongo_db):
        src = save_weather_source(_custom_source(name="API CUJAE"))
        set_active_weather_source(src["_id"])

        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )

            async def _run():
                return await schema.execute(WEATHER_QUERY, context_value=_make_gql_ctx())

            result = _run_async(_run())

        assert result.errors is None or result.errors == []
        assert result.data["weather"]["temperature"] == pytest.approx(29.3, abs=0.01)

    def test_weather_query_proveedor_es_nombre_fuente_propia(self, mongo_db):
        src = save_weather_source(_custom_source(name="WeatherAPI Cuba"))
        set_active_weather_source(src["_id"])

        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )

            async def _run():
                return await schema.execute(WEATHER_QUERY, context_value=_make_gql_ctx())

            result = _run_async(_run())

        assert result.data["weather"]["provider"] == "WeatherAPI Cuba"

    def test_weather_query_humedad_correcta(self, mongo_db):
        src = save_weather_source(_custom_source())
        set_active_weather_source(src["_id"])

        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )

            async def _run():
                return await schema.execute(WEATHER_QUERY, context_value=_make_gql_ctx())

            result = _run_async(_run())

        assert result.data["weather"]["humidity"] == 73

    def test_weather_query_radiacion_solar_correcta(self, mongo_db):
        src = save_weather_source(_custom_source())
        set_active_weather_source(src["_id"])

        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )

            async def _run():
                return await schema.execute(WEATHER_QUERY, context_value=_make_gql_ctx())

            result = _run_async(_run())

        assert result.data["weather"]["solarRadiation"] == 612

    def test_weather_query_pronostico_7_dias(self, mongo_db):
        src = save_weather_source(_custom_source())
        set_active_weather_source(src["_id"])

        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )

            async def _run():
                return await schema.execute(WEATHER_QUERY, context_value=_make_gql_ctx())

            result = _run_async(_run())

        assert len(result.data["weather"]["forecast"]) == 7

    def test_weather_query_sin_fuente_activa_usa_fallback(self, mongo_db):
        """Sin fuente activa, la query no falla: devuelve datos de fallback."""
        with respx.mock:
            # Ninguna URL mocked: ambas fallan → datos generados
            async def _run():
                return await schema.execute(WEATHER_QUERY, context_value=_make_gql_ctx())

            result = _run_async(_run())

        assert result.data is not None
        assert result.data["weather"]["temperature"] is not None

    def test_weather_query_fuente_propia_con_error_usa_fallback(self, mongo_db):
        """Si la fuente propia devuelve 503, la query no falla: usa fallback."""
        src = save_weather_source(_custom_source())
        set_active_weather_source(src["_id"])

        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(return_value=httpx.Response(503))
            # Open-Meteo no mocked → ConnectError → datos generados

            async def _run():
                return await schema.execute(WEATHER_QUERY, context_value=_make_gql_ctx())

            result = _run_async(_run())

        assert result.data is not None
        assert result.data["weather"] is not None

    def test_weather_query_campo_lastUpdated_presente(self, mongo_db):
        src = save_weather_source(_custom_source())
        set_active_weather_source(src["_id"])

        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )

            async def _run():
                return await schema.execute(WEATHER_QUERY, context_value=_make_gql_ctx())

            result = _run_async(_run())

        assert result.data["weather"]["lastUpdated"] is not None

    def test_weather_query_produccion_estimada_positiva(self, mongo_db):
        """El pronóstico diario debe incluir predictedProduction >= 0."""
        src = save_weather_source(_custom_source())
        set_active_weather_source(src["_id"])

        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )

            async def _run():
                return await schema.execute(WEATHER_QUERY, context_value=_make_gql_ctx())

            result = _run_async(_run())

        for day in result.data["weather"]["forecast"]:
            assert day["predictedProduction"] >= 0

    def test_weather_query_cambio_de_fuente_activa(self, mongo_db):
        """
        Cambiando la fuente activa de A a B, la próxima query usa B.
        Verifica que el switch de API funciona end-to-end.
        """
        OTRA_URL = "https://api.servicioB.cu/clima"
        src_a = save_weather_source(_custom_source(name="Fuente A"))
        src_b = save_weather_source({
            "name": "Fuente B",
            "baseUrl": OTRA_URL,
            "authType": "none",
            "fieldMapping": _CUSTOM_MAPPING,
            "enabled": True,
            "isActive": False,
        })

        set_active_weather_source(src_a["_id"])
        set_active_weather_source(src_b["_id"])  # ahora B es la activa

        with respx.mock:
            route_b = respx.get(OTRA_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )

            async def _run():
                return await schema.execute(WEATHER_QUERY, context_value=_make_gql_ctx())

            result = _run_async(_run())

        assert route_b.called
        assert result.data["weather"]["provider"] == "Fuente B"


# ─────────────────────────────────────────────────────────────────────────────
# 5. sourceError — el error de la fuente activa llega al operador
#    Antes: excepción tragada con "except Exception: pass".
#    Ahora: error capturado, fallback a Open-Meteo, sourceError poblado.
# ─────────────────────────────────────────────────────────────────────────────

WEATHER_QUERY_WITH_ERROR = """
query {
  weather {
    temperature provider sourceError
    forecast { date predictedProduction }
  }
}
"""


class TestSourceErrorPropagation:
    """
    Verifica que cuando una fuente activa falla el sistema:
      - No swallow la excepción silenciosamente.
      - Continúa con Open-Meteo como respaldo (el dashboard no queda en blanco).
      - Expone sourceError con el mensaje exacto del fallo.
    """

    @pytest.mark.asyncio
    async def test_http_error_popula_source_error(self, mongo_db):
        """HTTP 401 de la fuente activa → sourceError contiene el mensaje."""
        src = save_weather_source(_custom_source(name="API Fallida"))
        set_active_weather_source(src["_id"])

        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(return_value=httpx.Response(401))
            respx.get(_OPENMETEO_URL).mock(
                return_value=httpx.Response(200, json=_OPENMETEO_PAYLOAD)
            )
            result = await get_weather_with_fallback(23.1136, -82.3666, 50.0, "CUJAE")

        assert result["sourceError"] is not None
        assert "API Fallida" in result["sourceError"]
        assert "401" in result["sourceError"]

    @pytest.mark.asyncio
    async def test_fieldmapping_incompleto_popula_source_error(self, mongo_db):
        """
        Fuente activa con fieldMapping sin forecastArrayPath →
        map_payload_to_weather_data lanza ValueError →
        get_weather_with_fallback lo captura y lo expone en sourceError.
        """
        src = save_weather_source({
            "name": "API Sin Pronóstico",
            "baseUrl": _CUSTOM_API_URL,
            "authType": "none",
            "fieldMapping": {
                "temperaturePath":    "datos.temp_c",
                "solarRadiationPath": "datos.irradiancia_wm2",
                "cloudCoverPath":     "datos.nubosidad_pct",
                "humidityPath":       "datos.humedad_pct",
                "windSpeedPath":      "datos.viento_kmh",
                # forecastArrayPath, forecastDatePath, forecastMaxTempPath,
                # forecastMinTempPath, forecastSolarRadiationPath,
                # forecastCloudCoverPath → todos ausentes
            },
            "enabled": True,
            "isActive": False,
        })
        set_active_weather_source(src["_id"])

        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )
            respx.get(_OPENMETEO_URL).mock(
                return_value=httpx.Response(200, json=_OPENMETEO_PAYLOAD)
            )
            result = await get_weather_with_fallback(23.1136, -82.3666, 50.0, "CUJAE")

        assert result["sourceError"] is not None
        assert "API Sin Pronóstico" in result["sourceError"]
        # El mensaje del ValueError de map_payload_to_weather_data debe llegar
        assert "falt" in result["sourceError"].lower() or "mapeo" in result["sourceError"].lower()

    @pytest.mark.asyncio
    async def test_fuente_sin_forecast_data_popula_source_error(self, mongo_db):
        """
        Fuente con forecastArrayPath configurado pero el payload no contiene
        la ruta indicada → lista vacía → ValueError → sourceError.
        """
        src = save_weather_source({
            "name": "API Ruta Inválida",
            "baseUrl": _CUSTOM_API_URL,
            "authType": "none",
            "fieldMapping": {
                **_CUSTOM_MAPPING,
                "forecastArrayPath": "ruta.que.no.existe",
            },
            "enabled": True,
            "isActive": False,
        })
        set_active_weather_source(src["_id"])

        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )
            respx.get(_OPENMETEO_URL).mock(
                return_value=httpx.Response(200, json=_OPENMETEO_PAYLOAD)
            )
            result = await get_weather_with_fallback(23.1136, -82.3666, 50.0, "CUJAE")

        assert result["sourceError"] is not None
        assert "API Ruta Inválida" in result["sourceError"]

    @pytest.mark.asyncio
    async def test_timeout_popula_source_error(self, mongo_db):
        """Timeout de la fuente activa → sourceError con nombre de la fuente."""
        src = save_weather_source(_custom_source(name="API Lenta"))
        set_active_weather_source(src["_id"])

        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(
                side_effect=httpx.TimeoutException("timeout simulado")
            )
            respx.get(_OPENMETEO_URL).mock(
                return_value=httpx.Response(200, json=_OPENMETEO_PAYLOAD)
            )
            result = await get_weather_with_fallback(23.1136, -82.3666, 50.0, "CUJAE")

        assert result["sourceError"] is not None
        assert "API Lenta" in result["sourceError"]

    @pytest.mark.asyncio
    async def test_fuente_exitosa_no_tiene_source_error(self, mongo_db):
        """Cuando la fuente funciona correctamente, sourceError debe ser None."""
        src = save_weather_source(_custom_source())
        set_active_weather_source(src["_id"])

        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )
            result = await get_weather_with_fallback(23.1136, -82.3666, 50.0, "CUJAE")

        assert result.get("sourceError") is None

    @pytest.mark.asyncio
    async def test_sin_fuente_activa_no_tiene_source_error(self, mongo_db):
        """Sin fuente personalizada activa, sourceError es None (Open-Meteo funciona)."""
        with respx.mock:
            respx.get(_OPENMETEO_URL).mock(
                return_value=httpx.Response(200, json=_OPENMETEO_PAYLOAD)
            )
            result = await get_weather_with_fallback(23.1136, -82.3666, 50.0, "CUJAE")

        assert result.get("sourceError") is None

    @pytest.mark.asyncio
    async def test_source_error_no_impide_recibir_datos(self, mongo_db):
        """
        Aunque la fuente activa falle, el sistema entrega datos de Open-Meteo.
        El operador ve el error Y los datos, no una pantalla en blanco.
        """
        src = save_weather_source(_custom_source(name="API Caída"))
        set_active_weather_source(src["_id"])

        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(return_value=httpx.Response(503))
            respx.get(_OPENMETEO_URL).mock(
                return_value=httpx.Response(200, json=_OPENMETEO_PAYLOAD)
            )
            result = await get_weather_with_fallback(23.1136, -82.3666, 50.0, "CUJAE")

        assert result["temperature"] is not None
        assert result["sourceError"] is not None

    def test_graphql_expone_source_error_cuando_fuente_falla(self, mongo_db):
        """
        La query GQL `weather { sourceError }` expone el error al frontend.
        Antes del fix: sourceError no existía → operador no se enteraba.
        """
        src = save_weather_source(_custom_source(name="API con 401"))
        set_active_weather_source(src["_id"])

        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(return_value=httpx.Response(401))
            respx.get(_OPENMETEO_URL).mock(
                return_value=httpx.Response(200, json=_OPENMETEO_PAYLOAD)
            )

            async def _run():
                return await schema.execute(
                    WEATHER_QUERY_WITH_ERROR, context_value=_make_gql_ctx()
                )

            result = _run_async(_run())

        assert result.errors is None or result.errors == []
        weather = result.data["weather"]
        assert weather["sourceError"] is not None
        assert "API con 401" in weather["sourceError"]
        assert "401" in weather["sourceError"]
        # El operador sabe qué fuente falló y por qué,
        # y además tiene datos de Open-Meteo para seguir operando.
        assert weather["temperature"] is not None

    def test_graphql_source_error_es_none_cuando_todo_va_bien(self, mongo_db):
        """sourceError = null en la respuesta GraphQL cuando no hay error."""
        src = save_weather_source(_custom_source())
        set_active_weather_source(src["_id"])

        with respx.mock:
            respx.get(_CUSTOM_API_URL).mock(
                return_value=httpx.Response(200, json=_CUSTOM_PAYLOAD)
            )

            async def _run():
                return await schema.execute(
                    WEATHER_QUERY_WITH_ERROR, context_value=_make_gql_ctx()
                )

            result = _run_async(_run())

        assert result.data["weather"]["sourceError"] is None
