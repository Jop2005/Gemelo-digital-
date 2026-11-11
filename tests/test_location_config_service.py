"""
Pruebas de integración para location_config_service.

Cubre guardar y obtener la configuración de ubicación geográfica,
incluyendo validaciones de latitud/longitud y el fallback al valor
por defecto cuando no hay datos en la BD.
"""
import pytest

from app.services.location_config_service import get_location_config, save_location_config

# Coordenadas del sistema de referencia (La Habana, Cuba)
_LA_HABANA = (23.1136, -82.3666, "La Habana, Cuba")


# ─────────────────────────────────────────────────────────────────────────────
# Obtención con fallback
# ─────────────────────────────────────────────────────────────────────────────

class TestObtenerConfiguracionUbicacion:

    def test_sin_datos_devuelve_config_por_defecto(self, mongo_db):
        config = get_location_config()
        assert "lat" in config
        assert "lon" in config
        assert "name" in config

    def test_fallback_lat_es_float(self, mongo_db):
        config = get_location_config()
        assert isinstance(config["lat"], float)

    def test_fallback_lon_es_float(self, mongo_db):
        config = get_location_config()
        assert isinstance(config["lon"], float)

    def test_despues_de_guardar_devuelve_dato_guardado(self, mongo_db):
        save_location_config(*_LA_HABANA)
        config = get_location_config()
        assert config["lat"] == pytest.approx(23.1136)
        assert config["lon"] == pytest.approx(-82.3666)
        assert config["name"] == "La Habana, Cuba"


# ─────────────────────────────────────────────────────────────────────────────
# Guardar configuración
# ─────────────────────────────────────────────────────────────────────────────

class TestGuardarConfiguracionUbicacion:

    def test_guardar_retorna_lat_correcta(self, mongo_db):
        result = save_location_config(23.1136, -82.3666, "La Habana")
        assert result["lat"] == pytest.approx(23.1136)

    def test_guardar_retorna_lon_correcta(self, mongo_db):
        result = save_location_config(23.1136, -82.3666, "La Habana")
        assert result["lon"] == pytest.approx(-82.3666)

    def test_guardar_retorna_nombre_correcto(self, mongo_db):
        result = save_location_config(23.1136, -82.3666, "La Habana")
        assert result["name"] == "La Habana"

    def test_guardar_retorna_updated_at(self, mongo_db):
        result = save_location_config(23.1136, -82.3666, "La Habana")
        assert result["updatedAt"] is not None

    def test_guardar_redondea_coordenadas_a_6_decimales(self, mongo_db):
        result = save_location_config(23.11363333, -82.36663333, "X")
        assert result["lat"] == pytest.approx(23.113633, abs=1e-5)
        assert result["lon"] == pytest.approx(-82.366633, abs=1e-5)

    def test_guardar_elimina_espacios_del_nombre(self, mongo_db):
        result = save_location_config(23.0, -82.0, "  La Habana  ")
        assert result["name"] == "La Habana"

    def test_segunda_llamada_sobreescribe_la_primera(self, mongo_db):
        save_location_config(23.0, -82.0, "Lugar A")
        result = save_location_config(22.0, -81.0, "Lugar B")
        assert result["name"] == "Lugar B"
        # Solo existe un documento singleton
        config = get_location_config()
        assert config["name"] == "Lugar B"

    # ── Validaciones ──────────────────────────────────────────────────────────

    def test_lat_mayor_que_90_lanza_error(self, mongo_db):
        with pytest.raises(ValueError, match="[Ll]atitud"):
            save_location_config(91.0, 0.0, "X")

    def test_lat_menor_que_menos90_lanza_error(self, mongo_db):
        with pytest.raises(ValueError, match="[Ll]atitud"):
            save_location_config(-91.0, 0.0, "X")

    def test_lon_mayor_que_180_lanza_error(self, mongo_db):
        with pytest.raises(ValueError, match="[Ll]ongitud"):
            save_location_config(0.0, 181.0, "X")

    def test_lon_menor_que_menos180_lanza_error(self, mongo_db):
        with pytest.raises(ValueError, match="[Ll]ongitud"):
            save_location_config(0.0, -181.0, "X")

    def test_nombre_vacio_lanza_error(self, mongo_db):
        with pytest.raises(ValueError, match="[Nn]ombre"):
            save_location_config(23.0, -82.0, "")

    def test_nombre_solo_espacios_lanza_error(self, mongo_db):
        with pytest.raises(ValueError, match="[Nn]ombre"):
            save_location_config(23.0, -82.0, "   ")

    # ── Límites exactos válidos ───────────────────────────────────────────────

    def test_lat_exactamente_90_es_valida(self, mongo_db):
        result = save_location_config(90.0, 0.0, "Polo Norte")
        assert result["lat"] == pytest.approx(90.0)

    def test_lat_exactamente_menos90_es_valida(self, mongo_db):
        result = save_location_config(-90.0, 0.0, "Polo Sur")
        assert result["lat"] == pytest.approx(-90.0)

    def test_lon_exactamente_180_es_valida(self, mongo_db):
        result = save_location_config(0.0, 180.0, "Línea de fecha")
        assert result["lon"] == pytest.approx(180.0)
