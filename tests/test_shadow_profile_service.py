"""
Pruebas de integración para shadow_profile_service.

Cubre guardar y obtener el perfil de sombras horario, validaciones de
slots, cálculo de promedios derivados y comportamiento del singleton.
"""
import pytest

from app.services.shadow_profile_service import get_shadow_profile, save_shadow_profile


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _slot(hour: int, shadow: float, override=None) -> dict:
    return {"hour": hour, "shadowPct": shadow, "prodOverride": override}


def _basic_slots():
    return [
        _slot(8, 0.0, 100.0),
        _slot(12, 10.0, 90.0),
        _slot(16, 20.0, None),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Obtención sin datos
# ─────────────────────────────────────────────────────────────────────────────

class TestObtenerPerfilSombra:

    def test_sin_datos_retorna_none(self, mongo_db):
        assert get_shadow_profile() is None

    def test_despues_de_guardar_devuelve_perfil(self, mongo_db):
        save_shadow_profile(_basic_slots())
        result = get_shadow_profile()
        assert result is not None

    def test_estructura_devuelta_tiene_campos_requeridos(self, mongo_db):
        save_shadow_profile(_basic_slots())
        result = get_shadow_profile()
        assert "slots" in result
        assert "avgShadow" in result
        assert "avgProd" in result
        assert "updatedAt" in result


# ─────────────────────────────────────────────────────────────────────────────
# Guardar perfil
# ─────────────────────────────────────────────────────────────────────────────

class TestGuardarPerfilSombra:

    def test_guardar_persiste_slots(self, mongo_db):
        save_shadow_profile(_basic_slots())
        result = get_shadow_profile()
        assert len(result["slots"]) == 3

    def test_guardar_calcula_avg_shadow_correctamente(self, mongo_db):
        slots = [
            _slot(8, 0.0),
            _slot(12, 20.0),
            _slot(16, 40.0),
        ]
        result = save_shadow_profile(slots)
        expected_avg = (0.0 + 20.0 + 40.0) / 3
        assert result["avgShadow"] == pytest.approx(expected_avg, abs=0.01)

    def test_guardar_calcula_avg_prod_con_override(self, mongo_db):
        slots = [
            _slot(8, 0.0, 100.0),
            _slot(12, 10.0, 90.0),
        ]
        result = save_shadow_profile(slots)
        expected_avg_prod = (100.0 + 90.0) / 2
        assert result["avgProd"] == pytest.approx(expected_avg_prod, abs=0.01)

    def test_guardar_calcula_avg_prod_sin_override(self, mongo_db):
        slots = [
            _slot(8, 20.0),   # prod = max(0, 100 - 20) = 80
            _slot(12, 40.0),  # prod = max(0, 100 - 40) = 60
        ]
        result = save_shadow_profile(slots)
        expected_avg_prod = (80.0 + 60.0) / 2
        assert result["avgProd"] == pytest.approx(expected_avg_prod, abs=0.01)

    def test_guardar_retorna_updated_at(self, mongo_db):
        result = save_shadow_profile(_basic_slots())
        assert result["updatedAt"] is not None

    def test_segunda_llamada_sobreescribe_perfil(self, mongo_db):
        save_shadow_profile([_slot(8, 0.0)])
        save_shadow_profile([_slot(10, 50.0), _slot(14, 30.0)])
        result = get_shadow_profile()
        assert len(result["slots"]) == 2

    # ── Validaciones ──────────────────────────────────────────────────────────

    def test_slots_vacios_lanza_error(self, mongo_db):
        with pytest.raises(ValueError, match="[Ff]ranja|al menos"):
            save_shadow_profile([])

    def test_hora_invalida_mayor_que_23_lanza_error(self, mongo_db):
        with pytest.raises(ValueError, match="[Hh]ora"):
            save_shadow_profile([_slot(24, 10.0)])

    def test_hora_invalida_negativa_lanza_error(self, mongo_db):
        with pytest.raises(ValueError, match="[Hh]ora"):
            save_shadow_profile([_slot(-1, 10.0)])

    def test_shadow_pct_mayor_que_100_lanza_error(self, mongo_db):
        with pytest.raises(ValueError, match="sombra"):
            save_shadow_profile([_slot(10, 101.0)])

    def test_shadow_pct_negativo_lanza_error(self, mongo_db):
        with pytest.raises(ValueError, match="sombra"):
            save_shadow_profile([_slot(10, -1.0)])

    def test_prod_override_mayor_que_100_lanza_error(self, mongo_db):
        with pytest.raises(ValueError, match="producci"):
            save_shadow_profile([_slot(10, 10.0, 101.0)])

    def test_prod_override_negativo_lanza_error(self, mongo_db):
        with pytest.raises(ValueError, match="producci"):
            save_shadow_profile([_slot(10, 10.0, -5.0)])

    # ── Valores límite válidos ────────────────────────────────────────────────

    def test_shadow_0_y_100_son_validos(self, mongo_db):
        slots = [_slot(6, 0.0), _slot(20, 100.0)]
        result = save_shadow_profile(slots)
        shadow_vals = {s["shadowPct"] for s in result["slots"]}
        assert 0.0 in shadow_vals
        assert 100.0 in shadow_vals

    def test_hora_0_y_23_son_validas(self, mongo_db):
        slots = [_slot(0, 0.0), _slot(23, 0.0)]
        result = save_shadow_profile(slots)
        assert len(result["slots"]) == 2

    def test_prod_override_0_y_100_son_validos(self, mongo_db):
        slots = [_slot(10, 0.0, 0.0), _slot(14, 0.0, 100.0)]
        result = save_shadow_profile(slots)
        overrides = {s.get("prodOverride") for s in result["slots"]}
        assert 0.0 in overrides
        assert 100.0 in overrides

    def test_prod_override_none_es_valido(self, mongo_db):
        slots = [_slot(10, 5.0, None)]
        result = save_shadow_profile(slots)
        assert result["slots"][0]["prodOverride"] is None
