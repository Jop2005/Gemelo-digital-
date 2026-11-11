"""
Pruebas de integración para inverter_service.

Cubre el CRUD completo de inversores con validaciones de negocio:
crear, listar, obtener, actualizar, eliminar y manejo de errores.
"""
import pytest
from bson import ObjectId

from app.services.inverter_service import (
    create_inverter,
    delete_inverter,
    get_inverter,
    list_inverters,
    update_inverter,
)

# ─────────────────────────────────────────────────────────────────────────────
# Payloads de prueba
# ─────────────────────────────────────────────────────────────────────────────

INVERTER_PAYLOAD = {
    "manufacturer": "SMA",
    "model": "Sunny Tripower 50000TL",
    "ratedPowerKw": 50.0,
    "quantity": 1,
    "efficiencyPercent": 98.3,
}


# ─────────────────────────────────────────────────────────────────────────────
# Creación
# ─────────────────────────────────────────────────────────────────────────────

class TestCrearInversor:

    def test_crear_retorna_documento_con_id(self, mongo_db):
        inv = create_inverter(INVERTER_PAYLOAD.copy())
        assert "_id" in inv
        assert isinstance(inv["_id"], str)

    def test_crear_persiste_fabricante(self, mongo_db):
        inv = create_inverter(INVERTER_PAYLOAD.copy())
        assert inv["manufacturer"] == "SMA"

    def test_crear_persiste_potencia(self, mongo_db):
        inv = create_inverter(INVERTER_PAYLOAD.copy())
        assert inv["ratedPowerKw"] == pytest.approx(50.0)

    def test_crear_persiste_eficiencia(self, mongo_db):
        inv = create_inverter(INVERTER_PAYLOAD.copy())
        assert inv["efficiencyPercent"] == pytest.approx(98.3)

    def test_crear_persiste_cantidad(self, mongo_db):
        inv = create_inverter(INVERTER_PAYLOAD.copy())
        assert inv["quantity"] == 1

    def test_crear_timestamps_presentes(self, mongo_db):
        inv = create_inverter(INVERTER_PAYLOAD.copy())
        assert inv["createdAt"] is not None
        assert inv["updatedAt"] is not None

    def test_crear_sin_eficiencia_acepta_none(self, mongo_db):
        payload = {k: v for k, v in INVERTER_PAYLOAD.items() if k != "efficiencyPercent"}
        inv = create_inverter(payload)
        assert inv["efficiencyPercent"] is None

    def test_crear_sin_modelo_acepta_none(self, mongo_db):
        payload = {k: v for k, v in INVERTER_PAYLOAD.items() if k != "model"}
        inv = create_inverter(payload)
        assert inv["model"] is None

    def test_crear_sin_fabricante_lanza_error(self, mongo_db):
        payload = {**INVERTER_PAYLOAD, "manufacturer": ""}
        with pytest.raises(ValueError):
            create_inverter(payload)

    def test_crear_fabricante_none_lanza_error(self, mongo_db):
        payload = {**INVERTER_PAYLOAD, "manufacturer": None}
        with pytest.raises(ValueError):
            create_inverter(payload)

    def test_crear_potencia_negativa_lanza_error(self, mongo_db):
        payload = {**INVERTER_PAYLOAD, "ratedPowerKw": -5.0}
        with pytest.raises(ValueError):
            create_inverter(payload)

    def test_crear_potencia_cero_lanza_error(self, mongo_db):
        payload = {**INVERTER_PAYLOAD, "ratedPowerKw": 0}
        with pytest.raises(ValueError):
            create_inverter(payload)

    def test_crear_cantidad_cero_lanza_error(self, mongo_db):
        payload = {**INVERTER_PAYLOAD, "quantity": 0}
        with pytest.raises(ValueError):
            create_inverter(payload)

    def test_crear_cantidad_negativa_lanza_error(self, mongo_db):
        payload = {**INVERTER_PAYLOAD, "quantity": -1}
        with pytest.raises(ValueError):
            create_inverter(payload)

    def test_multiples_inversores_tienen_ids_distintos(self, mongo_db):
        inv1 = create_inverter({**INVERTER_PAYLOAD, "manufacturer": "SMA"})
        inv2 = create_inverter({**INVERTER_PAYLOAD, "manufacturer": "Fronius"})
        assert inv1["_id"] != inv2["_id"]


# ─────────────────────────────────────────────────────────────────────────────
# Listado
# ─────────────────────────────────────────────────────────────────────────────

class TestListarInversores:

    def test_listar_vacio_retorna_lista_vacia(self, mongo_db):
        assert list_inverters() == []

    def test_listar_devuelve_inversor_creado(self, mongo_db):
        create_inverter(INVERTER_PAYLOAD.copy())
        result = list_inverters()
        assert len(result) == 1

    def test_listar_devuelve_todos_los_inversores(self, mongo_db):
        create_inverter({**INVERTER_PAYLOAD, "manufacturer": "SMA"})
        create_inverter({**INVERTER_PAYLOAD, "manufacturer": "Fronius"})
        create_inverter({**INVERTER_PAYLOAD, "manufacturer": "Huawei"})
        result = list_inverters()
        assert len(result) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Obtención por ID
# ─────────────────────────────────────────────────────────────────────────────

class TestObtenerInversor:

    def test_obtener_por_id_existente(self, mongo_db):
        created = create_inverter(INVERTER_PAYLOAD.copy())
        found = get_inverter(created["_id"])
        assert found is not None
        assert found["_id"] == created["_id"]

    def test_obtener_por_id_trae_datos_correctos(self, mongo_db):
        created = create_inverter(INVERTER_PAYLOAD.copy())
        found = get_inverter(created["_id"])
        assert found["manufacturer"] == "SMA"
        assert found["ratedPowerKw"] == pytest.approx(50.0)

    def test_obtener_id_inexistente_retorna_none(self, mongo_db):
        result = get_inverter(str(ObjectId()))
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# Actualización
# ─────────────────────────────────────────────────────────────────────────────

class TestActualizarInversor:

    def test_actualizar_fabricante(self, mongo_db):
        created = create_inverter(INVERTER_PAYLOAD.copy())
        updated = update_inverter(created["_id"], {"manufacturer": "Fronius"})
        assert updated["manufacturer"] == "Fronius"

    def test_actualizar_potencia(self, mongo_db):
        created = create_inverter(INVERTER_PAYLOAD.copy())
        updated = update_inverter(created["_id"], {"ratedPowerKw": 60.0})
        assert updated["ratedPowerKw"] == pytest.approx(60.0)

    def test_actualizar_eficiencia(self, mongo_db):
        created = create_inverter(INVERTER_PAYLOAD.copy())
        updated = update_inverter(created["_id"], {"efficiencyPercent": 99.0})
        assert updated["efficiencyPercent"] == pytest.approx(99.0)

    def test_actualizar_eficiencia_a_none(self, mongo_db):
        created = create_inverter(INVERTER_PAYLOAD.copy())
        updated = update_inverter(created["_id"], {"efficiencyPercent": None})
        assert updated["efficiencyPercent"] is None

    def test_actualizar_no_modifica_campos_no_enviados(self, mongo_db):
        created = create_inverter(INVERTER_PAYLOAD.copy())
        update_inverter(created["_id"], {"manufacturer": "Fronius"})
        found = get_inverter(created["_id"])
        assert found["ratedPowerKw"] == pytest.approx(50.0)
        assert found["quantity"] == 1

    def test_actualizar_payload_vacio_retorna_inversor_sin_cambios(self, mongo_db):
        created = create_inverter(INVERTER_PAYLOAD.copy())
        result = update_inverter(created["_id"], {})
        assert result["manufacturer"] == "SMA"

    def test_actualizar_id_inexistente_retorna_none(self, mongo_db):
        result = update_inverter(str(ObjectId()), {"manufacturer": "X"})
        assert result is None

    def test_actualizar_fabricante_a_vacio_lanza_error(self, mongo_db):
        created = create_inverter(INVERTER_PAYLOAD.copy())
        with pytest.raises(ValueError):
            update_inverter(created["_id"], {"manufacturer": ""})

    def test_actualizar_potencia_negativa_lanza_error(self, mongo_db):
        created = create_inverter(INVERTER_PAYLOAD.copy())
        with pytest.raises(ValueError):
            update_inverter(created["_id"], {"ratedPowerKw": -10.0})


# ─────────────────────────────────────────────────────────────────────────────
# Eliminación
# ─────────────────────────────────────────────────────────────────────────────

class TestEliminarInversor:

    def test_eliminar_existente_retorna_true(self, mongo_db):
        created = create_inverter(INVERTER_PAYLOAD.copy())
        assert delete_inverter(created["_id"]) is True

    def test_eliminar_quita_de_la_lista(self, mongo_db):
        created = create_inverter(INVERTER_PAYLOAD.copy())
        delete_inverter(created["_id"])
        assert list_inverters() == []

    def test_eliminar_uno_no_afecta_otros(self, mongo_db):
        inv1 = create_inverter({**INVERTER_PAYLOAD, "manufacturer": "SMA"})
        create_inverter({**INVERTER_PAYLOAD, "manufacturer": "Fronius"})
        delete_inverter(inv1["_id"])
        remaining = list_inverters()
        assert len(remaining) == 1
        assert remaining[0]["manufacturer"] == "Fronius"

    def test_eliminar_id_inexistente_retorna_false(self, mongo_db):
        assert delete_inverter(str(ObjectId())) is False
