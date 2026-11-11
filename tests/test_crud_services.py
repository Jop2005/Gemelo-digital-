"""
Pruebas de integración para los servicios CRUD de equipos.

Usan la fixture `mongo_db` de conftest.py, que sustituye la base de datos real
por una instancia MongoDB completamente en memoria (mongomock). Los datos no
persisten entre tests y la base de datos de producción nunca se toca.

Cubre:
- panel_service: crear, listar, obtener, actualizar, eliminar, validaciones
- battery_service: crear, listar, obtener, actualizar, eliminar, validaciones
- appliance_service: crear, listar, obtener, actualizar, eliminar
"""
import pytest
from app.services.panel_service import (
    create_panel,
    list_panels,
    get_panel,
    update_panel,
    delete_panel,
)
from app.services.battery_service import (
    create_battery,
    list_batteries,
    get_battery,
    update_battery,
    delete_battery,
)
from app.services.appliance_service import (
    create_appliance,
    list_appliances,
    get_appliance,
    update_appliance,
    delete_appliance,
)
from app.services.panel_cleanliness_service import (
    save_cleanliness_reading,
    get_latest_cleanliness,
    list_cleanliness_readings,
)


# ─────────────────────────────────────────────────────────────────────────────
# Payloads de prueba
# ─────────────────────────────────────────────────────────────────────────────

PANEL_PAYLOAD = {
    "manufacturer": "Longi Solar",
    "model": "Hi-MO 6",
    "ratedPowerKw": 0.54,
    "quantity": 93,
    "tiltDegrees": 15.0,
    "orientation": "Sur",
    "efficiencyPercent": 22.0,
    "areaM2": 2.56,
}

BATTERY_PAYLOAD = {
    "manufacturer": "CATL",
    "model": "EnerOne",
    "capacityKwh": 100.0,
    "quantity": 1,
    "maxDepthOfDischargePercent": 90.0,
    "chargeRateKw": 25.0,
    "dischargeRateKw": 25.0,
    "efficiencyPercent": 96.0,
}

APPLIANCE_PAYLOAD = {
    "name": "Aire acondicionado",
    "category": "Climatización",
    "modes": [
        {"name": "Frío", "averagePowerW": 1200.0, "maxPowerW": 1800.0},
    ],
    "dailyUsageHours": 8.0,
    "quantity": 2,
}


# ─────────────────────────────────────────────────────────────────────────────
# Panel service
# ─────────────────────────────────────────────────────────────────────────────

class TestPanelService:

    def test_crear_panel_devuelve_documento_con_id(self, mongo_db):
        panel = create_panel(PANEL_PAYLOAD.copy())
        assert "_id" in panel
        assert isinstance(panel["_id"], str)

    def test_crear_panel_persiste_campos_correctamente(self, mongo_db):
        panel = create_panel(PANEL_PAYLOAD.copy())
        assert panel["manufacturer"] == "Longi Solar"
        assert panel["ratedPowerKw"] == pytest.approx(0.54)
        assert panel["quantity"] == 93

    def test_listar_devuelve_panel_creado(self, mongo_db):
        create_panel(PANEL_PAYLOAD.copy())
        panels = list_panels()
        assert len(panels) == 1

    def test_listar_sin_datos_devuelve_lista_vacia(self, mongo_db):
        assert list_panels() == []

    def test_obtener_panel_por_id(self, mongo_db):
        created = create_panel(PANEL_PAYLOAD.copy())
        found = get_panel(created["_id"])
        assert found is not None
        assert found["_id"] == created["_id"]

    def test_obtener_id_inexistente_devuelve_none(self, mongo_db):
        from bson import ObjectId
        result = get_panel(str(ObjectId()))
        assert result is None

    def test_actualizar_campo_manufacturer(self, mongo_db):
        created = create_panel(PANEL_PAYLOAD.copy())
        updated = update_panel(created["_id"], {"manufacturer": "Canadian Solar"})
        assert updated["manufacturer"] == "Canadian Solar"

    def test_actualizar_no_modifica_campos_no_enviados(self, mongo_db):
        created = create_panel(PANEL_PAYLOAD.copy())
        update_panel(created["_id"], {"manufacturer": "JA Solar"})
        found = get_panel(created["_id"])
        assert found["ratedPowerKw"] == pytest.approx(0.54)
        assert found["quantity"] == 93

    def test_eliminar_panel_existente_retorna_true(self, mongo_db):
        created = create_panel(PANEL_PAYLOAD.copy())
        result = delete_panel(created["_id"])
        assert result is True

    def test_eliminar_panel_lo_quita_de_la_lista(self, mongo_db):
        created = create_panel(PANEL_PAYLOAD.copy())
        delete_panel(created["_id"])
        assert list_panels() == []

    def test_eliminar_id_inexistente_retorna_false(self, mongo_db):
        from bson import ObjectId
        result = delete_panel(str(ObjectId()))
        assert result is False

    def test_crear_sin_manufacturer_lanza_error(self, mongo_db):
        payload = {**PANEL_PAYLOAD, "manufacturer": None}
        with pytest.raises(ValueError):
            create_panel(payload)

    def test_crear_con_potencia_negativa_lanza_error(self, mongo_db):
        payload = {**PANEL_PAYLOAD, "ratedPowerKw": -1.0}
        with pytest.raises(ValueError):
            create_panel(payload)

    def test_crear_con_cantidad_cero_lanza_error(self, mongo_db):
        payload = {**PANEL_PAYLOAD, "quantity": 0}
        with pytest.raises(ValueError):
            create_panel(payload)

    def test_multiples_paneles_son_independientes(self, mongo_db):
        p1 = create_panel({**PANEL_PAYLOAD, "manufacturer": "Longi"})
        p2 = create_panel({**PANEL_PAYLOAD, "manufacturer": "JA Solar"})
        assert p1["_id"] != p2["_id"]
        assert len(list_panels()) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Battery service
# ─────────────────────────────────────────────────────────────────────────────

class TestBatteryService:

    def test_crear_bateria_devuelve_documento_con_id(self, mongo_db):
        battery = create_battery(BATTERY_PAYLOAD.copy())
        assert "_id" in battery
        assert isinstance(battery["_id"], str)

    def test_crear_bateria_persiste_capacidad(self, mongo_db):
        battery = create_battery(BATTERY_PAYLOAD.copy())
        assert battery["capacityKwh"] == pytest.approx(100.0)

    def test_listar_devuelve_bateria_creada(self, mongo_db):
        create_battery(BATTERY_PAYLOAD.copy())
        assert len(list_batteries()) == 1

    def test_obtener_bateria_por_id(self, mongo_db):
        created = create_battery(BATTERY_PAYLOAD.copy())
        found = get_battery(created["_id"])
        assert found is not None
        assert found["manufacturer"] == "CATL"

    def test_actualizar_capacidad(self, mongo_db):
        created = create_battery(BATTERY_PAYLOAD.copy())
        updated = update_battery(created["_id"], {"capacityKwh": 200.0})
        assert updated["capacityKwh"] == pytest.approx(200.0)

    def test_eliminar_bateria(self, mongo_db):
        created = create_battery(BATTERY_PAYLOAD.copy())
        assert delete_battery(created["_id"]) is True
        assert list_batteries() == []

    def test_crear_sin_fabricante_lanza_error(self, mongo_db):
        payload = {**BATTERY_PAYLOAD, "manufacturer": ""}
        with pytest.raises(ValueError):
            create_battery(payload)

    def test_crear_con_capacidad_cero_lanza_error(self, mongo_db):
        payload = {**BATTERY_PAYLOAD, "capacityKwh": 0}
        with pytest.raises(ValueError):
            create_battery(payload)

    def test_campos_opcionales_pueden_ser_none(self, mongo_db):
        payload = {**BATTERY_PAYLOAD}
        payload.pop("chargeRateKw")
        payload.pop("dischargeRateKw")
        battery = create_battery(payload)
        assert battery["chargeRateKw"] is None
        assert battery["dischargeRateKw"] is None


# ─────────────────────────────────────────────────────────────────────────────
# Appliance service
# ─────────────────────────────────────────────────────────────────────────────

class TestApplianceService:

    def test_crear_electrodomestico_devuelve_id(self, mongo_db):
        appliance = create_appliance(APPLIANCE_PAYLOAD.copy())
        assert "_id" in appliance

    def test_crear_electrodomestico_persiste_nombre(self, mongo_db):
        appliance = create_appliance(APPLIANCE_PAYLOAD.copy())
        assert appliance["name"] == "Aire acondicionado"

    def test_listar_devuelve_electrodomestico_creado(self, mongo_db):
        create_appliance(APPLIANCE_PAYLOAD.copy())
        assert len(list_appliances()) == 1

    def test_obtener_electrodomestico_por_id(self, mongo_db):
        created = create_appliance(APPLIANCE_PAYLOAD.copy())
        found = get_appliance(created["_id"])
        assert found is not None
        assert found["category"] == "Climatización"

    def test_actualizar_nombre(self, mongo_db):
        created = create_appliance(APPLIANCE_PAYLOAD.copy())
        updated = update_appliance(created["_id"], {"name": "AC Inverter"})
        assert updated["name"] == "AC Inverter"

    def test_eliminar_electrodomestico(self, mongo_db):
        created = create_appliance(APPLIANCE_PAYLOAD.copy())
        assert delete_appliance(created["_id"]) is True
        assert list_appliances() == []

    def test_crear_sin_nombre_lanza_error(self, mongo_db):
        payload = {**APPLIANCE_PAYLOAD, "name": ""}
        with pytest.raises(ValueError):
            create_appliance(payload)


# ─────────────────────────────────────────────────────────────────────────────
# Panel cleanliness service (registro histórico de limpieza de paneles)
# ─────────────────────────────────────────────────────────────────────────────

class TestPanelCleanlinessService:

    def test_guardar_lectura_devuelve_documento_con_id(self, mongo_db):
        reading = save_cleanliness_reading("limpio", 92.5, 7.5)
        assert "_id" in reading
        assert isinstance(reading["_id"], str)

    def test_guardar_lectura_persiste_campos(self, mongo_db):
        reading = save_cleanliness_reading("sucio", 30.0, 70.0)
        assert reading["clasificacion"] == "sucio"
        assert reading["porcentajeLimpio"] == pytest.approx(30.0)
        assert reading["porcentajeSucio"] == pytest.approx(70.0)

    def test_source_por_defecto_es_manual(self, mongo_db):
        reading = save_cleanliness_reading("limpio", 80.0, 20.0)
        assert reading["source"] == "manual"

    def test_source_se_puede_especificar(self, mongo_db):
        reading = save_cleanliness_reading("limpio", 80.0, 20.0, source="camara")
        assert reading["source"] == "camara"

    def test_ultima_lectura_sin_datos_devuelve_none(self, mongo_db):
        assert get_latest_cleanliness() is None

    def test_ultima_lectura_devuelve_la_mas_reciente(self, mongo_db):
        save_cleanliness_reading("limpio", 90.0, 10.0)
        save_cleanliness_reading("sucio", 25.0, 75.0)
        latest = get_latest_cleanliness()
        assert latest is not None
        assert latest["clasificacion"] == "sucio"
        assert latest["porcentajeLimpio"] == pytest.approx(25.0)

    def test_listar_devuelve_lecturas_mas_reciente_primero(self, mongo_db):
        save_cleanliness_reading("limpio", 90.0, 10.0)
        save_cleanliness_reading("sucio", 40.0, 60.0)
        readings = list_cleanliness_readings()
        assert len(readings) == 2
        assert readings[0]["clasificacion"] == "sucio"

    def test_listar_sin_datos_devuelve_lista_vacia(self, mongo_db):
        assert list_cleanliness_readings() == []

    def test_lectura_incluye_timestamp(self, mongo_db):
        reading = save_cleanliness_reading("limpio", 88.0, 12.0)
        assert "timestamp" in reading
        assert isinstance(reading["timestamp"], str)
