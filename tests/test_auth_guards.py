"""
Pruebas unitarias para las guardas de autenticación y autorización.

Verifica que:
- require_auth acepta tokens válidos y rechaza peticiones anónimas
- require_admin acepta admins y rechaza usuarios con rol 'user'
- Los servicios CRUD rechazan operaciones de escritura a usuarios no-admin
  (simulando el contexto GraphQL con un token inyectado directamente)

No requiere base de datos ni red — las guardas son funciones puras sobre el
contexto GraphQL (un dict con la clave 'current_user').
"""
import pytest
from app.auth import require_auth, require_admin


# ─── Contextos de prueba ──────────────────────────────────────────────────────

def ctx_admin() -> dict:
    return {"current_user": {"sub": "admin@cujae.edu.cu", "role": "admin"}}

def ctx_user() -> dict:
    return {"current_user": {"sub": "operador@cujae.edu.cu", "role": "user"}}

def ctx_anon() -> dict:
    return {"current_user": None}

def ctx_sin_clave() -> dict:
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# require_auth
# ─────────────────────────────────────────────────────────────────────────────

class TestRequireAuth:

    def test_acepta_contexto_admin(self):
        user = require_auth(ctx_admin())
        assert user["role"] == "admin"

    def test_acepta_contexto_user(self):
        user = require_auth(ctx_user())
        assert user["role"] == "user"

    def test_devuelve_el_payload_del_usuario(self):
        user = require_auth(ctx_admin())
        assert user["sub"] == "admin@cujae.edu.cu"

    def test_rechaza_contexto_anonimo(self):
        with pytest.raises(Exception, match="[Aa]utenti"):
            require_auth(ctx_anon())

    def test_rechaza_contexto_sin_clave_current_user(self):
        with pytest.raises(Exception):
            require_auth(ctx_sin_clave())


# ─────────────────────────────────────────────────────────────────────────────
# require_admin
# ─────────────────────────────────────────────────────────────────────────────

class TestRequireAdmin:

    def test_acepta_admin(self):
        user = require_admin(ctx_admin())
        assert user["role"] == "admin"

    def test_rechaza_usuario_con_rol_user(self):
        with pytest.raises(Exception, match="[Dd]enegado|[Aa]dmin"):
            require_admin(ctx_user())

    def test_rechaza_usuario_anonimo(self):
        with pytest.raises(Exception):
            require_admin(ctx_anon())

    def test_rechaza_contexto_vacio(self):
        with pytest.raises(Exception):
            require_admin(ctx_sin_clave())

    def test_no_acepta_rol_inventado(self):
        ctx = {"current_user": {"sub": "x@x.cu", "role": "superadmin"}}
        with pytest.raises(Exception):
            require_admin(ctx)


# ─────────────────────────────────────────────────────────────────────────────
# Integración: guardas + servicio CRUD (con BD en memoria)
#
# Simula lo que hace el resolver GraphQL: primero valida el contexto y
# luego llama al servicio. Si la guarda falla, el servicio nunca se invoca.
# ─────────────────────────────────────────────────────────────────────────────

class TestCrudRequiereAdmin:
    """
    Verifica que el patrón require_admin(info.context) → service() aplicado en
    todos los resolvers de escritura bloquea correctamente a usuarios no-admin.

    Estos tests replican lo que hacen los resolvers en schema.py:
        require_admin(info.context)
        create_panel(...)
    """

    PANEL_PAYLOAD = {
        "manufacturer": "Test Solar",
        "ratedPowerKw": 0.4,
        "quantity": 10,
    }

    def _try_create_panel(self, context: dict, mongo_db) -> None:
        require_admin(context)
        from app.services.panel_service import create_panel
        create_panel(self.PANEL_PAYLOAD.copy())

    def test_admin_puede_crear_panel(self, mongo_db):
        self._try_create_panel(ctx_admin(), mongo_db)
        from app.services.panel_service import list_panels
        assert len(list_panels()) == 1

    def test_user_no_puede_crear_panel(self, mongo_db):
        with pytest.raises(Exception, match="[Dd]enegado|[Aa]dmin"):
            self._try_create_panel(ctx_user(), mongo_db)
        from app.services.panel_service import list_panels
        assert list_panels() == []  # la BD sigue vacía

    def test_anonimo_no_puede_crear_panel(self, mongo_db):
        with pytest.raises(Exception):
            self._try_create_panel(ctx_anon(), mongo_db)
        from app.services.panel_service import list_panels
        assert list_panels() == []

    def test_admin_puede_eliminar_panel(self, mongo_db):
        # Primero crea como admin
        require_admin(ctx_admin())
        from app.services.panel_service import create_panel, delete_panel, list_panels
        created = create_panel(self.PANEL_PAYLOAD.copy())
        # Elimina como admin
        require_admin(ctx_admin())
        result = delete_panel(created["_id"])
        assert result is True
        assert list_panels() == []

    def test_user_no_puede_eliminar_panel(self, mongo_db):
        from app.services.panel_service import create_panel, delete_panel
        created = create_panel(self.PANEL_PAYLOAD.copy())
        with pytest.raises(Exception, match="[Dd]enegado|[Aa]dmin"):
            require_admin(ctx_user())
            delete_panel(created["_id"])

    def test_admin_puede_crear_bateria(self, mongo_db):
        require_admin(ctx_admin())
        from app.services.battery_service import create_battery, list_batteries
        create_battery({"manufacturer": "CATL", "model": "X", "capacityKwh": 50.0, "quantity": 1})
        assert len(list_batteries()) == 1

    def test_user_no_puede_crear_bateria(self, mongo_db):
        with pytest.raises(Exception):
            require_admin(ctx_user())
        from app.services.battery_service import list_batteries
        assert list_batteries() == []

    def test_admin_puede_crear_electrodomestico(self, mongo_db):
        require_admin(ctx_admin())
        from app.services.appliance_service import create_appliance, list_appliances
        create_appliance({
            "name": "Nevera",
            "modes": [{"name": "Normal", "averagePowerW": 150.0, "maxPowerW": 200.0}],
            "dailyUsageHours": 24.0,
            "quantity": 1,
        })
        assert len(list_appliances()) == 1

    def test_user_no_puede_crear_electrodomestico(self, mongo_db):
        with pytest.raises(Exception):
            require_admin(ctx_user())
        from app.services.appliance_service import list_appliances
        assert list_appliances() == []
