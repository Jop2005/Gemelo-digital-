"""
Fixtures compartidos para todos los tests.

La fixture `mongo_db` reemplaza app.database.get_database con una base de datos
MongoDB completamente en memoria (mongomock). Ningún test toca la base de datos
real — cada test obtiene una BD limpia y vacía.
"""
import pytest
import mongomock
import app.database as _db_module


@pytest.fixture(autouse=True)
def _reset_weather_cache():
    """
    El caché de clima es un singleton de módulo (hoy+mañana, TTL 5 min) que
    persiste entre tests. Se vacía antes y después de cada test para que las
    llamadas mockeadas con respx siempre se ejerzan y no haya contaminación
    cruzada por datos cacheados.
    """
    from app.services.weather_cache_service import weather_cache

    weather_cache.reset()
    yield
    weather_cache.reset()


@pytest.fixture()
def mongo_db(monkeypatch):
    """
    Sustituye get_database() por una instancia mongomock en memoria.

    - Aislado: los datos no persisten entre tests.
    - Sin efectos secundarios: no toca MongoDB real ni ningún archivo externo.
    """
    client = mongomock.MongoClient()
    test_db = client["gemelo_test"]

    monkeypatch.setattr(_db_module, "get_database", lambda: test_db)
    # Forzamos que el módulo olvide cualquier conexión real previa.
    monkeypatch.setattr(_db_module, "_db", test_db)

    yield test_db

    client.close()
