"""
Prueba simple de carga de configuración desde el .env.

Verifica que load_dotenv() puebla el entorno y que la clase Settings expone
las variables con los tipos esperados.
"""
import os

from app.config import Settings, settings


class TestEnvLoading:
    """Comprueba que las variables del .env cargan correctamente."""

    def test_dotenv_pobla_jwt_secret(self):
        # load_dotenv() en app.config debe haber dejado JWT_SECRET en el entorno.
        assert os.getenv("JWT_SECRET"), "JWT_SECRET no se cargó desde el .env"

    def test_settings_se_instancia(self):
        # Crear Settings no debe lanzar (todas las variables requeridas presentes).
        s = Settings()
        assert s is not None

    def test_tipos_de_variables(self):
        assert isinstance(settings.HOST, str)
        assert isinstance(settings.PORT, int)
        assert isinstance(settings.CORS_ORIGINS, list)
        assert isinstance(settings.MONGODB_URI, str)
        assert isinstance(settings.MONGODB_DB, str)
        assert isinstance(settings.JWT_SECRET, str) and settings.JWT_SECRET

    def test_valores_por_defecto(self):
        # PORT debe ser un entero válido y CORS_ORIGINS no debe estar vacío.
        assert settings.PORT > 0
        assert len(settings.CORS_ORIGINS) >= 1
