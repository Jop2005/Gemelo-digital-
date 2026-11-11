"""
Application configuration
"""
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings"""

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # CORS
    CORS_ORIGINS: list[str] = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")

    # MongoDB
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017/GemeloDigitalCujai")
    MONGODB_DB: str = os.getenv("MONGODB_DB", "GemeloDigitalCujai")

    # Location (La Habana, Cuba)
    LATITUDE: float = 23.1136
    LONGITUDE: float = -82.3666

    # LDAP authentication is configured from the admin panel and stored in
    # MongoDB (see app/services/ldap_config_service.py) — not via env vars.

    # JWT
    JWT_SECRET: str = "una_clave_secreta_larga_y_segura"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_DAYS: int = 7

    # ML de consumo — calibración respecto al sistema configurado.
    # El modelo se entrenó con datos del medidor 55 del campus CUJAE (un edificio
    # cuyo consumo nominal es ~10× el sistema base del gemelo). Estos parámetros
    # permiten reapuntar el modelo a otro medidor y reescalar la salida.
    ML_CONSUMPTION_CAMPUS_ID: Optional[int] = (
        int(os.getenv("ML_CONSUMPTION_CAMPUS_ID")) if os.getenv("ML_CONSUMPTION_CAMPUS_ID") else None
    )
    ML_CONSUMPTION_METER_ID: Optional[int] = (
        int(os.getenv("ML_CONSUMPTION_METER_ID")) if os.getenv("ML_CONSUMPTION_METER_ID") else None
    )
    ML_CONSUMPTION_SCALE_DIVISOR: float = float(
        os.getenv("ML_CONSUMPTION_SCALE_DIVISOR", "10.0")
    )


settings = Settings()
