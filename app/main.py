"""
FastAPI application with GraphQL endpoint using Strawberry
"""
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from strawberry.fastapi import GraphQLRouter
from contextlib import asynccontextmanager
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from pydantic import BaseModel

from app.schema import schema
from app.config import settings
from app.database import get_database, close_database
from app.auth import get_context, verify_token
from app.services.panel_classifier_service import panel_classifier_service
from app.services.ml_model_service import ml_model_service
from app.services import test_runner_service

limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])


async def _periodic_snapshot_saver():
    """Save solar production to MongoDB every 5 minutes for historical trend analysis."""
    while True:
        try:
            await asyncio.sleep(300)  # 5 minutes
            from app.services.solar_service import get_solar_snapshot
            from app.services.lectura_service import save_reading
            from app.services.weather_cache_service import weather_cache
            from app.services.system_config import get_system_config

            # Refresca el caché de clima (hoy+mañana) cada 5 min, sobrescribiendo.
            cfg = get_system_config()
            await weather_cache.preload(cfg["location"]["lat"], cfg["location"]["lon"])

            snapshot_data = await get_solar_snapshot()
            production_kw = snapshot_data.get("current", {}).get("production", 0)
            save_reading(production_kw)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"⚠️  Background snapshot error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    """
    # Startup
    print("🚀 Starting Digital Twin GraphQL API...")
    get_database()  # Initialize database connection

    # Load ML model for panel classification
    try:
        panel_classifier_service.load_model()
    except Exception as e:
        print(f"⚠️  Warning: Could not load panel classifier model: {e}")
        print("   Panel classification endpoint will not be available.")

    # Load ML model for solar production prediction (La Habana, capacity factor)
    try:
        ml_model_service.load_model(model_name="havana_v1")
    except Exception as e:
        print(f"⚠️  Warning: Could not load solar production prediction model: {e}")
        print("   Solar production prediction endpoint will not be available.")
        print("   Please run the training script: backend/notebooks/train_solar_havana.py")

    # Load error estimators for post-hoc revision (one per domain)
    try:
        from app.services.error_estimator_singleton import (
            error_estimator_consumo,
            error_estimator_generacion,
        )
        error_estimator_consumo.load("models/error_estimator_consumo.joblib")
        error_estimator_generacion.load("models/error_estimator_generacion.joblib")
        print("✅ Error estimators loaded (consumo, generacion)")
    except Exception as e:
        print(f"⚠️  Warning: Could not load error estimators: {e}")
        print("   Post-hoc revision will not be available.")

    # Attach post-hoc revision to the ensemble engines
    try:
        from app.services.error_estimator_singleton import (
            error_estimator_consumo,
            error_estimator_generacion,
        )
        from app.services.instance_revision import InstanceRevision
        from app.services.production_forecast_service import production_forecast_service
        from app.services.prediction_service import get_consumption_engine

        production_forecast_service.engine.set_revision(
            InstanceRevision(error_estimator_generacion)
        )
        get_consumption_engine().set_revision(
            InstanceRevision(error_estimator_consumo)
        )
        print("✅ Post-hoc revision attached to engines")
    except Exception as e:
        print(f"⚠️  Warning: Could not attach post-hoc revision: {e}")
        
    # Warm the global weather cache (today + tomorrow) with a single call so the
    # first dashboard/predictor request is served from memory.
    try:
        from app.services.weather_cache_service import weather_cache
        from app.services.system_config import get_system_config

        cfg = get_system_config()
        await weather_cache.preload(cfg["location"]["lat"], cfg["location"]["lon"])
        print("🌤️  Weather cache warmed (today + tomorrow).")
    except Exception as e:
        print(f"⚠️  Warning: could not warm weather cache: {e}")

    # Start background task for historical data collection
    task = asyncio.create_task(_periodic_snapshot_saver())

    yield

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Shutdown
    print("👋 Shutting down Digital Twin GraphQL API...")
    close_database()


# Create FastAPI application
app = FastAPI(
    title="Digital Twin GraphQL API",
    description="GraphQL API for photovoltaic microgrid digital twin system",
    version="1.0.0",
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"error": "Demasiadas solicitudes. Por favor espera un momento antes de continuar."},
    )

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create GraphQL router
graphql_app = GraphQLRouter(
    schema,
    context_getter=get_context,
    graphiql=True,  # Enable GraphiQL interface in development
)

# Mount GraphQL endpoint
app.include_router(graphql_app, prefix="/graphql")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Digital Twin GraphQL API",
        "version": "1.0.0",
        "graphql_endpoint": "/graphql",
        "graphiql_interface": "/graphql (in browser)",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        db = get_database()
        # Test database connection
        db.command("ping")
        return {
            "status": "healthy",
            "database": "connected",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
        }


@app.post("/api/classify-panel")
async def classify_panel(file: UploadFile = File(...)):
    """
    Classify a solar panel image as clean or dusty.

    Args:
        file: Image file to classify (JPEG, PNG, etc.)

    Returns:
        JSON with:
        - clasificacion: "limpio" or "sucio"
        - porcentaje_limpio: Clean probability (0-100)
        - porcentaje_sucio: Dusty probability (0-100)

    Example:
        curl -X POST "http://localhost:8000/api/classify-panel" \
             -F "file=@panel_image.jpg"
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, etc.)"
        )

    try:
        # Read image bytes
        image_bytes = await file.read()

        # Classify using the service
        result = panel_classifier_service.classify_panel(image_bytes)

        # Persist the reading so the dashboard can surface panel state over time.
        # A DB failure must not break the classification response the frontend expects.
        try:
            from app.services.panel_cleanliness_service import save_cleanliness_reading

            save_cleanliness_reading(
                clasificacion=result["clasificacion"],
                porcentaje_limpio=result["porcentaje_limpio"],
                porcentaje_sucio=result["porcentaje_sucio"],
                source="manual",
            )
        except Exception as e:
            print(f"⚠️  Could not persist panel cleanliness reading: {e}")

        return result

    except RuntimeError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


def _require_admin_rest(request: Request) -> dict:
    """
    Guard de administrador para endpoints REST (equivalente a require_admin del
    contexto GraphQL). Lee el JWT del header Authorization, lo valida, comprueba
    que la sesión no esté revocada y exige rol 'admin'.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Autenticación requerida. Por favor inicia sesión.")
    payload = verify_token(auth_header[7:])
    if not payload:
        raise HTTPException(status_code=401, detail="Token inválido o expirado.")
    if payload.get("jti"):
        from app.services.session_service import is_token_revoked
        if is_token_revoked(payload["jti"]):
            raise HTTPException(status_code=401, detail="Token inválido o expirado.")
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Acceso denegado. Se requieren privilegios de administrador.")
    return payload


class RunTestsRequest(BaseModel):
    """Cuerpo para ejecutar pruebas: suite ('backend'|'frontend') + lista de ids."""
    suite: str
    ids: list[str]


@app.get("/api/dev/tests")
async def list_dev_tests(request: Request):
    """
    Catálogo de pruebas (solo administradores) agrupado por categoría, para la
    pantalla /dev/test. Incluye backend (pytest) y frontend (vitest).
    """
    _require_admin_rest(request)
    return test_runner_service.build_catalog()


@app.post("/api/dev/tests/run")
async def run_dev_tests(payload: RunTestsRequest, request: Request):
    """
    Ejecuta un grupo de pruebas o una prueba puntual (solo administradores).
    El cuerpo indica la suite y los ids (node ids de pytest, o '<archivo>::<nombre>'
    para vitest).
    """
    _require_admin_rest(request)
    if payload.suite == "backend":
        return test_runner_service.run_backend(payload.ids)
    if payload.suite == "frontend":
        return test_runner_service.run_frontend(payload.ids)
    raise HTTPException(status_code=400, detail="Suite desconocida. Use 'backend' o 'frontend'.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
