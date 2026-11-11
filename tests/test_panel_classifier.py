"""
Pruebas para el servicio de clasificación de paneles solares y el endpoint REST
`/api/classify-panel`.

TensorFlow no está disponible en el entorno de pruebas, por lo que se mockea
en `sys.modules` ANTES de cualquier importación del servicio.

El endpoint se prueba a través de una app FastAPI mínima que replica la lógica
de main.py sin importar main.py (cuya inicialización de GraphQLRouter es
incompatible con la versión de Strawberry instalada en este entorno).
"""
import sys
import io
from unittest.mock import MagicMock

# ─────────────────────────────────────────────────────────────────────────────
# Mock de TensorFlow — debe ejecutarse antes de importar cualquier módulo que
# use TF (incluyendo panel_classifier_service).
# ─────────────────────────────────────────────────────────────────────────────

# Si TensorFlow está instalado (entorno con el modelo real), se usa el real para
# no contaminar `sys.modules` — así la prueba e2e con el modelo real
# (test_panel_cleanliness_e2e.py) funciona en la misma corrida. Solo si TF no
# está disponible se sustituye por un mock para poder importar el servicio.
try:  # pragma: no cover - depende del entorno
    import tensorflow  # noqa: F401
except Exception:  # pragma: no cover
    _tf_mock = MagicMock()
    sys.modules["tensorflow"] = _tf_mock
    sys.modules["tensorflow.keras"] = _tf_mock.keras
    sys.modules["tensorflow.keras.models"] = _tf_mock.keras.models

# ─────────────────────────────────────────────────────────────────────────────
# Importaciones normales (TF ya está mockeado en sys.modules)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pytest
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.testclient import TestClient

from app.services.panel_classifier_service import (
    PanelClassifierService,
    panel_classifier_service,
)


# App mínima que replica el endpoint de main.py sin importar main.py.
_test_app = FastAPI()


@_test_app.post("/api/classify-panel")
async def _classify_panel_route(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, etc.)")
    try:
        image_bytes = await file.read()
        return panel_classifier_service.classify_panel(image_bytes)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error processing image: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_fake_image(color: tuple = (100, 150, 200)) -> bytes:
    img = Image.new("RGB", (50, 50), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_mock_model(prediction_value: float) -> MagicMock:
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[prediction_value]])
    return mock_model


# ─────────────────────────────────────────────────────────────────────────────
# Tests: PanelClassifierService
# ─────────────────────────────────────────────────────────────────────────────

class TestPanelClassifierService:

    def test_preprocess_image_devuelve_array_correcto(self):
        service = PanelClassifierService()
        result = service.preprocess_image(make_fake_image())
        assert result.shape == (1, 224, 224, 3)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_classify_panel_sin_modelo_lanza_error(self):
        service = PanelClassifierService()
        assert service.model is None
        with pytest.raises(RuntimeError, match="not loaded"):
            service.classify_panel(make_fake_image())

    def test_classify_panel_limpio(self):
        service = PanelClassifierService()
        service.model = make_mock_model(0.2)
        result = service.classify_panel(make_fake_image())
        assert result["clasificacion"] == "limpio"
        assert result["porcentaje_limpio"] == pytest.approx(80.0, abs=0.1)
        assert result["porcentaje_sucio"] == pytest.approx(20.0, abs=0.1)

    def test_classify_panel_sucio(self):
        service = PanelClassifierService()
        service.model = make_mock_model(0.8)
        result = service.classify_panel(make_fake_image())
        assert result["clasificacion"] == "sucio"
        assert result["porcentaje_sucio"] == pytest.approx(80.0, abs=0.1)
        assert result["porcentaje_limpio"] == pytest.approx(20.0, abs=0.1)

    def test_threshold_exacto_050_es_limpio(self):
        # prediction == 0.5 no es estrictamente > 0.5 → "limpio"
        service = PanelClassifierService()
        service.model = make_mock_model(0.5)
        result = service.classify_panel(make_fake_image())
        assert result["clasificacion"] == "limpio"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: endpoint REST /api/classify-panel
# ─────────────────────────────────────────────────────────────────────────────

class TestClasificarPanelEndpoint:

    @pytest.fixture(autouse=True)
    def client(self):
        return TestClient(_test_app, raise_server_exceptions=False)

    def test_archivo_no_imagen_devuelve_400(self, client):
        response = client.post(
            "/api/classify-panel",
            files={"file": ("document.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400

    def test_modelo_no_cargado_devuelve_503(self, client, monkeypatch):
        monkeypatch.setattr(panel_classifier_service, "model", None)
        response = client.post(
            "/api/classify-panel",
            files={"file": ("panel.jpg", make_fake_image(), "image/jpeg")},
        )
        assert response.status_code == 503

    def test_clasificacion_exitosa(self, client, monkeypatch):
        monkeypatch.setattr(panel_classifier_service, "model", make_mock_model(0.2))
        response = client.post(
            "/api/classify-panel",
            files={"file": ("panel.jpg", make_fake_image(), "image/jpeg")},
        )
        assert response.status_code == 200
        body = response.json()
        assert "clasificacion" in body
        assert "porcentaje_limpio" in body
        assert "porcentaje_sucio" in body

    def test_clasificacion_limpio_valores_correctos(self, client, monkeypatch):
        monkeypatch.setattr(panel_classifier_service, "model", make_mock_model(0.3))
        response = client.post(
            "/api/classify-panel",
            files={"file": ("panel.jpg", make_fake_image(), "image/jpeg")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["clasificacion"] == "limpio"
        assert body["porcentaje_limpio"] == pytest.approx(70.0, abs=0.1)
