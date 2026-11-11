"""
Prueba de extremo a extremo del comprobador de limpieza de paneles.

A diferencia de `test_panel_classifier.py` (que mockea TensorFlow), esta prueba
usa el **modelo real** y golpea el **mismo endpoint REST** que consume el
frontend: `POST /api/classify-panel` con un archivo `multipart/form-data` en el
campo `file` (ver frontend/src/app/components/DevicesView.tsx).

Objetivo: si esta clase pasa, queda garantizado que cuando un usuario entre por
el frontend a "comprobar limpieza" y suba una foto, el flujo completo funciona:
carga del modelo Keras → preprocesado de la imagen → inferencia → respuesta JSON
con la clasificación correcta.

Usa dos imágenes reales de muestra (tests/fixtures/panel_samples/) y verifica
que cada una se clasifique como corresponde: limpio.jpg → "limpio",
sucio.jpeg → "sucio".

La clase se omite limpiamente (skip) si faltan el modelo entrenado o las
imágenes, para no romper la suite en entornos sin esos artefactos.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_FIXTURES = Path(__file__).parent / "fixtures" / "panel_samples"
_LIMPIO = _FIXTURES / "limpio.jpg"
_SUCIO = _FIXTURES / "sucio.jpeg"


@pytest.fixture(autouse=True)
def _cleanup_real_db_readings():
    """
    El endpoint /api/classify-panel persiste cada clasificación en la BD real
    (no usa mongomock, porque este test ejerce la app real de extremo a extremo).
    Para no ensuciar la base con las fotos de prueba, borramos al terminar las
    lecturas insertadas durante el test, identificadas por su ventana de tiempo.
    """
    from datetime import datetime, timezone

    started_at = datetime.now(timezone.utc)
    yield
    try:
        from app.services.panel_cleanliness_service import _col

        _col().delete_many({"timestamp": {"$gte": started_at}})
    except Exception as e:
        print(f"⚠️  No se pudieron limpiar las lecturas de prueba: {e}")


@pytest.fixture(scope="module")
def client() -> TestClient:
    """
    Cliente HTTP sobre la app real, con el modelo de clasificación cargado
    exactamente como en producción (mismo singleton que usa el endpoint).
    """
    from app.services.panel_classifier_service import panel_classifier_service

    if not panel_classifier_service.model_path.exists():
        pytest.skip(f"Modelo no disponible en {panel_classifier_service.model_path}")
    if not _LIMPIO.exists() or not _SUCIO.exists():
        pytest.skip("Faltan las imágenes de muestra en tests/fixtures/panel_samples/")

    # Carga el modelo en el singleton global (lo que hace el lifespan al arrancar).
    if panel_classifier_service.model is None:
        panel_classifier_service.load_model()

    from app.main import app

    return TestClient(app)


def _classify(client: TestClient, path: Path, mime: str) -> dict:
    """Sube la imagen al endpoint igual que el frontend y devuelve el JSON."""
    with path.open("rb") as fh:
        response = client.post(
            "/api/classify-panel",
            files={"file": (path.name, fh, mime)},
        )
    assert response.status_code == 200, response.text
    return response.json()


class TestComprobadorLimpiezaPaneles:
    """
    Verifica el endpoint /api/classify-panel con el modelo real y fotos reales,
    replicando lo que hace el frontend al comprobar la limpieza de un panel.
    """

    def test_panel_limpio_se_clasifica_como_limpio(self, client):
        """Una foto de panel limpio debe devolver clasificacion == 'limpio'."""
        result = _classify(client, _LIMPIO, "image/jpeg")
        assert result["clasificacion"] == "limpio", (
            f"Se esperaba 'limpio' pero el modelo dijo {result}"
        )

    def test_panel_sucio_se_clasifica_como_sucio(self, client):
        """Una foto de panel sucio debe devolver clasificacion == 'sucio'."""
        result = _classify(client, _SUCIO, "image/jpeg")
        assert result["clasificacion"] == "sucio", (
            f"Se esperaba 'sucio' pero el modelo dijo {result}"
        )

    def test_respuesta_tiene_la_estructura_que_espera_el_frontend(self, client):
        """La respuesta debe traer las tres claves que consume el frontend."""
        result = _classify(client, _LIMPIO, "image/jpeg")
        for key in ("clasificacion", "porcentaje_limpio", "porcentaje_sucio"):
            assert key in result, f"Falta la clave '{key}' en la respuesta"

    def test_porcentajes_suman_cien(self, client):
        """porcentaje_limpio + porcentaje_sucio deben sumar ~100%."""
        result = _classify(client, _SUCIO, "image/jpeg")
        total = result["porcentaje_limpio"] + result["porcentaje_sucio"]
        assert total == pytest.approx(100.0, abs=0.5)

    def test_clasificacion_coincide_con_el_porcentaje_dominante(self, client):
        """La etiqueta debe ser coherente con el porcentaje más alto."""
        result = _classify(client, _LIMPIO, "image/jpeg")
        if result["clasificacion"] == "limpio":
            assert result["porcentaje_limpio"] >= result["porcentaje_sucio"]
        else:
            assert result["porcentaje_sucio"] > result["porcentaje_limpio"]

    def test_rechaza_archivo_que_no_es_imagen(self, client):
        """Subir un archivo no-imagen debe devolver 400 (igual que valida el front)."""
        response = client.post(
            "/api/classify-panel",
            files={"file": ("nota.txt", b"esto no es una imagen", "text/plain")},
        )
        assert response.status_code == 400
