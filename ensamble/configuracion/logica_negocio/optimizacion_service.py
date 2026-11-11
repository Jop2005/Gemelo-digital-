"""
Servicio de optimización de pesos para el ensamblaje.
Usa búsqueda discreta de pesos por segmento operativo (features).
"""
import numpy as np
import pandas as pd
import os
import glob
import joblib
import warnings
from .segmentation.segmenter import Segmenter
from .excepciones import SegmentacionNoEncontrada, EvaluacionNoEncontrada, ModelosNoEncontrados
from ..acceso_datos.config_repository import ConfigRepository
from ..acceso_datos.data_splitter import DataSplitter
from ..acceso_datos.config import get_config

PESOS_A_PROBAR = [(1.0, 0.0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.0, 1.0)]


class OptimizacionService:

    def __init__(self, context, model_repo):
        self._context = context
        self._model_repo = model_repo

    def ejecutar(self, config_loader) -> dict:
        segmenter = self._cargar_segmentador(config_loader)
        if not segmenter:
            raise SegmentacionNoEncontrada("No se encontró configuración de segmentación")

        evaluacion = self._cargar_evaluacion_reciente()
        if not evaluacion:
            raise EvaluacionNoEncontrada("No se encontró evaluación reciente")
        X_test, y_test = self._cargar_datos_prueba()
        modelos = self._model_repo.cargar_todos()
        if not modelos:
            raise ModelosNoEncontrados("No hay modelos entrenados disponibles")

        nombres = list(modelos.keys())
        if len(nombres) < 2:
            raise ModelosNoEncontrados("Se necesitan al menos 2 modelos")

        rf, xgb = modelos[nombres[0]], modelos[nombres[1]]
        pred_rf = rf.predict(X_test)
        pred_xgb = xgb.predict(X_test)

        segments = segmenter.transform(X_test)
        config_segmentos = self._elegir_mejores_pesos(
            np.asarray(y_test), pred_rf, pred_xgb, segments, nombres
        )
        return self._guardar(config_segmentos, evaluacion)

    def _cargar_segmentador(self, config_loader):
        config_seg = config_loader.cargar_segmentacion()
        return Segmenter.from_config(config_seg) if config_seg else None

    def _cargar_evaluacion_reciente(self):
        carpeta = str(get_config().carpeta_resultados)
        eval_files = glob.glob(f'{carpeta}/evaluacion_*.joblib')
        if not eval_files:
            return None
        return joblib.load(max(eval_files, key=os.path.getctime))

    def _cargar_datos_prueba(self):
        loader = self._context.get_loader()
        X, y = loader.obtener_datos_completos()
        _, X_test, _, y_test = DataSplitter(test_size=0.2).split(X, y)
        return X_test, y_test

    def _elegir_mejores_pesos(self, y_test, pred_rf, pred_xgb, segments, nombres):
        n_seg = int(max(segments.max(), 0)) + 1
        config = {}
        for seg in range(n_seg):
            mask = segments == seg
            if mask.sum() < 10:
                warnings.warn(f"Segmento {seg} tiene solo {mask.sum()} muestras.")
                continue
            mejor_mae = float("inf")
            mejor_pesos = (1.0, 0.0)
            for w_rf, w_xgb in PESOS_A_PROBAR:
                pred = w_rf * pred_rf[mask] + w_xgb * pred_xgb[mask]
                mae = np.mean(np.abs(y_test[mask] - pred))
                if mae < mejor_mae:
                    mejor_mae = mae
                    mejor_pesos = (w_rf, w_xgb)
            w_rf, w_xgb = mejor_pesos
            if w_rf > 0 and w_xgb > 0:
                config[seg] = {
                    "modelo": "Ensemble",
                    "pesos": {nombres[0]: float(w_rf), nombres[1]: float(w_xgb)},
                }
            elif w_rf > 0:
                config[seg] = {"modelo": nombres[0]}
            else:
                config[seg] = {"modelo": nombres[1]}
        return config

    def _guardar(self, config_segmentos, evaluacion):
        config_final = {
            "fecha": pd.Timestamp.now().isoformat(),
            "tipo_segmentacion": evaluacion.get("tipo_segmentacion", "unknown"),
            "variables_segmentacion": evaluacion.get("variables_segmentacion", []),
            "segmentos": config_segmentos,
            "n_segments": len(config_segmentos),
            "evaluacion_usada": evaluacion.get("fecha", ""),
        }
        ConfigRepository().guardar_versionado(config_final, "config_ensamblaje")
        return config_final