import numpy as np
import pandas as pd
import joblib
import os
from .segmentation.segmenter import Segmenter
from .evaluation import ModelEvaluator
from .excepciones import SegmentacionNoEncontrada, ModelosNoEncontrados
from ..acceso_datos.config import get_config

class EvaluacionService:
    
    def __init__(self, context, model_repo):
        self._context = context
        self._model_repo = model_repo
    
    def ejecutar(self, config_loader) -> dict:
        config_seg = config_loader.cargar_segmentacion()
        if not config_seg:
            raise SegmentacionNoEncontrada("No se encontró configuración de segmentación")
        
        segmenter = Segmenter.from_config(config_seg)
        modelos = self._model_repo.cargar_todos()
        if not modelos:
            raise ModelosNoEncontrados("No hay modelos entrenados disponibles")
        
        loader = self._context.get_loader()
        X, y = loader.obtener_datos_completos()
        
        split_idx = int(len(X) * 0.8)
        _, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
        X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
        
        segments = segmenter.transform(X_test.copy())
        segmentos_validos = segments[segments >= 0]
        if len(segmentos_validos) == 0:
            return {
                'error': 'No hay segmentos válidos para evaluar',
                'modelos_evaluados': [],
                'n_segments': 0
            }
        
        predicciones = {nombre: modelo.predict(X_test) for nombre, modelo in modelos.items()}
        
        evaluator = ModelEvaluator()
        metricas_por_modelo = {}
        for nombre, pred in predicciones.items():
            metricas_por_modelo[nombre] = evaluator.evaluar_por_segmentos(
                nombre, y_test.values, pred, segments, y_train=y_train.values
            )
        
        resultados = {
            'fecha': pd.Timestamp.now().isoformat(),
            'tipo_segmentacion': config_seg.get('tipo', 'unknown'),
            'variables_segmentacion': config_seg.get('variables', []),
            'n_segments': segmenter.n_segments,
            'modelos_evaluados': list(modelos.keys()),
            'metricas': {nombre: {str(k): v for k, v in m.items()} for nombre, m in metricas_por_modelo.items()},
            'distribucion': np.bincount(segmentos_validos).tolist()
        }
        
        config = get_config()
        carpeta = str(config.carpeta_resultados)
        os.makedirs(carpeta, exist_ok=True)
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        joblib.dump(resultados, f"{carpeta}/evaluacion_{timestamp}.joblib")
        
        return resultados