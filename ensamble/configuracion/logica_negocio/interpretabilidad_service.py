import os
import json
import pandas as pd
from datetime import datetime
from typing import Optional
from configuracion.acceso_datos.joblib_model_repository import JoblibModelRepository
from configuracion.acceso_datos.model_repository_interface import ModeloRepositoryInterface
from configuracion.acceso_datos.config import get_config
from .interpretability import InterpretabilityModule

class InterpretabilidadService:
    """Servicio de lógica de negocio para generación de interpretabilidad."""

    def __init__(self, context, model_repo: Optional[ModeloRepositoryInterface] = None):
        self._context = context
        self._model_repo = model_repo or JoblibModelRepository()

    def ejecutar(self) -> dict:
        modelos = self._model_repo.cargar_todos()
        if not modelos:
            return {}

        features = self._obtener_features()
        importancias = self._calcular_importancias(modelos, features)
        self._guardar_resultados(importancias)

        return importancias

    def explicar_prediccion(self, datos: dict, features: list, modelo_nombre: str = None) -> dict:
        modelos = self._model_repo.cargar_todos()
        if not modelos:
            return {'error': 'No hay modelos disponibles'}

        if modelo_nombre and modelo_nombre in modelos:
            modelo = {modelo_nombre: modelos[modelo_nombre]}
        else:
            modelo = modelos

        instancia = pd.DataFrame([datos])
        interp = InterpretabilityModule(modelo, features)

        nombre = modelo_nombre if modelo_nombre else next(iter(modelo.keys()))
        return interp.explain_prediction(nombre, instancia)

    def _obtener_features(self) -> list:
        loader = self._context.get_loader()
        return loader.config.get('features', [])

    def _calcular_importancias(self, modelos: dict, features: list) -> dict:
        interp = InterpretabilityModule(modelos, features)
        return interp.compare_importances()

    def _guardar_resultados(self, importancias: dict) -> None:
        config = get_config()
        carpeta_resultados = str(config.carpeta_resultados)
        os.makedirs(carpeta_resultados, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        df = pd.DataFrame(importancias).T
        df.to_csv(
            f'{carpeta_resultados}/importancia_caracteristicas_{timestamp}.csv',
            float_format='%.4f'
        )

        with open(
            f'{carpeta_resultados}/interpretabilidad_{timestamp}.json',
            'w',
            encoding='utf-8'
        ) as f:
            json.dump(importancias, f, indent=2, default=self._convertir_serializable)

    @staticmethod
    def _convertir_serializable(obj):
        return obj.tolist() if hasattr(obj, 'tolist') else str(obj)