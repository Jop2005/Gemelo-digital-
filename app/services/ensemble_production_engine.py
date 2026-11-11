"""
Motor de generación del ensamble dinámico.
app llama al ensamble; la revisión post-hoc se aplica aquí, en app.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from app.services.production_engine import ProductionEngine

ENSAMBLE_ROOT = Path(__file__).parent.parent.parent / "ensamble"
if str(ENSAMBLE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENSAMBLE_ROOT))


class EnsembleProductionEngine(ProductionEngine):
    name = "ensemble"

    def __init__(self):
        self._controller = None
        self._features = None
        self._inicializado = False

    def _inicializar(self):
        if self._inicializado:
            return

        from configuracion.acceso_datos.config import get_config
        from configuracion.acceso_datos.strategies_generacion import GeneracionStrategy
        from configuracion.acceso_datos.context import DatasetContext
        from configuracion.acceso_datos.joblib_model_repository import JoblibModelRepository
        from configuracion.acceso_datos.joblib_imputer_repository import JoblibImputerRepository
        from configuracion.acceso_datos.historial_repository import HistorialRepository
        from configuracion.acceso_datos.file_reader import FileReader
        from configuracion.acceso_datos.config_loader import ConfigLoader
        from prediccion.logica_negocio.prediccion_controller import PrediccionController

        get_config().set_dominio("generacion")
        strategy = GeneracionStrategy()
        context = DatasetContext(strategy)

        self._controller = PrediccionController(
            context=context,
            model_repo=JoblibModelRepository(),
            imputer_repo=JoblibImputerRepository(),
            historial_repo=HistorialRepository(),
            file_reader=FileReader(),
            config_loader=ConfigLoader(),
        )
        loader = context.get_loader()
        self._features = loader.config.get("features", [])
        self._inicializado = True

    def _load_historical(self):
        """
        Carga el histórico de referencia para la revisión.
        Sustituye este cuerpo por la carga real desde tu base.
        """
        raise NotImplementedError("Implementar carga real del histórico")

    def capacity_factor(self, features_df, lat, lon):
        self._inicializar()

        if not self._controller.esta_listo():
            raise RuntimeError("El ensamble de generación no está listo.")

        df_input = features_df.copy().reset_index(drop=True)
        missing = [f for f in self._features if f not in df_input.columns]
        if missing:
            raise ValueError(f"Columnas faltantes en el DataFrame: {missing}")

        resultados = []
        for _, row in df_input.iterrows():
            datos = {f: row[f] for f in self._features}
            res = self._controller.predecir_manual(datos)
            if isinstance(res, dict) and "error" in res:
                raise RuntimeError(f"Error en el ensamble de generación: {res['error']}")
            resultados.append(float(res["prediccion"]))

        predicciones = np.array(resultados)

        return np.clip(predicciones, 0.0, 1.0)

    def predict_batch(self, features_df):
        """
        Predice el factor de capacidad para un DataFrame completo usando el método
        de predicción por archivo del ensamble (mucho más rápido que iterar).
        """
        import tempfile
        import os
        import pandas as pd
        
        self._inicializar()  # Aseguramos que el controlador esté listo
        
        # Guardamos las features en un CSV temporal
        temp_csv = "temp_prod_features.csv"
        df_to_save = features_df.copy().reset_index(drop=True)
        df_to_save.to_csv(temp_csv, index=False)
        
        try:
            # Llamamos al método de archivo del controlador
            resultados = self._controller.predecir_archivo(temp_csv, self._features)
            
            # Extraemos las predicciones
            predicciones = [r['prediccion'] for r in resultados]
            return np.array(predicciones)
        finally:
            # Limpiamos el archivo temporal
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
