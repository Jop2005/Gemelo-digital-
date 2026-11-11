"""
Motor de consumo del ensamble dinámico.
app llama al ensamble; la revisión post-hoc se aplica aquí, en app.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from app.services.consumption_engine import ConsumptionEngine

ENSAMBLE_ROOT = Path(__file__).parent.parent.parent / "ensamble"
if str(ENSAMBLE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENSAMBLE_ROOT))


class EnsembleConsumptionEngine(ConsumptionEngine):
    name = "ensemble"

    def __init__(self):
        self._controller = None
        self._features = None
        self._inicializado = False

    def _inicializar(self):
        if self._inicializado:
            return

        from configuracion.acceso_datos.config import get_config
        from configuracion.acceso_datos.strategies_consumo import ConsumoStrategy
        from configuracion.acceso_datos.context import DatasetContext
        from configuracion.acceso_datos.joblib_model_repository import JoblibModelRepository
        from configuracion.acceso_datos.joblib_imputer_repository import JoblibImputerRepository
        from configuracion.acceso_datos.historial_repository import HistorialRepository
        from configuracion.acceso_datos.file_reader import FileReader
        from configuracion.acceso_datos.config_loader import ConfigLoader
        from prediccion.logica_negocio.prediccion_controller import PrediccionController

        get_config().set_dominio("consumo")
        strategy = ConsumoStrategy()
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

    def _build_feature_matrix(self, datetimes: List[str]) -> np.ndarray:
        rows = []
        for dt in pd.to_datetime(datetimes, utc=True):
            dt_local = dt.tz_convert("America/Havana") if dt.tz is not None else dt
            rows.append({
                "Hora": dt_local.hour,
                "Mes": dt_local.month,
                "tipo_dia_bin": 1 if dt_local.weekday() < 5 else 0,
            })
        df = pd.DataFrame(rows)
        return df[self._features].values

    def _load_historical(self):
        """
        Carga el histórico de referencia para la revisión.
        Sustituye este cuerpo por la carga real desde tu base.
        """
        raise NotImplementedError("Implementar carga real del histórico")

    def predict(self, datetimes: List[str], config: Dict) -> List[float]:
        self._inicializar()

        if not self._controller.esta_listo():
            raise RuntimeError("El ensamble de consumo no está listo.")

        predicciones = []
        for dt in pd.to_datetime(datetimes, utc=True):
            dt_local = dt.tz_convert("America/Havana") if dt.tz is not None else dt
            datos = {
                "Hora": dt_local.hour,
                "Mes": dt_local.month,
                "tipo_dia_bin": 1 if dt_local.weekday() < 5 else 0,
            }
            res = self._controller.predecir_manual(datos)
            if isinstance(res, dict) and "error" in res:
                raise RuntimeError(f"Error en el ensamble de consumo: {res['error']}")
            predicciones.append(float(res["prediccion"]))

        return predicciones

    def predict_batch(self, datetimes):
        """
        Predice el consumo para una lista de datetimes usando el método
        de predicción por archivo del ensamble (mucho más rápido que iterar).
        """
        import tempfile
        import os
        import pandas as pd
        
        self._inicializar()  # Aseguramos que el controlador esté listo
        
        # Preparamos las features de consumo
        rows = []
        for dt in pd.to_datetime(datetimes, utc=True):
            dt_local = dt.tz_convert("America/Havana") if dt.tz is not None else dt
            rows.append({
                "Hora": dt_local.hour,
                "Mes": dt_local.month,
                "tipo_dia_bin": 1 if dt_local.weekday() < 5 else 0,
            })
        df_consumo = pd.DataFrame(rows)
        
        # Guardamos en CSV temporal
        temp_csv = "temp_cons_features.csv"
        df_consumo.to_csv(temp_csv, index=False)
        
        try:
            # Llamamos al método de archivo del controlador
            resultados = self._controller.predecir_archivo(temp_csv, self._features)
            
            # Extraemos las predicciones
            predicciones = [r['prediccion'] for r in resultados]
            return predicciones
        finally:
            # Limpiamos el archivo temporal
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
