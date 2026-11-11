import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import pandas as pd

from .file_reader import FileReader
from .data_preprocessor import DataPreprocessor

class DataLoader:
    
    def __init__(self, config: Optional[Dict] = None):
        self._config = config or self._crear_config_default()
        self._df: Optional[pd.DataFrame] = None
        self._X: Optional[pd.DataFrame] = None
        self._y: Optional[pd.Series] = None
        self._file_reader = FileReader()
        self._preprocessor = DataPreprocessor(
            features=self._config.get('features', []),
            target=self._config.get('target'),
            month_map=self._config.get('month_map')
        )
    
    def _crear_config_default(self) -> Dict[str, Any]:
        return {
            'data_path': None,
            'features': [],
            'target': None,
            'month_map': None
        }
    
    @classmethod
    def from_json(cls, ruta_json: str) -> 'DataLoader':
        if not os.path.exists(ruta_json):
            raise FileNotFoundError(f"Configuración no encontrada: {ruta_json}")
        with open(ruta_json, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return cls(config)
    
    def cargar_datos(self, ruta: Optional[str] = None) -> pd.DataFrame:
        ruta = ruta or self._config.get('data_path')
        if not ruta:
            raise ValueError("No se especificó ruta de datos")
        self._df = self._file_reader.read(ruta)
        return self._df
    
    def preparar_features(self) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        if self._df is None:
            raise ValueError("No hay datos cargados")
        self._X, self._y = self._preprocessor.prepare(self._df)
        return self._X, self._y
    
    def obtener_features(self) -> pd.DataFrame:
        if self._X is None:
            self.cargar_datos()
            self.preparar_features()
        return self._X
    
    def obtener_datos_completos(self) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        if self._X is None or self._y is None:
            self.cargar_datos()
            self.preparar_features()
        return self._X, self._y
    
    @property
    def config(self) -> Dict[str, Any]:
        return self._config.copy()