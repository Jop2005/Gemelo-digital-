"""
Cargador de datos - Fachada para el subsistema de carga de datos.

Responsabilidad: Coordinar los componentes de carga, preprocesamiento y split.
Mantiene compatibilidad con el código existente.
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import pandas as pd

from .file_reader import FileReader
from .data_preprocessor import DataPreprocessor
from .data_splitter import DataSplitter
from .config_repository import ConfigRepository


class DataLoader:
    """
    Fachada para el subsistema de carga de datos.
    
    Coordina:
    - FileReader: Lectura de archivos
    - DataPreprocessor: Limpieza y validación
    - DataSplitter: División train/test
    
    Mantiene la misma interfaz pública para compatibilidad.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Diccionario de configuración (opcional)
        """
        self._config = config or self._crear_config_default()
        self._df: Optional[pd.DataFrame] = None
        self._X: Optional[pd.DataFrame] = None
        self._y: Optional[pd.Series] = None
        
        self._root = Path(__file__).parent.parent.parent.resolve()
        # Componentes internos
        self._file_reader = FileReader()
        self._preprocessor = self._crear_preprocessor()
        self._splitter = DataSplitter(test_size=self._config.get('test_size', 0.2))
    
    def _crear_config_default(self) -> Dict[str, Any]:
        """Crea configuración por defecto."""
        return {
            'data_path': None,
            'features': [],
            'target': None,
            'month_map': None,
            'test_size': 0.2,
            'random_state': 42
        }
    
    def _crear_preprocessor(self) -> DataPreprocessor:
        """Crea el preprocesador con la configuración actual."""
        return DataPreprocessor(
            features=self._config.get('features', []),
            target=self._config.get('target'),
            month_map=self._config.get('month_map')
        )
    
    # ==================== MÉTODOS PÚBLICOS (Interfaz compatible) ====================
    
    @classmethod
    def from_json(cls, ruta_json: str) -> 'DataLoader':
        """Crea un DataLoader desde archivo JSON."""
        if not os.path.exists(ruta_json):
            raise FileNotFoundError(f"Configuración no encontrada: {ruta_json}")
        
        with open(ruta_json, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return cls(config)
    
    def guardar_configuracion(self, nombre: str = "data_config") -> None:
        """Guarda la configuración actual."""
        repo = ConfigRepository()
        repo.guardar(self._config, nombre)
    
    @classmethod
    def cargar_configuracion(cls, nombre: str = "data_config") -> 'DataLoader':
        """Carga configuración guardada."""
        repo = ConfigRepository()
        config = repo.cargar(nombre)
        return cls(config)
    
    def cargar_datos(self, ruta: Optional[str] = None) -> pd.DataFrame:
        """
        Carga datos desde archivo.
        
        Args:
            ruta: Ruta al archivo (usa config si es None)
        
        Returns:
            DataFrame con los datos cargados
        """
        ruta = ruta or self._config.get('data_path')
        if not ruta:
            raise ValueError("No se especificó ruta de datos")
        
        ruta_path = Path(ruta)
        if not ruta_path.is_absolute():
            ruta_path =  self._root / ruta_path

        self._df = self._file_reader.read(str(ruta_path))
        return self._df
    
    def establecer_dataframe(self, df: pd.DataFrame) -> None:
        self._df = df
        self._X = None
        self._y = None

    def preparar_features(self) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
       if self._df is None:
           raise ValueError("No hay datos cargados")
       self._X, self._y = self._preprocessor.prepare(self._df)
       return self._X, self._y
    
    def obtener_datos_completos(self) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Retorna X e y preparados."""
        if self._X is None or self._y is None:
            self.cargar_datos()
            self.preparar_features()
        return self._X, self._y
    
    def obtener_features(self) -> pd.DataFrame:
        """Retorna solo las features preparadas."""
        if self._X is None:
            self.cargar_datos()
            self.preparar_features()
        return self._X
    
    def obtener_train_test(
        self,
        test_size: Optional[float] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Divide datos en entrenamiento y prueba.
        
        Args:
            test_size: Proporción para test (usa config si es None)
        
        Returns:
            Tupla (X_train, X_test, y_train, y_test)
        """
        if self._X is None or self._y is None:
            self.cargar_datos()
            self.preparar_features()
        
        if test_size is not None:
            splitter = DataSplitter(test_size=test_size)
        else:
            splitter = self._splitter
        
        return splitter.split(self._X, self._y)
    
    def cambiar_dataset(
        self,
        ruta: str,
        features: Optional[List[str]] = None,
        target: Optional[str] = None,
        month_map: Optional[Dict] = None
    ) -> None:
        """Cambia la configuración del dataset."""
        self._config['data_path'] = ruta
        if features:
            self._config['features'] = features
        if target:
            self._config['target'] = target
        if month_map is not None:
            self._config['month_map'] = month_map
        
        self._preprocessor = self._crear_preprocessor()
        self._df = None
        self._X = None
        self._y = None
    
    # ==================== PROPIEDADES ====================
    
    @property
    def config(self) -> Dict[str, Any]:
        """Retorna una copia de la configuración."""
        return self._config.copy()
    
    @property
    def dataframe(self) -> Optional[pd.DataFrame]:
        """Retorna el DataFrame cargado."""
        return self._df