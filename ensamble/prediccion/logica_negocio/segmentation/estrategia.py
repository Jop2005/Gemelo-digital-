# domain/segmentation/estrategia.py
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


class EstrategiaSegmentacion(ABC):
    """Interfaz Strategy para segmentación."""
    
    @abstractmethod
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_variables(self) -> List[str]:
        pass
    
    @abstractmethod
    def get_num_segmentos(self) -> int:
        pass
    
    @abstractmethod
    def get_nombre_archivo(self) -> str:
        """Retorna el nombre base para el archivo de configuración."""
        pass