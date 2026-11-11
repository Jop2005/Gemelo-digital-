import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .estrategia import EstrategiaSegmentacion
from .fabrica import FabricaSegmentacion

class Segmenter:
    """
    Contexto para segmentación.
    Delega el trabajo en una estrategia.
    """
    
    def __init__(self, estrategia: Optional[EstrategiaSegmentacion] = None):
        self._estrategia = estrategia
    
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> 'Segmenter':
        """Configura la estrategia según la configuración."""
        self._estrategia = FabricaSegmentacion.crear(config)
        self._estrategia.fit(df, config)
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Delega en la estrategia."""
        if self._estrategia is None:
            raise ValueError("Segmentador no configurado. Llame a fit() primero.")
        return self._estrategia.transform(X)
    
    def get_config(self) -> Dict[str, Any]:
        """Delega en la estrategia."""
        if self._estrategia is None:
            return {'tipo': 'none', 'n_segments': 0}
        return self._estrategia.get_config()
    
    @property
    def variables(self) -> list:
        if self._estrategia is None:
            return []
        return self._estrategia.get_variables()
    
    @property
    def n_segments(self) -> int:
        if self._estrategia is None:
            return 0
        return self._estrategia.get_num_segmentos()
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Segmenter':
        """Crea un segmentador desde una configuración persistida."""
        estrategia = FabricaSegmentacion.crear(config)
        return cls(estrategia)