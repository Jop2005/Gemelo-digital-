import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from .estrategia import EstrategiaSegmentacion

class SegmentacionPercentiles(EstrategiaSegmentacion):
    
    def __init__(self):
        self._variables: List[str] = []
        self._num_segmentos: int = 0
        self._umbrales: Dict[str, np.ndarray] = {}
        self._combinaciones: List[tuple] = []
    
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> None:
        # Método vacío intencional - se carga desde configuración
        # Las estrategias de segmentación pueden cargarse desde config sin entrenamiento
        # Este método existe para cumplir con la interfaz EstrategiaSegmentacion
        pass
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if len(self._variables) == 1:
            variable = self._variables[0]
            valores = X[variable].values if hasattr(X[variable], 'values') else X[variable]
            return np.digitize(valores, self._umbrales[variable][1:-1], right=False)
        num_filas = len(X)
        segmentos = np.zeros(num_filas, dtype=int)
        filas = X.to_dict('records')
        for indice, fila in enumerate(filas):
            indices = [self._encontrar_intervalo(fila[v], self._umbrales[v]) for v in self._variables]
            segmentos[indice] = self._buscar_combinacion(tuple(indices))
        return segmentos
    
    def _encontrar_intervalo(self, valor, umbrales):
        for i in range(len(umbrales) - 1):
            if umbrales[i] <= valor <= umbrales[i+1]:
                return i
        return 0
    
    def _buscar_combinacion(self, combinacion):
        for segmento, comb in enumerate(self._combinaciones):
            if comb == combinacion:
                return segmento
        return -1
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'tipo': 'percentiles',
            'variables': self._variables,
            'n_segments': self._num_segmentos,
            'umbrales': {k: v.tolist() for k, v in self._umbrales.items()},
            'combinaciones': self._combinaciones
        }
    
    def get_variables(self) -> List[str]:
        return self._variables.copy()
    
    def get_num_segmentos(self) -> int:
        return self._num_segmentos
    
    def get_nombre_archivo(self) -> str:
        variables = '_'.join(self._variables).replace(' ', '_').replace('/', '_')
        return f"segmentacion_{variables}"
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SegmentacionPercentiles':
        instancia = cls()
        instancia._variables = config.get('variables', [])
        instancia._num_segmentos = config.get('n_segments', 1)
        instancia._umbrales = {k: np.array(v) for k, v in config.get('umbrales', {}).items()}
        instancia._combinaciones = config.get('combinaciones', [])
        return instancia