import numpy as np
import pandas as pd
from itertools import product
from typing import Dict, Any, List, Optional
from .estrategia import EstrategiaSegmentacion

class SegmentacionPercentiles(EstrategiaSegmentacion):
    
    def __init__(self):
        self._variables: List[str] = []
        self._num_segmentos: int = 0
        self._umbrales: Dict[str, np.ndarray] = {}
        self._combinaciones: List[tuple] = []
    
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> None:
        self._variables = config.get('variables', [])
        self._calcular_umbrales(df)
        if len(self._variables) > 1:
            self._generar_combinaciones()
    
    def _calcular_umbrales(self, df: pd.DataFrame) -> None:
        for variable in self._variables:
            valores = df[variable]
            umbrales = np.unique(np.percentile(valores, np.linspace(0, 100, 101)))
            if len(umbrales) < 2:
                umbrales = np.array([valores.min(), valores.max()])
            self._umbrales[variable] = umbrales
        self._num_segmentos = len(self._umbrales[self._variables[0]]) - 1
    
    def _generar_combinaciones(self) -> None:
        intervalos_por_variable = []
        for variable in self._variables:
            intervalos = self._crear_intervalos(self._umbrales[variable])
            intervalos_por_variable.append(intervalos)
        self._combinaciones = list(product(*intervalos_por_variable))
        self._num_segmentos = len(self._combinaciones)
    
    def _crear_intervalos(self, umbrales: np.ndarray) -> List[tuple]:
        return [(umbrales[i], umbrales[i+1]) for i in range(len(umbrales) - 1)]
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if len(self._variables) == 1:
            return self._transformar_unico(X)
        return self._transformar_multivariable(X)
    
    def _transformar_unico(self, X: pd.DataFrame) -> np.ndarray:
        variable = self._variables[0]
        valores = X[variable].values if hasattr(X[variable], 'values') else X[variable]
        umbrales = self._umbrales[variable]
        return np.digitize(valores, umbrales[1:-1], right=False)
    
    def _transformar_multivariable(self, X: pd.DataFrame) -> np.ndarray:
        num_filas = len(X)
        segmentos = np.zeros(num_filas, dtype=int)
        filas = X.to_dict('records')
        for indice, fila in enumerate(filas):
            indices = self._obtener_indices_intervalo(fila)
            segmentos[indice] = self._buscar_combinacion(tuple(indices))
        return segmentos
    
    def _obtener_indices_intervalo(self, fila: dict) -> List[int]:
        indices = []
        for variable in self._variables:
            valor = fila[variable]
            indice = self._encontrar_intervalo(valor, self._umbrales[variable])
            indices.append(indice)
        return indices
    
    def _encontrar_intervalo(self, valor: float, umbrales: np.ndarray) -> int:
        for i in range(len(umbrales) - 1):
            if umbrales[i] <= valor <= umbrales[i+1]:
                return i
        return 0
    
    def _buscar_combinacion(self, combinacion: tuple) -> int:
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
    
    def get_umbrales(self) -> Optional[np.ndarray]:
        if len(self._variables) == 1:
            return self._umbrales[self._variables[0]]
        return None
    
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