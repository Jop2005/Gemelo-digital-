import numpy as np
import pandas as pd
from typing import Dict, Any, List
from .estrategia import EstrategiaSegmentacion


class SegmentacionReglas(EstrategiaSegmentacion):
    
    def __init__(self):
        self._reglas: List[Dict] = []
        self._variables: List[str] = []
        self._num_segmentos: int = 0
    
    def fit(self, df: pd.DataFrame = None, config: Dict[str, Any] = None) -> None:
        _ = df
        self._reglas = config.get('reglas', [])
        self._num_segmentos = len(self._reglas)
        self._variables = self._extraer_variables()
    
    def _extraer_variables(self) -> List[str]:
        variables = set()
        for regla in self._reglas:
            for clave in regla.keys():
                if clave != 'segmento':
                    variables.add(clave)
        return list(variables)
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        num_filas = len(X)
        segmentos = np.full(num_filas, -1, dtype=int)
        filas = X.to_dict('records')
        
        for indice, fila in enumerate(filas):
            segmentos[indice] = self._obtener_segmento(fila)
        
        return segmentos
    
    def _obtener_segmento(self, fila: dict) -> int:
        for regla in self._reglas:
            if self._regla_cumple(fila, regla):
                return regla['segmento']
        return -1
    
    def _regla_cumple(self, fila: dict, regla: Dict) -> bool:
        for variable, condicion in regla.items():
            if variable == 'segmento':
                continue
            valor_actual = fila.get(variable)
            if valor_actual is None:
                return False
            if isinstance(condicion, list):
                if valor_actual not in condicion:
                    return False
            else:
                if valor_actual != condicion:
                    return False
        return True
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'tipo': 'rules',
            'variables': self._variables,
            'reglas': self._reglas,
            'n_segments': self._num_segmentos
        }
    
    def get_variables(self) -> List[str]:
        return self._variables.copy()
    
    def get_num_segmentos(self) -> int:
        return self._num_segmentos
    
    def get_nombre_archivo(self) -> str:
        return "segmentacion_reglas"
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SegmentacionReglas':
        instancia = cls()
        instancia._reglas = config.get('reglas', [])
        instancia._num_segmentos = len(instancia._reglas)
        instancia._variables = config.get('variables', [])
        return instancia