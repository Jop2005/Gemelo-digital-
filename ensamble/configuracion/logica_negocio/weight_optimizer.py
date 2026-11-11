import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Optional, Any


class WeightOptimizer:
    """
    Optimizador de pesos para ensamblaje de modelos.
    Encuentra la combinación óptima de pesos que minimiza el error.
    """
    
    def __init__(self):
        self._pesos_por_segmento: Dict[int, Dict[str, float]] = {}
        
    def optimizar(
        self,
        valores_reales: np.ndarray,
        predicciones_por_modelo: Dict[str, np.ndarray],
        segmento_id: Optional[int] = None
    ) -> Optional[Dict[str, float]]:
        """
        Encuentra los pesos óptimos que minimizan el error.
        """
        nombres_modelos = list(predicciones_por_modelo.keys())
        num_modelos = len(nombres_modelos)
        
        if num_modelos == 0:
            return None
        
        if num_modelos == 1:
            pesos = {nombres_modelos[0]: 1.0}
            if segmento_id is not None:
                self._pesos_por_segmento[segmento_id] = pesos
            return pesos
        
        # Preparar datos para optimización
        matriz_predicciones = self._preparar_matriz_predicciones(predicciones_por_modelo, nombres_modelos)
        pesos_iniciales = self._pesos_iniciales(num_modelos)
        
        # Configurar optimización
        restricciones = self._crear_restriccion_suma()
        limites = self._crear_limites(num_modelos)
        
        # Ejecutar optimización
        resultado = minimize(
            self._funcion_objetivo,
            pesos_iniciales,
            args=(matriz_predicciones, valores_reales),
            method='SLSQP',
            bounds=limites,
            constraints=restricciones
        )
        
        if not resultado.success:
            return None
        
        pesos_optimos = self._normalizar_pesos(resultado.x)
        pesos = dict(zip(nombres_modelos, pesos_optimos))
        
        if segmento_id is not None:
            self._pesos_por_segmento[segmento_id] = pesos
        
        return pesos
    
    def _preparar_matriz_predicciones(
        self,
        predicciones_por_modelo: Dict[str, np.ndarray],
        nombres_modelos: List[str]
    ) -> np.ndarray:
        """Convierte el diccionario de predicciones en una matriz"""
        return np.array([predicciones_por_modelo[nombre] for nombre in nombres_modelos])
    
    def _pesos_iniciales(self, num_modelos: int) -> np.ndarray:
        """Pesos iniciales igualmente distribuidos"""
        return np.ones(num_modelos) / num_modelos
    
    def _crear_restriccion_suma(self) -> List[Dict]:
        """Restricción: la suma de los pesos debe ser 1"""
        return [{'type': 'eq', 'fun': self._restriccion_suma_unitaria}]
    
    def _crear_limites(self, num_modelos: int) -> List[tuple]:
        """Límites: cada peso debe estar entre 0 y 1"""
        return [(0, 1) for _ in range(num_modelos)]
    
    def _funcion_objetivo(
        self,
        pesos: np.ndarray,
        matriz_predicciones: np.ndarray,
        valores_reales: np.ndarray
    ) -> float:
        """
        Función objetivo: minimiza el MAE.
        """
        prediccion_ponderada = np.dot(matriz_predicciones.T, pesos)
        return np.mean(np.abs(valores_reales - prediccion_ponderada))
    
    def _restriccion_suma_unitaria(self, pesos: np.ndarray) -> float:
        """Restricción: sum(pesos) - 1 = 0"""
        return np.sum(pesos) - 1
    
    def _normalizar_pesos(self, pesos: np.ndarray) -> np.ndarray:
        """Normaliza los pesos para que sumen exactamente 1"""
        suma = np.sum(pesos)
        if suma > 0:
            return pesos / suma
        return pesos
        
    def obtener_pesos(self, segmento_id: Optional[int] = None) -> Dict:
        """Obtiene los pesos almacenados para un segmento o todos."""
        if segmento_id is not None:
            return self._pesos_por_segmento.get(segmento_id, {})
        return self._pesos_por_segmento
    
    def limpiar_pesos(self) -> None:
        """Limpia todos los pesos almacenados"""
        self._pesos_por_segmento.clear()
        
    @property
    def pesos(self) -> Dict[int, Dict[str, float]]:
        """Retorna una copia de los pesos por segmento"""
        return self._pesos_por_segmento.copy()