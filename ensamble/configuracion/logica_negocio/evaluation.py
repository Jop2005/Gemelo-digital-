import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Any, Tuple, Optional

class ModelEvaluator:
    
    def __init__(self):
        self._resultados_globales: Dict[str, Dict[str, float]] = {}
        self._resultados_por_segmento: Dict[str, Dict[int, Dict[str, float]]] = {}
    
    def calcular_metricas(
        self,
        valores_reales: np.ndarray,
        valores_predichos: np.ndarray,
        y_train: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        metricas = {
            'MAE': self._calcular_mae(valores_reales, valores_predichos),
            'RMSE': self._calcular_rmse(valores_reales, valores_predichos),
            'R2': self._calcular_r2(valores_reales, valores_predichos),
            'MAPE': self._calcular_mape(valores_reales, valores_predichos)
        }
        if y_train is not None:
            metricas['MASE'] = self._calcular_mase(valores_reales, valores_predichos, y_train)
        return metricas
    
    def _calcular_mae(self, reales, predichos):
        return mean_absolute_error(reales, predichos)
    
    def _calcular_rmse(self, reales, predichos):
        return np.sqrt(mean_squared_error(reales, predichos))
    
    def _calcular_r2(self, reales, predichos):
        return r2_score(reales, predichos)
    
    def _calcular_mape(self, reales, predichos):
        mascara = reales != 0
        if not mascara.any():
            return np.inf
        return np.mean(np.abs((reales[mascara] - predichos[mascara]) / reales[mascara])) * 100
    
    def _calcular_mase(self, reales, predichos, y_train):
        if len(y_train) < 2:
            return np.inf
        mae_naive = mean_absolute_error(y_train[1:], y_train[:-1])
        if mae_naive == 0:
            return np.inf
        return self._calcular_mae(reales, predichos) / mae_naive
    
    def evaluar_modelo(self, nombre, reales, predichos, y_train=None):
        metricas = self.calcular_metricas(reales, predichos, y_train)
        self._resultados_globales[nombre] = metricas
        return metricas
    
    def evaluar_por_segmentos(self, nombre, reales, predichos, segmentos, y_train=None):
        if nombre not in self._resultados_por_segmento:
            self._resultados_por_segmento[nombre] = {}
        for seg in np.unique(segmentos):
            if seg < 0:
                continue
            mask = segmentos == seg
            if mask.sum() == 0:
                continue
            self._resultados_por_segmento[nombre][int(seg)] = self.calcular_metricas(
                reales[mask], predichos[mask], y_train
            )
        return self._resultados_por_segmento[nombre]
    
    def obtener_mejor_por_segmento(self):
        if not self._resultados_por_segmento:
            return {}
        todos = set()
        for r in self._resultados_por_segmento.values():
            todos.update(r.keys())
        mejor = {}
        for seg in sorted(todos):
            mejor_mae, mejor_modelo = float('inf'), None
            for nombre, resultados in self._resultados_por_segmento.items():
                if seg in resultados and resultados[seg]['MAE'] < mejor_mae:
                    mejor_mae = resultados[seg]['MAE']
                    mejor_modelo = nombre
            if mejor_modelo:
                mejor[seg] = {'model': mejor_modelo, 'mae': mejor_mae}
        return mejor
    
    def obtener_resumen(self):
        return {
            'global': self._resultados_globales,
            'segmentos': self._resultados_por_segmento,
            'mejores': self.obtener_mejor_por_segmento()
        }
    
    def limpiar_resultados(self):
        self._resultados_globales.clear()
        self._resultados_por_segmento.clear()