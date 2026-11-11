from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from .models.base import ModeloBase


class Orchestrator:
    """
    Orquestador inteligente que gestiona la ejecución de modelos predictivos.
    
    Responsabilidades:
    - Seleccionar la estrategia de predicción según el segmento detectado
    - Ejecutar modelos individuales o ensamblajes ponderados
    - Proveer mecanismo de fallback cuando no hay configuración específica
    """
    
    def __init__(self):
        self._segmenter = None
        self._configuracion_por_segmento: Dict[int, dict] = {}
        self._modelos: Dict[str, ModeloBase] = {}
        self._variable_segmentacion: Optional[str] = None
        self._modelo_fallback_cache: Optional[str] = None
        self._nombre_modelo_fallback: Optional[str] = None
        self._metricas_evaluacion: Optional[dict] = None
    
    # ==================== MÉTODOS PÚBLICOS ====================
    
    def initialize(
        self,
        segmenter,
        configuracion_segmentos: dict,
        modelos: Dict[str, ModeloBase],
        variable_segmentacion: str = None,
        metricas_evaluacion: dict = None
    ) -> None:
        """Inicializa el orquestador con la configuración necesaria."""
        self._segmenter = segmenter
        self._modelos = modelos
        self._configuracion_por_segmento = self._normalizar_configuracion(configuracion_segmentos)
        self._variable_segmentacion = self._determinar_variable_segmentacion(
            variable_segmentacion, segmenter
        )
        self._metricas_evaluacion = metricas_evaluacion or {}
        self._modelo_fallback_cache = None
        self._nombre_modelo_fallback = None
    
    def predict(self, muestra: pd.DataFrame) -> dict:
        """Genera una predicción para una sola muestra."""
        self._validar_inicializacion()
        return self._predict_impl(muestra)
    
    def predict_batch(self, muestras: pd.DataFrame) -> List[dict]:
        """Genera predicciones para múltiples muestras en lote."""
        self._validar_inicializacion()
        segmentos = self._segmenter.transform(muestras)
        resultados = [None] * len(muestras)
        indices_por_segmento = self._agrupar_por_segmento(segmentos)
        
        for segmento, indices in indices_por_segmento.items():
            self._procesar_lote_por_segmento(muestras, indices, segmento, resultados)
        
        return resultados
    
    # ==================== MÉTODOS PRIVADOS - INICIALIZACIÓN ====================
    
    def _validar_inicializacion(self) -> None:
        """Valida que el orquestador esté correctamente inicializado."""
        if self._segmenter is None:
            raise ValueError("Orquestador no inicializado. Llame a initialize() primero.")
    
    def _normalizar_configuracion(self, configuracion: dict) -> Dict[int, dict]:
        """Convierte las claves de configuración a enteros."""
        normalizada = {}
        for key, value in configuracion.items():
            try:
                normalizada[int(key)] = value
            except (ValueError, TypeError):
                normalizada[key] = value
        return normalizada
    
    def _determinar_variable_segmentacion(
        self, 
        variable_explicita: Optional[str], 
        segmenter
    ) -> str:
        """Determina la variable usada para segmentación."""
        if variable_explicita:
            return variable_explicita
        variables = segmenter.variables if segmenter else []
        if len(variables) == 1:
            return variables[0]
        if len(variables) > 1:
            return '+'.join(variables)
        return 'desconocida'
    
    # ==================== MÉTODOS PRIVADOS - PREDICCIÓN ====================
    
    def _predict_impl(self, muestra: pd.DataFrame) -> dict:
        """Implementación interna de predicción para una muestra."""
        segmento = self._obtener_segmento(muestra)
        configuracion = self._obtener_configuracion(segmento)
        predicciones = self._ejecutar_estrategia_batch(muestra, configuracion)
        return self._construir_resultado(
            predicciones[0], segmento, configuracion
        )
    
    def _procesar_lote_por_segmento(
        self,
        muestras: pd.DataFrame,
        indices: List[int],
        segmento: int,
        resultados: List[dict]
    ) -> None:
        """Procesa un lote de muestras para un segmento específico."""
        indices_lista = [int(i) for i in indices]
        muestras_segmento = muestras.iloc[indices_lista].reset_index(drop=True)
        configuracion = self._obtener_configuracion(segmento)
        predicciones = self._ejecutar_estrategia_batch(
            muestras_segmento, configuracion
        )
        
        for i, idx in enumerate(indices_lista):
            resultados[idx] = self._construir_resultado(
                predicciones[i], segmento, configuracion
            )
    
    def _obtener_segmento(self, muestra: pd.DataFrame) -> int:
        """Determina el segmento de una muestra."""
        segmento_raw = self._segmenter.transform(muestra)
        return int(segmento_raw[0]) if len(segmento_raw) > 0 else -1
    
    def _obtener_configuracion(self, segmento: int) -> dict:
        """Obtiene la configuración para un segmento."""
        return self._configuracion_por_segmento.get(segmento, {})
    
    def _agrupar_por_segmento(self, segmentos: np.ndarray) -> Dict[int, List[int]]:
        """Agrupa índices por segmento para procesamiento por lotes."""
        indices_por_segmento = {}
        for idx, segmento in enumerate(segmentos):
            if segmento not in indices_por_segmento:
                indices_por_segmento[segmento] = []
            indices_por_segmento[segmento].append(idx)
        return indices_por_segmento
    
    # ==================== MÉTODOS PRIVADOS - EJECUCIÓN DE ESTRATEGIAS ====================
    
    def _ejecutar_estrategia_batch(
        self,
        muestras: pd.DataFrame,
        configuracion: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ejecuta la estrategia de predicción según la configuración."""
        if configuracion.get('modelo') == 'Ensemble':
            return self._ejecutar_ensamblaje(muestras, configuracion.get('pesos', {}))
        return self._ejecutar_modelo_individual(muestras, configuracion)
    
    def _ejecutar_modelo_individual(
        self,
        muestras: pd.DataFrame,
        configuracion: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ejecuta un modelo individual."""
        nombre_modelo = configuracion.get('modelo')
        if nombre_modelo and nombre_modelo in self._modelos:
            prediccion = self._modelos[nombre_modelo].predict(muestras)
            return self._extraer_vector(prediccion)
        return self._ejecutar_fallback(muestras)
    
    def _ejecutar_ensamblaje(
        self,
        muestras: pd.DataFrame,
        pesos: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ejecuta un ensamblaje ponderado de modelos.
        """
        n_muestras = len(muestras)
        pesos_activos, predicciones_activas = self._recopilar_predicciones_activas(
            muestras, pesos
        )
        
        if not predicciones_activas:
            return np.zeros(n_muestras)
        
        return self._calcular_prediccion_ponderada(
            predicciones_activas, pesos_activos
        )
    
    def _recopilar_predicciones_activas(
        self,
        muestras: pd.DataFrame,
        pesos: Dict[str, float]
    ) -> Tuple[List[float], List[np.ndarray]]:
        """
        Recopila las predicciones de los modelos con peso > 0.
        Retorna (pesos_activos, predicciones_activas).
        """
        pesos_activos = []
        predicciones_activas = []
        
        for nombre_modelo, peso in pesos.items():
            if peso > 0 and nombre_modelo in self._modelos:
                prediccion = self._modelos[nombre_modelo].predict(muestras)
                predicciones_activas.append(self._extraer_vector(prediccion))
                pesos_activos.append(peso)
        
        return pesos_activos, predicciones_activas
    
    def _calcular_prediccion_ponderada(
        self,
        predicciones_activas: List[np.ndarray],
        pesos_activos: List[float]
        ) -> np.ndarray:
        """
        Calcula la predicción ponderada
        """
        suma_pesos = sum(pesos_activos)
        prediccion_total = self._sumar_predicciones_ponderadas(
            predicciones_activas, pesos_activos
        )
        return prediccion_total / suma_pesos
    
    def _sumar_predicciones_ponderadas(
        self,
        predicciones_activas: List[np.ndarray],
        pesos_activos: List[float]
    ) -> np.ndarray:
        """Suma las predicciones ponderadas."""
        total = None
        for pred, peso in zip(predicciones_activas, pesos_activos):
            if total is None:
                total = pred * peso
            else:
                total += pred * peso
        return total
    
    # ==================== MÉTODOS PRIVADOS - FALLBACK ====================
    
    def _ejecutar_fallback(self, muestras: pd.DataFrame) -> np.ndarray:
        """Ejecuta el modelo de fallback cuando no hay configuración."""
        if not self._modelos:
            raise ValueError("No hay modelos disponibles para predicción")
        
        mejor_modelo = self._obtener_mejor_modelo_fallback()
        self._nombre_modelo_fallback = mejor_modelo
        prediccion = self._modelos[mejor_modelo].predict(muestras)
        
        return self._extraer_vector(prediccion)
    
    def _obtener_mejor_modelo_fallback(self) -> str:
        """Obtiene el mejor modelo según MAE o el primero disponible."""
        if self._modelo_fallback_cache is not None:
            return self._modelo_fallback_cache
        
        if self._metricas_evaluacion:
            mejor_modelo = self._seleccionar_por_mae(self._metricas_evaluacion)
            if mejor_modelo:
                self._modelo_fallback_cache = mejor_modelo
                return mejor_modelo
        
        self._modelo_fallback_cache = min(self._modelos.keys())
        return self._modelo_fallback_cache
    
    def _seleccionar_por_mae(self, metricas: dict) -> Optional[str]:
        """Selecciona el modelo con menor MAE."""
        mejor_modelo = None
        mejor_mae = float('inf')
        
        for nombre in self._modelos.keys():
            mae_global = metricas.get(nombre, {}).get('MAE', float('inf'))
            if mae_global < mejor_mae:
                mejor_mae = mae_global
                mejor_modelo = nombre
        
        return mejor_modelo
    
    
    
    # ==================== MÉTODOS PRIVADOS - UTILIDADES ====================
    
    @staticmethod
    def _extraer_vector(prediccion) -> np.ndarray:
        """Convierte la predicción en un array numpy."""
        if hasattr(prediccion, '__len__') and not isinstance(prediccion, str):
            return np.array(prediccion)
        return np.array([prediccion])
    
    def _construir_resultado(
        self,
        prediccion: float,
        segmento: int,
        configuracion: dict
        ) -> dict:
        """Construye el objeto de resultado con todos los metadatos."""
        
        return {
            'prediccion': float(prediccion),
            'segmento': int(segmento),
            'modelo': self._obtener_nombre_modelo(configuracion),
            'var_segmentacion': self._variable_segmentacion
        }
    
    def _obtener_nombre_modelo(self, configuracion: dict) -> str:
        if configuracion.get('modelo') == 'Ensemble':
            pesos = configuracion.get('pesos', {})
            modelos_participantes = ', '.join(pesos.keys())
            return f"Ensemble({modelos_participantes})"
        
        if not configuracion.get('modelo') and self._nombre_modelo_fallback:
            return self._nombre_modelo_fallback
        
        return configuracion.get('modelo', 'desconocido')
    
    def _obtener_rmse_modelo(self, configuracion: dict) -> float:
        nombre = configuracion.get('modelo', '')
        if not nombre and self._nombre_modelo_fallback:
            nombre = self._nombre_modelo_fallback
        return self._metricas_evaluacion.get(nombre, {}).get('RMSE', 0.0)