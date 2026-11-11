from typing import List, Dict, Any
from .base_controller import BaseController
from .segmentacion_controller import SegmentacionController
from .evaluacion_controller import EvaluacionController
from .optimizacion_controller import OptimizacionController
from .interpretabilidad_controller import InterpretabilidadController
from ..acceso_datos.config_loader import ConfigLoader


class PipelineController(BaseController):
   
    def __init__(self, context):
        super().__init__(context)
        self._config_loader = ConfigLoader()
        self._segmentacion_controller = SegmentacionController(context)
        self._evaluacion_controller = EvaluacionController(context)
        self._optimizacion_controller = OptimizacionController(context)
        self._interpretabilidad_controller = InterpretabilidadController(context)
    
    def ejecutar_segmentacion(self) -> Any:
        return self._segmentacion_controller.ejecutar()
    
    def ejecutar_evaluacion(self) -> Dict[str, Any]:
        return self._evaluacion_controller.ejecutar(self._config_loader)
    
    def ejecutar_optimizacion(self) -> Dict[str, Any]:
        return self._optimizacion_controller.ejecutar(self._config_loader)
    
    def ejecutar_interpretabilidad(self) -> Dict[str, Any]:
        return self._interpretabilidad_controller.ejecutar()
    
    def ejecutar_entrenamiento_ml(self, entrenadores: List[Any]) -> List[Dict[str, Any]]:
        resultados = []
        for entrenador in entrenadores:
            resultado = entrenador.ejecutar()
            resultados.append({
                'nombre': entrenador.get_nombre(),
                'resultado': resultado
            })
        return resultados
    
    def ejecutar_entrenamiento_imputer(self, entrenador_imputer) -> Any:
        return entrenador_imputer.ejecutar()