"""
Controlador para entrenamiento de Random Forest.
"""

from .base_controller import BaseController
from .entrenamiento.entrenador_rf import EntrenadorRandomForest
from configuracion.acceso_datos.model_repository_interface import ModeloRepositoryInterface
from .resultado_entrenamiento import ResultadoEntrenamiento


class EntrenadorRFController(BaseController):
    
    def __init__(self, context, repository: ModeloRepositoryInterface):
        """
        Args:
            context: Contexto del dataset
            repository: Implementación de ModeloRepositoryInterface
        """
        super().__init__(context)
        self._repository = repository
    
    def get_nombre(self) -> str:
        return "Random Forest"
    
    def ejecutar(self, test_size: float = 0.2) -> ResultadoEntrenamiento:
        """
        Ejecuta el entrenamiento del modelo Random Forest.
        
        Returns:
            ResultadoEntrenamiento: Value object con modelo, params y métricas.
        """
        entrenador = EntrenadorRandomForest(
            self.context,
            self._repository,
            test_size
        )
        return entrenador.entrenar()