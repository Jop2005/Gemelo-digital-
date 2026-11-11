from .base_controller import BaseController
from .entrenamiento_knn_service import EntrenamientoKNNService
from configuracion.acceso_datos.imputer_repository_interface import ImputerRepositoryInterface


class EntrenadorKNNController(BaseController):

    def __init__(self, context, imputer_repo: ImputerRepositoryInterface):
        super().__init__(context)
        self._service = EntrenamientoKNNService(context, imputer_repo)

    def get_nombre(self) -> str:
        return "KNN Imputer"

    def ejecutar(self):
        return self._service.ejecutar()
