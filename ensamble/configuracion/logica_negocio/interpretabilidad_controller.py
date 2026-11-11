from typing import Optional
from .base_controller import BaseController
from .interpretabilidad_service import InterpretabilidadService
from configuracion.acceso_datos.model_repository_interface import ModeloRepositoryInterface


class InterpretabilidadController(BaseController):

    def __init__(
        self,
        context,
        model_repo: Optional[ModeloRepositoryInterface] = None
    ):
        super().__init__(context)
        self._service = InterpretabilidadService(context, model_repo)

    def ejecutar(self) -> dict:
        return self._service.ejecutar()

    def explicar_prediccion(self, datos: dict, features: list, modelo_nombre: str = None) -> dict:
        return self._service.explicar_prediccion(datos, features, modelo_nombre)