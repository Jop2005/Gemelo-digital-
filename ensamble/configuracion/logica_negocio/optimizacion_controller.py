from .base_controller import BaseController
from .optimizacion_service import OptimizacionService
from .excepciones import OptimizacionError
from ..acceso_datos.joblib_model_repository import JoblibModelRepository


class OptimizacionController(BaseController):
    
    def __init__(self, context, model_repo=None):
        super().__init__(context)
        self._service = OptimizacionService(context, model_repo or JoblibModelRepository())
    
    def ejecutar(self, config_loader):
        try:
            return self._service.ejecutar(config_loader)
        except OptimizacionError as e:
            return {'error': str(e)}