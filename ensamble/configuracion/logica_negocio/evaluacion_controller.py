from .base_controller import BaseController
from .evaluacion_service import EvaluacionService
from .excepciones import EvaluacionError
from ..acceso_datos.joblib_model_repository import JoblibModelRepository

class EvaluacionController(BaseController):
    
    def __init__(self, context, model_repo=None):
        super().__init__(context)
        self._service = EvaluacionService(context, model_repo or JoblibModelRepository())
    
    def ejecutar(self, config_loader):
        try:
            return self._service.ejecutar(config_loader)
        except EvaluacionError as e:
            return {'error': str(e)}