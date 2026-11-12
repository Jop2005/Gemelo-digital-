from typing import Optional, Tuple, List, Dict, Any
from .base_controller import BaseController
from .prediccion_service import PrediccionService
from .excepciones import PrediccionError


class PrediccionController(BaseController):
    
    def __init__(self, context, model_repo, imputer_repo, historial_repo, file_reader, config_loader):
        super().__init__(context)
        self._servicio = PrediccionService(context, model_repo, imputer_repo, historial_repo, file_reader, config_loader)
    
    def predecir_archivo(self, ruta, features):
        try:
            return self._servicio.predecir_archivo(ruta, features)
        except (PrediccionError, ValueError) as e:
            return {'error': str(e)}, None
    
    def predecir_manual(self, datos):
        try:
            return self._servicio.predecir_manual(datos)
        except (PrediccionError, ValueError) as e:
            return {'error': str(e)}, None
    
    def explicar_prediccion(self, datos, features, modelo_nombre=None):
        try:
            return self._servicio.explicar_prediccion(datos, features, modelo_nombre)
        except PrediccionError as e:
            return {'error': str(e)}
    
    def esta_listo(self):
        return self._servicio.esta_listo()