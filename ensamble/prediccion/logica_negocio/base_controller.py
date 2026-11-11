from ..acceso_datos.context import DatasetContext
from ..acceso_datos.config import get_config

class BaseController:
    """Controlador base con dependencias comunes"""
    
    def __init__(self, context: DatasetContext):
        self.context = context
        self.config = get_config()
    
    def _get_loader(self):
        return self.context.get_loader()