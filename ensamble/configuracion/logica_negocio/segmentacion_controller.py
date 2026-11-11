from .base_controller import BaseController
from .segmentacion_service import SegmentacionService


class SegmentacionController(BaseController):
    """Controlador que delega la generación de segmentación en el servicio."""

    def __init__(self, context):
        super().__init__(context)
        self._service = SegmentacionService(context)

    def ejecutar(self):
        return self._service.ejecutar()
