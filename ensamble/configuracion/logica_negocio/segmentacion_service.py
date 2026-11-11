from datetime import datetime
from .segmentation.segmenter import Segmenter
from ..acceso_datos.config_repository import ConfigRepository


class SegmentacionService:
    """Servicio de lógica de negocio para generación de segmentación."""

    def __init__(self, context):
        self._context = context

    def ejecutar(self):
        loader = self._context.get_loader()
        config_dataset = loader.config
        config_segmentacion = self._extraer_config_segmentacion(config_dataset)

        if config_segmentacion is None:
            return None

        segmenter = Segmenter()

        if config_segmentacion.get('tipo') == 'rules':
            segmenter.fit(None, config_segmentacion)
        else:
            df = loader.cargar_datos()
            loader.preparar_features()
            segmenter.fit(df, config_segmentacion)

        self._guardar_configuracion(segmenter)
        return segmenter

    def _extraer_config_segmentacion(self, config_dataset: dict) -> dict:
        return config_dataset.get('segmentacion')

    def _guardar_configuracion(self, segmenter: Segmenter) -> None:
        config = segmenter.get_config()
        config['fecha'] = datetime.now().isoformat()
        repo = ConfigRepository()
        nombre = segmenter._estrategia.get_nombre_archivo()
        repo.guardar_versionado(config, nombre)
