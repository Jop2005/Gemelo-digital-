from typing import Dict, Any
from.estrategia import EstrategiaSegmentacion
from .percentiles import SegmentacionPercentiles
from .reglas import SegmentacionReglas
from .clustering import SegmentacionClustering

class FabricaSegmentacion:
    """Fábrica para crear estrategias de segmentación."""
    
    _REGISTRO = {
        'percentiles': SegmentacionPercentiles,
        'rules': SegmentacionReglas,
        'clustering': SegmentacionClustering
    }
    
    @classmethod
    def crear(cls, config: Dict[str, Any]) -> EstrategiaSegmentacion:
        """
        Crea la estrategia según la configuración.
        Si la clase tiene método from_config, lo usa. Si no, usa constructor vacío.
        """
        tipo = config.get('tipo')
        
        if tipo not in cls._REGISTRO:
            raise ValueError(f"Tipo de segmentación desconocido: {tipo}")
        
        clase_estrategia = cls._REGISTRO[tipo]
        
        # Usar from_config si existe (para cargar desde persistencia)
        if hasattr(clase_estrategia, 'from_config'):
            return clase_estrategia.from_config(config)
        
        # Fallback: constructor vacío
        return clase_estrategia()
    
    @classmethod
    def registrar(cls, tipo: str, clase):
        """Registra nuevos tipos dinámicamente."""
        cls._REGISTRO[tipo] = clase