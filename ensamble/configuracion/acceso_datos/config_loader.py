import os
import glob
from typing import Optional, Dict, Any, Tuple
from .config import get_config
from .config_repository import ConfigRepository

class ConfigLoader:
    """Carga configuraciones de ensamblaje y segmentación desde archivos JSON."""
    
    def __init__(self):
        self._config = get_config()
        self._repo = ConfigRepository(base_path=str(self._config.carpeta_config))
        self._cache: Dict[str, Any] = {}
    
    def cargar_ensamblaje(self, usar_cache: bool = True) -> Optional[Dict]:
        """Carga la configuración de ensamblaje más reciente."""
        if usar_cache and 'ensamblaje' in self._cache:
            return self._cache['ensamblaje']
        
        archivo = self._obtener_archivo_mas_reciente('config_ensamblaje_*.json')
        if not archivo:
            return None
        
        config = self._repo.cargar(archivo)
        self._cache['ensamblaje'] = config
        return config
    
    def cargar_segmentacion(self, usar_cache: bool = True) -> Optional[Dict]:
        """Carga la configuración de segmentación más reciente."""
        if usar_cache and 'segmentacion' in self._cache:
            return self._cache['segmentacion']
        
        archivo = self._obtener_archivo_mas_reciente('segmentacion_*.json')
        if not archivo:
            return None
        
        config = self._repo.cargar(archivo)
        self._cache['segmentacion'] = config
        return config
    
    def cargar_por_nombre(self, nombre: str, usar_cache: bool = True) -> Optional[Dict]:
        """Carga una configuración específica por su nombre base."""
        if usar_cache and nombre in self._cache:
            return self._cache[nombre]
        
        try:
            config = self._repo.cargar(nombre)
            self._cache[nombre] = config
            return config
        except Exception:
            return None
    
    def listar_todas(self) -> list:
        """Lista todas las configuraciones disponibles."""
        return self._repo.listar()
    
    def limpiar_cache(self) -> None:
        """Limpia la caché de configuraciones."""
        self._cache.clear()
    
    def recargar(self) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Recarga todas las configuraciones (limpia caché y vuelve a cargar)."""
        self.limpiar_cache()
        return self.cargar_ensamblaje(), self.cargar_segmentacion()
        
    def _obtener_archivo_mas_reciente(self, patron: str) -> Optional[str]:
        """Retorna el nombre base del archivo más reciente que coincide con el patrón."""
        patron_completo = str(self._config.carpeta_config / patron)
        archivos = glob.glob(patron_completo)
        
        if not archivos:
            return None
        
        archivo_reciente = max(archivos, key=os.path.getctime)
        return os.path.basename(archivo_reciente).replace('.json', '')