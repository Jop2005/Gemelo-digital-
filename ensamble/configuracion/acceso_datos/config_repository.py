"""
Repositorio para almacenar y recuperar configuraciones en formato JSON.
"""

import os
import json
import glob
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

from .config import get_config


class ConfigRepository:
    """Repositorio para almacenar y recuperar configuraciones en formato JSON."""
    
    def __init__(self, base_path: str = None):
        if base_path is None:
            config = get_config()
            base_path = str(config.carpeta_config)
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Any] = {}
    
    def guardar(self, config: Dict, nombre: str) -> Path:
        """Guarda una configuración sin versionado (sobrescribe)."""
        archivo = self._base_path / f"{nombre}.json"
        self._escribir_json(archivo, config)
        return archivo
    
    def guardar_versionado(self, config: Dict, nombre: str) -> Path:
        """Guarda una configuración con timestamp en el nombre."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archivo = self._base_path / f"{nombre}_{timestamp}.json"
        self._escribir_json(archivo, config)
        return archivo
    
    def cargar(self, nombre: str) -> Optional[Dict]:
        """Carga una configuración por nombre (con o sin timestamp)."""
        if nombre in self._cache:
            return self._cache[nombre]
        
        archivo = self._buscar_archivo(nombre)
        if not archivo:
            return None
        
        with open(archivo, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self._cache[nombre] = config
        return config
    
    def listar(self, patron: Optional[str] = None) -> List[str]:
        """Lista los nombres de todas las configuraciones."""
        if patron:
            archivos = glob.glob(str(self._base_path / f"{patron}*.json"))
        else:
            archivos = glob.glob(str(self._base_path / "*.json"))
        
        return [self._nombre_sin_extension(a) for a in archivos]
    
    def eliminar(self, nombre: str) -> bool:
        """Elimina una configuración por nombre."""
        archivo = self._base_path / f"{nombre}.json"
        if not archivo.exists():
            return False
        
        archivo.unlink()
        self._cache.pop(nombre, None)
        return True
    
    def limpiar_cache(self) -> None:
        """Limpia la caché de configuraciones."""
        self._cache.clear()
        
    def _escribir_json(self, archivo: Path, config: Dict) -> None:
        """Escribe un diccionario como JSON en el archivo."""
        with open(archivo, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def _buscar_archivo(self, nombre: str) -> Optional[Path]:
        """Busca el archivo (exacto o versionado) más reciente."""
        archivo_exacto = self._base_path / f"{nombre}.json"
        if archivo_exacto.exists():
            return archivo_exacto
        
        pattern = self._base_path / f"{nombre}_*.json"
        archivos = list(glob.glob(str(pattern)))
        
        if not archivos:
            return None
        
        return Path(max(archivos, key=os.path.getctime))
    
    @staticmethod
    def _nombre_sin_extension(ruta: str) -> str:
        """Retorna el nombre del archivo sin extensión .json."""
        return os.path.basename(ruta).replace('.json', '')