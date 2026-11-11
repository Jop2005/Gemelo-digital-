import joblib
import os
import glob
from pathlib import Path
from typing import Any, Dict, Optional, List

from .model_repository_interface import ModeloRepositoryInterface

class JoblibModelRepository(ModeloRepositoryInterface):
    
    def __init__(self, carpeta_modelos: str = None):
        if carpeta_modelos is None:
            from .config import get_config
            self._carpeta = get_config().carpeta_modelos
        else:
            self._carpeta = Path(carpeta_modelos)
        self._cache: Optional[Dict[str, Any]] = None
    
    def cargar(self, nombre: str) -> Optional[Any]:
        modelos = self.cargar_todos()
        return modelos.get(nombre)
    
    def cargar_todos(self, forzar_recarga: bool = False) -> Dict[str, Any]:
        if not forzar_recarga and self._cache is not None:
            return self._cache
        if not self._carpeta.exists():
            return {}
        modelos = {}
        for archivo in glob.glob(str(self._carpeta / "*.joblib")):
            nombre = os.path.basename(archivo).split('_')[0]
            modelo = self._cargar_archivo(archivo)
            if modelo is not None:
                modelos[nombre] = modelo
        self._cache = modelos
        return modelos
    
    def listar_modelos(self) -> List[str]:
        return list(self.cargar_todos().keys())
    
    def limpiar_cache(self) -> None:
        self._cache = None
    
    def _cargar_archivo(self, ruta_archivo: str) -> Optional[Any]:
        try:
            data = joblib.load(ruta_archivo)
            if isinstance(data, dict) and 'model' in data:
                return data['model']
            return data
        except (FileNotFoundError, EOFError, KeyError, ValueError):
            return None