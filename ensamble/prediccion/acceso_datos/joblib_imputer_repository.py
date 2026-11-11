import joblib
from pathlib import Path
from typing import Optional

from .imputer_repository_interface import ImputerRepositoryInterface

class JoblibImputerRepository(ImputerRepositoryInterface):
    
    def __init__(self, carpeta_imputers: str = None):
        if carpeta_imputers is None:
            from .config import get_config
            self._carpeta = get_config().carpeta_imputers
        else:
            self._carpeta = Path(carpeta_imputers)
        self._cache = {}
    
    def cargar(self, nombre: str = 'knn_imputer') -> Optional[object]:
        if nombre in self._cache:
            return self._cache[nombre]
        archivo = self._carpeta / f"{nombre}.joblib"
        if not archivo.exists():
            return None
        try:
            imputer = joblib.load(archivo)
            self._cache[nombre] = imputer
            return imputer
        except (FileNotFoundError, EOFError, KeyError, ValueError):
            return None
    
    def limpiar_cache(self) -> None:
        self._cache.clear()