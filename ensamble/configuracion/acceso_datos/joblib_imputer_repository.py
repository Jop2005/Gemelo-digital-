"""
Repositorio de imputadores usando joblib.
ÚNICA implementación concreta de ImputerRepositoryInterface.
"""

import joblib
from pathlib import Path
from typing import Optional

from .imputer_repository_interface import ImputerRepositoryInterface
from .config import get_config


class JoblibImputerRepository(ImputerRepositoryInterface):
    """
    Repositorio de imputadores usando joblib.
    
    Responsabilidades:
    - Guardar imputadores entrenados
    - Cargar imputadores desde disco
    
    DIP: Esta es la ÚNICA clase que conoce joblib para imputadores.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache = {}
            config = get_config()
            cls._instance._carpeta = config.carpeta_imputers
        return cls._instance
    
    # ==================== MÉTODOS DE LA INTERFAZ ====================
    
    def guardar(self, imputer, nombre: str) -> str:
        """
        Guarda un imputador entrenado usando joblib.
        
        Args:
            imputer: Objeto del imputador
            nombre: Nombre para guardarlo
        
        Returns:
            Ruta del archivo guardado
        """
        self._carpeta.mkdir(parents=True, exist_ok=True)
        archivo = self._carpeta / f"{nombre}.joblib"
        
        joblib.dump(imputer, archivo)
        self._cache[nombre] = imputer
        
        return str(archivo)
    
    def cargar(self, nombre: str = 'knn_imputer', forzar_recarga: bool = False) -> Optional[object]:
        """
        Carga un imputador por nombre usando joblib.
        
        Args:
            nombre: Nombre del imputador
            forzar_recarga: Si True, ignora la caché
        
        Returns:
            Imputador cargado o None
        """
        if not forzar_recarga and nombre in self._cache:
            return self._cache[nombre]
        
        archivo = self._carpeta / f"{nombre}.joblib"
        if not archivo.exists():
            return None
        
        try:
            imputer = joblib.load(archivo)
            self._cache[nombre] = imputer
            return imputer
        except Exception:
            return None
    
    # ==================== MÉTODOS ESPECÍFICOS ====================
    
    def cambiar_carpeta(self, carpeta: str) -> None:
        """Cambia la carpeta de imputadores."""
        self._carpeta = Path(carpeta)
        self._cache.clear()
    
    def limpiar_cache(self) -> None:
        """Limpia la caché de imputadores."""
        self._cache.clear()
    
    def recargar(self, nombre: str) -> Optional[object]:
        """Recarga un imputador específico."""
        self._cache.pop(nombre, None)
        return self.cargar(nombre, forzar_recarga=True)