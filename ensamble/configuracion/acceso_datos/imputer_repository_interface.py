"""
Interfaz para el repositorio de imputadores (ISP + DIP).

Define el contrato específico para operaciones con imputadores.
"""

from abc import ABC, abstractmethod
from typing import Optional


class ImputerRepositoryInterface(ABC):
    """
    Interfaz específica para repositorios de imputadores.
    
    ISP: Solo métodos relacionados con imputadores.
    DIP: Los controladores dependen de esta abstracción.
    """
    
    @abstractmethod
    def guardar(self, imputer, nombre: str) -> str:
        """
        Guarda un imputador entrenado.
        
        Args:
            imputer: Objeto del imputador
            nombre: Nombre para guardarlo
        
        Returns:
            Ruta del archivo guardado
        """
        pass
    
    @abstractmethod
    def cargar(self, nombre: str) -> Optional[object]:
        """
        Carga un imputador por nombre.
        
        Args:
            nombre: Nombre del imputador
        
        Returns:
            Imputador cargado o None
        """
        pass