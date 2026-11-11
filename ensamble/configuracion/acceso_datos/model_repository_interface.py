"""
Interfaz para el repositorio de modelos (ISP + DIP).

Define el contrato específico para operaciones con modelos ML.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class ModeloRepositoryInterface(ABC):
    """
    Interfaz específica para repositorios de modelos.
    
    ISP: Solo métodos relacionados con modelos ML.
    DIP: Los controladores dependen de esta abstracción.
    """
    
    @abstractmethod
    def guardar(
        self,
        modelo: Any,
        params: Dict[str, Any],
        mae: float,
        rmse: float,
        r2: float
    ) -> str:
        """
        Guarda un modelo entrenado con sus metadatos.
        
        Args:
            modelo: Objeto del modelo entrenado
            params: Hiperparámetros utilizados
            mae: Error Absoluto Medio
            rmse: Raíz del Error Cuadrático Medio
            r2: Coeficiente de determinación
        
        Returns:
            Ruta del archivo guardado
        """
        pass
    
    @abstractmethod
    def cargar(self, nombre: str) -> Any:
        """
        Carga un modelo específico por nombre.
        
        Args:
            nombre: Nombre del modelo
        
        Returns:
            Modelo cargado o None
        """
        pass
    
    @abstractmethod
    def cargar_todos(self) -> Dict[str, Any]:
        """
        Carga todos los modelos disponibles.
        
        Returns:
            Diccionario nombre -> modelo entrenado
        """
        pass
    
    @abstractmethod
    def listar_modelos(self) -> List[str]:
        """
        Lista los nombres de los modelos disponibles.
        
        Returns:
            Lista de nombres de modelos
        """
        pass