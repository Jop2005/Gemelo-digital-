from abc import ABC, abstractmethod
from typing import Dict, Any, List

class ModeloRepositoryInterface(ABC):
    
    @abstractmethod
    def cargar(self, nombre: str) -> Any:
        pass
    
    @abstractmethod
    def cargar_todos(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def listar_modelos(self) -> List[str]:
        pass