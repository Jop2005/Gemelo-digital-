from abc import ABC, abstractmethod
from typing import Optional

class ImputerRepositoryInterface(ABC):
    
    @abstractmethod
    def cargar(self, nombre: str) -> Optional[object]:
        pass