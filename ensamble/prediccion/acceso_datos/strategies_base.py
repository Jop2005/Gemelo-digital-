from abc import ABC, abstractmethod
from .data_loader import DataLoader

class DatasetStrategy(ABC):
    """Interfaz Strategy para carga de datasets"""
    
    @abstractmethod
    def get_loader(self) -> DataLoader:
        pass