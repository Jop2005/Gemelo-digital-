from .strategies_base import DatasetStrategy

class DatasetContext:
    def __init__(self, strategy: DatasetStrategy = None):
        self._strategy = strategy

    def set_strategy(self, strategy: DatasetStrategy):
        self._strategy = strategy

    def get_loader(self):
        if self._strategy is None:
            raise ValueError("No se ha definido una estrategia de dataset")
        return self._strategy.get_loader()