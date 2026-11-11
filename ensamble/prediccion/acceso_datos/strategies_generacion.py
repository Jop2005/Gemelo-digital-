from pathlib import Path
from .strategies_base import DatasetStrategy
from .data_loader import DataLoader


class GeneracionStrategy(DatasetStrategy):
    def get_loader(self) -> DataLoader:
        # ==== RUTA ABSOLUTA (ajustado para integración) ====
        root = Path(__file__).parent.parent.parent.resolve()
        ruta = root / 'artifacts' / 'config' / 'generacion' / 'dataset_generacion.json'
        return DataLoader.from_json(str(ruta))
        # ==== FIN ====