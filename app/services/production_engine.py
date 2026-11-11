from __future__ import annotations

import abc

import numpy as np
import pandas as pd

class ProductionEngine(abc.ABC):
    """Interfaz del motor que estima el factor de capacidad base (0-1)."""

    name: str = "base"

    @abc.abstractmethod
    def capacity_factor(
        self, features_df: pd.DataFrame, lat: float, lon: float
    ) -> np.ndarray:
        """
        Devuelve el factor de capacidad (0-1) por fila, en la geometría de
        REFERENCIA (20°/sur). La corrección por geometría real se aplica después
        como factor de orientación, de modo que ambos motores son comparables.
        """

        raise NotImplementedError