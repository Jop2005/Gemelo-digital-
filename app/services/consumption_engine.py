"""
================================================================================
  INTERFAZ DE MOTOR DE CONSUMO (añadido para la integración del ensamble)
================================================================================
  Define el contrato que deben cumplir los motores de predicción de consumo.
  Permite intercambiar entre el método actual (perfiles) y el ensamble dinámico.
================================================================================
"""
from __future__ import annotations

import abc
from typing import Dict, List, Optional

import pandas as pd


class ConsumptionEngine(abc.ABC):
    """
    Interfaz del motor que estima el consumo eléctrico (kW) para un datetime.
    """

    name: str = "base"

    @abc.abstractmethod
    def predict(self, datetimes: List[str], config: Dict) -> List[float]:
        """
        Predice el consumo (kW) para una lista de datetimes ISO.

        Args:
            datetimes: Lista de strings ISO (ej. "2026-10-19T12:00:00").
            config: Configuración del sistema (para acceder a los electrodomésticos).

        Returns:
            Lista de consumos (kW) en el mismo orden que los datetimes.
        """
        raise NotImplementedError
