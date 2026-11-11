"""
Divisor de datos para series temporales.

Responsabilidad única: Dividir datos en train/test respetando orden cronológico.
"""

import pandas as pd
from typing import Tuple


class DataSplitter:
    """
    Divisor de datos para series temporales.
    
    Responsabilidad: Crear splits train/test que respeten el orden temporal.
    No usa validación cruzada estándar para evitar data leakage.
    """
    
    def __init__(self, test_size: float = 0.2):
        """
        Args:
            test_size: Proporción de datos para test (0.0 a 1.0)
        """
        if not 0.0 < test_size < 1.0:
            raise ValueError(f"test_size debe estar entre 0 y 1, recibido: {test_size}")
        
        self._test_size = test_size
    
    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Divide los datos en entrenamiento y prueba.
        
        Asume que los datos están en orden cronológico.
        Los primeros (1 - test_size)% van a train, el resto a test.
        
        Args:
            X: DataFrame con features
            y: Series con target
        
        Returns:
            Tupla (X_train, X_test, y_train, y_test)
        """
        if len(X) != len(y):
            raise ValueError(f"X y y deben tener misma longitud: {len(X)} vs {len(y)}")
        
        split_index = int(len(X) * (1 - self._test_size))
        
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]
        
        return X_train, X_test, y_train, y_test