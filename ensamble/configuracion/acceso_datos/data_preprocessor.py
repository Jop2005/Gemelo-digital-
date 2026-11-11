"""
Preprocesador de datos.

Responsabilidad única: Limpiar, validar y transformar datos para ML.
"""

import pandas as pd
from typing import List, Optional, Dict, Any


class DataPreprocessor:
    """
    Preprocesador de datos para modelos de ML.
    
    Responsabilidades:
    - Validar que las features existan en el DataFrame
    - Limpiar datos (eliminar nulos)
    - Aplicar transformaciones (mapa de meses, etc.)
    - Separar features (X) y target (y)
    """
    
    def __init__(
        self,
        features: List[str],
        target: Optional[str] = None,
        month_map: Optional[Dict[str, int]] = None
    ):
        """
        Args:
            features: Lista de nombres de columnas a usar como features
            target: Nombre de la columna objetivo (opcional)
            month_map: Diccionario para mapear nombres de meses a números
        """
        self._features = features
        self._target = target
        self._month_map = month_map or {}
    
    def prepare(
        self,
        df: pd.DataFrame
    ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Prepara los datos para entrenamiento o predicción.
        
        Args:
            df: DataFrame con los datos crudos
        
        Returns:
            Tupla (X, y) donde X son las features e y es el target (o None)
        
        Raises:
            ValueError: Si falta alguna feature requerida
        """
        df = df.copy()
        
        self._apply_month_mapping(df)
        self._validate_features(df)
        df = self._clean_data(df)
        
        X = df[self._features]
        y = df[self._target] if self._target and self._target in df.columns else None
        
        return X, y
    
    def _apply_month_mapping(self, df: pd.DataFrame) -> None:
        """Aplica el mapeo de meses si la columna existe."""
        if self._month_map and 'month' in df.columns:
            df['month'] = df['month'].astype(str).str.strip()
            df['month'] = df['month'].map(self._month_map)
    
    def _validate_features(self, df: pd.DataFrame) -> None:
        """Verifica que todas las features requeridas existan."""
        missing = [f for f in self._features if f not in df.columns]
        if missing:
            raise ValueError(f"Features faltantes en el dataset: {missing}")
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina filas con valores nulos en features o target."""
        cols_to_check = self._features.copy()
        if self._target and self._target in df.columns:
            cols_to_check.append(self._target)
        
        return df[cols_to_check].dropna()