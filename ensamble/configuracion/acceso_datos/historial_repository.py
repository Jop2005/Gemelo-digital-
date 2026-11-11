"""
Repositorio para almacenar y recuperar el historial de predicciones en CSV.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from .config import get_config


class HistorialRepository:
    """Almacena y recupera el historial de predicciones en CSV."""
    
    def __init__(self, archivo: str = None):
        if archivo is None:
            config = get_config()
            archivo = str(config.archivo_historial)
        self._archivo = Path(archivo)
    
    def guardar(self, entrada: Dict, prediccion: float, segmento: int, modelo: str) -> None:
        """
        Guarda una nueva predicción en el historial.
        Usa modo append para no cargar todo el archivo en memoria.
        """
        self._archivo.parent.mkdir(parents=True, exist_ok=True)
        nueva_fila = self._crear_fila(entrada, prediccion, segmento, modelo)
        df_nuevo = pd.DataFrame([nueva_fila])
        
        # Escribir en modo append sin leer todo el archivo
        if self._archivo.exists() and self._archivo.stat().st_size > 0:
            df_nuevo.to_csv(self._archivo, mode='a', header=False, index=False)
        else:
            df_nuevo.to_csv(self._archivo, mode='w', header=True, index=False)
    
    def leer_todas(self) -> pd.DataFrame:
        """Lee todas las predicciones del historial."""
        if self._existe_y_valido():
            return pd.read_csv(self._archivo)
        return pd.DataFrame()
    
    def obtener_ultimas(self, n: int = 10) -> pd.DataFrame:
        """Obtiene las últimas n predicciones."""
        df = self.leer_todas()
        return df.tail(n) if len(df) > n else df
    
    def limpiar(self) -> None:
        """Elimina el archivo de historial."""
        if self._archivo.exists():
            self._archivo.unlink()
    
    # ==================== MÉTODOS PRIVADOS ====================
    
    def _crear_fila(self, entrada: Dict, prediccion: float, segmento: int, modelo: str) -> Dict:
        """Crea una fila para el historial."""
        fila = {
            'timestamp': datetime.now().isoformat(),
            'prediccion': f"{prediccion:.2f}",
            'segmento': segmento,
            'modelo': modelo,
        }
        fila.update(entrada)
        return fila
    
    def _existe_y_valido(self) -> bool:
        """Verifica si el archivo existe y tiene contenido."""
        return self._archivo.exists() and self._archivo.stat().st_size > 0