"""
Lector de archivos de datos.

Responsabilidad única: Leer datos desde diferentes formatos de archivo.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


class FileReader:
    """
    Lector de archivos CSV y Excel.
    
    Responsabilidad: Cargar datos desde el sistema de archivos.
    No valida, no limpia, no transforma. Solo lee.
    """
    
    def read(self, filepath: str) -> pd.DataFrame:
        """
        Lee un archivo CSV o Excel y retorna un DataFrame.
        
        Args:
            filepath: Ruta al archivo (.csv, .xlsx, .xls)
        
        Returns:
            DataFrame con los datos leídos
        
        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el formato no es soportado
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
        
        if path.suffix.lower() == '.csv':
            return self._read_csv(filepath)
        elif path.suffix.lower() in ['.xlsx', '.xls']:
            return self._read_excel(filepath)
        else:
            raise ValueError(f"Formato no soportado: {path.suffix}. Use .csv o .xlsx")
    
    def _read_csv(self, filepath: str) -> pd.DataFrame:
        """Lee archivo CSV con diferentes codificaciones."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                df.columns = df.columns.str.strip()
                return df
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"No se pudo leer el archivo CSV: {filepath}")
    
    def _read_excel(self, filepath: str) -> pd.DataFrame:
        """Lee archivo Excel."""
        df = pd.read_excel(filepath)
        df.columns = df.columns.str.strip()
        return df