import joblib
import os
import glob
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

from .model_repository_interface import ModeloRepositoryInterface
from .config import get_config

class JoblibModelRepository(ModeloRepositoryInterface):
    
    def __init__(self):
        self._config = get_config()
        self._carpeta = self._config.carpeta_modelos
        self._cache: Optional[Dict[str, Any]] = None
    
    def guardar(
        self,
        modelo: Any,
        params: Dict[str, Any],
        mae: float,
        rmse: float,
        r2: float
    ) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        nombre_modelo = params.get('nombre_modelo', 'modelo')
        model_data = {
            'model': modelo,
            'metadata': {
                'tipo': nombre_modelo,
                'params': params,
                'mae_test': float(mae),
                'rmse_test': float(rmse),
                'r2_test': float(r2),
                'fecha': datetime.now().isoformat()
            }
        }
        self._carpeta.mkdir(parents=True, exist_ok=True)
        nombre_archivo = f"{nombre_modelo}_{timestamp}.joblib"
        ruta_archivo = self._carpeta / nombre_archivo
        joblib.dump(model_data, ruta_archivo)
        self._cache = None
        return str(ruta_archivo)
    
    def cargar(self, nombre: str) -> Optional[Any]:
        modelos = self.cargar_todos()
        return modelos.get(nombre)
    
    def cargar_todos(self, forzar_recarga: bool = False) -> Dict[str, Any]:
        if not forzar_recarga and self._cache is not None:
            return self._cache
        if not self._carpeta.exists():
            return {}
        modelos = {}
        for archivo in glob.glob(str(self._carpeta / "*.joblib")):
            nombre = self._extraer_nombre_modelo(archivo)
            modelo = self._cargar_archivo(archivo)
            if modelo is not None:
                modelos[nombre] = modelo
        self._cache = modelos
        return modelos
    
    def listar_modelos(self) -> List[str]:
        modelos = self.cargar_todos()
        return list(modelos.keys())
    
    def limpiar_cache(self) -> None:
        self._cache = None
    
    def recargar(self) -> Dict[str, Any]:
        self.limpiar_cache()
        return self.cargar_todos(forzar_recarga=True)
    
    def _extraer_nombre_modelo(self, ruta_archivo: str) -> str:
        nombre_base = os.path.basename(ruta_archivo)
        return nombre_base.split('_')[0]
    
    def _cargar_archivo(self, ruta_archivo: str) -> Optional[Any]:
        try:
            data = joblib.load(ruta_archivo)
            if isinstance(data, dict) and 'model' in data:
                return data['model']
            return data
        except (FileNotFoundError, EOFError, KeyError, ValueError):
            return None