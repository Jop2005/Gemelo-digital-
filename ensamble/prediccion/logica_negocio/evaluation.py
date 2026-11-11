import os
import glob
import joblib
import numpy as np
from typing import Dict

class ModelEvaluator:
    
    def __init__(self, carpeta_resultados: str = None):
        if carpeta_resultados is None:
            from pathlib import Path
            root = Path(__file__).parent.parent.parent.parent
            carpeta_resultados = str(root / 'artifacts' / 'resultados')
        self._carpeta = carpeta_resultados
    
    def cargar_metricas(self) -> Dict[str, Dict[str, float]]:
        eval_files = glob.glob(f'{self._carpeta}/evaluacion_*.joblib')
        if not eval_files:
            return {}
        eval_path = max(eval_files, key=os.path.getctime)
        evaluacion = joblib.load(eval_path)
        metricas_globales = {}
        for nombre, metricas in evaluacion.get('metricas', {}).items():
            if 'global' in metricas:
                metricas_globales[nombre] = metricas['global']
            elif metricas:
                maes = [m.get('MAE', 0) for m in metricas.values() if isinstance(m, dict)]
                rmses = [m.get('RMSE', 0) for m in metricas.values() if isinstance(m, dict)]
                if maes:
                    metricas_globales[nombre] = {
                        'MAE': sum(maes) / len(maes),
                        'RMSE': sum(rmses) / len(rmses)
                    }
        return metricas_globales