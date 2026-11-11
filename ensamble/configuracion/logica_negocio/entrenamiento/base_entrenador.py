from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from configuracion.acceso_datos.context import DatasetContext
from configuracion.acceso_datos.config import get_config
from configuracion.acceso_datos.model_repository_interface import ModeloRepositoryInterface
from ..resultado_entrenamiento import ResultadoEntrenamiento

class BaseEntrenador(ABC):
    
    def __init__(
        self,
        context: DatasetContext,
        modelo_repository: ModeloRepositoryInterface,
        test_size: float = None
    ):
        self._context = context
        self._config = get_config()
        self._modelo_repository = modelo_repository
        self._test_size = test_size or self._config.test_size
        self._cargar_datos()
    
    def _cargar_datos(self) -> None:
        loader = self._context.get_loader()
        self._X_train, self._X_test, self._y_train, self._y_test = loader.obtener_train_test(
            test_size=self._test_size
        )
    
    @abstractmethod
    def _get_combinaciones(self) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def _crear_modelo(self, params: Dict[str, Any]):
        pass
    
    @abstractmethod
    def _get_nombre_modelo(self) -> str:
        pass
    
    def _calcular_mape(self, reales: np.ndarray, predichos: np.ndarray) -> float:
        mask = reales != 0
        if not mask.any():
            return float('inf')
        return np.mean(np.abs((reales[mask] - predichos[mask]) / reales[mask])) * 100
    
    def _calcular_mae_naive(self) -> float:
        y = self._y_train.values if hasattr(self._y_train, 'values') else np.array(self._y_train)
        if len(y) < 2:
            return 0.0
        return mean_absolute_error(y[1:], y[:-1])
    
    def _entrenar_y_evaluar(self, params: Dict[str, Any]) -> Tuple[Any, float, float, float, float, float]:
        modelo = self._crear_modelo(params)
        modelo.fit(self._X_train, self._y_train)
        y_pred = modelo.predict(self._X_test)
        
        mae = mean_absolute_error(self._y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self._y_test, y_pred))
        r2 = r2_score(self._y_test, y_pred)
        mape = self._calcular_mape(self._y_test.values, y_pred)
        mase = mae / self._calcular_mae_naive() if self._calcular_mae_naive() > 0 else float('inf')
        
        return modelo, mae, rmse, r2, mape, mase
    
    def _encontrar_mejor_modelo(self):
        combinaciones = self._get_combinaciones()
        mejor = {'mae': float('inf')}
        
        for params in combinaciones:
            modelo, mae, rmse, r2, mape, mase = self._entrenar_y_evaluar(params)
            if mae < mejor['mae']:
                mejor = {
                    'modelo': modelo,
                    'params': params,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'mape': mape,
                    'mase': mase
                }
        
        return mejor
    
    def _guardar_modelo(self, modelo, params, mae, rmse, r2):
        params_con_nombre = params.copy()
        params_con_nombre['nombre_modelo'] = self._get_nombre_modelo()
        return self._modelo_repository.guardar(modelo, params_con_nombre, mae, rmse, r2)
    
    def entrenar(self) -> ResultadoEntrenamiento:
        mejor = self._encontrar_mejor_modelo()
        self._guardar_modelo(mejor['modelo'], mejor['params'], mejor['mae'], mejor['rmse'], mejor['r2'])
        
        return ResultadoEntrenamiento(
            modelo=mejor['modelo'],
            params=mejor['params'],
            mae=mejor['mae'],
            rmse=mejor['rmse'],
            r2=mejor['r2'],
            mape=mejor['mape'],
            mase=mejor['mase'],
            tipo='ml_model'
        )