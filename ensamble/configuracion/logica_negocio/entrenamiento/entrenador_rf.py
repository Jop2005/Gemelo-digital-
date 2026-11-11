from sklearn.ensemble import RandomForestRegressor
from configuracion.logica_negocio.entrenamiento.base_entrenador import BaseEntrenador
from configuracion.acceso_datos.model_repository_interface import ModeloRepositoryInterface
from configuracion.acceso_datos.context import DatasetContext

class EntrenadorRandomForest(BaseEntrenador):
    
    def __init__(self, context: DatasetContext, 
                 modelo_repository: ModeloRepositoryInterface,
                 test_size: float = None):
        super().__init__(context, modelo_repository, test_size)
    
    def _get_nombre_modelo(self) -> str:
        return 'RandomForest'
    
    def _get_combinaciones(self):
        return [
        {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2},
        {'n_estimators': 150, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2},
        {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 2},
        {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 1},
        {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 2},
        {'n_estimators': 200, 'max_depth': 8, 'min_samples_split': 5, 'min_samples_leaf': 2},
        {'n_estimators': 150, 'max_depth': 12, 'min_samples_split': 4, 'min_samples_leaf': 2},
        {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 10, 'min_samples_leaf': 4},
        {'n_estimators': 150, 'max_depth': 20, 'min_samples_split': 2, 'min_samples_leaf': 1},
        {'n_estimators': 250, 'max_depth': 25, 'min_samples_split': 2, 'min_samples_leaf': 1},
        {'n_estimators': 300, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 2},        
        {'n_estimators': 400, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2},        
        {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},        
        {'n_estimators': 800, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 2},
        ]
    
    def _crear_modelo(self, params):
        return RandomForestRegressor(
            **params,
            max_features=params.get('max_features', 'sqrt'),
            random_state=42,
            n_jobs=-1
        )