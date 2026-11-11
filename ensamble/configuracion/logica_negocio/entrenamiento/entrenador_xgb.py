from xgboost import XGBRegressor
from configuracion.logica_negocio.entrenamiento.base_entrenador import BaseEntrenador
from configuracion.acceso_datos.model_repository_interface import ModeloRepositoryInterface
from configuracion.acceso_datos.context import DatasetContext

class EntrenadorXGBoost(BaseEntrenador):
    
    def __init__(self, context: DatasetContext, 
                 modelo_repository: ModeloRepositoryInterface,
                 test_size: float = None):
        super().__init__(context, modelo_repository, test_size)
    
    def _get_nombre_modelo(self) -> str:
        return 'XGBoost'
    
    def _get_combinaciones(self):
        return [
        # 1. Learning rate bajo + muchos estimadores
        {'n_estimators': 1000, 'max_depth': 5, 'learning_rate': 0.01, 
         'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_lambda': 2.0, 'reg_alpha': 0.5},
        
        # 2. Regularización fuerte
        {'n_estimators': 800, 'max_depth': 4, 'learning_rate': 0.02, 
         'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_lambda': 5.0, 'reg_alpha': 1.0, 'gamma': 0.3},
        
        # 3. Control de sobreajuste
        {'n_estimators': 1200, 'max_depth': 6, 'learning_rate': 0.015, 
         'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_lambda': 3.0, 'min_child_weight': 5},
        
        # 4. Equilibrado
        {'n_estimators': 600, 'max_depth': 5, 'learning_rate': 0.025, 
         'subsample': 0.75, 'colsample_bytree': 0.75, 'reg_lambda': 2.0, 'gamma': 0.1},
        
        # 5. Learning rate muy bajo
        {'n_estimators': 1500, 'max_depth': 5, 'learning_rate': 0.008, 
         'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_lambda': 2.0, 'reg_alpha': 0.3},
        
        # 6. Regularización L1 dominante
        {'n_estimators': 700, 'max_depth': 5, 'learning_rate': 0.02, 
         'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_lambda': 1.0, 'reg_alpha': 1.5, 'gamma': 0.2},
        
        # 7. Árboles poco profundos
        {'n_estimators': 900, 'max_depth': 3, 'learning_rate': 0.02, 
         'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 2.0, 'min_child_weight': 3},
        
        # 8. Gamma alto para poda
        {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.03, 
         'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_lambda': 1.5, 'gamma': 0.5, 'min_child_weight': 4},
        
        # 9. Subsample bajo
        {'n_estimators': 800, 'max_depth': 5, 'learning_rate': 0.02, 
         'subsample': 0.5, 'colsample_bytree': 0.5, 'reg_lambda': 3.0, 'reg_alpha': 0.7},
        
        # 10. Colsample_bytree bajo
        {'n_estimators': 700, 'max_depth': 6, 'learning_rate': 0.025, 
         'subsample': 0.8, 'colsample_bytree': 0.4, 'reg_lambda': 2.0},
        
        # 11. Máxima regularización
        {'n_estimators': 600, 'max_depth': 4, 'learning_rate': 0.03, 
         'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_lambda': 8.0, 'reg_alpha': 2.0, 'gamma': 0.4, 'min_child_weight': 7},
        
        # 12. Learning rate medio + muchos árboles
        {'n_estimators': 1300, 'max_depth': 5, 'learning_rate': 0.012, 
         'subsample': 0.65, 'colsample_bytree': 0.65, 'reg_lambda': 2.5, 'reg_alpha': 0.6},
        
        # 13. Balanceado con gamma
        {'n_estimators': 550, 'max_depth': 6, 'learning_rate': 0.028, 
         'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_lambda': 2.0, 'gamma': 0.15, 'min_child_weight': 3},
        
        # 14. Ultra conservador
        {'n_estimators': 500, 'max_depth': 7, 'learning_rate': 0.01, 
         'subsample': 0.7,  'reg_lambda': 5.0},
    ]
    
    def _crear_modelo(self, params):
        return XGBRegressor(**params, random_state=42, verbosity=0)