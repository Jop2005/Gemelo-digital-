from .entrenamiento_rf_controller import EntrenadorRFController
from .entrenamiento_xgb_controller import EntrenadorXGBController

def descubrir_entrenadores_ml(context, repository):
    return [
        EntrenadorRFController(context, repository),
        EntrenadorXGBController(context, repository),
    ]