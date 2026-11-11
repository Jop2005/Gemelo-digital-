from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass(frozen=True)
class ResultadoEntrenamiento:
    modelo: Any
    params: Dict[str, Any]
    mae: Optional[float]
    rmse: Optional[float]
    r2: Optional[float]
    mape: Optional[float]
    mase: Optional[float]
    tipo: str
    
    def es_modelo_ml(self) -> bool:
        return self.tipo == 'ml_model'
    
    def es_imputer(self) -> bool:
        return self.tipo == 'imputer'
    
    def tiene_metricas(self) -> bool:
        return self.mae is not None and self.rmse is not None and self.r2 is not None