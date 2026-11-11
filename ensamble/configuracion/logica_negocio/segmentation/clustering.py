import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import Dict, Any, List, Optional
from .estrategia import EstrategiaSegmentacion


class SegmentacionClustering(EstrategiaSegmentacion):
    
    def __init__(self):
        self._variables: List[str] = []
        self._num_segmentos: int = 0
        self._centroides: Optional[np.ndarray] = None
        self._orden: Optional[np.ndarray] = None
    
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> None:
        self._variables = config.get('variables', [])
        self._num_segmentos = config.get('n_segments', 4)
        
        X = df[self._variables].values
        kmeans = KMeans(n_clusters=self._num_segmentos, random_state=42, n_init='auto')
        kmeans.fit(X)
        
        self._centroides = kmeans.cluster_centers_
        self._orden = np.argsort(self._centroides.flatten())
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        valores = X[self._variables].values
        distancias = np.abs(valores - self._centroides.flatten())
        clusters = np.argmin(distancias, axis=1)
        segmentos = np.zeros(len(X), dtype=int)
        for i, cluster in enumerate(clusters):
            segmentos[i] = int(np.nonzero(self._orden == cluster)[0][0])
        return segmentos
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'tipo': 'clustering',
            'variables': self._variables,
            'n_segments': self._num_segmentos,
            'centroides': self._centroides.flatten().tolist() if self._centroides is not None else [],
            'orden': self._orden.tolist() if self._orden is not None else []
        }
    
    def get_variables(self) -> List[str]:
        return self._variables.copy()
    
    def get_num_segmentos(self) -> int:
        return self._num_segmentos
    
    def get_nombre_archivo(self) -> str:
        variables = '_'.join(self._variables).replace(' ', '_').replace('/', '_')
        return f"segmentacion_{variables}"
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SegmentacionClustering':
        instancia = cls()
        instancia._variables = config.get('variables', [])
        instancia._num_segmentos = config.get('n_segments', 4)
        centroides = config.get('centroides', [])
        orden = config.get('orden', [])
        if centroides and orden:
            instancia._centroides = np.array(centroides).reshape(-1, 1)
            instancia._orden = np.array(orden)
        return instancia