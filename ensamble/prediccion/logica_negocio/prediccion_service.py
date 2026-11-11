import os
import glob
import joblib
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from .segmentation.segmenter import Segmenter
from .interpretability import InterpretabilityModule
from .orchestrator import Orchestrator
from .excepciones import OrquestadorNoInicializado, ModelosNoEncontrados, PrediccionError
from ..acceso_datos.config import get_config


class PrediccionService:
    
    MENSAJE_ORQUESTADOR_NO_INICIALIZADO = "El orquestador no está inicializado"
    
    def __init__(self, context, model_repo, imputer_repo, historial_repo, file_reader, config_loader):
        self._context = context
        self._model_repo = model_repo
        self._imputer_repo = imputer_repo
        self._historial = historial_repo
        self._file_reader = file_reader
        self._config_loader = config_loader
        self._orchestrator: Optional[Orchestrator] = None
        self._knn_imputer: Optional[KNNImputer] = None
        self._inicializar()
    
    def _inicializar(self):
        try:
            self._cargar_o_entrenar_imputer()
            self._inicializar_orchestrator()
        except FileNotFoundError as e:
             raise OrquestadorNoInicializado(
            f"Archivo de configuración no encontrado: {str(e)}. "
            "Ejecute primero el pipeline de configuración."
             )
    
    def _cargar_o_entrenar_imputer(self):
        self._knn_imputer = self._imputer_repo.cargar('knn_imputer')
        if self._knn_imputer is None:
            self._entrenar_knn_imputer()
    
    def _entrenar_knn_imputer(self):
        loader = self._context.get_loader()
        df = loader.obtener_features()
        knn = KNNImputer(n_neighbors=5)
        knn.fit(df)
        self._imputer_repo.guardar(knn, 'knn_imputer')
        self._knn_imputer = knn
    
    def _cargar_metricas_evaluacion(self) -> dict:
        try:
            config = get_config()
            carpeta = str(config.carpeta_resultados)
            eval_files = glob.glob(f'{carpeta}/evaluacion_*.joblib')
            if not eval_files:
                return {}
            eval_path = max(eval_files, key=os.path.getctime)
            evaluacion = joblib.load(eval_path)
            metricas = {}
            for nombre, m in evaluacion.get('metricas', {}).items():
                if 'global' in m:
                    metricas[nombre] = m['global']
                elif m:
                    maes = [v.get('MAE', 0) for v in m.values() if isinstance(v, dict)]
                    rmses = [v.get('RMSE', 0) for v in m.values() if isinstance(v, dict)]
                    if maes:
                        metricas[nombre] = {'MAE': sum(maes)/len(maes), 'RMSE': sum(rmses)/len(rmses)}
            return metricas
        except (FileNotFoundError, KeyError, ValueError):
            return {}
    
    def _inicializar_orchestrator(self):
        config_ensamblaje = self._config_loader.cargar_ensamblaje()
        if not config_ensamblaje:
            raise OrquestadorNoInicializado(
                "No se encontró configuración de ensamblaje. Ejecute primero el pipeline de configuración."
            )
        
        config_seg = self._config_loader.cargar_segmentacion()
        if not config_seg:
            raise OrquestadorNoInicializado(
                "No se encontró configuración de segmentación. Ejecute primero el pipeline de configuración."
            )
        
        modelos = self._model_repo.cargar_todos()
        if not modelos:
            raise OrquestadorNoInicializado(
                "No hay modelos entrenados. Ejecute primero el pipeline de configuración."
            )
        
        segmenter = Segmenter.from_config(config_seg)
        metricas = self._cargar_metricas_evaluacion()
        self._orchestrator = Orchestrator()
        self._orchestrator.initialize(segmenter, config_ensamblaje.get('segmentos', {}), modelos, metricas_evaluacion=metricas)
    
    def predecir_archivo(self, ruta, features):
        if not self._orchestrator:
            raise OrquestadorNoInicializado(self.MENSAJE_ORQUESTADOR_NO_INICIALIZADO)
        try:
            df = self._file_reader.read(ruta)
        except Exception as e:
            raise PrediccionError(f"Error al leer archivo: {str(e)}")
        if not self._validar_columnas(df, features):
            raise PrediccionError("Columnas requeridas no encontradas en el archivo")
        df_input = self._imputar(df[features].copy())
        resultados = self._orchestrator.predict_batch(df_input)
        self._guardar_historial_lote(df, resultados)
        return resultados
    
    def predecir_manual(self, datos):
        if not self._orchestrator:
            raise OrquestadorNoInicializado(self.MENSAJE_ORQUESTADOR_NO_INICIALIZADO)
        df_input = self._imputar(pd.DataFrame([datos]))
        resultado = self._orchestrator.predict(df_input)
        self._historial.guardar(datos, resultado['prediccion'], resultado['segmento'], resultado['modelo'])
        return resultado
    
    def explicar_prediccion(self, datos, features, modelo_nombre=None):
        if not self._orchestrator:
            raise OrquestadorNoInicializado(self.MENSAJE_ORQUESTADOR_NO_INICIALIZADO)
        modelos = self._model_repo.cargar_todos()
        if not modelos:
            raise ModelosNoEncontrados("No hay modelos disponibles")
        if modelo_nombre and modelo_nombre in modelos:
            modelo = {modelo_nombre: modelos[modelo_nombre]}
        else:
            modelo = modelos
            modelo_nombre = min(modelo.keys())
        interp = InterpretabilityModule(modelo, features)
        return interp.explain_prediction(modelo_nombre, pd.DataFrame([datos]))
    
    def esta_listo(self):
        return self._orchestrator is not None
    
    def _validar_columnas(self, df, features):
        return len([f for f in features if f not in df.columns]) == 0
    
    def _imputar(self, df):
        if self._knn_imputer is None:
            return df
        if df.isnull().any().any():
            X = self._knn_imputer.transform(df)
            return pd.DataFrame(X, columns=df.columns)
        return df
    
    def _guardar_historial_lote(self, df, resultados):
        for idx, row in df.iterrows():
            self._historial.guardar(
                {col: row[col] for col in df.columns},
                resultados[idx]['prediccion'],
                resultados[idx]['segmento'],
                resultados[idx]['modelo']
            )