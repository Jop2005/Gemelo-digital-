"""
Pruebas de Caja Negra para el dominio CONSUMO
Ejecutar con: pytest test/consumo.py -v
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock

# Agregar directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# IMPORTS - CORREGIDOS: USAR REPOSITORIOS DE CONFIGURACIÓN
# =============================================================================

# Para PREDICCIÓN (usar repositorios de predicción - solo cargar)
from prediccion.acceso_datos.context import DatasetContext as PrediccionContext
from prediccion.acceso_datos.config import get_config
from prediccion.acceso_datos.joblib_model_repository import JoblibModelRepository as PrediccionModelRepo
from prediccion.acceso_datos.joblib_imputer_repository import JoblibImputerRepository as PrediccionImputerRepo
from prediccion.acceso_datos.historial_repository import HistorialRepository
from prediccion.acceso_datos.file_reader import FileReader
from prediccion.acceso_datos.config_loader import ConfigLoader
from prediccion.logica_negocio.prediccion_controller import PrediccionController
from prediccion.logica_negocio.orchestrator import Orchestrator

# Para CONFIGURACIÓN (usar repositorios de configuración - tienen guardar)
from configuracion.acceso_datos.context import DatasetContext as ConfigContext
from configuracion.acceso_datos.strategies_consumo import ConsumoStrategy
from configuracion.acceso_datos.joblib_model_repository import JoblibModelRepository as ConfigModelRepo
from configuracion.acceso_datos.joblib_imputer_repository import JoblibImputerRepository as ConfigImputerRepo

# Controladores de configuración
from configuracion.logica_negocio.segmentacion_controller import SegmentacionController
from configuracion.logica_negocio.entrenamiento_knn_controller import EntrenadorKNNController
from configuracion.logica_negocio.entrenamiento_rf_controller import EntrenadorRFController
from configuracion.logica_negocio.entrenamiento_xgb_controller import EntrenadorXGBController
from configuracion.logica_negocio.evaluacion_controller import EvaluacionController
from configuracion.logica_negocio.optimizacion_controller import OptimizacionController
from configuracion.logica_negocio.interpretabilidad_controller import InterpretabilidadController
from configuracion.logica_negocio.weight_optimizer import WeightOptimizer


# =============================================================================
# CONSTANTES
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"

def ruta_archivo(nombre: str) -> str:
    return str(DATA_DIR / nombre)

def archivo_existe(nombre: str) -> bool:
    return os.path.exists(ruta_archivo(nombre))


# =============================================================================
# FIXTURES - CORREGIDOS
# =============================================================================

@pytest.fixture(scope="session")
def config_consumo():
    """Configura el dominio de consumo"""
    get_config().set_dominio('consumo')
    return get_config()


@pytest.fixture(scope="session")
def controller_consumo(config_consumo):
    """
    Crea un controlador de PREDICCIÓN para el dominio de consumo.
    Este usa repositorios de predicción (solo cargar).
    """
    try:
        # Para predicción usamos el context de predicción con estrategia de consumo
        from configuracion.acceso_datos.strategies_consumo import ConsumoStrategy as ConfigConsumoStrategy
        strategy = ConfigConsumoStrategy()
        context = PrediccionContext(strategy)
        
        model_repo = PrediccionModelRepo()
        imputer_repo = PrediccionImputerRepo()
        historial_repo = HistorialRepository()
        file_reader = FileReader()
        config_loader = ConfigLoader()
        
        return PrediccionController(
            context, model_repo, imputer_repo, historial_repo, file_reader, config_loader
        )
    except FileNotFoundError:
        pytest.skip("Pipeline de configuración de consumo no ejecutado")
        return None


@pytest.fixture
def context_consumo_configuracion(config_consumo):
    """
    Contexto de CONFIGURACIÓN para el dominio de consumo.
    Usa repositorios de configuración (con guardar).
    """
    try:
        return ConfigContext(ConsumoStrategy())
    except FileNotFoundError:
        pytest.skip("Configuración de consumo no encontrada")
        return None


@pytest.fixture
def model_repo_configuracion(config_consumo):
    """
    Repositorio de modelos de CONFIGURACIÓN (tiene guardar).
    """
    return ConfigModelRepo()


@pytest.fixture
def imputer_repo_configuracion(config_consumo):
    """
    Repositorio de imputadores de CONFIGURACIÓN (tiene guardar).
    """
    return ConfigImputerRepo()


@pytest.fixture
def features_consumo():
    return ['Hora', 'Mes', 'tipo_dia_bin']


@pytest.fixture
def orchestrator():
    return Orchestrator()


@pytest.fixture
def weight_optimizer():
    return WeightOptimizer()


# =============================================================================
# FUNCIONES PARA CREAR ARCHIVOS DE PRUEBA
# =============================================================================

def crear_archivo_prueba(nombre, datos, columnas):
    """Crea un archivo CSV de prueba"""
    DATA_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(datos, columns=columnas)
    ruta = DATA_DIR / nombre
    df.to_csv(ruta, index=False)
    return ruta


def crear_datos_100_filas():
    """CP-10: Crea 100 filas para consumo"""
    np.random.seed(42)
    datos = {
        'Hora': np.random.choice(range(24), 100),
        'Mes': np.random.choice(range(1, 13), 100),
        'tipo_dia_bin': np.random.choice([0, 1], 100)
    }
    return crear_archivo_prueba('datos_100_filas.csv', datos, ['Hora', 'Mes', 'tipo_dia_bin'])


def crear_datos_sin_hora():
    """CP-12: Datos sin columna Hora"""
    np.random.seed(43)
    datos = {
        'Mes': np.random.choice(range(1, 13), 50),
        'tipo_dia_bin': np.random.choice([0, 1], 50)
    }
    return crear_archivo_prueba('datos_sin_hora.csv', datos, ['Mes', 'tipo_dia_bin'])


def crear_datos_con_nulos():
    """CP-13: Datos con 10% de valores nulos"""
    np.random.seed(44)
    n = 100
    datos = {
        'Hora': [np.random.choice(range(24)) if np.random.random() > 0.1 else np.nan for _ in range(n)],
        'Mes': [np.random.choice(range(1, 13)) if np.random.random() > 0.1 else np.nan for _ in range(n)],
        'tipo_dia_bin': [np.random.choice([0, 1]) if np.random.random() > 0.1 else np.nan for _ in range(n)]
    }
    return crear_archivo_prueba('datos_con_nulos.csv', datos, ['Hora', 'Mes', 'tipo_dia_bin'])


# =============================================================================
# PRUEBAS DE CAJA NEGRA - CONSUMO
# =============================================================================

class TestConsumo:
    """
    Pruebas de Caja Negra para el dominio Consumo.
    CP-01 a CP-15 (excepto CP-08, CP-09, CP-14, CP-15 que son de generación)
    """

    # CP-01: Generar segmentación (consumo)
    def test_cp01_generar_segmentacion_consumo(self, context_consumo_configuracion):
        if context_consumo_configuracion is None:
            pytest.skip("Contexto no disponible")
        controller = SegmentacionController(context_consumo_configuracion)
        segmenter = controller.ejecutar()
        assert segmenter is not None
        config = segmenter.get_config()
        assert config.get('tipo') == 'rules'
        assert segmenter.n_segments == 8

    # CP-02: Entrenar KNN Imputer - CORREGIDO: usa repositorio de configuración
    def test_cp02_entrenar_knn_imputer(self, context_consumo_configuracion, imputer_repo_configuracion):
        if context_consumo_configuracion is None:
            pytest.skip("Contexto no disponible")
        controller = EntrenadorKNNController(context_consumo_configuracion, imputer_repo_configuracion)
        imputer = controller.ejecutar()
        assert imputer is not None
        assert hasattr(imputer, 'transform')

    # CP-03: Entrenar Random Forest - CORREGIDO: usa repositorio de configuración
    def test_cp03_entrenar_random_forest(self, context_consumo_configuracion, model_repo_configuracion):
        if context_consumo_configuracion is None:
            pytest.skip("Contexto no disponible")
        controller = EntrenadorRFController(context_consumo_configuracion, model_repo_configuracion)
        resultado = controller.ejecutar()
        assert resultado is not None
        assert resultado.modelo is not None
        assert resultado.mae is not None

    # CP-04: Entrenar XGBoost - CORREGIDO: usa repositorio de configuración
    def test_cp04_entrenar_xgboost(self, context_consumo_configuracion, model_repo_configuracion):
        if context_consumo_configuracion is None:
            pytest.skip("Contexto no disponible")
        controller = EntrenadorXGBController(context_consumo_configuracion, model_repo_configuracion)
        resultado = controller.ejecutar()
        assert resultado is not None
        assert resultado.modelo is not None
        assert resultado.mae is not None

    # CP-05: Evaluar modelos - CORREGIDO: usa repositorio de configuración
    def test_cp05_evaluar_modelos(self, context_consumo_configuracion):
        if context_consumo_configuracion is None:
            pytest.skip("Contexto no disponible")
        evaluador = EvaluacionController(context_consumo_configuracion)
        resultado = evaluador.ejecutar(ConfigLoader())
        assert resultado is not None
        assert 'metricas' in resultado

    # CP-06: Optimizar ensamblaje - CORREGIDO: usa repositorio de configuración
    def test_cp06_optimizar_ensamblaje(self, context_consumo_configuracion):
        if context_consumo_configuracion is None:
            pytest.skip("Contexto no disponible")
        optimizador = OptimizacionController(context_consumo_configuracion)
        resultado = optimizador.ejecutar(ConfigLoader())
        assert resultado is not None
        assert 'segmentos' in resultado

    # CP-07: Generar interpretabilidad
    def test_cp07_generar_interpretabilidad(self, context_consumo_configuracion):
        if context_consumo_configuracion is None:
            pytest.skip("Contexto no disponible")
        controller = InterpretabilidadController(context_consumo_configuracion)
        resultado = controller.ejecutar()
        assert resultado is not None
        assert len(resultado) > 0

    # CP-10: Predicción desde archivo
    def test_cp10_prediccion_desde_archivo(self, controller_consumo, features_consumo):
        if controller_consumo is None:
            pytest.skip("Controlador no disponible")
        
        if not archivo_existe("datos_100_filas.csv"):
            crear_datos_100_filas()
        
        resultados = controller_consumo.predecir_archivo(
            ruta_archivo("datos_100_filas.csv"), features_consumo
        )
        
        assert resultados is not None
        assert not isinstance(resultados, dict) or 'error' not in resultados
        assert len(resultados) == 100

    # CP-11: Predicción manual
    def test_cp11_prediccion_manual(self, controller_consumo):
        if controller_consumo is None:
            pytest.skip("Controlador no disponible")
        
        datos = {'Hora': 10, 'Mes': 7, 'tipo_dia_bin': 1}
        res = controller_consumo.predecir_manual(datos)
        
        assert res is not None
        assert not isinstance(res, dict) or 'error' not in res
        assert 'prediccion' in res
        assert 'segmento' in res
        assert 'modelo' in res

    # CP-12: Archivo con columna faltante
    def test_cp12_columna_faltante(self, controller_consumo, features_consumo):
        if controller_consumo is None:
            pytest.skip("Controlador no disponible")
        
        if not archivo_existe("datos_sin_hora.csv"):
            crear_datos_sin_hora()
        
        resultados, _ = controller_consumo.predecir_archivo(
            ruta_archivo("datos_sin_hora.csv"), features_consumo
        )
        
        assert isinstance(resultados, dict)
        assert 'error' in resultados
        assert "Columnas requeridas" in resultados['error'] or "faltan" in resultados['error'].lower()

    # CP-13: Archivo con valores nulos
    def test_cp13_valores_nulos(self, controller_consumo, features_consumo):
        if controller_consumo is None:
            pytest.skip("Controlador no disponible")
        
        if not archivo_existe("datos_con_nulos.csv"):
            crear_datos_con_nulos()
        
        resultado = controller_consumo.predecir_archivo(
            ruta_archivo("datos_con_nulos.csv"), features_consumo
        )

        if isinstance(resultado, tuple):
            resultados, _ = resultado
        else:
            resultados = resultado
        
        assert resultados is not None
        assert not isinstance(resultados, dict) or 'error' not in resultados
        assert len(resultados) > 0


# =============================================================================
# PRUEBAS DE CAJA BLANCA - CONSUMO (CP-16 a CP-22)
# =============================================================================

class TestCajaBlancaConsumo:
    """
    Pruebas de Caja Blanca para el dominio Consumo.
    CP-16 a CP-22
    """

    def test_cp16_orchestrator_configuracion_existente(self, orchestrator):
        mock_segmenter = Mock()
        mock_segmenter.transform.return_value = np.array([0])
        mock_segmenter.variables = ['Hora', 'tipo_dia_bin']
        mock_modelo = Mock()
        mock_modelo.predict.return_value = np.array([150.5])
        mock_modelo.get_nombre.return_value = "RandomForest"
        
        orchestrator.initialize(
            segmenter=mock_segmenter,
            configuracion_segmentos={0: {'modelo': 'RandomForest'}},
            modelos={'RandomForest': mock_modelo},
            metricas_evaluacion={}
        )
        muestra = pd.DataFrame({'Hora': [10], 'Mes': [7], 'tipo_dia_bin': [1]})
        resultado = orchestrator.predict(muestra)
        
        assert resultado is not None
        assert 'prediccion' in resultado
        assert resultado['segmento'] == 0

    def test_cp17_orchestrator_fallback(self, orchestrator):
        mock_segmenter = Mock()
        mock_segmenter.transform.return_value = np.array([99])
        mock_segmenter.variables = ['Hora', 'tipo_dia_bin']
        mock_modelo = Mock()
        mock_modelo.predict.return_value = np.array([160.0])
        mock_modelo.get_nombre.return_value = "XGBoost"
        
        metricas = {
            'RandomForest': {'MAE': 10.0, 'RMSE': 15.0},
            'XGBoost': {'MAE': 8.0, 'RMSE': 12.0}
        }
        
        orchestrator.initialize(
            segmenter=mock_segmenter,
            configuracion_segmentos={},
            modelos={'RandomForest': mock_modelo, 'XGBoost': mock_modelo},
            metricas_evaluacion=metricas
        )
        muestra = pd.DataFrame({'Hora': [10], 'Mes': [7], 'tipo_dia_bin': [1]})
        resultado = orchestrator.predict(muestra)
        
        assert resultado is not None
        assert 'prediccion' in resultado
        assert 'modelo' in resultado
        assert resultado['modelo'] != 'desconocido'

    def test_cp18_orchestrator_ensamblaje(self, orchestrator):
        mock_segmenter = Mock()
        mock_segmenter.transform.return_value = np.array([0])
        mock_segmenter.variables = ['Hora', 'tipo_dia_bin']
        
        mock_rf = Mock()
        mock_rf.predict.return_value = np.array([150.0])
        mock_rf.get_nombre.return_value = "RandomForest"
        
        mock_xgb = Mock()
        mock_xgb.predict.return_value = np.array([160.0])
        mock_xgb.get_nombre.return_value = "XGBoost"
        
        config = {0: {'modelo': 'Ensemble', 'pesos': {'RandomForest': 0.7, 'XGBoost': 0.3}}}
        
        orchestrator.initialize(
            segmenter=mock_segmenter,
            configuracion_segmentos=config,
            modelos={'RandomForest': mock_rf, 'XGBoost': mock_xgb},
            metricas_evaluacion={}
        )
        muestra = pd.DataFrame({'Hora': [10], 'Mes': [7], 'tipo_dia_bin': [1]})
        resultado = orchestrator.predict(muestra)
        
        assert resultado is not None
        assert 'prediccion' in resultado
        assert 'Ensemble' in resultado['modelo']

    def test_cp19_orchestrator_sin_inicializar(self, orchestrator):
        muestra = pd.DataFrame({'Hora': [10], 'Mes': [7], 'tipo_dia_bin': [1]})
        with pytest.raises(ValueError) as excinfo:
            orchestrator.predict(muestra)
        assert "no inicializado" in str(excinfo.value).lower() or "initialize" in str(excinfo.value).lower()

    def test_cp20_weight_optimizer_un_modelo(self, weight_optimizer):
        valores_reales = np.array([100, 110, 105])
        predicciones = {'RandomForest': np.array([102, 108, 106])}
        pesos = weight_optimizer.optimizar(valores_reales, predicciones, segmento_id=0)
        
        assert pesos is not None
        assert 'RandomForest' in pesos
        assert pesos['RandomForest'] == 1.0

    def test_cp21_weight_optimizer_dos_modelos(self, weight_optimizer):
        valores_reales = np.array([100, 110, 105, 95, 115])
        predicciones = {
            'RandomForest': np.array([105, 108, 107, 98, 112]),
            'XGBoost': np.array([98, 112, 103, 94, 118])
        }
        pesos = weight_optimizer.optimizar(valores_reales, predicciones, segmento_id=0)
        
        assert pesos is not None
        assert 'RandomForest' in pesos
        assert 'XGBoost' in pesos
        suma = sum(pesos.values())
        assert abs(suma - 1.0) < 0.001
        for p in pesos.values():
            assert p >= 0
            assert p <= 1

    def test_cp22_weight_optimizer_predicciones_vacias(self, weight_optimizer):
        valores_reales = np.array([100, 110, 105])
        predicciones = {}
        pesos = weight_optimizer.optimizar(valores_reales, predicciones, segmento_id=0)
        assert pesos is None


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])