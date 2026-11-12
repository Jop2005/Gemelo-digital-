"""
Pruebas de Caja Negra y Caja Blanca para el dominio GENERACIÓN
Ejecutar con: pytest test/generacion.py -v
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# IMPORTS
# =============================================================================

from prediccion.acceso_datos.context import DatasetContext as PrediccionContext
from prediccion.acceso_datos.config import get_config
from prediccion.acceso_datos.joblib_model_repository import JoblibModelRepository as PrediccionModelRepo
from prediccion.acceso_datos.joblib_imputer_repository import JoblibImputerRepository as PrediccionImputerRepo
from prediccion.acceso_datos.historial_repository import HistorialRepository
from prediccion.acceso_datos.file_reader import FileReader
from prediccion.acceso_datos.config_loader import ConfigLoader
from prediccion.logica_negocio.prediccion_controller import PrediccionController

from configuracion.acceso_datos.context import DatasetContext as ConfigContext
from configuracion.acceso_datos.strategies_generacion import GeneracionStrategy
from configuracion.acceso_datos.joblib_model_repository import JoblibModelRepository as ConfigModelRepo

from configuracion.logica_negocio.segmentacion_controller import SegmentacionController
from configuracion.logica_negocio.entrenamiento_rf_controller import EntrenadorRFController
from configuracion.logica_negocio.entrenamiento_xgb_controller import EntrenadorXGBController


# =============================================================================
# CONSTANTES
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"

FEATURES_MODELO = [
    "shortwave_radiation", "clearsky_ghi", "clearsky_index",
    "solar_elevation", "effective_irradiance", "cloud_cover",
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "temp_loss_factor", "hour_sin", "hour_cos", "month_sin", "month_cos"
]

def ruta_archivo(nombre: str) -> str:
    return str(DATA_DIR / nombre)

def archivo_existe(nombre: str) -> bool:
    return os.path.exists(ruta_archivo(nombre))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def config_generacion():
    get_config().set_dominio('generacion')
    return get_config()


@pytest.fixture(scope="session")
def controller_generacion(config_generacion):
    try:
        from configuracion.acceso_datos.strategies_generacion import GeneracionStrategy as ConfigGenStrategy
        strategy = ConfigGenStrategy()
        context = PrediccionContext(strategy)
        model_repo = PrediccionModelRepo()
        imputer_repo = PrediccionImputerRepo()
        historial_repo = HistorialRepository()
        file_reader = FileReader()
        config_loader = ConfigLoader()
        return PrediccionController(context, model_repo, imputer_repo, historial_repo, file_reader, config_loader)
    except FileNotFoundError:
        pytest.skip("Pipeline de configuración de generación no ejecutado")
        return None


@pytest.fixture
def context_generacion_configuracion(config_generacion):
    try:
        return ConfigContext(GeneracionStrategy())
    except FileNotFoundError:
        pytest.skip("Configuración de generación no encontrada")
        return None


@pytest.fixture
def model_repo_configuracion(config_generacion):
    return ConfigModelRepo()


@pytest.fixture
def features_generacion():
    return FEATURES_MODELO.copy()


# =============================================================================
# FUNCIONES PARA CREAR ARCHIVOS DE PRUEBA
# =============================================================================

def crear_archivo_prueba(nombre, datos, columnas):
    DATA_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(datos, columns=columnas)
    ruta = DATA_DIR / nombre
    df.to_csv(ruta, index=False)
    return ruta


def crear_datos_meteorologicos_forzado():
    archivo = DATA_DIR / 'datos_meteorologicos.csv'
    if archivo.exists():
        archivo.unlink()
    np.random.seed(45)
    n = 50
    datos = {
        'temperature_2m': np.random.uniform(15, 35, n),
        'relative_humidity_2m': np.random.uniform(40, 90, n),
        'cloud_cover': np.random.uniform(0, 100, n),
        'wind_speed_10m': np.random.uniform(0, 20, n),
        'shortwave_radiation': np.random.uniform(0, 1000, n)
    }
    df = pd.DataFrame(datos)
    ruta = DATA_DIR / 'datos_meteorologicos.csv'
    df.to_csv(ruta, index=False)
    return str(ruta)


def construir_features_completas(df_meteo: pd.DataFrame) -> pd.DataFrame:
    df = df_meteo.copy()
    df['clearsky_ghi'] = df['shortwave_radiation'] * 1.2
    df['clearsky_index'] = (df['shortwave_radiation'] / df['clearsky_ghi'].replace(0, np.nan)).clip(0, 1.2).fillna(0.0)
    df['solar_elevation'] = 45.0
    df['effective_irradiance'] = df['shortwave_radiation'] * (1.0 - df['cloud_cover'] / 100.0)
    df['temp_loss_factor'] = 1.0 - 0.004 * (df['temperature_2m'] - 25).clip(lower=0)
    df['hour_sin'] = 0.0
    df['hour_cos'] = 1.0
    df['month_sin'] = 0.0
    df['month_cos'] = 1.0
    return df[FEATURES_MODELO]


def crear_archivo_formato_incorrecto():
    ruta = DATA_DIR / 'datos_gen.txt'
    with open(ruta, 'w') as f:
        f.write("esto no es un csv")
    return ruta


def crear_archivo_vacio():
    ruta = DATA_DIR / 'datos_gen_vacios.csv'
    pd.DataFrame().to_csv(ruta, index=False)
    return ruta


# =============================================================================
# PRUEBAS DE CAJA NEGRA - GENERACIÓN (CP-08, CP-09, CP-14, CP-15, CP-27 a CP-30)
# =============================================================================

class TestGeneracion:

    def test_cp08_generar_segmentacion_generacion(self, context_generacion_configuracion):
        if context_generacion_configuracion is None:
            pytest.skip("Contexto no disponible")
        controller = SegmentacionController(context_generacion_configuracion)
        segmenter = controller.ejecutar()
        assert segmenter is not None
        config = segmenter.get_config()
        assert config.get('tipo') == 'clustering'
        assert segmenter.n_segments == 6

    def test_cp09_entrenar_modelos_generacion(self, context_generacion_configuracion, model_repo_configuracion):
        if context_generacion_configuracion is None:
            pytest.skip("Contexto no disponible")
        rf = EntrenadorRFController(context_generacion_configuracion, model_repo_configuracion)
        xgb = EntrenadorXGBController(context_generacion_configuracion, model_repo_configuracion)
        resultado_rf = rf.ejecutar()
        resultado_xgb = xgb.ejecutar()
        assert resultado_rf is not None
        assert resultado_rf.modelo is not None
        assert resultado_rf.mae is not None
        assert resultado_xgb is not None
        assert resultado_xgb.modelo is not None
        assert resultado_xgb.mae is not None

    def test_cp14_prediccion_generacion_archivo(self, controller_generacion, features_generacion):
        if controller_generacion is None:
            pytest.skip("Controlador no disponible")
        ruta = crear_datos_meteorologicos_forzado()
        assert os.path.exists(ruta), f"Archivo no creado: {ruta}"
        df_meteo = pd.read_csv(ruta)
        df_features = construir_features_completas(df_meteo)
        ruta_features = DATA_DIR / 'datos_meteorologicos_features.csv'
        df_features.to_csv(ruta_features, index=False)
        resultados = controller_generacion.predecir_archivo(str(ruta_features), features_generacion)
        assert resultados is not None
        if isinstance(resultados, dict) and 'error' in resultados:
            pytest.fail(f"Error al predecir: {resultados['error']}")
        assert len(resultados) > 0, "No se generaron predicciones"
        for res in resultados:
            assert 'prediccion' in res
            assert 'segmento' in res
            assert 'modelo' in res

    def test_cp15_prediccion_generacion_manual(self, controller_generacion):
        if controller_generacion is None:
            pytest.skip("Controlador no disponible")
        datos_meteo = {
            'temperature_2m': 25.0,
            'relative_humidity_2m': 60.0,
            'cloud_cover': 50.0,
            'wind_speed_10m': 10.0,
            'shortwave_radiation': 500.0
        }
        df_meteo = pd.DataFrame([datos_meteo])
        df_features = construir_features_completas(df_meteo)
        datos = df_features.iloc[0].to_dict()
        res = controller_generacion.predecir_manual(datos)
        assert res is not None
        assert not isinstance(res, dict) or 'error' not in res
        assert 'prediccion' in res
        assert 'segmento' in res
        assert 'modelo' in res

    # ===== CASOS NEGATIVOS (CP-27 a CP-30) =====

    def test_cp27_archivo_formato_incorrecto_generacion(self, controller_generacion, features_generacion):
        if controller_generacion is None:
            pytest.skip("Controlador no disponible")
        ruta = crear_archivo_formato_incorrecto()
        resultado = controller_generacion.predecir_archivo(str(ruta), features_generacion)
        if isinstance(resultado, tuple):
            resultados, _ = resultado
        else:
            resultados = resultado
        assert isinstance(resultados, dict)
        assert 'error' in resultados
        assert "Formato no soportado" in resultados['error'] or "formato" in resultados['error'].lower()

    def test_cp28_archivo_vacio_generacion(self, controller_generacion, features_generacion):
        if controller_generacion is None:
            pytest.skip("Controlador no disponible")
        ruta = crear_archivo_vacio()
        resultado = controller_generacion.predecir_archivo(str(ruta), features_generacion)
        if isinstance(resultado, tuple):
            resultados, _ = resultado
        else:
            resultados = resultado
        assert isinstance(resultados, dict)
        assert 'error' in resultados
        assert any(kw in resultados['error'].lower() for kw in ["vacío", "vacio", "empty", "no columns"])

    def test_cp29_radiacion_negativa(self, controller_generacion):
        if controller_generacion is None:
            pytest.skip("Controlador no disponible")
        datos_meteo = {
            'temperature_2m': 25.0,
            'relative_humidity_2m': 60.0,
            'cloud_cover': 50.0,
            'wind_speed_10m': 10.0,
            'shortwave_radiation': -100.0
        }
        df_meteo = pd.DataFrame([datos_meteo])
        df_features = construir_features_completas(df_meteo)
        datos = df_features.iloc[0].to_dict()
        res = controller_generacion.predecir_manual(datos)
        assert res is not None
        assert 'prediccion' in res
        assert res['prediccion'] == 0.0 or res['prediccion'] >= 0.0

    def test_cp30_humedad_fuera_rango(self, controller_generacion):
        if controller_generacion is None:
            pytest.skip("Controlador no disponible")
        datos_meteo = {
            'temperature_2m': 25.0,
            'relative_humidity_2m': 150.0,
            'cloud_cover': 50.0,
            'wind_speed_10m': 10.0,
            'shortwave_radiation': 500.0
        }
        df_meteo = pd.DataFrame([datos_meteo])
        df_features = construir_features_completas(df_meteo)
        datos = df_features.iloc[0].to_dict()
        res = controller_generacion.predecir_manual(datos)
        assert res is not None
        assert 'prediccion' in res


if __name__ == "__main__":
    pytest.main([__file__, "-v"])