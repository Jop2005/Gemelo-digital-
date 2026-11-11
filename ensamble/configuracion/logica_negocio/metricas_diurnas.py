"""
Cálculo de métricas diurnas para comparar con Havana v1.

Filtra las muestras a horas de día (solar_elevation > 0) y calcula:
- R² diurno
- nRMSE diurno (RMSE / capacidad de referencia)
- nMAE diurno (MAE / capacidad de referencia)
- MASE diurno

Uso:
    python -m configuracion.logica_negocio.metricas_diurnas
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Asegurar imports desde la raíz
RAIZ = Path(__file__).parent.parent.parent
sys.path.insert(0, str(RAIZ))
os.chdir(RAIZ)

from configuracion.acceso_datos.config import get_config
from configuracion.acceso_datos.strategies_generacion import GeneracionStrategy
from configuracion.acceso_datos.context import DatasetContext
from configuracion.acceso_datos.config_loader import ConfigLoader
from configuracion.acceso_datos.data_splitter import DataSplitter
from configuracion.acceso_datos.joblib_model_repository import JoblibModelRepository
from configuracion.logica_negocio.segmentation.segmenter import Segmenter


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Capacidad nominal de referencia del dataset
# Como el target es capacity_factor (0-1), la "capacidad" para normalizar es 1.0
CAPACIDAD_REFERENCIA = 1.0


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def cargar_datos_prueba():
    """Carga los datos de prueba respetando el split cronológico 80/20."""
    strategy = GeneracionStrategy()
    context = DatasetContext(strategy)
    loader = context.get_loader()

    X, y = loader.obtener_datos_completos()
    X_train, X_test, y_train, y_test = DataSplitter(test_size=0.2).split(X, y)
    return X_test, y_test, X_train, y_train


def cargar_modelos():
    """Carga todos los modelos entrenados del dominio generación."""
    repo = JoblibModelRepository()
    return repo.cargar_todos()


def calcular_metricas_diurnas(y_true, y_pred, mascara_diurna):
    """Calcula métricas sobre las muestras diurnas."""
    y_t = np.asarray(y_true)[mascara_diurna]
    y_p = np.asarray(y_pred)[mascara_diurna]

    if len(y_t) == 0:
        return {
            'n_muestras': 0,
            'R2': float('nan'),
            'RMSE': float('nan'),
            'MAE': float('nan'),
            'nRMSE': float('nan'),
            'nMAE': float('nan'),
        }

    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    mae = float(mean_absolute_error(y_t, y_p))
    r2 = float(r2_score(y_t, y_p))

    return {
        'n_muestras': int(len(y_t)),
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'nRMSE': rmse / CAPACIDAD_REFERENCIA,
        'nMAE': mae / CAPACIDAD_REFERENCIA,
    }


def calcular_mase_diurno(y_true, y_pred, y_train, mascara_diurna):
    """MASE sobre muestras diurnas, usando naive de y_train."""
    y_t = np.asarray(y_true)[mascara_diurna]
    y_p = np.asarray(y_pred)[mascara_diurna]
    y_tr = np.asarray(y_train)

    if len(y_tr) < 2 or len(y_t) == 0:
        return float('nan')

    mae_naive = mean_absolute_error(y_tr[1:], y_tr[:-1])
    if mae_naive == 0:
        return float('inf')

    return float(mean_absolute_error(y_t, y_p) / mae_naive)


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    print("=" * 70)
    print("  MÉTRICAS DIURNAS — GENERACIÓN")
    print("=" * 70)

    # Configurar dominio
    get_config().set_dominio('generacion')

    # Cargar datos
    print("\n📥 Cargando datos de prueba...")
    X_test, y_test, X_train, y_train = cargar_datos_prueba()
    print(f"   Test: {len(X_test)} muestras")
    print(f"   Train: {len(X_train)} muestras")

    # Máscara diurna
    if 'solar_elevation' not in X_test.columns:
        print("\n❌ ERROR: La columna 'solar_elevation' no está en las features.")
        print("   No se puede calcular la máscara diurna.")
        return

    mascara_diurna = X_test['solar_elevation'].values > 0
    n_diurnas = int(mascara_diurna.sum())
    n_total = len(X_test)
    print(f"\n☀️  Muestras diurnas: {n_diurnas} de {n_total} ({n_diurnas/n_total*100:.1f}%)")

    # Cargar modelos
    print("\n📦 Cargando modelos...")
    modelos = cargar_modelos()
    if not modelos:
        print("❌ No se encontraron modelos entrenados.")
        return
    print(f"   Modelos: {', '.join(modelos.keys())}")

    # Calcular métricas por modelo
    print("\n" + "=" * 70)
    print("  RESULTADOS")
    print("=" * 70)

    resultados = {}

    for nombre, modelo in modelos.items():
        y_pred = modelo.predict(X_test)

        # Métricas globales
        r2_global = float(r2_score(y_test, y_pred))
        rmse_global = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae_global = float(mean_absolute_error(y_test, y_pred))

        # Métricas diurnas
        metricas_d = calcular_metricas_diurnas(y_test, y_pred, mascara_diurna)
        mase_d = calcular_mase_diurno(y_test, y_pred, y_train, mascara_diurna)

        resultados[nombre] = {
            'R2_global': r2_global,
            'RMSE_global': rmse_global,
            'MAE_global': mae_global,
            'R2_diurno': metricas_d['R2'],
            'RMSE_diurno': metricas_d['RMSE'],
            'MAE_diurno': metricas_d['MAE'],
            'nRMSE_diurno': metricas_d['nRMSE'],
            'nMAE_diurno': metricas_d['nMAE'],
            'MASE_diurno': mase_d,
            'n_muestras_diurnas': metricas_d['n_muestras'],
        }

    # Imprimir tabla comparativa
    print(f"\n{'Modelo':<15} {'R² glob':>10} {'R² día':>10} {'nRMSE día':>12} {'nMAE día':>12} {'MASE día':>10}")
    print("-" * 75)
    for nombre, r in resultados.items():
        print(
            f"{nombre:<15} "
            f"{r['R2_global']:>10.4f} "
            f"{r['R2_diurno']:>10.4f} "
            f"{r['nRMSE_diurno']*100:>11.2f}% "
            f"{r['nMAE_diurno']*100:>11.2f}% "
            f"{r['MASE_diurno']:>10.4f}"
        )

    # Comparación con Havana v1
    print("\n" + "=" * 70)
    print("  COMPARACIÓN CON HAVANA v1 (del gemelo)")
    print("=" * 70)
    print(f"  Havana v1 — R² diurno:   0,7895")
    print(f"  Havana v1 — nRMSE diurno: 11,40%")
    print(f"  Havana v1 — nMAE diurno:   7,50%")
    print()

    for nombre, r in resultados.items():
        diff_r2 = r['R2_diurno'] - 0.7895
        diff_nrmse = r['nRMSE_diurno'] * 100 - 11.40
        diff_nmae = r['nMAE_diurno'] * 100 - 7.50
        signo_r2 = "✅ mejora" if diff_r2 > 0 else "❌ empeora"
        signo_nrmse = "✅ mejora" if diff_nrmse < 0 else "❌ empeora"
        print(f"  {nombre}:")
        print(f"    R² diurno:   {r['R2_diurno']:.4f} (Δ {diff_r2:+.4f}) {signo_r2}")
        print(f"    nRMSE diurno: {r['nRMSE_diurno']*100:.2f}% (Δ {diff_nrmse:+.2f} pp) {signo_nrmse}")
        print()

    # Guardar resultados en JSON
    carpeta_resultados = RAIZ / 'artifacts' / 'resultados' / 'generacion'
    carpeta_resultados.mkdir(parents=True, exist_ok=True)

    ruta_salida = carpeta_resultados / 'metricas_diurnas.json'
    import json
    with open(ruta_salida, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2, default=str)

    print(f"📁 Resultados guardados en: {ruta_salida}")
    print("=" * 70)


if __name__ == "__main__":
    main()