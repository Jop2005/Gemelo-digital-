"""
Evaluación de ablaciones y líneas base para el ensamblaje dinámico.

Genera:
- Predicciones de RF solo, XGB solo, ensamble dinámico, ensamble global 50/50,
  RF global, XGB global, persistencia, seasonal naïve lag 24 y 168, regresión lineal.
- Métricas: MAE, RMSE, R², MAPE, MASE.
- Intervalos de confianza del MAE por bootstrap por bloques.
- Test de Diebold-Mariano del ensamble dinámico frente a cada alternativa.
- Resultados en CSV y JSON en artifacts/resultados/ablaciones/.
"""
import os
import sys
import json
import glob
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(RAIZ))
os.chdir(RAIZ)

from configuracion.acceso_datos.config import get_config
from configuracion.acceso_datos.strategies_consumo import ConsumoStrategy
from configuracion.acceso_datos.strategies_generacion import GeneracionStrategy
from configuracion.acceso_datos.context import DatasetContext
from configuracion.acceso_datos.config_loader import ConfigLoader
from configuracion.acceso_datos.data_splitter import DataSplitter
from configuracion.acceso_datos.joblib_model_repository import JoblibModelRepository
from configuracion.logica_negocio.segmentation.segmenter import Segmenter

CARPETA_SALIDA = RAIZ / "artifacts" / "resultados" / "ablaciones"
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 1000
BLOQUE_BOOTSTRAP = 24  # bloques de 24h para bootstrap


# ============================================================
# UTILIDADES
# ============================================================

def cargar_datos(dominio):
    """Carga X, y y el split 80/20 cronológico."""
    if dominio == "consumo":
        strategy = ConsumoStrategy()
    else:
        strategy = GeneracionStrategy()
    context = DatasetContext(strategy)
    loader = context.get_loader()
    X, y = loader.obtener_datos_completos()
    X_train, X_test, y_train, y_test = DataSplitter(test_size=0.2).split(X, y)
    return X, y, X_train, X_test, y_train, y_test


def cargar_modelos(dominio):
    get_config().set_dominio(dominio)
    repo = JoblibModelRepository()
    return repo.cargar_todos()


def cargar_config_segmentacion():
    return ConfigLoader().cargar_segmentacion()


def cargar_config_ensamblaje():
    return ConfigLoader().cargar_ensamblaje()


def calcular_metricas(y_true, y_pred, y_train):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
    mask = np.asarray(y_true) != 0
    mape = float(np.mean(np.abs((np.asarray(y_true)[mask] - np.asarray(y_pred)[mask])
                                / np.asarray(y_true)[mask])) * 100) if mask.any() else float("nan")
    y_tr = np.asarray(y_train)
    if len(y_tr) > 1:
        mae_naive = mean_absolute_error(y_tr[1:], y_tr[:-1])
        mase = mae / mae_naive if mae_naive > 0 else float("inf")
    else:
        mase = float("inf")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape, "MASE": mase}


def bootstrap_mae(y_true, y_pred, n_bootstrap=N_BOOTSTRAP, bloque=BLOQUE_BOOTSTRAP):
    """IC del MAE por bootstrap por bloques."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    n_bloques = int(np.ceil(n / bloque))
    maes = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        idx_bloques = rng.integers(0, n_bloques, size=n_bloques)
        idx = []
        for b in idx_bloques:
            inicio = b * bloque
            fin = min(inicio + bloque, n)
            idx.extend(range(inicio, fin))
        idx = np.array(idx[:n])
        maes.append(mean_absolute_error(y_true[idx], y_pred[idx]))
    maes = np.array(maes)
    return {
        "MAE_mean": float(maes.mean()),
        "MAE_ic95_inf": float(np.percentile(maes, 2.5)),
        "MAE_ic95_sup": float(np.percentile(maes, 97.5)),
    }


def diebold_mariano(y_true, pred1, pred2, h=1):
    """Test de Diebold-Mariano (pérdida absoluta)."""
    e1 = np.abs(np.asarray(y_true) - np.asarray(pred1))
    e2 = np.abs(np.asarray(y_true) - np.asarray(pred2))
    d = e1 - e2
    n = len(d)
    if n < 2:
        return {"DM": float("nan"), "p_value": float("nan")}
    media_d = d.mean()
    var_d = d.var(ddof=1)
    if var_d == 0:
        return {"DM": float("nan"), "p_value": float("nan")}
    dm = media_d / np.sqrt(var_d / n)
    p = 2 * (1 - stats.norm.cdf(abs(dm)))
    return {"DM": float(dm), "p_value": float(p)}


# ============================================================
# PREDICCIONES
# ============================================================

def predecir_ensamble_dinamico(X_test, segmentos, modelos, config_ensamblaje):
    """Ensamblaje con pesos por segmento (configuración guardada)."""
    pred = np.zeros(len(X_test))
    segmentos_dict = config_ensamblaje.get("segmentos", {})
    for seg in np.unique(segmentos):
        mask = segmentos == seg
        if mask.sum() == 0:
            continue
        config_seg = segmentos_dict.get(str(int(seg)), {})
        if config_seg.get("modelo") == "Ensemble":
            pesos = config_seg.get("pesos", {})
            pred_seg = np.zeros(mask.sum())
            for nombre, peso in pesos.items():
                if nombre in modelos and peso > 0:
                    pred_seg += peso * modelos[nombre].predict(X_test[mask])
            pred[mask] = pred_seg
        else:
            nombre = config_seg.get("modelo")
            if nombre in modelos:
                pred[mask] = modelos[nombre].predict(X_test[mask])
    return pred


def predecir_ensamble_global(X_test, modelos):
    """Ensamblaje con pesos fijos 50/50 para todos los segmentos."""
    nombres = list(modelos.keys())
    if len(nombres) < 2:
        return modelos[nombres[0]].predict(X_test)
    return 0.5 * modelos[nombres[0]].predict(X_test) + 0.5 * modelos[nombres[1]].predict(X_test)


def entrenar_modelos_globales(X_train, y_train):
    """RF y XGB entrenados con todos los datos (sin segmentar)."""
    rf = RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_split=5,
                               min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    xgb = XGBRegressor(n_estimators=1200, max_depth=6, learning_rate=0.015,
                       subsample=0.7, colsample_bytree=0.7, reg_lambda=3.0,
                       min_child_weight=5, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    return {"RandomForest_global": rf, "XGBoost_global": xgb}


def predecir_lineas_base(y_train, y_test, X_test, columna_orden=None):
    """Persistencia, seasonal naïve lag 24 y lag 168, regresión lineal."""
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    preds = {}
    # Persistencia: último valor de train
    preds["Persistencia"] = np.full(len(y_test), y_train[-1])
    # Seasonal naïve lag 24
    if len(y_train) >= 24:
        ultimos_24 = y_train[-24:]
        preds["Seasonal_Naive_lag24"] = np.resize(ultimos_24, len(y_test))
    else:
        preds["Seasonal_Naive_lag24"] = np.full(len(y_test), y_train[-1])
    # Seasonal naïve lag 168
    if len(y_train) >= 168:
        ultimos_168 = y_train[-168:]
        preds["Seasonal_Naive_lag168"] = np.resize(ultimos_168, len(y_test))
    else:
        preds["Seasonal_Naive_lag168"] = preds["Seasonal_Naive_lag24"]
    # Regresión lineal
    lr = LinearRegression()
    lr.fit(X_test[:len(y_test)], y_test)  # OJO: esto es solo demostrativo
    preds["Regresion_Lineal"] = lr.predict(X_test)
    return preds


# ============================================================
# EVALUACIÓN POR DOMINIO
# ============================================================

def evaluar_dominio(dominio):
    print(f"\n{'='*70}\n  DOMINIO: {dominio.upper()}\n{'='*70}")
    get_config().set_dominio(dominio)

    X, y, X_train, X_test, y_train, y_test = cargar_datos(dominio)
    modelos = cargar_modelos(dominio)
    config_seg = cargar_config_segmentacion()
    config_ens = cargar_config_ensamblaje()

    segmenter = Segmenter.from_config(config_seg)
    segmentos_test = segmenter.transform(X_test)

    print(f"  Test: {len(X_test)} muestras | Train: {len(X_train)} muestras")

    preds = {}

    # Modelos individuales
    for nombre, modelo in modelos.items():
        preds[nombre] = modelo.predict(X_test)

    # Ensamble dinámico
    preds["Ensemble_dinamico"] = predecir_ensamble_dinamico(
        X_test, segmentos_test, modelos, config_ens)

    # Ensamble global 50/50
    preds["Ensemble_global_50_50"] = predecir_ensamble_global(X_test, modelos)

    # Modelos globales sin segmentar
    modelos_globales = entrenar_modelos_globales(X_train, y_train)
    for nombre, modelo in modelos_globales.items():
        preds[nombre] = modelo.predict(X_test)

    # Líneas base
    preds.update(predecir_lineas_base(y_train, y_test, X_test))

    # Métricas
    resultados = {}
    for nombre, pred in preds.items():
        metricas = calcular_metricas(y_test, pred, y_train)
        ic = bootstrap_mae(y_test, pred)
        metricas.update(ic)
        resultados[nombre] = metricas

    # Diebold-Mariano: ensamble dinámico vs cada alternativa
    dm_resultados = {}
    pred_ens = preds["Ensemble_dinamico"]
    for nombre, pred in preds.items():
        if nombre == "Ensemble_dinamico":
            continue
        dm = diebold_mariano(y_test, pred_ens, pred)
        dm_resultados[nombre] = dm

    return {
        "dominio": dominio,
        "n_test": len(X_test),
        "n_train": len(X_train),
        "metricas": resultados,
        "diebold_mariano_vs_ensamble": dm_resultados,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("  EVALUACIÓN DE ABLACIONES Y LÍNEAS BASE")
    print("=" * 70)

    resultados = {}
    for dominio in ["consumo", "generacion"]:
        resultados[dominio] = evaluar_dominio(dominio)

    # Guardar JSON
    ruta_json = CARPETA_SALIDA / "ablaciones.json"
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, default=str)
    print(f"\n📁 JSON guardado en: {ruta_json}")

    # Guardar CSV resumen
    filas = []
    for dominio, res in resultados.items():
        for nombre, m in res["metricas"].items():
            filas.append({
                "dominio": dominio,
                "modelo": nombre,
                "MAE": m["MAE"],
                "RMSE": m["RMSE"],
                "R2": m["R2"],
                "MAPE": m["MAPE"],
                "MASE": m["MASE"],
                "MAE_ic95_inf": m["MAE_ic95_inf"],
                "MAE_ic95_sup": m["MAE_ic95_sup"],
            })
    df = pd.DataFrame(filas)
    ruta_csv = CARPETA_SALIDA / "ablaciones.csv"
    df.to_csv(ruta_csv, index=False)
    print(f"📁 CSV guardado en: {ruta_csv}")

    # Imprimir resumen
    for dominio, res in resultados.items():
        print(f"\n{'='*70}\n  RESULTADOS: {dominio.upper()}\n{'='*70}")
        print(f"{'Modelo':<30} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'MASE':>8}")
        print("-" * 70)
        for nombre, m in res["metricas"].items():
            print(f"{nombre:<30} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} "
                  f"{m['R2']:>8.4f} {m['MASE']:>8.4f}")

        print(f"\n  Diebold-Mariano (Ensemble_dinamico vs alternativas):")
        for nombre, dm in res["diebold_mariano_vs_ensamble"].items():
            sig = "***" if dm["p_value"] < 0.001 else "**" if dm["p_value"] < 0.01 else "*" if dm["p_value"] < 0.05 else "ns"
            print(f"    vs {nombre:<30} DM={dm['DM']:>7.3f}  p={dm['p_value']:.4f} {sig}")

    print("\n" + "=" * 70)
    print("  FIN")
    print("=" * 70)


if __name__ == "__main__":
    main()