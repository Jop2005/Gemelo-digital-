"""
Evaluación con segmentación por rango del target (como Mystakidis et al. 2023).

Prueba:
- Segmentación por cuantiles del target (6 segmentos, como el paper).
- Para cada segmento, elige el mejor modelo o una combinación de pesos
  probando a mano (trial and error): 100/0, 75/25, 50/50, 25/75, 0/100.
- Compara con RF solo, XGB solo y ensamble dinámico original.
- Reporta MAE, RMSE, R², MASE e IC 95% por bootstrap.
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

CARPETA_SALIDA = RAIZ / "artifacts" / "resultados" / "ablaciones_target"
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 1000
BLOQUE_BOOTSTRAP = 24
N_SEGMENTOS_TARGET = 6
PESOS_A_PROBAR = [
    (1.0, 0.0),
    (0.75, 0.25),
    (0.5, 0.5),
    (0.25, 0.75),
    (0.0, 1.0),
]


def cargar_datos(dominio):
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


def diebold_mariano(y_true, pred1, pred2):
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


def segmentar_por_cuantiles(y_train, y_test, n_segmentos):
    """Segmenta train y test por cuantiles del target."""
    cuantiles = np.linspace(0, 100, n_segmentos + 1)
    umbrales = np.percentile(y_train, cuantiles)
    umbrales[0] = -np.inf
    umbrales[-1] = np.inf
    segmentos_train = np.digitize(y_train, umbrales[1:-1], right=False)
    segmentos_test = np.digitize(y_test, umbrales[1:-1], right=False)
    return segmentos_train, segmentos_test, umbrales


def elegir_mejor_combinacion_por_segmento(
    y_train, y_test, pred_rf_train, pred_xgb_train, pred_rf_test, pred_xgb_test,
    segmentos_train, segmentos_test
):
    """Para cada segmento, prueba combinaciones de pesos y elige la de menor MAE en train."""
    n_seg = int(max(segmentos_train.max(), segmentos_test.max())) + 1
    config = {}
    for seg in range(n_seg):
        mask_train = segmentos_train == seg
        mask_test = segmentos_test == seg
        if mask_train.sum() == 0:
            continue
        mejor_mae = float("inf")
        mejor_pesos = (1.0, 0.0)
        for w_rf, w_xgb in PESOS_A_PROBAR:
            pred = w_rf * pred_rf_train[mask_train] + w_xgb * pred_xgb_train[mask_train]
            mae = mean_absolute_error(y_train[mask_train], pred)
            if mae < mejor_mae:
                mejor_mae = mae
                mejor_pesos = (w_rf, w_xgb)
        config[seg] = {"pesos": mejor_pesos, "mae_train": mejor_mae, "n_train": int(mask_train.sum())}
    return config


def predecir_ensamble_target(y_test, pred_rf_test, pred_xgb_test, segmentos_test, config):
    pred = np.zeros(len(y_test))
    for seg, cfg in config.items():
        mask = segmentos_test == seg
        if mask.sum() == 0:
            continue
        w_rf, w_xgb = cfg["pesos"]
        pred[mask] = w_rf * pred_rf_test[mask] + w_xgb * pred_xgb_test[mask]
    return pred


def evaluar_dominio(dominio):
    print(f"\n{'='*70}\n  DOMINIO: {dominio.upper()}\n{'='*70}")
    get_config().set_dominio(dominio)

    X, y, X_train, X_test, y_train, y_test = cargar_datos(dominio)
    modelos = cargar_modelos(dominio)
    nombres = list(modelos.keys())
    if len(nombres) < 2:
        print("  ⚠️ Se necesitan al menos 2 modelos")
        return None

    rf, xgb = modelos[nombres[0]], modelos[nombres[1]]
    pred_rf_train = rf.predict(X_train)
    pred_xgb_train = xgb.predict(X_train)
    pred_rf_test = rf.predict(X_test)
    pred_xgb_test = xgb.predict(X_test)

    # Segmentación por cuantiles del target
    seg_train, seg_test, umbrales = segmentar_por_cuantiles(
        np.asarray(y_train), np.asarray(y_test), N_SEGMENTOS_TARGET
    )

    # Elegir mejor combinación por segmento
    config = elegir_mejor_combinacion_por_segmento(
        np.asarray(y_train), np.asarray(y_test),
        pred_rf_train, pred_xgb_train, pred_rf_test, pred_xgb_test,
        seg_train, seg_test
    )

    print(f"  Segmentos por cuantiles del target: {len(config)}")
    for seg, cfg in config.items():
        w_rf, w_xgb = cfg["pesos"]
        print(f"    Segmento {seg}: RF={w_rf:.2f}, XGB={w_xgb:.2f} (MAE train={cfg['mae_train']:.4f}, n={cfg['n_train']})")

    # Predicciones
    preds = {
        "RandomForest": pred_rf_test,
        "XGBoost": pred_xgb_test,
        "Ensemble_target": predecir_ensamble_target(
            np.asarray(y_test), pred_rf_test, pred_xgb_test, seg_test, config
        ),
    }

    # Métricas
    resultados = {}
    for nombre, pred in preds.items():
        metricas = calcular_metricas(y_test, pred, y_train)
        ic = bootstrap_mae(y_test, pred)
        metricas.update(ic)
        resultados[nombre] = metricas

    # Diebold-Mariano
    dm_resultados = {}
    pred_ens = preds["Ensemble_target"]
    for nombre, pred in preds.items():
        if nombre == "Ensemble_target":
            continue
        dm = diebold_mariano(y_test, pred_ens, pred)
        dm_resultados[nombre] = dm

    return {
        "dominio": dominio,
        "n_test": len(X_test),
        "n_train": len(X_train),
        "umbrales": umbrales.tolist(),
        "config_segmentos": {str(k): v for k, v in config.items()},
        "metricas": resultados,
        "diebold_mariano_vs_ensamble_target": dm_resultados,
    }


def main():
    print("=" * 70)
    print("  EVALUACIÓN CON SEGMENTACIÓN POR RANGO DEL TARGET")
    print("=" * 70)

    resultados = {}
    for dominio in ["consumo", "generacion"]:
        res = evaluar_dominio(dominio)
        if res:
            resultados[dominio] = res

    # Guardar JSON
    ruta_json = CARPETA_SALIDA / "ablaciones_target.json"
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, default=str)
    print(f"\n📁 JSON guardado en: {ruta_json}")

    # Guardar CSV
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
    ruta_csv = CARPETA_SALIDA / "ablaciones_target.csv"
    df.to_csv(ruta_csv, index=False)
    print(f"📁 CSV guardado en: {ruta_csv}")

    # Resumen
    for dominio, res in resultados.items():
        print(f"\n{'='*70}\n  RESULTADOS: {dominio.upper()}\n{'='*70}")
        print(f"{'Modelo':<25} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'MASE':>8}")
        print("-" * 70)
        for nombre, m in res["metricas"].items():
            print(f"{nombre:<25} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} "
                  f"{m['R2']:>8.4f} {m['MASE']:>8.4f}")

        print(f"\n  Diebold-Mariano (Ensemble_target vs alternativas):")
        for nombre, dm in res["diebold_mariano_vs_ensamble_target"].items():
            sig = "***" if dm["p_value"] < 0.001 else "**" if dm["p_value"] < 0.01 else "*" if dm["p_value"] < 0.05 else "ns"
            print(f"    vs {nombre:<25} DM={dm['DM']:>7.3f}  p={dm['p_value']:.4f} {sig}")

    print("\n" + "=" * 70)
    print("  FIN")
    print("=" * 70)


if __name__ == "__main__":
    main()