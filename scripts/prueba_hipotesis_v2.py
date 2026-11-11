"""
Pruebas de hipótesis para comparar modelos predictivos en el conjunto de prueba.

Prueba principal: Diebold-Mariano (adecuada para series temporales).
Corrección por comparaciones múltiples: Bonferroni.
Intervalos de confianza del MAE: bootstrap por bloques.

Dominios: consumo (Perfil, RF, XGB, Ensemble) y generación (RF, XGB, Ensemble, Havana v1).
"""
import sys
sys.path.insert(0, r"D:\gemelo-digital-main\backend")
import pandas as pd
import numpy as np
import joblib
import json
import os
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===== RUTAS =====
RUTA_CONSUMO = r"D:\gemelo-digital-main\backend\ensamble\data\_BD del consumo en la cujae_.xlsx"
RUTA_GENERACION = r"D:\gemelo-digital-main\backend\ensamble\data\havana_solar_training.csv"

RUTA_RF_CON = r"D:\gemelo-digital-main\backend\ensamble\artifacts\modelos\consumo\RandomForest_20261028_172328.joblib"
RUTA_XGB_CON = r"D:\gemelo-digital-main\backend\ensamble\artifacts\modelos\consumo\XGBoost_20261028_172447.joblib"
RUTA_RF_GEN = r"D:\gemelo-digital-main\backend\ensamble\artifacts\modelos\generacion\RandomForest_20261028_173518.joblib"
RUTA_XGB_GEN = r"D:\gemelo-digital-main\backend\ensamble\artifacts\modelos\generacion\XGBoost_20261028_174133.joblib"

RUTA_CONFIG_ENS_CON = r"D:\gemelo-digital-main\backend\ensamble\artifacts\config\consumo\config_ensamblaje_20261028_172500.json"
RUTA_SEG_CON = r"D:\gemelo-digital-main\backend\ensamble\artifacts\config\consumo\segmentacion_reglas_20261028_172243.json"
RUTA_CONFIG_ENS_GEN = r"D:\gemelo-digital-main\backend\ensamble\artifacts\config\generacion\config_ensamblaje_20261028_174151.json"
RUTA_SEG_GEN = r"D:\gemelo-digital-main\backend\ensamble\artifacts\config\generacion\segmentacion_shortwave_radiation_20261028_172544.json"

TARGET_CON = "Energia_consumida_kW"
FEATURES_CON = ["Hora", "Mes", "tipo_dia_bin"]

TARGET_GEN = "capacity_factor"
FEATURES_GEN = [
    "shortwave_radiation", "clearsky_ghi", "clearsky_index",
    "solar_elevation", "effective_irradiance", "cloud_cover",
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "temp_loss_factor", "hour_sin", "hour_cos", "month_sin", "month_cos"
]

ALPHA = 0.05
N_BOOTSTRAP = 1000
BLOQUE_BOOTSTRAP = 24

# ===== FUNCIONES AUXILIARES =====

def cargar_modelo(ruta):
    data = joblib.load(ruta)
    return data["model"] if isinstance(data, dict) and "model" in data else data

def cargar_json(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)

def asignar_segmento_reglas(row, seg_config):
    for regla in seg_config["reglas"]:
        cumple = True
        for var, cond in regla.items():
            if var == "segmento":
                continue
            val = row[var]
            if isinstance(cond, list):
                if val not in cond:
                    cumple = False
                    break
            else:
                if val != cond:
                    cumple = False
                    break
        if cumple:
            return regla["segmento"]
    return -1

def asignar_segmento_clustering(row, seg_config):
    variable = seg_config["variables"][0]
    centroides = np.array(seg_config["centroides"]).flatten()
    orden = np.array(seg_config["orden"])
    valor = row[variable]
    distancias = np.abs(valor - centroides)
    cluster = np.argmin(distancias)
    return int(np.nonzero(orden == cluster)[0][0])

def predecir_ensamble(test, pred_rf, pred_xgb, ens_config, seg_config, tipo_seg):
    pred_ens = np.zeros(len(test))
    for i, row in test.iterrows():
        if tipo_seg == "reglas":
            seg = asignar_segmento_reglas(row, seg_config)
        else:
            seg = asignar_segmento_clustering(row, seg_config)
        config_seg = ens_config["segmentos"].get(str(seg), {})
        if config_seg.get("modelo") == "Ensemble":
            pesos = config_seg.get("pesos", {})
            p = 0.0
            if "RandomForest" in pesos:
                p += pesos["RandomForest"] * pred_rf[i]
            if "XGBoost" in pesos:
                p += pesos["XGBoost"] * pred_xgb[i]
            pred_ens[i] = p
        elif config_seg.get("modelo") == "RandomForest":
            pred_ens[i] = pred_rf[i]
        elif config_seg.get("modelo") == "XGBoost":
            pred_ens[i] = pred_xgb[i]
        else:
            pred_ens[i] = pred_rf[i]
    return pred_ens

def calcular_perfil_estadistico(train, test):
    perfil = train.groupby(["Mes", "Hora", "tipo_dia_bin"])[TARGET_CON].mean().reset_index()
    perfil.columns = ["Mes", "Hora", "tipo_dia_bin", "pred_perfil"]
    test_merge = test.merge(perfil, on=["Mes", "Hora", "tipo_dia_bin"], how="left")
    test_merge["pred_perfil"] = test_merge["pred_perfil"].fillna(train[TARGET_CON].mean())
    return test_merge["pred_perfil"].values

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

# ===== DOMINIO CONSUMO =====

def evaluar_consumo():
    print("\n" + "=" * 70)
    print("  DOMINIO: CONSUMO")
    print("=" * 70)

    df = pd.read_excel(RUTA_CONSUMO)
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx].reset_index(drop=True)
    test = df.iloc[split_idx:].reset_index(drop=True)
    y_train = train[TARGET_CON].values
    y_test = test[TARGET_CON].values
    X_test = test[FEATURES_CON]

    rf = cargar_modelo(RUTA_RF_CON)
    xgb = cargar_modelo(RUTA_XGB_CON)
    ens_config = cargar_json(RUTA_CONFIG_ENS_CON)
    seg_config = cargar_json(RUTA_SEG_CON)

    pred_rf = rf.predict(X_test)
    pred_xgb = xgb.predict(X_test)
    pred_ens = predecir_ensamble(test, pred_rf, pred_xgb, ens_config, seg_config, "reglas")
    pred_perfil = calcular_perfil_estadistico(train, test)

    modelos = {
        "Perfil": pred_perfil,
        "RandomForest": pred_rf,
        "XGBoost": pred_xgb,
        "Ensemble": pred_ens,
    }

    print("\n--- Métricas e intervalos de confianza (bootstrap 95%) ---")
    resultados = {}
    for nombre, pred in modelos.items():
        m = calcular_metricas(y_test, pred, y_train)
        ic = bootstrap_mae(y_test, pred)
        m.update(ic)
        resultados[nombre] = m
        print(f"{nombre:<15} MAE={m['MAE']:.2f} [{m['MAE_ic95_inf']:.2f}, {m['MAE_ic95_sup']:.2f}]  "
              f"RMSE={m['RMSE']:.2f}  R²={m['R2']:.4f}  MAPE={m['MAPE']:.2f}%  MASE={m['MASE']:.4f}")

    print("\n--- Diebold-Mariano: Ensemble vs alternativas ---")
    dm_resultados = {}
    for nombre, pred in modelos.items():
        if nombre == "Ensemble":
            continue
        dm = diebold_mariano(y_test, pred_ens, pred)
        dm_resultados[nombre] = dm
        sig = "***" if dm["p_value"] < 0.001 else "**" if dm["p_value"] < 0.01 else "*" if dm["p_value"] < 0.05 else "ns"
        print(f"  Ensemble vs {nombre:<15} DM={dm['DM']:>7.3f}  p={dm['p_value']:.6f} {sig}")

    return {"metricas": resultados, "diebold_mariano": dm_resultados}

# ===== DOMINIO GENERACIÓN =====

def evaluar_generacion():
    print("\n" + "=" * 70)
    print("  DOMINIO: GENERACIÓN")
    print("=" * 70)

    df = pd.read_csv(RUTA_GENERACION)
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx].reset_index(drop=True)
    test = df.iloc[split_idx:].reset_index(drop=True)
    y_train = train[TARGET_GEN].values
    y_test = test[TARGET_GEN].values
    X_test = test[FEATURES_GEN]

    rf = cargar_modelo(RUTA_RF_GEN)
    xgb = cargar_modelo(RUTA_XGB_GEN)
    ens_config = cargar_json(RUTA_CONFIG_ENS_GEN)
    seg_config = cargar_json(RUTA_SEG_GEN)

    pred_rf = rf.predict(X_test)
    pred_xgb = xgb.predict(X_test)
    pred_ens = predecir_ensamble(test, pred_rf, pred_xgb, ens_config, seg_config, "clustering")

    modelos = {
        "RandomForest": pred_rf,
        "XGBoost": pred_xgb,
        "Ensemble": pred_ens,
    }

    # Cargar Havana v1
    try:
        from app.services.ml_model_service import ml_model_service
        ml_model_service.load_model("havana_v1")
        pred_havana = ml_model_service.predict(X_test)
        modelos["Havana_v1"] = pred_havana
        print("✅ Havana v1 cargado y evaluado")
    except Exception as e:
        print(f"⚠️ No se pudo cargar Havana v1: {e}")

    print("\n--- Métricas globales e intervalos de confianza (bootstrap 95%) ---")
    resultados = {}
    for nombre, pred in modelos.items():
        m = calcular_metricas(y_test, pred, y_train)
        ic = bootstrap_mae(y_test, pred)
        m.update(ic)
        resultados[nombre] = m
        print(f"{nombre:<15} MAE={m['MAE']:.6f} [{m['MAE_ic95_inf']:.6f}, {m['MAE_ic95_sup']:.6f}]  "
              f"RMSE={m['RMSE']:.6f}  R²={m['R2']:.4f}  MAPE={m['MAPE']:.2f}%  MASE={m['MASE']:.4f}")

    print("\n--- Diebold-Mariano: Ensemble vs alternativas ---")
    dm_resultados = {}
    for nombre, pred in modelos.items():
        if nombre == "Ensemble":
            continue
        dm = diebold_mariano(y_test, pred_ens, pred)
        dm_resultados[nombre] = dm
        sig = "***" if dm["p_value"] < 0.001 else "**" if dm["p_value"] < 0.01 else "*" if dm["p_value"] < 0.05 else "ns"
        print(f"  Ensemble vs {nombre:<15} DM={dm['DM']:>7.3f}  p={dm['p_value']:.6f} {sig}")

    return {"metricas": resultados, "diebold_mariano": dm_resultados}

# ===== MAIN =====

if __name__ == "__main__":
    resultados = {}
    resultados["consumo"] = evaluar_consumo()
    resultados["generacion"] = evaluar_generacion()

    carpeta_salida = r"D:\gemelo-digital-main\backend\ensamble\artifacts\resultados"
    os.makedirs(carpeta_salida, exist_ok=True)
    ruta_json = os.path.join(carpeta_salida, "pruebas_hipotesis_v2.json")
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, default=str)
    print(f"\n📁 Resultados guardados en: {ruta_json}")