"""
Calcula pruebas de hipótesis (paramétricas y no paramétricas) para comparar
modelos predictivos en el conjunto de prueba.

Dos dominios:
- Generación: RF vs XGB vs Havana v1 (cargado vía ml_model_service)
- Consumo: RF vs XGB

Uso:
    python pruebas_hipotesis.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

# Añadir el backend al path para importar app.services
BACKEND_DIR = Path(r"D:\gemelo-digital-main\backend")
sys.path.insert(0, str(BACKEND_DIR))

# ============================================================
# CONFIGURACIÓN: RUTAS ABSOLUTAS
# ============================================================

# --- GENERACIÓN ---
RUTA_DATASET_GEN = r"D:\gemelo-digital-main\backend\ensamble\data\havana_solar_training.csv"
RUTA_RF_GEN = r"D:\gemelo-digital-main\backend\ensamble\artifacts\modelos\generacion\RandomForest_20261028_173518.joblib"
RUTA_XGB_GEN = r"D:\gemelo-digital-main\backend\ensamble\artifacts\modelos\generacion\XGBoost_20261028_174133.joblib"

TARGET_GEN = "capacity_factor"
FEATURES_GEN = [
    "shortwave_radiation", "clearsky_ghi", "clearsky_index",
    "solar_elevation", "effective_irradiance", "cloud_cover",
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "temp_loss_factor", "hour_sin", "hour_cos", "month_sin", "month_cos"
]

# --- CONSUMO ---
RUTA_DATASET_CON = r"D:\gemelo-digital-main\backend\ensamble\data\_BD del consumo en la cujae_.xlsx"
RUTA_RF_CON = r"D:\gemelo-digital-main\backend\ensamble\artifacts\modelos\consumo\RandomForest_20261028_172328.joblib"
RUTA_XGB_CON = r"D:\gemelo-digital-main\backend\ensamble\artifacts\modelos\consumo\XGBoost_20261028_172447.joblib"

TARGET_CON = "Energia_consumida_kW"
FEATURES_CON = ["Hora", "Mes", "tipo_dia_bin"]

# --- GENERAL ---
TEST_SIZE = 0.2
ALPHA = 0.05

# ============================================================
# CARGA DE DATOS Y MODELOS
# ============================================================

def cargar_datos_csv(ruta, target, features):
    df = pd.read_csv(ruta)
    split_idx = int(len(df) * (1 - TEST_SIZE))
    df_test = df.iloc[split_idx:].reset_index(drop=True)
    X_test = df_test[features]
    y_test = df_test[target].values
    return X_test, y_test

def cargar_datos_excel(ruta, target, features):
    df = pd.read_excel(ruta)
    split_idx = int(len(df) * (1 - TEST_SIZE))
    df_test = df.iloc[split_idx:].reset_index(drop=True)
    X_test = df_test[features]
    y_test = df_test[target].values
    return X_test, y_test

def cargar_modelo_joblib(ruta):
    modelo = joblib.load(ruta)
    if isinstance(modelo, dict) and "model" in modelo:
        return modelo["model"]
    return modelo

def calcular_errores(y_true, y_pred):
    return np.abs(y_true - y_pred)

# ============================================================
# PRUEBAS DE HIPÓTESIS
# ============================================================

def prueba_t_pareada(e1, e2, nombre1, nombre2):
    """
    Prueba t pareada. H0: mu_d = 0 (no hay diferencia).
    Determina cuál modelo es mejor según el signo de t.
    """
    t_stat, p_val = stats.ttest_rel(e1, e2)
    if p_val < ALPHA:
        mejor = f"{nombre1} mejor" if t_stat < 0 else f"{nombre2} mejor"
    else:
        mejor = "Empate (no significativo)"
    return {
        "comparacion": f"{nombre1} vs {nombre2}",
        "prueba": "t pareada (paramétrica)",
        "estadistico": t_stat,
        "p_valor": p_val,
        "decision": "Rechazar H0" if p_val < ALPHA else "No rechazar H0",
        "mejor": mejor,
    }

def prueba_wilcoxon(e1, e2, nombre1, nombre2):
    """
    Prueba de Wilcoxon de rango con signo. H0: mediana de diferencias = 0.
    Imprime W+, W- y W. Determina cuál modelo es mejor.
    """
    from scipy.stats import wilcoxon, rankdata
    
    d = e1 - e2
    d_nonzero = d[d != 0]
    n = len(d_nonzero)
    
    if n == 0:
        return {
            "comparacion": f"{nombre1} vs {nombre2}",
            "prueba": "Wilcoxon (no paramétrica)",
            "W+": 0, "W-": 0, "W": 0,
            "p_valor": 1.0,
            "decision": "No rechazar H0 (sin diferencias)",
            "mejor": "Empate",
        }
    
    abs_d = np.abs(d_nonzero)
    rangos = rankdata(abs_d)
    W_plus = np.sum(rangos[d_nonzero > 0])
    W_minus = np.sum(rangos[d_nonzero < 0])
    W = min(W_plus, W_minus)
    
    _, p_val = wilcoxon(e1, e2)
    
    if p_val < ALPHA:
        mejor = f"{nombre1} mejor" if W_plus < W_minus else f"{nombre2} mejor"
    else:
        mejor = "Empate (no significativo)"
    
    return {
        "comparacion": f"{nombre1} vs {nombre2}",
        "prueba": "Wilcoxon (no paramétrica)",
        "W+": W_plus,
        "W-": W_minus,
        "W": W,
        "p_valor": p_val,
        "decision": "Rechazar H0" if p_val < ALPHA else "No rechazar H0",
        "mejor": mejor,
    }

def prueba_f_varianzas(e1, e2, nombre1, nombre2):
    """
    Prueba F de igualdad de varianzas. H0: sigma1^2 = sigma2^2.
    No determina cuál modelo es mejor (solo si las varianzas son iguales).
    """
    f_stat = np.var(e1, ddof=1) / np.var(e2, ddof=1)
    gl1 = len(e1) - 1
    gl2 = len(e2) - 1
    p_val = 2 * min(stats.f.cdf(f_stat, gl1, gl2), 1 - stats.f.cdf(f_stat, gl1, gl2))
    return {
        "comparacion": f"{nombre1} vs {nombre2}",
        "prueba": "F de varianzas (paramétrica)",
        "estadistico": f_stat,
        "p_valor": p_val,
        "decision": "Rechazar H0" if p_val < ALPHA else "No rechazar H0",
        "mejor": "N/A (no compara medias)",
    }

def prueba_chi2_normalidad(errores, nombre):
    """
    Prueba chi-cuadrado de bondad de ajuste. H0: los errores son normales.
    No determina cuál modelo es mejor.
    """
    z = (errores - np.mean(errores)) / np.std(errores, ddof=1)
    bins = np.linspace(-3, 3, 11)
    observados, _ = np.histogram(z, bins=bins)
    esperados = []
    for i in range(len(bins) - 1):
        prob = stats.norm.cdf(bins[i+1]) - stats.norm.cdf(bins[i])
        esperados.append(prob * len(z))
    esperados = np.array(esperados)
    chi2 = np.sum((observados - esperados)**2 / esperados)
    gl = len(observados) - 1 - 2
    p_val = 1 - stats.chi2.cdf(chi2, gl)
    return {
        "comparacion": f"Normalidad de errores ({nombre})",
        "prueba": "Chi-cuadrado de bondad de ajuste (paramétrica)",
        "estadistico": chi2,
        "p_valor": p_val,
        "decision": "Rechazar H0 (no normal)" if p_val < ALPHA else "No rechazar H0 (normal)",
        "mejor": "N/A (no compara modelos)",
    }

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("  PRUEBAS DE HIPÓTESIS PARA MODELOS PREDICTIVOS")
    print("=" * 70)

    resultados = []

    # ---------- GENERACIÓN ----------
    print("\n" + "=" * 70)
    print("  DOMINIO: GENERACIÓN")
    print("=" * 70)

    X_test_gen, y_test_gen = cargar_datos_csv(RUTA_DATASET_GEN, TARGET_GEN, FEATURES_GEN)
    print(f"   Conjunto de prueba: {len(y_test_gen)} observaciones")

    modelos_gen = {}
    if os.path.exists(RUTA_RF_GEN):
        modelos_gen["RF"] = cargar_modelo_joblib(RUTA_RF_GEN)
        print(f"   ✅ RF cargado")
    if os.path.exists(RUTA_XGB_GEN):
        modelos_gen["XGB"] = cargar_modelo_joblib(RUTA_XGB_GEN)
        print(f"   ✅ XGB cargado")

    # Cargar Havana v1 con ml_model_service
    try:
        from app.services.ml_model_service import ml_model_service
        ml_model_service.load_model("havana_v1")
        modelos_gen["Havana_v1"] = ml_model_service
        print(f"   ✅ Havana v1 cargado desde ml_model_service")
    except Exception as e:
        print(f"   ⚠️ No se pudo cargar Havana v1: {e}")
        print(f"   → Se omite la comparación con Havana v1")

    errores_gen = {}
    for nombre, modelo in modelos_gen.items():
        try:
            if nombre == "Havana_v1":
                y_pred = modelo.predict(X_test_gen)
            else:
                y_pred = modelo.predict(X_test_gen)
            errores_gen[nombre] = calcular_errores(y_test_gen, y_pred)
            print(f"   {nombre}: MAE = {np.mean(errores_gen[nombre]):.6f}")
        except Exception as e:
            print(f"   ⚠️ Error al predecir con {nombre}: {e}")

    # Pruebas de hipótesis para generación
    if "RF" in errores_gen and "XGB" in errores_gen:
        resultados.append(prueba_t_pareada(errores_gen["RF"], errores_gen["XGB"], "RF", "XGB"))
        resultados.append(prueba_wilcoxon(errores_gen["RF"], errores_gen["XGB"], "RF", "XGB"))
        resultados.append(prueba_f_varianzas(errores_gen["RF"], errores_gen["XGB"], "RF", "XGB"))
    if "RF" in errores_gen and "Havana_v1" in errores_gen:
        resultados.append(prueba_t_pareada(errores_gen["RF"], errores_gen["Havana_v1"], "RF", "Havana v1"))
        resultados.append(prueba_wilcoxon(errores_gen["RF"], errores_gen["Havana_v1"], "RF", "Havana v1"))
    if "XGB" in errores_gen and "Havana_v1" in errores_gen:
        resultados.append(prueba_t_pareada(errores_gen["XGB"], errores_gen["Havana_v1"], "XGB", "Havana v1"))
        resultados.append(prueba_wilcoxon(errores_gen["XGB"], errores_gen["Havana_v1"], "XGB", "Havana v1"))
    for nombre, e in errores_gen.items():
        resultados.append(prueba_chi2_normalidad(e, nombre))

    # ---------- CONSUMO ----------
    print("\n" + "=" * 70)
    print("  DOMINIO: CONSUMO")
    print("=" * 70)

    X_test_con, y_test_con = cargar_datos_excel(RUTA_DATASET_CON, TARGET_CON, FEATURES_CON)
    print(f"   Conjunto de prueba: {len(y_test_con)} observaciones")

    modelos_con = {}
    if os.path.exists(RUTA_RF_CON):
        modelos_con["RF"] = cargar_modelo_joblib(RUTA_RF_CON)
        print(f"   ✅ RF cargado")
    if os.path.exists(RUTA_XGB_CON):
        modelos_con["XGB"] = cargar_modelo_joblib(RUTA_XGB_CON)
        print(f"   ✅ XGB cargado")

    errores_con = {}
    for nombre, modelo in modelos_con.items():
        try:
            y_pred = modelo.predict(X_test_con)
            errores_con[nombre] = calcular_errores(y_test_con, y_pred)
            print(f"   {nombre}: MAE = {np.mean(errores_con[nombre]):.6f}")
        except Exception as e:
            print(f"   ⚠️ Error al predecir con {nombre}: {e}")

    if "RF" in errores_con and "XGB" in errores_con:
        resultados.append(prueba_t_pareada(errores_con["RF"], errores_con["XGB"], "RF (consumo)", "XGB (consumo)"))
        resultados.append(prueba_wilcoxon(errores_con["RF"], errores_con["XGB"], "RF (consumo)", "XGB (consumo)"))
        resultados.append(prueba_f_varianzas(errores_con["RF"], errores_con["XGB"], "RF (consumo)", "XGB (consumo)"))
    for nombre, e in errores_con.items():
        resultados.append(prueba_chi2_normalidad(e, f"{nombre} (consumo)"))

    # ---------- RESULTADOS ----------
    print("\n" + "=" * 70)
    print("  RESULTADOS")
    print("=" * 70)
    df_resultados = pd.DataFrame(resultados)
    print(df_resultados.to_string(index=False))

    ruta_salida = Path(r"D:\gemelo-digital-main\backend\ensamble\artifacts\resultados") / "pruebas_hipotesis.csv"
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df_resultados.to_csv(ruta_salida, index=False)
    print(f"\n📁 Resultados guardados en: {ruta_salida}")

if __name__ == "__main__":
    main()