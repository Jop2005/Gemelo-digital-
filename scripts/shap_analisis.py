"""
Análisis de interpretabilidad con SHAP para los modelos de consumo y generación.
Con prints de progreso para identificar cuellos de botella.
"""
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import time

# ===== RUTAS =====
RUTA_CONSUMO = r"D:\gemelo-digital-main\backend\ensamble\data\_BD del consumo en la cujae_.xlsx"
RUTA_GENERACION = r"D:\gemelo-digital-main\backend\ensamble\data\havana_solar_training.csv"
RUTA_RF_CON = r"D:\gemelo-digital-main\backend\ensamble\artifacts\modelos\consumo\RandomForest_20261028_172328.joblib"
RUTA_XGB_CON = r"D:\gemelo-digital-main\backend\ensamble\artifacts\modelos\consumo\XGBoost_20261028_172447.joblib"
RUTA_RF_GEN = r"D:\gemelo-digital-main\backend\ensamble\artifacts\modelos\generacion\RandomForest_20261028_173518.joblib"
RUTA_XGB_GEN = r"D:\gemelo-digital-main\backend\ensamble\artifacts\modelos\generacion\XGBoost_20261028_174133.joblib"

CARPETA_SALIDA = r"D:\gemelo-digital-main\backend\ensamble\artifacts\resultados\shap"
os.makedirs(CARPETA_SALIDA, exist_ok=True)

TARGET_CON = "Energia_consumida_kW"
FEATURES_CON = ["Hora", "Mes", "tipo_dia_bin"]
TARGET_GEN = "capacity_factor"
FEATURES_GEN = [
    "shortwave_radiation", "clearsky_ghi", "clearsky_index",
    "solar_elevation", "effective_irradiance", "cloud_cover",
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "temp_loss_factor", "hour_sin", "hour_cos", "month_sin", "month_cos"
]

def cargar_modelo(ruta):
    print(f"  Cargando modelo: {ruta}")
    t0 = time.time()
    data = joblib.load(ruta)
    modelo = data["model"] if isinstance(data, dict) and "model" in data else data
    print(f"  Modelo cargado en {time.time() - t0:.2f}s")
    return modelo

def shap_summary(modelo, X, nombre_modelo, nombre_dominio, n_grafico=300, max_display=6):
    print(f"\n=== SHAP: {nombre_modelo} ({nombre_dominio}) ===")
    print(f"  Calculando valores SHAP para {len(X)} registros...")
    t0 = time.time()
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X)
    print(f"  Valores SHAP calculados en {time.time() - t0:.2f}s")
    print(f"  Guardando CSV...")
    t0 = time.time()
    df_shap = pd.DataFrame(shap_values, columns=X.columns)
    ruta_csv = os.path.join(CARPETA_SALIDA, f"shap_values_{nombre_dominio}_{nombre_modelo}.csv")
    df_shap.to_csv(ruta_csv, index=False)
    print(f"  CSV guardado en {time.time() - t0:.2f}s: {ruta_csv}")
    print(f"  Generando summary plot...")
    t0 = time.time()
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=min(n_grafico, len(X)), replace=False)
    plt.figure()
    shap.summary_plot(shap_values[idx], X.iloc[idx], show=False, max_display=max_display)
    ruta_png = os.path.join(CARPETA_SALIDA, f"shap_summary_{nombre_dominio}_{nombre_modelo}.png")
    plt.savefig(ruta_png, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Summary plot guardado en {time.time() - t0:.2f}s: {ruta_png}")

if __name__ == "__main__":
    # Consumo
    print("\n=== CONSUMO ===")
    print("Cargando datos de consumo...")
    t0 = time.time()
    df_con = pd.read_excel(RUTA_CONSUMO)
    print(f"Datos cargados en {time.time() - t0:.2f}s ({len(df_con)} registros)")
    print("Muestreando 500 registros...")
    X_con = df_con[FEATURES_CON].sample(n=min(50, len(df_con)), random_state=42)
    rf_con = cargar_modelo(RUTA_RF_CON)
    xgb_con = cargar_modelo(RUTA_XGB_CON)
    shap_summary(rf_con, X_con, "RandomForest", "consumo", n_grafico=50)
    shap_summary(xgb_con, X_con, "XGBoost", "consumo", n_grafico=50)

    # Generación
    print("\n=== GENERACIÓN ===")
    print("Cargando datos de generación...")
    t0 = time.time()
    df_gen = pd.read_csv(RUTA_GENERACION)
    print(f"Datos cargados en {time.time() - t0:.2f}s ({len(df_gen)} registros)")
    print("Muestreando 500 registros...")
    X_gen = df_gen[FEATURES_GEN].sample(n=min(50, len(df_gen)), random_state=42)
    rf_gen = cargar_modelo(RUTA_RF_GEN)
    xgb_gen = cargar_modelo(RUTA_XGB_GEN)
    shap_summary(rf_gen, X_gen, "RandomForest", "generacion", n_grafico=50)
    shap_summary(xgb_gen, X_gen, "XGBoost", "generacion", n_grafico=50)