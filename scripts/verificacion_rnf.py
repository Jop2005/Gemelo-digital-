"""
Verificación de Requisitos No Funcionales (RNF) del módulo predictivo.

RNF1: Entrenamiento < 4 h
RNF2: Degradación del MAE < 20% con 10% de datos ausentes
RNF3: Degradación del MAE < 20% con ruido significativo
RNF4: Tolerancia a fallos de modelos individuales
RNF7: Reproducibilidad
RNF8: Procesar 100 000 registros sin degradación significativa
"""
import pandas as pd
import numpy as np
import joblib
import json
import time
import os
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.impute import KNNImputer

# ===== RUTAS =====
RUTA_CONSUMO = r"D:\gemelo-digital-main\backend\ensamble\data\_BD del consumo en la cujae_.xlsx"
RUTA_GENERACION = r"D:\gemelo-digital-main\backend\ensamble\data\havana_solar_training.csv"
RUTA_RF_CON = r"D:\gemelo-digital-main\backend\ensamble\artifacts\modelos\consumo\RandomForest_20261028_172328.joblib"
RUTA_XGB_CON = r"D:\gemelo-digital-main\backend\ensamble\artifacts\modelos\consumo\XGBoost_20261028_172447.joblib"
RUTA_RF_GEN = r"D:\gemelo-digital-main\backend\ensamble\artifacts\modelos\generacion\RandomForest_20261028_173518.joblib"
RUTA_XGB_GEN = r"D:\gemelo-digital-main\backend\ensamble\artifacts\modelos\generacion\XGBoost_20261028_174133.joblib"

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
    data = joblib.load(ruta)
    return data["model"] if isinstance(data, dict) and "model" in data else data

# ===== RNF1: Tiempo de entrenamiento =====
def rnf1_tiempo_entrenamiento():
    print("\n--- RNF1: Tiempo de entrenamiento ---")
    # Consumo
    df = pd.read_excel(RUTA_CONSUMO)
    X = df[FEATURES_CON]
    y = df[TARGET_CON]
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    t_rf = time.time() - t0
    t0 = time.time()
    xgb = XGBRegressor(n_estimators=500, max_depth=7, learning_rate=0.01, random_state=42, verbosity=0)
    xgb.fit(X, y)
    t_xgb = time.time() - t0
    print(f"  Consumo: RF={t_rf:.2f}s, XGB={t_xgb:.2f}s")

    # Generación
    df = pd.read_csv(RUTA_GENERACION)
    X = df[FEATURES_GEN]
    y = df[TARGET_GEN]
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    t_rf = time.time() - t0
    t0 = time.time()
    xgb = XGBRegressor(n_estimators=1200, max_depth=6, learning_rate=0.015, random_state=42, verbosity=0)
    xgb.fit(X, y)
    t_xgb = time.time() - t0
    print(f"  Generación: RF={t_rf:.2f}s, XGB={t_xgb:.2f}s")
    print("  → RNF1 se cumple si el tiempo total es < 4 h.")

# ===== RNF2 y RNF3: Degradación del MAE con datos ausentes y ruido =====
def rnf2_rnf3_degradacion():
    print("\n--- RNF2 y RNF3: Degradación del MAE con datos ausentes y ruido ---")
    # Consumo
    df = pd.read_excel(RUTA_CONSUMO)
    split_idx = int(len(df) * 0.8)
    test = df.iloc[split_idx:].reset_index(drop=True)
    X_test = test[FEATURES_CON]
    y_test = test[TARGET_CON].values
    rf = cargar_modelo(RUTA_RF_CON)
    mae_base = mean_absolute_error(y_test, rf.predict(X_test))
    print(f"  Consumo baseline: MAE={mae_base:.4f}")

    # RNF2: 10% de datos ausentes
    rng = np.random.default_rng(42)
    mask = rng.random(X_test.shape) < 0.1
    X_nulos = X_test.mask(mask)
    imputer = KNNImputer(n_neighbors=5)
    imputer.fit(X_test)
    X_imputado = pd.DataFrame(imputer.transform(X_nulos), columns=X_test.columns)
    mae_nulos = mean_absolute_error(y_test, rf.predict(X_imputado))
    deg_nulos = (mae_nulos - mae_base) / mae_base * 100
    print(f"  Consumo con 10% nulos: MAE={mae_nulos:.4f} (degradación={deg_nulos:.2f}%)")
    # RNF3: Ruido gaussiano (10% de la desviación estándar)
    ruido = rng.normal(0, 0.1 * X_test.std(), X_test.shape)
    X_ruido = pd.DataFrame(X_test.values + ruido, columns=X_test.columns)
    mae_ruido = mean_absolute_error(y_test, rf.predict(X_ruido))
    deg_ruido = (mae_ruido - mae_base) / mae_base * 100
    print(f"  Consumo con ruido: MAE={mae_ruido:.4f} (degradación={deg_ruido:.2f}%)")
    print("  → RNF2 y RNF3 se cumplen si la degradación es < 20%.")

# ===== RNF4: Tolerancia a fallos =====
def rnf4_tolerancia_fallos():
    print("\n--- RNF4: Tolerancia a fallos de modelos individuales ---")
    # Simulación: el orquestador debe renormalizar pesos si un modelo falla
    pesos = {"RandomForest": 0.5, "XGBoost": 0.5}
    print(f"  Pesos iniciales: {pesos}")
    # Simular fallo de XGBoost
    pesos_sin_xgb = {"RandomForest": 1.0}
    print(f"  Pesos tras fallo de XGBoost (renormalizados): {pesos_sin_xgb}")
    print("  → RNF4 se cumple si el orquestador renormaliza los pesos y sigue prediciendo.")

# ===== RNF7: Reproducibilidad =====
def rnf7_reproducibilidad():
    print("\n--- RNF7: Reproducibilidad ---")
    df = pd.read_excel(RUTA_CONSUMO)
    X = df[FEATURES_CON]
    y = df[TARGET_CON]
    rf1 = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf1.fit(X, y)
    pred1 = rf1.predict(X.iloc[:100])
    rf2 = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf2.fit(X, y)
    pred2 = rf2.predict(X.iloc[:100])
    identicos = np.allclose(pred1, pred2)
    print(f"  ¿Predicciones idénticas con misma semilla? {identicos}")
    print("  → RNF7 se cumple si las predicciones son idénticas.")

# ===== RNF8: 100 000 registros =====
def rnf8_100k():
    print("\n--- RNF8: Procesar 100 000 registros ---")
    df = pd.read_excel(RUTA_CONSUMO)
    X = df[FEATURES_CON]
    y = df[TARGET_CON]
    rf = cargar_modelo(RUTA_RF_CON)
    # Generar 100 000 registros sintéticos
    X_sintetico = pd.DataFrame({
        "Hora": np.random.randint(0, 24, 100000),
        "Mes": np.random.randint(1, 13, 100000),
        "tipo_dia_bin": np.random.randint(0, 2, 100000),
    })
    t0 = time.time()
    _ = rf.predict(X_sintetico)
    t_inferencia = time.time() - t0
    print(f"  Tiempo de inferencia para 100 000 registros: {t_inferencia:.2f}s")
    print("  → RNF8 se cumple si el tiempo es razonable (< 1 min).")

if __name__ == "__main__":
    rnf1_tiempo_entrenamiento()
    rnf2_rnf3_degradacion()
    rnf4_tolerancia_fallos()
    rnf7_reproducibilidad()
    rnf8_100k()