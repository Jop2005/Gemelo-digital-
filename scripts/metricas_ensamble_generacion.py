import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===== RUTAS ABSOLUTAS (cambia solo estas si es necesario) =====
RUTA_GENERACION = r"D:\gemelo-digital-main\backend\ensamble\data\havana_solar_training.csv"
RUTA_RF = r"D:\gemelo-digital-main\backend\ensamble\artifacts\modelos\generacion\RandomForest_20261028_173518.joblib"
RUTA_XGB = r"D:\gemelo-digital-main\backend\ensamble\artifacts\modelos\generacion\XGBoost_20261028_174133.joblib"
RUTA_CONFIG_ENSAMBLE = r"D:\gemelo-digital-main\backend\ensamble\artifacts\config\generacion\config_ensamblaje_20261028_174151.json"
RUTA_SEGMENTACION = r"D:\gemelo-digital-main\backend\ensamble\artifacts\config\generacion\segmentacion_shortwave_radiation_20261028_172544.json"
TARGET = "capacity_factor"
FEATURES = [
    "shortwave_radiation", "clearsky_ghi", "clearsky_index",
    "solar_elevation", "effective_irradiance", "cloud_cover",
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "temp_loss_factor", "hour_sin", "hour_cos", "month_sin", "month_cos"
]
# ===============================================================

# 1. Cargar datos y split 80/20
df = pd.read_csv(RUTA_GENERACION)
split_idx = int(len(df) * 0.8)
test = df.iloc[split_idx:].reset_index(drop=True)
y_true = test[TARGET].values
X_test = test[FEATURES]

# 2. Cargar modelos
def cargar_modelo(ruta):
    data = joblib.load(ruta)
    return data["model"] if isinstance(data, dict) and "model" in data else data

rf = cargar_modelo(RUTA_RF)
xgb = cargar_modelo(RUTA_XGB)

# 3. Cargar configuración de segmentación y ensamble
with open(RUTA_SEGMENTACION, "r", encoding="utf-8") as f:
    seg_config = json.load(f)
with open(RUTA_CONFIG_ENSAMBLE, "r", encoding="utf-8") as f:
    ens_config = json.load(f)

# 4. Función para asignar segmento (clustering)
def asignar_segmento_clustering(row, seg_config):
    variable = seg_config["variables"][0]
    centroides = np.array(seg_config["centroides"]).flatten()
    orden = np.array(seg_config["orden"])
    valor = row[variable]
    distancias = np.abs(valor - centroides)
    cluster = np.argmin(distancias)
    return int(np.nonzero(orden == cluster)[0][0])

# 5. Predecir con RF y XGB
pred_rf = rf.predict(X_test)
pred_xgb = xgb.predict(X_test)

# 6. Calcular predicción del ensamble
pred_ensamble = np.zeros(len(test))
for i, row in test.iterrows():
    seg = asignar_segmento_clustering(row, seg_config)
    config_seg = ens_config["segmentos"].get(str(seg), {})
    if config_seg.get("modelo") == "Ensemble":
        pesos = config_seg.get("pesos", {})
        p = 0.0
        if "RandomForest" in pesos:
            p += pesos["RandomForest"] * pred_rf[i]
        if "XGBoost" in pesos:
            p += pesos["XGBoost"] * pred_xgb[i]
        pred_ensamble[i] = p
    elif config_seg.get("modelo") == "RandomForest":
        pred_ensamble[i] = pred_rf[i]
    elif config_seg.get("modelo") == "XGBoost":
        pred_ensamble[i] = pred_xgb[i]
    else:
        pred_ensamble[i] = pred_rf[i]  # fallback

# 7. Calcular métricas globales
mae = mean_absolute_error(y_true, pred_ensamble)
rmse = np.sqrt(mean_squared_error(y_true, pred_ensamble))
r2 = r2_score(y_true, pred_ensamble)

mask_nonzero = y_true != 0
mape = np.mean(np.abs((y_true[mask_nonzero] - pred_ensamble[mask_nonzero]) / y_true[mask_nonzero])) * 100

# MASE: naive de persistencia sobre train
y_train = df.iloc[:split_idx][TARGET].values
mae_naive = mean_absolute_error(y_train[1:], y_train[:-1])
mase = mae / mae_naive if mae_naive > 0 else float("inf")

print("=== MÉTRICAS GLOBALES DEL ENSAMBLE (GENERACIÓN) ===")
print(f"MAE:  {mae:.6f} (factor de capacidad)")
print(f"RMSE: {rmse:.6f}")
print(f"R²:   {r2:.4f}")
print(f"MAPE: {mape:.2f} %")
print(f"MASE: {mase:.4f}")

# 8. Calcular métricas diurnas (solar_elevation > 0)
mask_diurno = X_test["solar_elevation"].values > 0
y_true_d = y_true[mask_diurno]
pred_d = pred_ensamble[mask_diurno]
mae_d = mean_absolute_error(y_true_d, pred_d)
rmse_d = np.sqrt(mean_squared_error(y_true_d, pred_d))
r2_d = r2_score(y_true_d, pred_d)

mask_d_nonzero = y_true_d != 0
mape_d = np.mean(np.abs((y_true_d[mask_d_nonzero] - pred_d[mask_d_nonzero]) / y_true_d[mask_d_nonzero])) * 100

# nRMSE y nMAE diurnos (normalizados por capacidad de referencia = 1.0)
nrmse_d = rmse_d / 1.0
nmae_d = mae_d / 1.0

# MASE diurno
mase_d = mae_d / mae_naive if mae_naive > 0 else float("inf")

print("\n=== MÉTRICAS DIURNAS DEL ENSAMBLE (GENERACIÓN) ===")
print(f"Muestras diurnas: {mask_diurno.sum()} de {len(y_true)}")
print(f"MAE:  {mae_d:.6f}")
print(f"RMSE: {rmse_d:.6f}")
print(f"R²:   {r2_d:.4f}")
print(f"nRMSE: {nrmse_d*100:.2f} %")
print(f"nMAE:  {nmae_d*100:.2f} %")
print(f"MASE:  {mase_d:.4f}")
print(f"MAPE:  {mape_d:.2f} %")