import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===== RUTAS ABSOLUTAS (cambia solo estas si es necesario) =====
RUTA_CONSUMO = r"D:\gemelo-digital-main\backend\ensamble\data\_BD del consumo en la cujae_.xlsx"
RUTA_RF = r"D:\gemelo-digital-main\backend\ensamble\artifacts\modelos\consumo\RandomForest_20261028_172328.joblib"
RUTA_XGB = r"D:\gemelo-digital-main\backend\ensamble\artifacts\modelos\consumo\XGBoost_20261028_172447.joblib"
RUTA_CONFIG_ENSAMBLE = r"D:\gemelo-digital-main\backend\ensamble\artifacts\config\consumo\config_ensamblaje_20261028_172500.json"
RUTA_SEGMENTACION = r"D:\gemelo-digital-main\backend\ensamble\artifacts\config\consumo\segmentacion_reglas_20261028_172243.json"
TARGET = "Energia_consumida_kW"
FEATURES = ["Hora", "Mes", "tipo_dia_bin"]
# ===============================================================

# 1. Cargar datos y split 80/20
df = pd.read_excel(RUTA_CONSUMO)
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

# 4. Función para asignar segmento (reglas)
def asignar_segmento(row):
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

# 5. Predecir con RF y XGB
pred_rf = rf.predict(X_test)
pred_xgb = xgb.predict(X_test)

# 6. Calcular predicción del ensamble
pred_ensamble = np.zeros(len(test))
for i, row in test.iterrows():
    seg = asignar_segmento(row)
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

# 7. Calcular métricas
mae = mean_absolute_error(y_true, pred_ensamble)
rmse = np.sqrt(mean_squared_error(y_true, pred_ensamble))
r2 = r2_score(y_true, pred_ensamble)
mape = np.mean(np.abs((y_true - pred_ensamble) / y_true)) * 100

# MASE: naive de persistencia sobre train
y_train = df.iloc[:split_idx][TARGET].values
mae_naive = mean_absolute_error(y_train[1:], y_train[:-1])
mase = mae / mae_naive if mae_naive > 0 else float("inf")

print("=== MÉTRICAS DEL ENSAMBLE (CONSUMO) ===")
print(f"MAE:  {mae:.2f} kW")
print(f"RMSE: {rmse:.2f} kW")
print(f"R²:   {r2:.4f}")
print(f"MAPE: {mape:.2f} %")
print(f"MASE: {mase:.4f}")