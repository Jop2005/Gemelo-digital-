"""
Reentrena el Random Forest de Havana v1 sobre el dataset del ensamblaje (2005-2015)
y calcula Diebold-Mariano contra el ensamblaje.
"""
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ===== RUTAS =====
BASE = Path(__file__).parent
DATASET = BASE / "data" / "havana_solar_training.csv"
RF_ENSAMBLE = BASE / "artifacts" / "modelos" / "generacion" / "RandomForest_20261028_173518.joblib"
XGB_ENSAMBLE = BASE / "artifacts" / "modelos" / "generacion" / "XGBoost_20261028_174133.joblib"
CONFIG_ENSAMBLE = BASE / "artifacts" / "config" / "generacion" / "config_ensamblaje_20261028_174151.json"
CONFIG_SEG = BASE / "artifacts" / "config" / "generacion" / "segmentacion_shortwave_radiation_20261028_172544.json"

# Features del ensamblaje
FEATURES = [
    "shortwave_radiation", "clearsky_ghi", "clearsky_index",
    "solar_elevation", "effective_irradiance", "cloud_cover",
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "temp_loss_factor", "hour_sin", "hour_cos", "month_sin", "month_cos"
]
TARGET = "capacity_factor"

# ===== CARGAR DATOS =====
print("Cargando dataset...")
df = pd.read_csv(DATASET, index_col="time")
df.index = pd.to_datetime(df.index, utc=True)
df = df.dropna(subset=FEATURES + [TARGET])

X = df[FEATURES]
y = df[TARGET].clip(0, 1)
daylight = X["solar_elevation"].values > 0

split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
day_test = daylight[split:]
print(f"Train: {len(X_train)}  Test: {len(X_test)}  (día: {day_test.sum()})")

# ===== REENTRENAR RANDOM FOREST DE HAVANA V1 =====
print("\nReentrenando Random Forest de Havana v1...")
rf_havana = RandomForestRegressor(
    n_estimators=200, max_depth=12, min_samples_leaf=4,
    max_features="sqrt", random_state=42, n_jobs=-1
)
rf_havana.fit(X_train, y_train)
pred_havana = np.clip(rf_havana.predict(X_test), 0, 1)

# ===== CARGAR ENSAMBLE =====
print("\nCargando ensamble...")
def cargar_modelo(ruta):
    data = joblib.load(ruta)
    return data["model"] if isinstance(data, dict) and "model" in data else data

rf_ens = cargar_modelo(RF_ENSAMBLE)
xgb_ens = cargar_modelo(XGB_ENSAMBLE)

with open(CONFIG_ENSAMBLE, "r", encoding="utf-8") as f:
    ens_config = json.load(f)
with open(CONFIG_SEG, "r", encoding="utf-8") as f:
    seg_config = json.load(f)

def asignar_segmento(row):
    variable = seg_config["variables"][0]
    centroides = np.array(seg_config["centroides"]).flatten()
    orden = np.array(seg_config["orden"])
    valor = row[variable]
    distancias = np.abs(valor - centroides)
    cluster = np.argmin(distancias)
    return int(np.nonzero(orden == cluster)[0][0])

pred_rf = rf_ens.predict(X_test)
pred_xgb = xgb_ens.predict(X_test)
pred_ens = np.zeros(len(X_test))
for i, (idx, row) in enumerate(X_test.iterrows()):
    seg = asignar_segmento(row)
    cfg = ens_config["segmentos"].get(str(seg), {})
    if cfg.get("modelo") == "Ensemble":
        pesos = cfg.get("pesos", {})
        pred_ens[i] = sum(pesos.get(m, 0) * (pred_rf[i] if m == "RandomForest" else pred_xgb[i]) for m in pesos)
    elif cfg.get("modelo") == "RandomForest":
        pred_ens[i] = pred_rf[i]
    elif cfg.get("modelo") == "XGBoost":
        pred_ens[i] = pred_xgb[i]
    else:
        pred_ens[i] = pred_rf[i]

# ===== MÉTRICAS =====
print("\n" + "="*60)
print("MÉTRICAS GLOBALES (día)")
print("="*60)
for name, pred in [("Havana v1 (reentrenado)", pred_havana), ("Ensamblaje", pred_ens)]:
    rmse = np.sqrt(mean_squared_error(y_test.values[day_test], pred[day_test]))
    mae = mean_absolute_error(y_test.values[day_test], pred[day_test])
    r2 = r2_score(y_test.values[day_test], pred[day_test])
    print(f"  {name:<25} RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

# ===== DIEBOLD-MARIANO =====
print("\n" + "="*60)
print("DIEBOLD-MARIANO: Ensamblaje vs Havana v1 (reentrenado)")
print("="*60)
e_havana = pred_havana[day_test] - y_test.values[day_test]
e_ens = pred_ens[day_test] - y_test.values[day_test]
d = np.abs(e_ens) - np.abs(e_havana)
n = len(d)
dm = d.mean() / np.sqrt(d.var(ddof=1) / n)
p = 2 * (1 - stats.norm.cdf(abs(dm)))
print(f"  DM = {dm:.4f}  p = {p:.6f}")
if p < 0.05:
    print(f"  → El ensamblaje es significativamente mejor (p < 0.05)")
else:
    print(f"  → No hay diferencia significativa (p >= 0.05)")