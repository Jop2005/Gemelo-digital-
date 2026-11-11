import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# ===== RUTAS ABSOLUTAS (cambia solo estas) =====
RUTA_CONSUMO = r'D:\gemelo-digital-main\backend\ensamble\data\_BD del consumo en la cujae_.xlsx'  # <-- ruta absoluta al Excel
HOJA = 0  # nombre o índice de la hoja (0 = primera hoja)
COL_TARGET = 'Energia_consumida_kW'
COL_MES = 'Mes'
COL_HORA = 'Hora'
COL_TIPO_DIA = 'tipo_dia_bin'
# ===============================================

# 1. Cargar el dataset desde Excel
df = pd.read_excel(RUTA_CONSUMO, sheet_name=HOJA)

# 2. Partición cronológica 80/20
n = len(df)
train = df.iloc[:int(0.8 * n)].copy()
test = df.iloc[int(0.8 * n):].copy()

# 3. Calcular la media histórica por (Mes, Hora, tipo_dia) SOLO con train
perfil = train.groupby([COL_MES, COL_HORA, COL_TIPO_DIA])[COL_TARGET].mean().reset_index()
perfil.columns = [COL_MES, COL_HORA, COL_TIPO_DIA, 'pred_perfil']

# 4. Unir el perfil al conjunto de prueba
test = test.merge(perfil, on=[COL_MES, COL_HORA, COL_TIPO_DIA], how='left')

# 5. Si algún grupo no existe en train, rellenar con la media global de train
test['pred_perfil'] = test['pred_perfil'].fillna(train[COL_TARGET].mean())

# 6. Calcular métricas
y_true = test[COL_TARGET]
y_pred = test['pred_perfil']

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# MASE: escalado por el error del naive (persistencia)
naive = y_true.shift(1).dropna()
mase_denom = np.mean(np.abs(y_true.iloc[1:] - naive))
mase = mae / mase_denom

print(f"Perfil estadístico (Mes-Hora-tipo_dia):")
print(f"MAE:  {mae:.2f} kW")
print(f"RMSE: {rmse:.2f} kW")
print(f"R²:   {r2:.4f}")
print(f"MAPE: {mape:.2f} %")
print(f"MASE: {mase:.4f}")