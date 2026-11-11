"""
Genera wind_8760.csv con la velocidad del viento en km/h a partir del archivo de Open-Meteo.
"""
import pandas as pd
import numpy as np
import os

# ============================================================
# 1. CONFIGURACIÓN: PON AQUÍ EL NOMBRE DE TU ARCHIVO DESCARGADO
# ============================================================
ARCHIVO_ENTRADA = "open-meteo-23.09N82.37W9m.csv"  # <--- CAMBIA SI ES NECESARIO
ARCHIVO_SALIDA = "wind_8760.csv"

# ============================================================
# 2. LEER EL ARCHIVO DE OPEN-METEO
# ============================================================
if not os.path.exists(ARCHIVO_ENTRADA):
    raise FileNotFoundError(f"No se encuentra el archivo: {ARCHIVO_ENTRADA}")

print(f"📂 Leyendo: {ARCHIVO_ENTRADA}")

with open(ARCHIVO_ENTRADA, 'r', encoding='utf-8') as f:
    lines = f.readlines()

start_line = 0
for i, line in enumerate(lines):
    if line.strip().lower().startswith('time'):
        start_line = i
        break

df = pd.read_csv(ARCHIVO_ENTRADA, skiprows=start_line, skip_blank_lines=True)
col_viento = None
for col in df.columns:
    if 'wind_speed' in col.lower():
        col_viento = col
        break

if col_viento is None:
    raise ValueError(f"No se encontró columna de viento. Columnas: {df.columns.tolist()}")

print(f"✅ Columna de viento: '{col_viento}'")
viento_ms = pd.to_numeric(df[col_viento], errors='coerce').values
print(f"✅ Datos leídos: {len(viento_ms)} horas")

# ============================================================
# 3. CONVERTIR DE m/s A km/h (lo que espera el notebook)
# ============================================================
viento_kmh = viento_ms * 3.6

# ============================================================
# 4. GUARDAR EL CSV CON EL FORMATO ESPERADO
# ============================================================
df_salida = pd.DataFrame({
    "datetime": df['time'],
    "wind_kmh": viento_kmh
})
df_salida.to_csv(ARCHIVO_SALIDA, index=False)

print(f"✅ {ARCHIVO_SALIDA} generado con {len(df_salida)} filas")
print(f"   Viento medio: {viento_kmh.mean():.2f} km/h")
print(f"   Viento mínimo: {viento_kmh.min():.2f} km/h")
print(f"   Viento máximo: {viento_kmh.max():.2f} km/h")