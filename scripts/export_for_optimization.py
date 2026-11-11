"""
Script de Integración: Exportar predicciones anuales para el optimizador NSGA-II.
Usa los métodos de predicción por lotes de los motores de ensamble.
"""
import sys
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
# Importaciones
from app.services.ensemble_consumption_engine import EnsembleConsumptionEngine
from app.services.ensemble_production_engine import EnsembleProductionEngine
from app.services.solar_features import build_features

# Configuración
YEAR_TO_SIMULATE = 2015
DATASET_PATH = Path(__file__).parent.parent / "ensamble" / "data" / "havana_solar_training.csv"

def cargar_datos_historicos():
    print(f"📂 Cargando dataset histórico desde: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    
    date_col = next((c for c in ['time', 'datetime', 'fecha', 'date'] if c in df.columns), None)
    if not date_col:
        raise ValueError("No se encontró columna de fecha en el dataset.")
    
    df['datetime'] = pd.to_datetime(df[date_col], utc=True)
    df = df.set_index('datetime')
    
    df_year = df[df.index.year == YEAR_TO_SIMULATE].copy()
    
    if len(df_year) < 8760:
        print(f"⚠️  Solo hay {len(df_year)} horas. Rellenando...")
        df_year = df_year.resample('h').mean().interpolate()
        if len(df_year) < 8760:
            df_year = df_year.reindex(pd.date_range(start=df_year.index[0], periods=8760, freq='h'), method='nearest')
    
    df_year = df_year.iloc[:8760]
    print(f"✅ Se encontraron {len(df_year)} horas para el año {YEAR_TO_SIMULATE}.")
    return df_year

async def generar_archivos():
    print("⏳ Iniciando generación de series anuales...")
    df_year = cargar_datos_historicos()
    datetimes = [dt.isoformat() for dt in df_year.index]
    
    # --- 1. GENERACIÓN SOLAR ---
    print("☀️  Prediciendo generación solar...")
    features_df = build_features(df_year)
    
    engine_prod = EnsembleProductionEngine()
    produccion = engine_prod.predict_batch(features_df)  # <-- Llamada limpia
    
    df_solar = pd.DataFrame({
        "datetime": datetimes,
        "solar_cf": np.clip(produccion, 0, 1)
    })
    df_solar.to_csv("solar_8760.csv", index=False)
    print(f"✅ solar_8760.csv generado con {len(df_solar)} filas.")
    
    # --- 2. CONSUMO ELÉCTRICO ---
    print("🔌 Prediciendo consumo eléctrico...")
    engine_cons = EnsembleConsumptionEngine()
    predicciones_consumo = engine_cons.predict_batch(datetimes)  # <-- Llamada limpia
    
    df_load = pd.DataFrame({
        "datetime": datetimes,
        "load_kW": [round(float(kw), 4) for kw in predicciones_consumo]
    })
    df_load.to_csv("load_8760.csv", index=False)
    print(f"✅ load_8760.csv generado con {len(df_load)} filas.")
    
    print("\n🎉 ¡Archivos listos!")

if __name__ == "__main__":
    asyncio.run(generar_archivos())
