"""
Configura la microrred del gemelo digital en MongoDB.
Ejecutar desde la carpeta backend con el venv activado:
    python configurar_microrred.py
"""
import sys
sys.path.insert(0, r"D:\gemelo-digital-main\backend")

from app.database import get_database

# ============================================================
# CONFIGURACIÓN: EDITA ESTOS VALORES CON LOS REALES DEL CAMPUS
# ============================================================

PANELES = [
    {
        "manufacturer": "Trina Solar",
        "model": "TSM-445DE09",
        "ratedPowerKw": 0.445,      # Potencia de cada panel en kW
        "quantity": 100,             # Cantidad de paneles
        "tiltDegrees": 20.0,         # Inclinación
        "orientation": "sur",        # Orientación
        "efficiencyPercent": 21.8,
        "areaM2": 2.0,
    },
]

BATERIAS = [
    {
        "manufacturer": "Pylontech",
        "model": "US5000",
        "capacityKwh": 4.8,          # Capacidad de cada batería en kWh
        "quantity": 20,              # Cantidad de baterías
        "maxDepthOfDischargePercent": 80.0,
        "chargeRateKw": 2.4,
        "dischargeRateKw": 2.4,
        "efficiencyPercent": 95.0,
    },
]

INVERSORES = [
    {
        "manufacturer": "Huawei",
        "model": "SUN2000-50KTL",
        "ratedPowerKw": 50.0,
        "quantity": 2,
        "efficiencyPercent": 98.0,
    },
]

# ============================================================
# FUNCIÓN PARA GUARDAR
# ============================================================
def configurar():
    db = get_database()
    print("🔌 Conectado a MongoDB")

    # Limpiar colecciones existentes (opcional, para no duplicar)
    db["paneles"].delete_many({})
    db["baterias"].delete_many({})
    db["inversores"].delete_many({})
    print("🧹 Colecciones limpiadas")

    # Insertar paneles
    if PANELES:
        db["paneles"].insert_many(PANELES)
        print(f"✅ {len(PANELES)} tipo(s) de panel insertado(s)")

    # Insertar baterías
    if BATERIAS:
        db["baterias"].insert_many(BATERIAS)
        print(f"✅ {len(BATERIAS)} tipo(s) de batería insertado(s)")

    # Insertar inversores
    if INVERSORES:
        db["inversores"].insert_many(INVERSORES)
        print(f"✅ {len(INVERSORES)} tipo(s) de inversor insertado(s)")

    # Mostrar resumen
    total_pv = sum(p["ratedPowerKw"] * p["quantity"] for p in PANELES)
    total_bat = sum(b["capacityKwh"] * b["quantity"] for b in BATERIAS)
    total_inv = sum(i["ratedPowerKw"] * i["quantity"] for i in INVERSORES)

    print("\n📊 RESUMEN DE LA CONFIGURACIÓN:")
    print(f"   Capacidad solar: {total_pv:.2f} kW")
    print(f"   Capacidad baterías: {total_bat:.2f} kWh")
    print(f"   Capacidad inversores: {total_inv:.2f} kW")
    print("\n🎉 Configuración guardada en MongoDB")


if __name__ == "__main__":
    configurar()