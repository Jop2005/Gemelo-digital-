"""
================================================================================
  SCRIPT DE PRUEBAS — INTEGRACIÓN DEL ENSAMBLE DINÁMICO
================================================================================
  Prueba la integración paso a paso:
    1. Registro del motor en el gemelo.
    2. Predicción aislada con datos del CSV.
    3. Pipeline completo del gemelo (con internet).
    4. Comparación ML vs Ensemble.

  Uso:
    python test_ensemble_integration.py

  Requisitos:
    - Estar en el directorio backend/.
    - Venv activado.
    - El pipeline de configuración del ensamble debe estar ejecutado
      (artifacts/ con modelos y configs).
================================================================================
"""
import sys
import asyncio
from pathlib import Path

# Asegurar que el backend esté en el path
BACKEND_ROOT = Path(__file__).parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def separador(titulo):
    print("\n" + "=" * 70)
    print(f"  {titulo}")
    print("=" * 70)


def test_1_registro_motor():
    """Paso 1: Verificar que el motor 'ensemble' está registrado."""
    separador("PASO 1 — REGISTRO DEL MOTOR")

    try:
        from app.services.production_forecast_service import (
            _ENGINE_REGISTRY,
            get_production_service,
        )

        motores = list(_ENGINE_REGISTRY.keys())
        print(f"Motores registrados: {motores}")

        if "ensemble" not in motores:
            print("❌ ERROR: El motor 'ensemble' NO está registrado.")
            print("   Revisa que hayas añadido la entrada en _ENGINE_REGISTRY.")
            return False

        print("✅ Motor 'ensemble' registrado")

        service = get_production_service()
        print(f"Motor activo: {service.engine.name}")

        if service.engine.name != "ensemble":
            print("⚠️  ADVERTENCIA: El motor activo NO es 'ensemble'.")
            print("   Revisa que el motor por defecto sea 'ensemble'.")
            return False

        print("✅ Motor activo: 'ensemble'")
        return True

    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_2_prediccion_aislada():
    """Paso 2: Probar el motor con datos del CSV."""
    separador("PASO 2 — PREDICCIÓN AISLADA CON CSV")

    try:
        import pandas as pd
        import numpy as np
        from app.services.production_forecast_service import EnsembleProductionEngine
        from app.services.solar_features import FEATURE_COLUMNS

        # Ruta del CSV del ensamble
        csv_path = BACKEND_ROOT / "ensamble" / "data" / "havana_solar_training.csv"
        if not csv_path.exists():
            print(f"❌ ERROR: No se encontró el CSV: {csv_path}")
            return False

        print(f"Cargando CSV: {csv_path}")
        df = pd.read_csv(csv_path, nrows=100)
        print(f"Filas cargadas: {len(df)}")

        # Verificar columnas
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            print(f"❌ ERROR: Faltan columnas en el CSV: {missing}")
            return False
        print("✅ Todas las features están presentes")

        X = df[FEATURE_COLUMNS].copy()
        y_real = df["capacity_factor"].values if "capacity_factor" in df.columns else None

        # Crear motor e inicializar
        print("\nInicializando motor...")
        engine = EnsembleProductionEngine()

        print("Prediciendo...")
        predicciones = engine.capacity_factor(X, lat=23.11, lon=-82.37)

        print(f"\nPrimeras 5 predicciones: {predicciones[:5]}")
        if y_real is not None:
            print(f"Primeros 5 valores reales: {y_real[:5]}")
            mae = np.mean(np.abs(predicciones - y_real))
            print(f"MAE: {mae:.4f}")

        if np.any(predicciones < 0) or np.any(predicciones > 1):
            print("⚠️  ADVERTENCIA: Hay predicciones fuera de [0, 1].")
            return False

        print("✅ Predicciones en rango [0, 1]")
        return True

    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_pipeline_completo():
    """Paso 3: Probar el pipeline completo del gemelo (necesita internet)."""
    separador("PASO 3 — PIPELINE COMPLETO DEL GEMELO")

    try:
        from app.services.production_forecast_service import get_production_service

        async def _run():
            service = get_production_service()
            print(f"Motor activo: {service.engine.name}")

            print("\nPrediciendo próximas 2 horas...")
            resultados = await service.predict([
                "2026-06-18T13:00:00",
                "2026-06-18T14:00:00",
            ])

            for r in resultados:
                print(f"  {r['datetime']}: {r['production_kw']:.4f} "
                      f"(engine: {r['engine']}, fuente: {r.get('weather_source', '?')})")

            return resultados

        resultados = asyncio.run(_run())

        if not resultados:
            print("❌ ERROR: No se obtuvieron resultados.")
            return False

        print("✅ Pipeline completo funciona")
        return True

    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Si el error es de red, verifica tu conexión a internet.")
        return False


def test_4_comparacion_motores():
    """Paso 4: Comparar motor 'ml' vs 'ensemble'."""
    separador("PASO 4 — COMPARACIÓN ML vs ENSEMBLE")

    try:
        from app.services.production_forecast_service import get_production_service
        # ==== CARGAR MODELO ML (añadido para el test) ====
        from app.services.ml_model_service import ml_model_service
        if not ml_model_service.model_loaded:
            print("Cargando modelo 'ml' (Havana v1)...")
            ml_model_service.load_model(model_name="havana_v1")
            print("✅ Modelo 'ml' cargado")
        # ==== FIN ====

        async def _run():
            service = get_production_service()
            fecha = ["2026-06-18T13:00:00"]

            # Probar motor ML
            service.set_engine("ml")
            print("=== Motor ML (Havana v1) ===")
            r_ml = await service.predict(fecha)
            prod_ml = r_ml[0]["production_kw"]
            print(f"  Producción: {prod_ml:.4f} kW")

            # Probar motor Ensemble
            service.set_engine("ensemble")
            print("\n=== Motor Ensemble ===")
            r_ens = await service.predict(fecha)
            prod_ens = r_ens[0]["production_kw"]
            print(f"  Producción: {prod_ens:.4f} kW")

            # Comparar
            print("\n=== Comparación ===")
            print(f"  ML:       {prod_ml:.4f} kW")
            print(f"  Ensemble: {prod_ens:.4f} kW")
            diff = abs(prod_ml - prod_ens)
            print(f"  Diferencia: {diff:.4f} kW")

            if prod_ml > 0:
                pct = (diff / prod_ml) * 100
                print(f"  Diferencia relativa: {pct:.1f}%")

                if pct < 20:
                    print("  ✅ Los motores dan resultados comparables")
                else:
                    print("  ⚠️  Diferencia grande. Verifica las unidades.")
                    return False

            return True

        return asyncio.run(_run())

    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("  PRUEBAS DE INTEGRACIÓN — ENSAMBLE DINÁMICO")
    print("=" * 70)

    resultados = []

    # Paso 1
    ok = test_1_registro_motor()
    resultados.append(("Paso 1: Registro del motor", ok))

    if not ok:
        print("\n❌ El Paso 1 falló. Deteniendo pruebas.")
        _resumen(resultados)
        return

    # Paso 2
    ok = test_2_prediccion_aislada()
    resultados.append(("Paso 2: Predicción aislada", ok))

    # Paso 3 (necesita internet)
    if ok:
        ok = test_3_pipeline_completo()
        resultados.append(("Paso 3: Pipeline completo", ok))

    # Paso 4 (necesita internet)
    if ok:
        ok = test_4_comparacion_motores()
        resultados.append(("Paso 4: Comparación motores", ok))

    _resumen(resultados)


def _resumen(resultados):
    separador("RESUMEN")
    for nombre, ok in resultados:
        estado = "✅ OK" if ok else "❌ FALLO"
        print(f"  {estado}: {nombre}")

    total = len(resultados)
    exitos = sum(1 for _, ok in resultados if ok)
    print(f"\n{exitos}/{total} pruebas exitosas")


if __name__ == "__main__":
    main()