import joblib

rutas = {
    "RF Consumo": r"D:\gemelo-digital-main\backend\ensamble\artifacts\modelos\consumo\RandomForest_20261028_172328.joblib",
    "XGB Consumo": r"D:\gemelo-digital-main\backend\ensamble\artifacts\modelos\consumo\XGBoost_20261028_172447.joblib",
    "RF Generación": r"D:\gemelo-digital-main\backend\ensamble\artifacts\modelos\generacion\RandomForest_20261028_173518.joblib",
    "XGB Generación": r"D:\gemelo-digital-main\backend\ensamble\artifacts\modelos\generacion\XGBoost_20261028_174133.joblib",
}

for nombre, ruta in rutas.items():
    data = joblib.load(ruta)
    print(f"\n=== {nombre} ===")
    print(data["metadata"]["params"])