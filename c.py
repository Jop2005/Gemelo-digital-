import pickle
with open(r"D:\gemelo-digital-main\backend\notebooks\modelos_train\solar_production_havana_v1.pkl", "rb") as f:
    obj = pickle.load(f)
print(type(obj))
print(hasattr(obj, "predict"))