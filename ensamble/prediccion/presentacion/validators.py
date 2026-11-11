import numpy as np
import os


def _validar_valor_numero_entrada(valor, min_valor, max_valor):
    if min_valor is not None and valor < min_valor:
        print(f"   ⚠️ El valor debe ser mayor o igual a {min_valor}")
        return False
    if max_valor is not None and valor > max_valor:
        print(f"   ⚠️ El valor debe ser menor o igual a {max_valor}")
        return False
    return True


def leer_numero(variable: str, obligatorio: bool = False, min_valor: float = None, max_valor: float = None):
    while True:
        entrada = input(f"{variable}: ").strip()
        if entrada == "":
            if obligatorio:
                print("⚠️ Este campo es obligatorio")
                continue
            return None
        try:
            if '.' in entrada:
                valor = float(entrada)
            else:
                valor = int(entrada)
            if _validar_valor_numero_entrada(valor, min_valor, max_valor):
                return valor
        except ValueError:
            print("   ❌ Debe ser un número")
            continue


def validar_archivo(filepath: str) -> bool:
    if not os.path.exists(filepath):
        print(f"❌ Archivo no encontrado: {filepath}")
        return False
    if not filepath.endswith(('.csv', '.xlsx', '.xls')):
        print("❌ Formato no soportado. Use .csv o .xlsx")
        return False
    return True


def validar_columnas(df, columnas_requeridas: list) -> bool:
    faltantes = [c for c in columnas_requeridas if c not in df.columns]
    if faltantes:
        print(f"   ❌ Columnas faltantes: {faltantes}")
        return False
    return True


def validar_rango(valor, min_valor=None, max_valor=None, nombre="Valor"):
    if min_valor is not None and valor < min_valor:
        print(f"   ❌ {nombre} debe ser >= {min_valor}")
        return False
    if max_valor is not None and valor > max_valor:
        print(f"   ❌ {nombre} debe ser <= {max_valor}")
        return False
    return True


def normalizar_ruta(ruta: str) -> str:
    ruta = ruta.strip()
    ruta = ruta.strip('"').strip("'")
    return ruta