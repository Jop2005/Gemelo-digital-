import os
import json

# ============================================================
# CONFIGURACIÓN
# ============================================================

RUTA_CARPETA = r"D:\gemelo-digital-main\backend\notebooks"

EXTENSIONES_EXCLUIDAS = [".tmp", ".log", ".bak", ".pyc"]

CARPETAS_EXCLUIDAS = ["pycache", ".git", "node_modules", "venv", "models","tests"]

NOMBRE_SALIDA = "notebooks.txt"

# ------------------------------------------------------------
# ÚNICA LISTA A EDITAR: extensiones cuyo contenido se extraerá
# ------------------------------------------------------------
EXTENSIONES_A_EXTRAER = [
    ".py",
    ".ipynb",
]


# ============================================================
# EXTRACTORES DE CONTENIDO
# ============================================================

def _leer_texto(ruta):
    """Lee un archivo de texto con fallback de codificación."""
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(ruta, "r", encoding="latin-1") as f:
            return f.read()


def extraer_contenido_py(ruta):
    """Lee un archivo .py y devuelve su contenido como texto."""
    return _leer_texto(ruta)


def extraer_contenido_ipynb(ruta):
    """
    Extrae el contenido de un notebook .ipynb.
    Devuelve solo el código de las celdas, separadas por marcadores.
    """
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        return f"[ERROR al leer el notebook: {e}]"

    partes = []
    celdas = nb.get("cells", [])

    for i, celda in enumerate(celdas, start=1):
        tipo = celda.get("cell_type", "desconocido")
        fuente = celda.get("source", "")
        if isinstance(fuente, list):
            codigo = "".join(fuente)
        else:
            codigo = fuente

        partes.append(f"# --- Celda {i} ({tipo}) ---")
        partes.append(codigo.rstrip())
        partes.append("")

    return "\n".join(partes)


def extraer_contenido_json(ruta):
    """
    Lee un .json. Si es válido, lo re-indenta para mejor lectura.
    Si no, devuelve el contenido crudo con un aviso.
    """
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            texto = f.read()
    except UnicodeDecodeError:
        with open(ruta, "r", encoding="latin-1") as f:
            texto = f.read()

    try:
        data = json.loads(texto)
        return json.dumps(data, indent=2, ensure_ascii=False)
    except json.JSONDecodeError as e:
        return f"[JSON inválido: {e}]\n\n--- Contenido crudo ---\n{texto}"


def extraer_contenido_texto(ruta):
    """Extractor genérico para archivos de texto plano."""
    return _leer_texto(ruta)


# Mapa extensión -> función extractora
# (las que no estén aquí usan el extractor genérico de texto)
_EXTRACTORES = {
    ".py": extraer_contenido_py,
    ".ipynb": extraer_contenido_ipynb,
    ".json": extraer_contenido_json,
}


def extraer_contenido(ruta):
    """Elige el extractor adecuado según la extensión."""
    _, ext = os.path.splitext(ruta)
    ext = ext.lower()
    extractor = _EXTRACTORES.get(ext, extraer_contenido_texto)
    try:
        return extractor(ruta)
    except Exception as e:
        return f"[ERROR al extraer contenido: {e}]"


# ============================================================
# LISTADO DE ARCHIVOS
# ============================================================

def listar_archivos(ruta_base, extensiones_excluidas, carpetas_excluidas):
    ext_excluidas = {e.lower() for e in extensiones_excluidas}
    carpetas_excluidas_set = set(carpetas_excluidas)

    resultados = []
    for raiz, dirs, archivos in os.walk(ruta_base):
        dirs[:] = [d for d in dirs if d not in carpetas_excluidas_set]

        for nombre in archivos:
            _, ext = os.path.splitext(nombre)
            if ext.lower() in ext_excluidas:
                continue
            resultados.append((nombre, os.path.join(raiz, nombre)))

    return resultados


# ============================================================
# MAIN
# ============================================================

def main():
    print(">>> Inicio del script")

    if not os.path.isdir(RUTA_CARPETA):
        print(f">>> La carpeta no existe: {RUTA_CARPETA}")
        return

    archivos = listar_archivos(RUTA_CARPETA, EXTENSIONES_EXCLUIDAS, CARPETAS_EXCLUIDAS)

    ruta_salida = os.path.join(RUTA_CARPETA, NOMBRE_SALIDA)
    ext_a_extraer = {e.lower() for e in EXTENSIONES_A_EXTRAER}

    with open(ruta_salida, "w", encoding="utf-8") as f:
        print(f"Carpeta analizada: {RUTA_CARPETA}", file=f)
        print(f"Total de archivos: {len(archivos)}", file=f)
        print(f"Extensiones con contenido extraído: {sorted(ext_a_extraer)}", file=f)
        print("=" * 80, file=f)
        print("", file=f)

        for i, (nombre, ruta) in enumerate(archivos, start=1):
            print(f"{i:>4}. Nombre: {nombre}", file=f)
            print(f"      Ruta:   {ruta}", file=f)

            _, ext = os.path.splitext(nombre)
            if ext.lower() in ext_a_extraer:
                print("-" * 80, file=f)
                print(f"CONTENIDO DE {nombre}:", file=f)
                print("-" * 80, file=f)

                contenido = extraer_contenido(ruta)
                print(contenido if contenido is not None else "[Sin contenido]", file=f)

                print("-" * 80, file=f)
            print("", file=f)

    print(f">>> Archivos encontrados: {len(archivos)}")
    print(f">>> Guardado en: {ruta_salida}")


# ============================================================
# EJECUCIÓN
# ============================================================

main()