"""
Servicio de descubrimiento y ejecución de pruebas para la pantalla /dev/test.

Expone dos capacidades, ambas reservadas a administradores (el guard vive en la
capa REST, ver app/main.py):

1. `build_catalog()` — descubre las pruebas existentes (backend con pytest +
   frontend con vitest) y las agrupa por categoría. Para el backend se parsea el
   AST de cada archivo `tests/test_*.py` para extraer las clases `Test*`, sus
   métodos `test_*` y los docstrings (así cada prueba "dice exactamente qué
   prueba"). Para el frontend se parsean los bloques `describe(...)` / `it(...)`.

2. `run_backend(...)` / `run_frontend(...)` — ejecutan un grupo o una prueba
   puntual como subproceso y devuelven el resultado (passed/failed + salida).

El descubrimiento es dinámico: los node ids que se usan para correr provienen del
propio AST/regex, de modo que siempre coinciden con las pruebas reales del repo.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Rutas del repositorio
# ---------------------------------------------------------------------------
# Este archivo: backend/app/services/test_runner_service.py
_BACKEND_DIR = Path(__file__).resolve().parents[2]          # backend/
_REPO_ROOT = _BACKEND_DIR.parent                            # repo raíz
_BACKEND_TESTS_DIR = _BACKEND_DIR / "tests"
_FRONTEND_DIR = _REPO_ROOT / "frontend"

# Límite de salida devuelta al cliente (evita payloads enormes).
_MAX_OUTPUT_CHARS = 20000
# Timeout por ejecución (segundos).
_RUN_TIMEOUT = 300


# ---------------------------------------------------------------------------
# Categorías (etiquetas amigables en español)
# ---------------------------------------------------------------------------
# Cada archivo de pruebas se asigna a una categoría. Si un archivo no está
# mapeado, se usa una etiqueta derivada del nombre como respaldo.
_BACKEND_CATEGORIES: Dict[str, str] = {
    "test_auth_guards": "Autenticación y usuarios",
    "test_session_service": "Autenticación y usuarios",
    "test_invitation_service": "Autenticación y usuarios",
    "test_user_service": "Autenticación y usuarios",
    "test_user_service_integration": "Autenticación y usuarios",
    "test_ldap_service": "Autenticación y usuarios",
    "test_crud_services": "CRUD de dispositivos",
    "test_inverter_service": "CRUD de dispositivos",
    "test_battery_discharge": "Baterías y apagones",
    "test_blackout_service": "Baterías y apagones",
    "test_weather_http": "Clima",
    "test_weather_source_service": "Clima",
    "test_ml_production": "Predicción y aprendizaje automático",
    "test_panel_classifier": "Predicción y aprendizaje automático",
    "test_prediction_service": "Predicción y aprendizaje automático",
    "test_solar_query": "Datos solares e históricos",
    "test_analytics": "Datos solares e históricos",
    "test_lectura_service": "Datos solares e históricos",
    "test_location_config_service": "Configuración del sistema",
    "test_shadow_profile_service": "Configuración del sistema",
    "test_graphql_schema": "API GraphQL",
    "test_appliance_measurement_service": "Mediciones de electrodomésticos",
}

_FRONTEND_CATEGORIES: Dict[str, str] = {
    "AuthGate": "Componentes de interfaz",
    "ConfirmDialog": "Componentes de interfaz",
    "FloatingBottomNav": "Componentes de interfaz",
    "BatteryStatus": "Componentes de visualización",
    "WeatherForecast": "Componentes de visualización",
    "permissions": "Permisos y lógica",
    "calculations": "Cálculos y predicciones",
    "predictions": "Cálculos y predicciones",
    "shadowCalc": "Cálculos y predicciones",
}


def _prettify(stem: str) -> str:
    """Convierte 'test_weather_source_service' en 'Weather source service'."""
    name = stem.replace("test_", "", 1).replace("_", " ").strip()
    return name[:1].upper() + name[1:] if name else stem


# ---------------------------------------------------------------------------
# Descubrimiento backend (AST de pytest)
# ---------------------------------------------------------------------------
def _clean_doc(doc: Optional[str]) -> str:
    if not doc:
        return ""
    # Primera línea/párrafo no vacío, compactado.
    lines = [ln.strip() for ln in doc.strip().splitlines()]
    text = " ".join(ln for ln in lines if ln)
    return text.strip()


def discover_backend_tests() -> List[Dict[str, Any]]:
    """Devuelve las categorías de pruebas del backend.

    Estructura: [{ key, label, suite, groups: [{ id, name, file, description,
    tests: [{ id, name, description }] }] }]
    """
    categories: Dict[str, Dict[str, Any]] = {}

    for path in sorted(_BACKEND_TESTS_DIR.glob("test_*.py")):
        rel_file = f"tests/{path.name}"
        stem = path.stem
        label = _BACKEND_CATEGORIES.get(stem, _prettify(stem))
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue

        groups: List[Dict[str, Any]] = []
        standalone: List[Dict[str, Any]] = []

        for node in tree.body:
            # Clases Test* → grupos
            if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                tests = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test"):
                        tests.append({
                            "id": f"{rel_file}::{node.name}::{item.name}",
                            "name": item.name,
                            "description": _clean_doc(ast.get_docstring(item)),
                        })
                groups.append({
                    "id": f"{rel_file}::{node.name}",
                    "name": node.name,
                    "file": rel_file,
                    "description": _clean_doc(ast.get_docstring(node)),
                    "tests": tests,
                })
            # Funciones test_* a nivel de módulo → grupo sintético por archivo
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                standalone.append({
                    "id": f"{rel_file}::{node.name}",
                    "name": node.name,
                    "description": _clean_doc(ast.get_docstring(node)),
                })

        if standalone:
            groups.append({
                "id": rel_file,
                "name": _prettify(stem),
                "file": rel_file,
                "description": _clean_doc(ast.get_docstring(tree)),
                "tests": standalone,
            })

        if not groups:
            continue

        cat = categories.setdefault(label, {
            "key": label,
            "label": label,
            "suite": "backend",
            "groups": [],
        })
        cat["groups"].extend(groups)

    return list(categories.values())


# ---------------------------------------------------------------------------
# Descubrimiento frontend (regex de vitest)
# ---------------------------------------------------------------------------
_DESCRIBE_RE = re.compile(r"""\bdescribe\s*\(\s*(['"`])(.+?)\1""")
_IT_RE = re.compile(r"""\b(?:it|test)\s*(?:\.\w+)?\s*\(\s*(['"`])(.+?)\1""")


def discover_frontend_tests() -> List[Dict[str, Any]]:
    """Descubre pruebas del frontend parseando bloques describe/it.

    El `id` de cada grupo/prueba codifica el archivo y, opcionalmente, el nombre
    del test: `<ruta-relativa>` o `<ruta-relativa>::<nombre del test>`.
    """
    categories: Dict[str, Dict[str, Any]] = {}
    src_dir = _FRONTEND_DIR / "src"
    if not src_dir.exists():
        return []

    files = sorted(
        list(src_dir.rglob("*.test.ts")) + list(src_dir.rglob("*.test.tsx"))
    )

    for path in files:
        rel_file = str(path.relative_to(_FRONTEND_DIR))
        base = path.name.replace(".test.tsx", "").replace(".test.ts", "")
        label = _FRONTEND_CATEGORIES.get(base, "Otras pruebas del frontend")
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        describe_titles = [m.group(2) for m in _DESCRIBE_RE.finditer(content)]
        it_titles = [m.group(2) for m in _IT_RE.finditer(content)]
        group_name = describe_titles[0] if describe_titles else base

        tests = [{
            "id": f"{rel_file}::{title}",
            "name": title,
            "description": "",
        } for title in it_titles]

        group = {
            "id": rel_file,
            "name": group_name,
            "file": rel_file,
            "description": f"Suite de pruebas de {base} ({len(tests)} casos).",
            "tests": tests,
        }

        cat = categories.setdefault(label, {
            "key": label,
            "label": label,
            "suite": "frontend",
            "groups": [],
        })
        cat["groups"].append(group)

    return list(categories.values())


def build_catalog() -> Dict[str, Any]:
    """Catálogo completo de pruebas para la pantalla /dev/test."""
    return {
        "backend": discover_backend_tests(),
        "frontend": discover_frontend_tests(),
    }


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------
def _truncate(text: str) -> str:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    return "…(salida truncada)…\n" + text[-_MAX_OUTPUT_CHARS:]


def _parse_pytest_summary(output: str) -> Dict[str, int]:
    """Extrae conteos passed/failed/etc. de la línea resumen de pytest."""
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    for key in counts:
        m = re.search(rf"(\d+)\s+{key}", output)
        if m:
            counts[key] = int(m.group(1))
    return counts


def run_backend(node_ids: List[str]) -> Dict[str, Any]:
    """Ejecuta uno o varios node ids de pytest dentro de backend/."""
    safe_ids = [nid for nid in node_ids if _is_safe_backend_id(nid)]
    if not safe_ids:
        return {"ok": False, "output": "No se indicaron pruebas válidas para ejecutar.", "summary": {}}

    cmd = [sys.executable, "-m", "pytest", *safe_ids, "-q", "--tb=short", "--no-header", "--color=no", "-p", "no:cacheprovider"]
    return _run(cmd, cwd=_BACKEND_DIR, summary_parser=_parse_pytest_summary)


def run_frontend(ids: List[str]) -> Dict[str, Any]:
    """Ejecuta pruebas de vitest. Cada id es `<archivo>` o `<archivo>::<nombre>`."""
    files: List[str] = []
    name_patterns: List[str] = []
    for raw in ids:
        if "::" in raw:
            file_part, name = raw.split("::", 1)
            name_patterns.append(name)
        else:
            file_part = raw
        if _is_safe_frontend_file(file_part) and file_part not in files:
            files.append(file_part)

    if not files:
        return {"ok": False, "output": "No se indicaron pruebas válidas para ejecutar.", "summary": {}}

    cmd = ["npx", "vitest", "run", *files]
    if name_patterns:
        # vitest -t acepta una sola regex; se unen los nombres con OR.
        pattern = "|".join(re.escape(n) for n in name_patterns)
        cmd += ["-t", pattern]
    return _run(cmd, cwd=_FRONTEND_DIR, summary_parser=_parse_vitest_summary)


def _parse_vitest_summary(output: str) -> Dict[str, int]:
    counts = {"passed": 0, "failed": 0, "skipped": 0}
    m = re.search(r"Tests\s+(?:(\d+)\s+failed[ ,|]*)?(\d+)\s+passed", output)
    if m:
        counts["failed"] = int(m.group(1) or 0)
        counts["passed"] = int(m.group(2) or 0)
    return counts


def _run(cmd: List[str], cwd: Path, summary_parser) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=_RUN_TIMEOUT,
            env={**os.environ, "CI": "true", "FORCE_COLOR": "0"},
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        return {
            "ok": proc.returncode == 0,
            "returnCode": proc.returncode,
            "output": _truncate(output.strip()),
            "summary": summary_parser(output),
            "command": " ".join(cmd[:3]) + " …",
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "output": f"La ejecución superó el límite de {_RUN_TIMEOUT}s.", "summary": {}}
    except FileNotFoundError as exc:
        return {"ok": False, "output": f"No se encontró el ejecutable: {exc}", "summary": {}}
    except Exception as exc:  # pragma: no cover - defensivo
        return {"ok": False, "output": f"Error al ejecutar las pruebas: {exc}", "summary": {}}


# ---------------------------------------------------------------------------
# Validación de entradas (evita inyección de argumentos / rutas)
# ---------------------------------------------------------------------------
_BACKEND_ID_RE = re.compile(r"^tests/test_[A-Za-z0-9_]+\.py(::[A-Za-z0-9_]+){0,2}$")


def _is_safe_backend_id(node_id: str) -> bool:
    return bool(_BACKEND_ID_RE.match(node_id))


def _is_safe_frontend_file(rel_path: str) -> bool:
    if rel_path.startswith("/") or ".." in rel_path:
        return False
    if not (rel_path.endswith(".test.ts") or rel_path.endswith(".test.tsx")):
        return False
    return (_FRONTEND_DIR / rel_path).is_file()
