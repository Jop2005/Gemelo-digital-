@echo off
REM Script para iniciar el servidor FastAPI GraphQL en Windows
REM Ejecutar desde el directorio backend: run.bat

cd /d "%~dp0"

REM Activar entorno virtual
call venv\Scripts\activate.bat

REM Iniciar servidor con hot-reload
echo 🚀 Iniciando servidor GraphQL en http://localhost:8000
echo 📊 GraphQL Playground: http://localhost:8000/graphql
echo.
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
