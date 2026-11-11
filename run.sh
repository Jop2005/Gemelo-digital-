#!/bin/bash

# Script para iniciar el servidor FastAPI GraphQL
# Ejecutar desde el directorio backend: ./run.sh

cd "$(dirname "$0")"

# Activar entorno virtual
source venv/bin/activate

# Iniciar servidor con hot-reload
echo "🚀 Iniciando servidor GraphQL en http://localhost:8000"
echo "📊 GraphQL Playground: http://localhost:8000/graphql"
echo ""
uvicorn app.main:app --host 0.0.0.0 --port 8000
