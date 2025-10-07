#!/usr/bin/env bash
# Script para iniciar el servidor FastAPI en producción

# Activar entorno virtual si existe
if [ -f "./venv/Scripts/activate" ]; then
    source ./venv/Scripts/activate
fi


# Ejecutar el servidor con Uvicorn en primer plano
echo "Iniciando servidor Uvicorn. Para detener, presiona Ctrl+C."
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
