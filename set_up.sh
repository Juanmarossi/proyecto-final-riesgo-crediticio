#!/usr/bin/env bash
set -euo pipefail

# ===================================
# Purpose: Script to setup a Python virtual environment, install requirements,
# Equivalente para macOS/Linux de set_up.bat (mismo flujo: leer project_code
# desde src/config.json, crear el venv, instalar requirements.txt y
# registrar el kernel de Jupyter).
# ===================================

echo
echo "=== Python Virtual Environment Setup ==="
echo

# Desactivar el ambiente virtual actual si esta activo
if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "Desactivando ambiente virtual actual: $VIRTUAL_ENV"
    deactivate 2>/dev/null || true
fi

echo "Buscando codigo del proyecto en config.json..."

if [ ! -f "src/config.json" ]; then
    echo 'Error: No se encontro "config.json" en el directorio "src". Asegurate de que la ruta es correcta.'
    exit 1
fi

cd src

PROJECT_CODE=$(grep '"project_code"' config.json | cut -d':' -f2- | tr -d ' ",')

cd ..

echo "Project code encontrado: [$PROJECT_CODE]"

if [ -z "$PROJECT_CODE" ]; then
    echo "Error: no se pudo leer project_code desde src/config.json"
    exit 1
fi

echo "Creando nuevo ambiente virtual: ${PROJECT_CODE}-venv"
python3 -m venv "${PROJECT_CODE}-venv"

echo "Activating virtual environment..."
# shellcheck disable=SC1091
source "${PROJECT_CODE}-venv/bin/activate"

echo
echo "Ambiente virtual creado con exito!."
echo "Python actual:"
which python

echo
echo "Directorio actual: $(pwd)"
echo "================================"
ls -1
echo "=== Instalando requisitos ==="
if [ -f "requirements.txt" ]; then
    echo "requirements.txt encontrado, instalando librerias..."
    "${PROJECT_CODE}-venv/bin/python" -m pip install --no-cache-dir -r requirements.txt

    echo
    echo "Todas las librerias instaladas correctamente."

    echo
    echo "=== Registrando ambiente virtual con Jupyter ==="
    echo "Registrando kernel con Jupyter..."
    python3 -m ipykernel install --user --name="${PROJECT_CODE}-venv" --display-name="${PROJECT_CODE}-venv Python ETL"

    echo "Ambiente virtual registrado como kernel de Jupyter correctamente."
    echo "Ahora podes seleccionar \"${PROJECT_CODE}-venv Python ETL\" en Jupyter notebook."
else
    echo
    echo "Advertencia: requirements.txt no fue encontrado en el directorio actual."
fi

echo
