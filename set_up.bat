@echo off
setlocal enabledelayedexpansion

rem Este script arma el entorno virtual del proyecto leyendo el nombre
rem desde src\config.json (clave "project_code"), asi el nombre del
rem entorno queda consistente con la config sin hardcodearlo dos veces.

set "CONFIG_FILE=src\config.json"

if not exist "%CONFIG_FILE%" (
    echo No se encontro %CONFIG_FILE%
    exit /b 1
)

for /f "tokens=1* delims=:" %%A in ('findstr /C:"\"project_code\":" "%CONFIG_FILE%"') do (
    set "PROJECT_CODE=%%B"
)

rem Limpieza: sacar comillas, comas y espacios sobrantes del valor
set "PROJECT_CODE=%PROJECT_CODE:"=%"
set "PROJECT_CODE=%PROJECT_CODE:,=%"
set "PROJECT_CODE=%PROJECT_CODE: =%"

if "%PROJECT_CODE%"=="" (
    echo No se pudo leer project_code desde %CONFIG_FILE%
    exit /b 1
)

echo Codigo de proyecto: %PROJECT_CODE%

set "VENV_DIR=%PROJECT_CODE%-venv"

if not exist "%VENV_DIR%" (
    echo Creando entorno virtual en %VENV_DIR% ...
    python -m venv "%VENV_DIR%"
)

call "%VENV_DIR%\Scripts\activate.bat"

echo Instalando dependencias de requirements.txt ...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Entorno listo. Para activarlo mas adelante correr:
echo call %VENV_DIR%\Scripts\activate.bat

endlocal
