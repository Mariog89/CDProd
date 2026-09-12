@echo off
setlocal
cd /d "%~dp0"

set "VENV_DIR=codigoproyecto-venv"
set "REQUIREMENTS=requirements.txt"

echo Verificando Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no esta disponible o no puede ejecutarse.
    echo Instala Python 3, agregalo al PATH y vuelve a intentarlo.
    exit /b 1
)

if not exist "%REQUIREMENTS%" (
    echo [ERROR] No se encontro requirements.txt en la raiz del proyecto.
    exit /b 1
)

if exist "%VENV_DIR%\Scripts\python.exe" (
    echo El entorno virtual "%VENV_DIR%" ya existe. Se reutilizara.
) else (
    if exist "%VENV_DIR%" (
        echo [ERROR] Existe la carpeta "%VENV_DIR%" pero no contiene un entorno virtual valido ^(falta Scripts\python.exe^).
        exit /b 1
    )
    echo Creando entorno virtual "%VENV_DIR%"...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Fallo la creacion del entorno virtual.
        exit /b 1
    )
)

echo Activando entorno virtual...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Fallo la activacion del entorno virtual.
    exit /b 1
)

echo Actualizando pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Fallo la actualizacion de pip.
    exit /b 1
)

echo Instalando dependencias desde "%REQUIREMENTS%"...
python -m pip install -r "%REQUIREMENTS%"
if errorlevel 1 (
    echo [ERROR] Fallo la instalacion de dependencias.
    exit /b 1
)

echo Registrando kernel de Jupyter...
python -m ipykernel install --user --name codigoproyecto-venv --display-name "Python CDProd"
if errorlevel 1 (
    echo [ERROR] Fallo el registro del kernel de Jupyter.
    exit /b 1
)

echo.
echo Configuracion completada correctamente.
endlocal
exit /b 0
