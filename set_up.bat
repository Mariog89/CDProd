@echo off

echo Creando entorno virtual...
python -m venv codigoproyecto-venv

echo Activando entorno virtual...
call codigoproyecto-venv\Scripts\activate

echo Actualizando pip...
python -m pip install --upgrade pip

echo Instalando dependencias...
pip install -r requirements.txt

echo Registrando kernel de Jupyter...
python -m ipykernel install --user --name codigoproyecto-venv --display-name "Python CDP"

echo.
echo Configuracion completada correctamente.
pause