# CDProd — Análisis y modelado de riesgo crediticio

## Objetivo

Analizar la información disponible sobre créditos y clientes con el fin de identificar problemas de calidad de datos, caracterizar el comportamiento de la cartera y encontrar factores asociados con el cumplimiento o incumplimiento oportuno de los pagos, como base para la futura construcción de un modelo predictivo.

## Estado actual

Disponible actualmente:

- `Base_de_datos.csv` en la raíz del repositorio.
- Pipeline de Stage 3 en `mlops_pipeline/src/`.
- `Cargar_datos.py`: carga reproducible de configuración y datos crudos.
- `ft_engineering.py`: feature engineering reproducible sin leakage (16 features v1, split 80/20 estratificado).
- Configuración canónica de Stage 3 en `mlops_pipeline/src/config.json`.
- Setup del entorno en Windows mediante `set_up.bat`.
- Dependencias directas reproducibles declaradas en `requirements.txt`.
- `comprension_eda.ipynb`: notebook de comprensión/EDA existente, migrado desde el análisis exploratorio previo, ubicado en `mlops_pipeline/src/comprension_eda.ipynb`.
- Calidad/CI con pytest, Ruff, GitHub Actions y SonarQube Cloud.

Estructura preparada, pendiente de implementación:
- `hueristic_model.py`
- `model_training.py`
- `model_deploy.py`
- `model_evaluation.py`
- `model_monitoring.py`

## Estructura del proyecto

```text
CDProd/
├── mlops_pipeline/
│   └── src/
│       ├── Cargar_datos.py
│       ├── comprension_eda.ipynb
│       ├── ft_engineering.py
│       ├── hueristic_model.py
│       ├── model_training.py
│       ├── model_deploy.py
│       ├── model_evaluation.py
│       ├── model_monitoring.py
│       └── config.json
├── tests/
├── .github/
├── Base_de_datos.csv
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── sonar-project.properties
├── .gitignore
├── readme.md
└── set_up.bat
```

El entorno virtual local `codigoproyecto-venv` no forma parte del repositorio: está ignorado por Git y se genera localmente con `set_up.bat`.

El análisis exploratorio histórico que originó el diseño actual se conserva únicamente en el historial Git; ya no forma parte de la estructura activa.

## Requisitos del entorno

- El proyecto ha sido validado con Python 3.13.
- Windows para el script automatizado `set_up.bat`.
- Las dependencias directas están declaradas con versiones exactas en `requirements.txt`.

## Setup en Windows

Flujo recomendado desde un clon limpio:

```bat
git clone https://github.com/Mariog89/CDProd.git
cd CDProd
.\set_up.bat
```

`set_up.bat`:

- Crea `codigoproyecto-venv` si no existe y lo reutiliza si ya existe.
- Instala las dependencias desde `requirements.txt`.
- Registra el kernel de Jupyter como "Python CDProd".

## Ejecución de pruebas y calidad

```bat
.\codigoproyecto-venv\Scripts\python.exe -m pytest -q
.\codigoproyecto-venv\Scripts\python.exe -m ruff check .
.\codigoproyecto-venv\Scripts\python.exe -m ruff format --check .
.\codigoproyecto-venv\Scripts\python.exe -m pip check
```

Para reproducir el entorno validado del proyecto, seleccione el kernel `Python CDProd`.

## Configuración

`mlops_pipeline/src/config.json` contiene la configuración actual de lectura de datos y variable objetivo:

- `data_file`: nombre del archivo de datos en la raíz del proyecto.
- `separator`: separador de columnas del CSV.
- `encoding`: codificación del archivo CSV.
- `target`: variable objetivo prevista para el modelado futuro.

## Datos

El archivo `Base_de_datos.csv` contiene registros históricos de créditos con información sobre:

- Características del crédito y el cliente.
- Historial crediticio y puntajes.
- Saldos, cuotas y cumplimiento de pagos.

El archivo se encuentra en la raíz del repositorio, que es la ubicación esperada por `Cargar_datos.py` a través de `mlops_pipeline/src/config.json`.

## Workflow Git

- `main` representa la rama estable.
- `develop` es la rama de integración.
- Los cambios se realizan mediante ramas `feature/*` y Pull Requests hacia `develop`.

## Próximos pasos

El Entregable 3 continuará con:

- Modelo heurístico / baseline.
- Entrenamiento y comparación de modelos.
- Deployment.
- Evaluation.
- Monitoring.

## Licencia

Uso educativo / interno.
