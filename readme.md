# CDProd — Análisis y modelado de riesgo crediticio

## Objetivo

Analizar la información disponible sobre créditos y clientes con el fin de identificar problemas de calidad de datos, caracterizar el comportamiento de la cartera y encontrar factores asociados con el cumplimiento o incumplimiento oportuno de los pagos, como base para la futura construcción de un modelo predictivo.

## Estado actual

Disponible actualmente:

- `Base_de_datos.csv` en la raíz del repositorio.
- Análisis exploratorio de datos (EDA) en `src/desarrollo/transformacion_eda.ipynb`.
- Configuración de lectura y variable objetivo en `src/config.json`.
- Setup del entorno en Windows mediante `set_up.bat`.
- Dependencias directas reproducibles declaradas en `requirements.txt`.

Todavía NO está implementado (próximos pasos del Entregable 3, no funcionalidades existentes):

- `ft_engineering.py`
- Heuristic / baseline model
- Model training
- Model deployment
- Model evaluation
- Model monitoring

## Estructura del proyecto

```text
CDProd/
├── Base_de_datos.csv
├── requirements.txt
├── set_up.bat
├── readme.md
├── .gitignore
└── src/
    ├── config.json
    └── desarrollo/
        └── transformacion_eda.ipynb
```

El entorno virtual local `codigoproyecto-venv` no forma parte del repositorio: está ignorado por Git y se genera localmente con `set_up.bat`.

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

## Ejecución del notebook

Ruta correcta del análisis actual:

```text
src/desarrollo/transformacion_eda.ipynb
```

Para reproducir el entorno validado del proyecto, seleccione el kernel `Python CDProd`.

Ejemplo:

```bash
jupyter notebook src/desarrollo/transformacion_eda.ipynb
```

## Configuración

`src/config.json` contiene la configuración actual de lectura de datos y variable objetivo:

- `data_file`: nombre del archivo de datos en la raíz del proyecto.
- `separator`: separador de columnas del CSV.
- `encoding`: codificación del archivo CSV.
- `target`: variable objetivo prevista para el modelado futuro.

## Datos

El archivo `Base_de_datos.csv` contiene registros históricos de créditos con información sobre:

- Características del crédito y el cliente.
- Historial crediticio y puntajes.
- Saldos, cuotas y cumplimiento de pagos.

El archivo se encuentra en la raíz del repositorio, que es la ubicación esperada por el notebook a través de `src/config.json`.

## Workflow Git

- `main` representa la rama estable.
- `develop` es la rama de integración.
- Los cambios se realizan mediante ramas `feature/*` y Pull Requests hacia `develop`.

## Próximos pasos

El Entregable 3 continuará con:

- Feature engineering reproducible.
- Modelo heurístico / baseline.
- Entrenamiento y comparación de modelos.
- Deployment.
- Evaluation.
- Monitoring.

Ninguno de estos componentes está implementado todavía.

## Licencia

Uso educativo / interno.
