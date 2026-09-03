# Análisis Exploratorio de Créditos

## Objetivo

Analizar la información disponible sobre créditos y clientes con el fin de identificar problemas de calidad de datos, caracterizar el comportamiento de la cartera y encontrar factores asociados con el cumplimiento o incumplimiento oportuno de los pagos.

## Características

- Análisis exploratorio de datos (EDA) sobre datos de créditos.
- Limpieza y transformación de variables provenientes de diferentes fuentes.
- Visualización de distribuciones, correlaciones y tendencias de la cartera.
- Identificación de factores asociados al pago a tiempo vs. mora.

## Estructura del proyecto

```text
Ejercicio2CDP/
├── Base_de_datos.csv
├── requirements.txt
├── README.md
└── src/
    ├── config.json
    └── desarrollo/
        └── transformacion_eda.ipynb
```

## Requisitos

- Python >= 3.8
- pip

Instala las dependencias:

```bash
pip install -r requirements.txt
```

## Uso

1. Asegúrate de tener configurado el entorno virtual (opcional pero recomendado).
2. Instala las dependencias listadas en `requirements.txt`.
3. Ejecuta el notebook de Jupyter ubicado en `src/desarrollo/transformacion_eda.ipynb` para realizar el análisis exploratorio y transformaciones:

```bash
jupyter notebook src/desarrollo/transformacion_eda.ipynb
```

También puedes usar JupyterLab si lo prefieres:

```bash
jupyter lab src/desarrollo/transformacion_eda.ipynb
```

## Datos

El archivo `Base_de_datos.csv` contiene registros históricos de créditos con información sobre:
- Características del crédito y el cliente.
- Historial crediticio y puntajes.
- Saldos, cuotas y cumplimiento de pagos.

Asegúrate de que el archivo de datos se encuentre en la ruta esperada por el notebook antes de ejecutarlo.

## Configuración

Revisa y ajusta `src/config.json` según las rutas y parámetros específicos de tu entorno.

## Licencia

Uso educativo / interno.
