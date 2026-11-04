"""Lanza el entrenamiento AutoML Tabular (clasificación) en Vertex AI.

- Dataset tabular desde BigQuery (vertex/preparar_tabla_vertex.py).
- Solo las 15 variables RAW del modelo del repo; fila_id y split no son features.
- Partición predefinida (columna split): TRAIN / VALIDATE / TEST = holdout local.
- Objetivo maximize-au-roc (umbral independiente; la clase de riesgo es minoritaria).
- Presupuesto 1 nodo-hora (1000 milli node hours), el mínimo de AutoML.

El job corre en GCP; este script solo lo crea y guarda su identificador en
vertex/resultados/automl_job.json. El estado se consulta con automl_evaluar.py.

Uso:
    python -m vertex.automl_entrenar
"""

import os

from google.cloud import aiplatform

from vertex import comun

FEATURES_NUMERICAS = (
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal",
    "promedio_ingresos_datacredito",
)
FEATURES_CATEGORICAS = ("tipo_credito", "tipo_laboral", "tendencia_ingresos")
BUDGET_MILLI_NODE_HOURS = 1000


def main() -> None:
    aiplatform.init(project=comun.PROJECT, location=comun.REGION, staging_bucket=comun.BUCKET)
    dataset = aiplatform.TabularDataset.create(
        display_name="cdprod-creditos-vertex",
        bq_source=f"bq://{comun.BQ_TABLA_VERTEX}",
    )
    column_specs = {c: "numeric" for c in FEATURES_NUMERICAS}
    column_specs.update({c: "categorical" for c in FEATURES_CATEGORICAS})
    job = aiplatform.AutoMLTabularTrainingJob(
        display_name="cdprod-automl-riesgo",
        optimization_prediction_type="classification",
        optimization_objective="maximize-au-roc",
        column_specs=column_specs,
    )
    job.run(
        dataset=dataset,
        target_column=comun.TARGET,
        predefined_split_column_name=comun.COLUMNA_SPLIT,
        budget_milli_node_hours=BUDGET_MILLI_NODE_HOURS,
        model_display_name="cdprod-automl-riesgo",
        disable_early_stopping=False,
        sync=False,
    )
    job.wait_for_resource_creation()
    info = {
        "dataset": dataset.resource_name,
        "training_pipeline": job.resource_name,
        "objetivo": "maximize-au-roc",
        "budget_milli_node_hours": BUDGET_MILLI_NODE_HOURS,
        "tabla": comun.BQ_TABLA_VERTEX,
        "consola": (
            "https://console.cloud.google.com/vertex-ai/locations/"
            f"{comun.REGION}/training/{job.resource_name.split('/')[-1]}?project={comun.PROJECT}"
        ),
    }
    comun.guardar_json("automl_job", info)
    print(info)
    # El job sigue en GCP; no esperar a que termine (tarda 1-2 h).
    os._exit(0)


if __name__ == "__main__":
    main()
