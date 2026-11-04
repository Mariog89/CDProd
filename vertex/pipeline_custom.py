"""Pipeline custom en Vertex Pipelines (KFP v2) para el modelo de riesgo crediticio.

Cada ejecución de `entrenar_evaluar` descarga el código del repositorio en un
commit fijo y corre el mismo flujo del Stage 3:

    cargar_datos (BigQuery) -> preparar_datos(random_state=semilla)
    -> entrenar (CV + tuning) -> evaluar (holdout sellado)

Un ParallelFor repite el flujo con varias semillas de partición train/holdout
para medir estabilidad; la semilla 42 reproduce el holdout de Python local y de
AutoML. `consolidar` resume media/desviación de cada métrica.

Uso:
    python -m vertex.pipeline_custom            # compila y envía a Vertex
    python -m vertex.pipeline_custom --compilar # solo compila el YAML
"""

import argparse
from pathlib import Path

from kfp import compiler, dsl
from kfp.dsl import ClassificationMetrics, Metrics, Model, Output

from vertex import comun

BASE_IMAGE = "python:3.13-slim"
PAQUETES = [
    "pandas==3.0.5",
    "numpy==2.5.3",
    "scikit-learn==1.9.0",
    "scipy==1.18.1",
    "joblib==1.6.0",
    "matplotlib==3.11.1",
    "google-cloud-bigquery==3.46.2",
    "pyarrow==26.0.0",
    "db-dtypes==1.8.0",
]
COMMIT_DEFAULT = "4a2fc811d7246e2d97413621066048ea829a7a7b"
YAML = Path(__file__).resolve().parent / "pipeline_custom.yaml"


@dsl.component(base_image=BASE_IMAGE, packages_to_install=PAQUETES)
def entrenar_evaluar(
    repo_zip: str,
    semilla_split: int,
    metricas: Output[Metrics],
    clasificacion: Output[ClassificationMetrics],
    modelo: Output[Model],
) -> str:
    """Entrena y evalúa el modelo del repo con una semilla de partición."""
    import io
    import json
    import os
    import platform
    import sys
    import time
    import urllib.request
    import zipfile

    import joblib
    import numpy as np
    from sklearn.metrics import confusion_matrix, roc_curve

    with urllib.request.urlopen(repo_zip, timeout=120) as resp:  # noqa: S310 - URL fija de GitHub
        zipfile.ZipFile(io.BytesIO(resp.read())).extractall("/tmp/repo")
    raiz = os.path.join("/tmp/repo", os.listdir("/tmp/repo")[0])
    sys.path.insert(0, raiz)

    from mlops_pipeline.src.Cargar_datos import cargar_config, cargar_datos
    from mlops_pipeline.src.ft_engineering import preparar_datos
    from mlops_pipeline.src.model_evaluation import evaluar, obtener_score_riesgo
    from mlops_pipeline.src.model_training import entrenar
    from scripts.validate_metrics import extraer_metricas

    config = cargar_config(__import__("pathlib").Path(raiz, "mlops_pipeline/src/config.json"))
    t0 = time.perf_counter()
    df = cargar_datos(config)
    t_carga = time.perf_counter() - t0
    datos = preparar_datos(df, random_state=semilla_split)
    t1 = time.perf_counter()
    entrenamiento = entrenar(datos.X_train, datos.y_train, datos.preprocessor)
    t_entrenamiento = time.perf_counter() - t1
    resultado = evaluar(entrenamiento, datos.X_eval, datos.y_eval)
    m = extraer_metricas(resultado)
    m.update(
        {
            "semilla_split": semilla_split,
            "tiempo_carga_s": round(t_carga, 2),
            "tiempo_entrenamiento_s": round(t_entrenamiento, 2),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
        }
    )

    for nombre in (
        "f1_0",
        "recall_0",
        "precision_0",
        "balanced_accuracy",
        "accuracy",
        "roc_auc_risk",
        "average_precision_risk",
        "cv_f1_0_mean",
        "cv_f1_0_std",
        "tiempo_entrenamiento_s",
    ):
        if m.get(nombre) is not None:
            metricas.log_metric(nombre, float(m[nombre]))
    metricas.log_metric("semilla_split", semilla_split)

    modelo_final = entrenamiento.modelo_final_fitted
    y_riesgo = (np.asarray(datos.y_eval) == 0).astype(int)
    score = obtener_score_riesgo(modelo_final, datos.X_eval)
    if score is not None:
        fpr, tpr, umbrales = roc_curve(y_riesgo, score)
        paso = max(1, len(fpr) // 200)
        umbrales = np.clip(umbrales[::paso], 0.0, 1.0)
        clasificacion.log_roc_curve(fpr[::paso].tolist(), tpr[::paso].tolist(), umbrales.tolist())
    y_pred = modelo_final.predict(datos.X_eval)
    matriz = confusion_matrix(datos.y_eval, y_pred, labels=[0, 1])
    clasificacion.log_confusion_matrix(["riesgo(0)", "pago(1)"], matriz.tolist())

    os.makedirs(modelo.path, exist_ok=True)
    joblib.dump(modelo_final, os.path.join(modelo.path, "model.joblib"))
    modelo.metadata.update({"framework": "scikit-learn", "modelo": m["modelo_final"]})
    return json.dumps(m)


@dsl.component(base_image=BASE_IMAGE, packages_to_install=["numpy==2.5.3"])
def consolidar(resultados: list, resumen: Output[Metrics]) -> str:
    """Media y desviación estándar de cada métrica entre semillas (estabilidad)."""
    import json

    import numpy as np

    filas = [json.loads(r) for r in resultados]
    claves = (
        "f1_0",
        "recall_0",
        "precision_0",
        "balanced_accuracy",
        "accuracy",
        "roc_auc_risk",
        "average_precision_risk",
        "tiempo_entrenamiento_s",
    )
    salida = {"n_semillas": len(filas), "por_semilla": filas}
    for clave in claves:
        valores = np.array([f[clave] for f in filas if f.get(clave) is not None], dtype=float)
        if len(valores):
            salida[f"{clave}_media"] = float(valores.mean())
            salida[f"{clave}_std"] = float(valores.std(ddof=1)) if len(valores) > 1 else 0.0
            resumen.log_metric(f"{clave}_media", salida[f"{clave}_media"])
            resumen.log_metric(f"{clave}_std", salida[f"{clave}_std"])
    return json.dumps(salida)


@dsl.pipeline(
    name="cdprod-riesgo-custom",
    description="Modelo de riesgo crediticio del repo CDProd entrenado en Vertex Pipelines.",
)
# KFP compila el default como valor literal del parámetro, no como objeto compartido.
def pipeline(repo_zip: str, semillas: list = list(comun.SEMILLAS_ESTABILIDAD)):  # noqa: B006
    with dsl.ParallelFor(items=semillas, parallelism=5) as semilla:
        tarea = (
            entrenar_evaluar(repo_zip=repo_zip, semilla_split=semilla)
            .set_cpu_limit("4")
            .set_memory_limit("16G")
            .set_display_name("entrenar-evaluar")
        )
    consolidar(resultados=dsl.Collected(tarea.outputs["Output"])).set_display_name(
        "consolidar-estabilidad"
    )


def compilar() -> Path:
    compiler.Compiler().compile(pipeline_func=pipeline, package_path=str(YAML))
    return YAML


def enviar(commit: str) -> dict:
    from google.cloud import aiplatform

    aiplatform.init(project=comun.PROJECT, location=comun.REGION, staging_bucket=comun.BUCKET)
    job = aiplatform.PipelineJob(
        display_name="cdprod-riesgo-custom",
        template_path=str(compilar()),
        pipeline_root=comun.PIPELINE_ROOT,
        parameter_values={
            "repo_zip": comun.REPO_ZIP.format(commit=commit),
            "semillas": list(comun.SEMILLAS_ESTABILIDAD),
        },
        enable_caching=False,
    )
    job.submit()
    info = {
        "pipeline_job": job.resource_name,
        "commit": commit,
        "semillas": list(comun.SEMILLAS_ESTABILIDAD),
        "consola": job._dashboard_uri(),
    }
    comun.guardar_json("pipeline_job", info)
    return info


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--compilar", action="store_true", help="Solo compila el YAML.")
    parser.add_argument("--commit", default=COMMIT_DEFAULT, help="Commit del repo a ejecutar.")
    args = parser.parse_args(argv)
    if args.compilar:
        print(compilar())
    else:
        print(enviar(args.commit))


if __name__ == "__main__":
    main()
