"""Evalúa el modelo AutoML sobre el mismo holdout del repositorio.

1. Lee el training pipeline guardado en vertex/resultados/automl_job.json.
2. Guarda la evaluación nativa de Vertex (auRoc, auPrc, logLoss) del split TEST.
3. Lanza una batch prediction sobre las filas VALIDATE + TEST (BigQuery -> BigQuery).
4. Calcula con vertex.metricas las mismas métricas que el modelo local:
   - umbral 0.5 (salida por defecto de AutoML);
   - umbral que maximiza F1_0 en VALIDATE, aplicado sin cambios a TEST.

Uso:
    python -m vertex.automl_evaluar            # estado del entrenamiento
    python -m vertex.automl_evaluar --evaluar  # evaluación completa (al terminar)
    python -m vertex.automl_evaluar --evaluar --reusar-batch  # sin nueva batch prediction
"""

import argparse

import pandas as pd
from google.cloud import aiplatform, bigquery

from vertex import comun, metricas
from vertex.automl_entrenar import FEATURES_NUMERICAS

COLUMNAS_EXCLUIDAS_INPUT = (comun.TARGET,)


def _pipeline() -> aiplatform.AutoMLTabularTrainingJob:
    info = comun.leer_json("automl_job")
    return aiplatform.AutoMLTabularTrainingJob.get(info["training_pipeline"])


def estado() -> str:
    job = _pipeline()
    gca = job.gca_resource
    duracion = None
    if gca.end_time and gca.start_time:
        duracion = (gca.end_time - gca.start_time).total_seconds() / 60
    print(f"estado: {gca.state.name}  inicio: {gca.start_time}  fin: {gca.end_time}")
    if duracion:
        print(f"duración: {duracion:.1f} min")
    return gca.state.name


def _crear_tabla_input(cliente: bigquery.Client) -> str:
    """Filas VALIDATE + TEST sin el target, con nulos numéricos imputados.

    AutoML rechaza en predicción las filas con nulos en columnas numéricas
    ("Null values are not allowed in transformations"). Se imputan con la
    mediana de TRAIN, igual que el SimpleImputer(median) del pipeline local, para
    evaluar AutoML sobre el holdout completo.
    """
    destino = f"{comun.PROJECT}.{comun.BQ_DATASET}.automl_input_val_test"
    columnas = [
        f.name
        for f in cliente.get_table(comun.BQ_TABLA_VERTEX).schema
        if f.name not in COLUMNAS_EXCLUIDAS_INPUT
    ]
    medianas = ", ".join(f"APPROX_QUANTILES({c}, 2)[OFFSET(1)] AS {c}" for c in FEATURES_NUMERICAS)
    seleccion = ", ".join(
        f"IFNULL(t.{c}, m.{c}) AS {c}" if c in FEATURES_NUMERICAS else f"t.{c}" for c in columnas
    )
    sql = (
        f"CREATE OR REPLACE TABLE `{destino}` AS "
        f"WITH m AS (SELECT {medianas} FROM `{comun.BQ_TABLA_VERTEX}` "
        f"WHERE {comun.COLUMNA_SPLIT} = 'TRAIN') "
        f"SELECT {seleccion} FROM `{comun.BQ_TABLA_VERTEX}` t CROSS JOIN m "
        f"WHERE t.{comun.COLUMNA_SPLIT} IN ('VALIDATE', 'TEST')"
    )
    cliente.query(sql).result()
    return destino


def _scores_riesgo(df: pd.DataFrame, columna_pred: str) -> pd.Series:
    def score(pred) -> float:
        clases = [str(c) for c in pred["classes"]]
        return float(pred["scores"][clases.index(str(comun.RISK_LABEL))])

    return df[columna_pred].apply(score)


BATCH_DISPLAY_NAME = "cdprod-automl-batch-val-test"


def _lanzar_batch(
    modelo: aiplatform.Model, cliente: bigquery.Client
) -> aiplatform.BatchPredictionJob:
    tabla_input = _crear_tabla_input(cliente)
    return modelo.batch_predict(
        job_display_name=BATCH_DISPLAY_NAME,
        bigquery_source=f"bq://{tabla_input}",
        bigquery_destination_prefix=f"bq://{comun.PROJECT}.{comun.BQ_DATASET}",
        instances_format="bigquery",
        predictions_format="bigquery",
        machine_type="n1-standard-4",
        starting_replica_count=1,
        max_replica_count=1,
        sync=True,
    )


def _ultimo_batch_exitoso() -> aiplatform.BatchPredictionJob:
    for job in aiplatform.BatchPredictionJob.list(
        filter=f'display_name="{BATCH_DISPLAY_NAME}"', order_by="create_time desc"
    ):
        if job.gca_resource.state.name == "JOB_STATE_SUCCEEDED":
            return job
    raise RuntimeError("No hay batch prediction exitosa para reutilizar.")


def evaluar(reusar_batch: bool = False) -> dict:
    job = _pipeline()
    modelo = job.get_model()
    cliente = bigquery.Client(project=comun.PROJECT)

    nativa = {}
    for ev in modelo.list_model_evaluations():
        m = dict(ev.metrics)
        nativa = {k: m.get(k) for k in ("auRoc", "auPrc", "logLoss") if k in m}
        break

    batch = _ultimo_batch_exitoso() if reusar_batch else _lanzar_batch(modelo, cliente)
    g_batch = batch.gca_resource
    t_batch = (g_batch.end_time - g_batch.start_time).total_seconds()
    tabla_pred = batch.output_info.bigquery_output_table
    dataset_pred = batch.output_info.bigquery_output_dataset.replace("bq://", "")
    pred = cliente.query(f"SELECT * FROM `{dataset_pred}.{tabla_pred}`").to_dataframe()
    real = cliente.query(
        f"SELECT {comun.COLUMNA_ID}, {comun.TARGET}, "
        "promedio_ingresos_datacredito IS NULL AS imputado "
        f"FROM `{comun.BQ_TABLA_VERTEX}`"
    ).to_dataframe()
    # La salida de batch prediction devuelve las columnas de entrada como texto.
    pred[comun.COLUMNA_ID] = pred[comun.COLUMNA_ID].astype("int64")
    real[comun.COLUMNA_ID] = real[comun.COLUMNA_ID].astype("int64")
    df = pred.merge(real, on=comun.COLUMNA_ID)
    n_input = cliente.get_table(
        f"{comun.PROJECT}.{comun.BQ_DATASET}.automl_input_val_test"
    ).num_rows
    if len(df) != n_input:
        raise RuntimeError(f"Batch prediction incompleta: {len(df)} de {n_input} filas.")
    col_pred = next(c for c in df.columns if c.startswith("predicted_"))
    df["score_riesgo"] = _scores_riesgo(df, col_pred)
    df["y"] = df[comun.TARGET].astype(int)

    val = df[df[comun.COLUMNA_SPLIT] == "VALIDATE"]
    test = df[df[comun.COLUMNA_SPLIT] == "TEST"]
    umbral_val = metricas.umbral_max_f1(val["y"], val["score_riesgo"])
    resultado = {
        "modelo": modelo.resource_name,
        "batch_prediction_job": batch.resource_name,
        "evaluacion_nativa_vertex_test": nativa,
        "n_validate": int(len(val)),
        "n_test": int(len(test)),
        "umbral_validacion": float(umbral_val),
        "test_umbral_0_5": metricas.evaluar_scores(test["y"], test["score_riesgo"], 0.5),
        "test_umbral_validacion": metricas.evaluar_scores(
            test["y"], test["score_riesgo"], umbral_val
        ),
        "tiempo_batch_prediction_s": round(t_batch, 1),
        "tabla_predicciones": f"{dataset_pred}.{tabla_pred}",
    }
    gca = job.gca_resource
    resultado["tiempo_entrenamiento_min"] = round(
        (gca.end_time - gca.start_time).total_seconds() / 60, 1
    )
    comun.guardar_json("automl_metricas", resultado)
    test[[comun.COLUMNA_ID, "y", "score_riesgo", "imputado"]].to_csv(
        comun.RESULTADOS / "automl_scores_test.csv", index=False
    )
    return resultado


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Estado y evaluación del modelo AutoML.")
    parser.add_argument("--evaluar", action="store_true")
    parser.add_argument(
        "--reusar-batch",
        action="store_true",
        help="Usa la última batch prediction exitosa en lugar de lanzar otra.",
    )
    args = parser.parse_args(argv)
    aiplatform.init(project=comun.PROJECT, location=comun.REGION, staging_bucket=comun.BUCKET)
    if args.evaluar:
        print(evaluar(reusar_batch=args.reusar_batch))
    else:
        estado()


if __name__ == "__main__":
    main()
