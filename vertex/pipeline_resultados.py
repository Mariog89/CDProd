"""Descarga los resultados de la corrida de Vertex Pipelines guardada en pipeline_job.json.

Extrae la salida de `consolidar` (métricas por semilla, media y desviación) y
los tiempos de cada tarea, y los guarda en vertex/resultados/pipeline_metricas.json.

Uso:
    python -m vertex.pipeline_resultados
"""

import json

from google.cloud import aiplatform

from vertex import comun


def main() -> None:
    aiplatform.init(project=comun.PROJECT, location=comun.REGION)
    info = comun.leer_json("pipeline_job")
    job = aiplatform.PipelineJob.get(info["pipeline_job"])
    gca = job.gca_resource
    consolidado = None
    tareas = []
    for tarea in gca.job_detail.task_details:
        duracion = None
        if tarea.end_time and tarea.start_time:
            duracion = round((tarea.end_time - tarea.start_time).total_seconds(), 1)
        tareas.append(
            {"tarea": tarea.task_name, "estado": tarea.state.name, "duracion_s": duracion}
        )
        if tarea.task_name == "consolidar":
            consolidado = json.loads(tarea.execution.metadata["output:Output"])
    resultado = {
        "pipeline_job": info["pipeline_job"],
        "estado": gca.state.name,
        "duracion_total_s": round((gca.end_time - gca.start_time).total_seconds(), 1),
        "tareas": tareas,
        "consolidado": consolidado,
    }
    comun.guardar_json("pipeline_metricas", resultado)
    print(json.dumps({k: v for k, v in resultado.items() if k != "consolidado"}, indent=1))
    for fila in consolidado["por_semilla"]:
        print(
            fila["semilla_split"],
            fila["modelo_final"],
            fila["hiperparametros"],
            round(fila["f1_0"], 4),
            round(fila["recall_0"], 4),
            round(fila["roc_auc_risk"], 4),
            fila["tiempo_entrenamiento_s"],
        )


if __name__ == "__main__":
    main()
