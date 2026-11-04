"""Configuración compartida de la Fase 2 (entrenamiento en GCP con Vertex AI).

Todas las piezas (tabla para Vertex, AutoML, Vertex Pipelines y evaluación)
usan el mismo proyecto, región, tabla de origen y partición train/holdout
que el pipeline local del repositorio, para que la comparación sea justa.
"""

import json
from pathlib import Path

PROJECT = "cdprod-upb-mg"
REGION = "us-central1"
BUCKET = "gs://cdprod-upb-mg-vertex"
PIPELINE_ROOT = f"{BUCKET}/pipeline-root"

BQ_DATASET = "novabank_dev"
BQ_TABLA_ORIGEN = f"{PROJECT}.{BQ_DATASET}.Creditos"
BQ_TABLA_VERTEX = f"{PROJECT}.{BQ_DATASET}.creditos_vertex"
BQ_TABLA_PREDICCIONES = f"{PROJECT}.{BQ_DATASET}.automl_predicciones_test"

TARGET = "Pago_atiempo"
COLUMNA_SPLIT = "split"
COLUMNA_ID = "fila_id"
RISK_LABEL = 0

# Semilla del split del repositorio (preparar_datos) y fracción de validación
# que AutoML toma del train para early stopping / selección interna.
SEMILLA_SPLIT = 42
FRACCION_VALIDACION = 0.10

# Semillas del split train/holdout para el análisis de estabilidad.
SEMILLAS_ESTABILIDAD = (42, 7, 13, 21, 99)

# Commit del repositorio que ejecutan los componentes de Vertex Pipelines.
REPO_ZIP = "https://github.com/Mariog89/CDProd/archive/{commit}.zip"

RESULTADOS = Path(__file__).resolve().parent / "resultados"


def guardar_json(nombre: str, datos: dict) -> Path:
    """Guarda un resultado de la Fase 2 en vertex/resultados/<nombre>.json."""
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    destino = RESULTADOS / f"{nombre}.json"
    destino.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
    return destino


def leer_json(nombre: str) -> dict:
    return json.loads((RESULTADOS / f"{nombre}.json").read_text(encoding="utf-8"))
