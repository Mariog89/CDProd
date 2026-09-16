"""API batch del mejor modelo (Stage 3).

Reutiliza exclusivamente model_deploy.cargar_modelo() y
model_deploy.predecir(). No entrena, no tunea, no evalua ni
recalcula thresholds dentro del request HTTP.
"""

import csv
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from mlops_pipeline.src.model_deploy import cargar_modelo, predecir

_LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "1.0.0"
DEFAULT_METRICS_PATH = REPO_ROOT / "runtime" / "metrics.html"
DEFAULT_LOG_PATH = REPO_ROOT / "runtime" / "prediction_log.csv"


def resolver_artifact_dir() -> Path:
    raw = os.environ.get("MODEL_ARTIFACT_DIR", str(DEFAULT_ARTIFACT_DIR))
    candidato = Path(raw)
    if not candidato.is_absolute():
        candidato = REPO_ROOT / candidato
    return candidato


def resolver_metrics_path() -> Path:
    raw = os.environ.get("METRICS_HTML_PATH", str(DEFAULT_METRICS_PATH))
    candidato = Path(raw)
    if not candidato.is_absolute():
        candidato = REPO_ROOT / candidato
    return candidato


def resolver_log_path() -> Path:
    raw = os.environ.get("PREDICTION_LOG_PATH", str(DEFAULT_LOG_PATH))
    candidato = Path(raw)
    if not candidato.is_absolute():
        candidato = REPO_ROOT / candidato
    return candidato


def registrar_log_batch(
    registros: list[dict[str, Any]],
    predicciones: list[dict[str, Any]],
    version: str,
    marca: str,
    ruta: Path | None = None,
) -> str:
    """Agrega un batch al prediction log append-only para monitoring.

    No guarda Pago_atiempo, secretos ni rutas internas. Solo inputs raw
    (prefijo raw_) más la salida de inferencia, con request_id común.
    """
    destino = Path(ruta) if ruta is not None else resolver_log_path()
    request_id = uuid.uuid4().hex
    destino.parent.mkdir(parents=True, exist_ok=True)
    claves_raw = sorted({k for r in registros for k in r if k != "Pago_atiempo"})
    columnas = [
        "request_id",
        "prediction_timestamp",
        "model_version",
        "prediction",
        "risk_probability",
        "risk_score",
        *[f"raw_{k}" for k in claves_raw],
    ]
    existe = destino.is_file()
    with open(destino, "a", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas, extrasaction="ignore")
        if not existe:
            escritor.writeheader()
        for registro, pred in zip(registros, predicciones, strict=True):
            fila = {
                "request_id": request_id,
                "prediction_timestamp": marca,
                "model_version": version,
                "prediction": pred.get("prediction"),
                "risk_probability": pred.get("risk_probability"),
                "risk_score": pred.get("risk_score"),
            }
            for clave in claves_raw:
                valor = registro.get(clave)
                fila[f"raw_{clave}"] = "" if valor is None else str(valor)
            escritor.writerow(fila)
    return request_id


def _sanitizar_detalle(mensaje: str) -> str:
    texto = str(mensaje)
    # Evita exponer rutas personales del servidor en respuestas HTTP.
    texto = texto.replace(str(REPO_ROOT), "<repo>")
    texto = texto.replace(str(Path.home()), "<home>")
    return texto[:500]


def _valor_json(valor: Any) -> Any:
    try:
        if valor is None:
            return None
        if isinstance(valor, float) and pd.isna(valor):
            return None
        if pd.isna(valor):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(valor, bool | int | float | str):
        return valor
    if hasattr(valor, "item"):
        try:
            return valor.item()
        except (ValueError, AttributeError):
            pass
    if hasattr(valor, "isoformat"):
        try:
            return valor.isoformat()
        except (ValueError, TypeError):
            pass
    return str(valor)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.artifact = None
    app.state.load_error = None
    app.state.artifact_dir = str(resolver_artifact_dir())
    try:
        app.state.artifact = cargar_modelo(resolver_artifact_dir())
    except Exception as exc:  # noqa: BLE001 - estado NOT READY controlado
        app.state.load_error = _sanitizar_detalle(exc)
    yield


app = FastAPI(title="CDProd Risk Batch API", version="1.0.0", lifespan=lifespan)


class BatchRequest(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1)


@app.get("/health")
def health(request: Request):
    artifact = getattr(request.app.state, "artifact", None)
    if artifact is not None:
        version = str(artifact.metadata.get("model_version", "desconocida"))
        return {"status": "ok", "model_loaded": True, "model_version": version}
    detail = getattr(request.app.state, "load_error", None) or "model not loaded"
    return JSONResponse(
        status_code=503,
        content={"status": "not_ready", "model_loaded": False, "detail": detail},
    )


@app.get("/metrics", response_class=HTMLResponse)
def metrics():
    """Sirve la pestaña de métricas si fue generada; nunca recalcula."""
    ruta = resolver_metrics_path()
    if not ruta.is_file():
        return JSONResponse(
            status_code=404,
            content={"detail": "metrics not generated yet"},
        )
    return HTMLResponse(content=ruta.read_text(encoding="utf-8"))


@app.post("/predict_batch")
def predict_batch(payload: BatchRequest, request: Request):
    artifact = getattr(request.app.state, "artifact", None)
    if artifact is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    try:
        df_crudo = pd.DataFrame(payload.records)
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422, detail=f"records invalidos: {_sanitizar_detalle(exc)}"
        ) from exc
    try:
        resultado = predecir(artifact, df_crudo)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=422, detail=f"schema invalido: {_sanitizar_detalle(exc)}"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - 500 generico sin stack trace
        raise HTTPException(status_code=500, detail="prediction failed") from exc
    try:
        marco = resultado.dataframe
        version = str(resultado.model_version)
        marca = str(resultado.prediction_timestamp)
        predicciones = []
        for _, fila in marco.iterrows():
            predicciones.append(
                {
                    "prediction": int(fila.get("prediction")),
                    "risk_probability": _valor_json(fila.get("risk_probability")),
                    "risk_score": _valor_json(fila.get("risk_score")),
                    "prediction_timestamp": _valor_json(fila.get("prediction_timestamp", marca)),
                }
            )
        for item in predicciones:
            if item["prediction"] not in (0, 1):
                raise ValueError("prediction fuera de {0,1}")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - serializacion controlada
        raise HTTPException(status_code=500, detail="prediction failed") from exc
    try:
        registrar_log_batch(payload.records, predicciones, version, marca)
    except Exception as exc:  # noqa: BLE001 - el log no bloquea la respuesta
        _LOG.warning("prediction log failed: %s", type(exc).__name__)
    return {"model_version": version, "count": len(predicciones), "predictions": predicciones}
