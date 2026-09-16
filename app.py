"""API batch del mejor modelo (Stage 3).

Reutiliza exclusivamente model_deploy.cargar_modelo() y
model_deploy.predecir(). No entrena, no tunea, no evalua ni
recalcula thresholds dentro del request HTTP.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from mlops_pipeline.src.model_deploy import cargar_modelo, predecir

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "1.0.0"


def resolver_artifact_dir() -> Path:
    raw = os.environ.get("MODEL_ARTIFACT_DIR", str(DEFAULT_ARTIFACT_DIR))
    candidato = Path(raw)
    if not candidato.is_absolute():
        candidato = REPO_ROOT / candidato
    return candidato


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
    return {"model_version": version, "count": len(predicciones), "predictions": predicciones}
