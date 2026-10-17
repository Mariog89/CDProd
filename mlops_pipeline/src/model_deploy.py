"""Persistencia, carga validada e inferencia del modelo final (Stage 3).

Deployment en este módulo es persistencia + carga + inferencia Python
reutilizable. No entrena, no tunea y no usa X_eval/y_eval para decidir.

Se serializa un único Pipeline fitted completo (preprocessor fitted +
estimator fitted) para reducir riesgo de desincronía de objetos aprendidos.

Training-serving skew: el Pipeline único evita desincronizar objetos
aprendidos (imputación/encoding ajustados solo con train), PERO
normalizar_tipos/crear_features/FEATURES_V1 viven en código. Por eso se
registra FEATURE_SCHEMA_VERSION y se valida al cargar y al predecir.
El skew no es imposible: un cambio de código sin versionar lo reintroduce.

SHA-256 valida integridad, NO autenticidad. Solo cargar artefactos
confiables generados por este proyecto.
"""

import hashlib
import json
import platform
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.utils.validation import check_is_fitted

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from mlops_pipeline.src.Cargar_datos import (  # noqa: E402
    cargar_config,
    cargar_datos,
    encontrar_raiz_proyecto,
)
from mlops_pipeline.src.ft_engineering import (  # noqa: E402
    FEATURE_SCHEMA_VERSION,
    FEATURES_V1,
    preparar_datos,
    preparar_features_inferencia,
)
from mlops_pipeline.src.model_training import entrenar  # noqa: E402

RISK_LABEL = 0
TARGET = "Pago_atiempo"
ARTIFACT_SCHEMA_VERSION = "1"
MODEL_VERSION_DEFAULT = "1.0.0"
N_FEATURES_INPUT = 16


@dataclass
class ArtefactoModelo:
    """Pipeline cargado con su metadata y marca de verificación."""

    modelo: object
    metadata: dict
    verificado: bool


@dataclass
class ResultadoPrediccion:
    """Salida de inferencia con metadata mínima."""

    dataframe: pd.DataFrame
    model_version: str
    prediction_timestamp: str


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _resumir_cv(resultado_cv) -> dict:
    if resultado_cv is None:
        return {}
    if isinstance(resultado_cv, dict):
        return dict(resultado_cv)
    resumido = {}
    for clave in (
        "mean_f1_0",
        "std_f1_0",
        "mean_recall_0",
        "std_recall_0",
        "mean_precision_0",
        "std_precision_0",
        "mean_balanced_accuracy",
        "std_balanced_accuracy",
    ):
        if hasattr(resultado_cv, clave):
            resumido[clave] = float(getattr(resultado_cv, clave))
    nombre = getattr(resultado_cv, "nombre", "final_tuned")
    resumido["nombre"] = str(nombre)
    return resumido


def _resumir_holdout(metricas_holdout) -> dict | None:
    if metricas_holdout is None:
        return None
    if isinstance(metricas_holdout, dict):
        return dict(metricas_holdout)
    resumido = {}
    for clave in (
        "precision_0",
        "recall_0",
        "f1_0",
        "balanced_accuracy",
        "accuracy_ctx",
    ):
        if hasattr(metricas_holdout, clave):
            resumido[clave] = float(getattr(metricas_holdout, clave))
    return resumido


def crear_metadata_modelo(
    nombre_modelo: str,
    clase_estimator: str,
    hiperparametros_finales: dict,
    resultado_cv_modelo_final=None,
    metricas_holdout_finales=None,
    model_version: str = MODEL_VERSION_DEFAULT,
    random_state: int = 42,
    fecha_entrenamiento: str | None = None,
    commit_sha: str | None = None,
    tiempos: dict | None = None,
    notas: str | None = None,
) -> dict:
    """Construye la metadata técnica sin datos personales ni filas.

    Semántica de metricas_holdout_finales: un artefacto release posterior
    a evaluation debe recibir las métricas finales ya obtenidas por
    model_evaluation.py; la demo standalone usa None. Este módulo nunca
    reejecuta evaluation para rellenarlo ni hardcodea métricas finales.
    Con None el JSON sigue siendo válido y el artefacto queda marcado
    como demo sin referencia holdout.
    """
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "model_version": str(model_version),
        "nombre_modelo": str(nombre_modelo),
        "clase_estimator": str(clase_estimator),
        "hiperparametros_finales": dict(hiperparametros_finales or {}),
        "fecha_entrenamiento": fecha_entrenamiento or _utc_now_iso(),
        "fecha_serializacion": None,
        "random_state": int(random_state),
        "sklearn_version": str(sklearn.__version__),
        "joblib_version": str(joblib.__version__),
        "python_version": str(platform.python_version()),
        "features_v1": list(FEATURES_V1),
        "target": TARGET,
        "risk_label": int(RISK_LABEL),
        "n_features_input": int(N_FEATURES_INPUT),
        "resultado_cv_modelo_final": _resumir_cv(resultado_cv_modelo_final),
        "metricas_holdout_finales": _resumir_holdout(metricas_holdout_finales),
        "artifact_sha256": None,
        "artifact_size_bytes": None,
        "commit_sha": commit_sha,
        "tiempos": dict(tiempos) if tiempos is not None else None,
        "notas": notas,
    }


def _sha256_archivo(ruta: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with open(ruta, "rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
            total += len(bloque)
    return digest.hexdigest(), int(total)


def guardar_modelo(
    modelo_fitted,
    metadata: dict,
    ruta: Path,
    overwrite: bool = False,
) -> dict:
    """Serializa el Pipeline fitted y completa sha/tamaño/fecha.

    Orden: verificar fitted -> crear directorio -> serializar ->
    calcular SHA-256 y bytes -> completar metadata -> escribir JSON.
    Falla si el destino ya existe salvo overwrite=True explícito.
    """
    if ruta is None or str(ruta).strip() == "":
        raise ValueError("ruta es obligatoria para guardar el artefacto.")
    if not isinstance(metadata, dict):
        raise TypeError("metadata debe ser un dict creado por crear_metadata_modelo.")
    check_is_fitted(modelo_fitted)
    destino = Path(ruta)
    destino.mkdir(parents=True, exist_ok=True)
    ruta_modelo = destino / "model.joblib"
    ruta_meta = destino / "metadata.json"
    if not overwrite and (ruta_modelo.exists() or ruta_meta.exists()):
        raise FileExistsError(
            f"El destino ya existe en {destino}. Use overwrite=True para reemplazar."
        )
    joblib.dump(modelo_fitted, ruta_modelo)
    sha, size = _sha256_archivo(ruta_modelo)
    final = dict(metadata)
    final["artifact_sha256"] = sha
    final["artifact_size_bytes"] = size
    final["fecha_serializacion"] = _utc_now_iso()
    with open(ruta_meta, "w", encoding="utf-8") as archivo:
        json.dump(final, archivo, indent=2, ensure_ascii=False)
    return final


def cargar_modelo(ruta: Path) -> ArtefactoModelo:
    """Carga validada: metadata primero, hash antes de deserializar.

    Valida schema, feature_schema y sklearn exacta antes de joblib.load.
    No descarga desde URL: solo rutas locales.
    """
    if ruta is None or str(ruta).strip() == "":
        raise ValueError("ruta es obligatoria para cargar el artefacto.")
    origen = Path(ruta)
    ruta_modelo = origen / "model.joblib"
    ruta_meta = origen / "metadata.json"
    if not ruta_modelo.is_file():
        raise FileNotFoundError(f"No existe model.joblib en {origen}.")
    if not ruta_meta.is_file():
        raise FileNotFoundError(f"No existe metadata.json en {origen}.")
    with open(ruta_meta, encoding="utf-8") as archivo:
        metadata = json.load(archivo)
    if metadata.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("artifact_schema_version no coincide con el esperado.")
    if metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("feature_schema_version no coincide con el código actual.")
    esperado_sklearn = metadata.get("sklearn_version")
    if esperado_sklearn != str(sklearn.__version__):
        raise RuntimeError(f"sklearn_version {esperado_sklearn} != actual {sklearn.__version__}.")
    sha_actual, _ = _sha256_archivo(ruta_modelo)
    if sha_actual != metadata.get("artifact_sha256"):
        raise ValueError("SHA-256 no coincide: artefacto corrupto o manipulado.")
    modelo = joblib.load(ruta_modelo)
    check_is_fitted(modelo)
    return ArtefactoModelo(modelo=modelo, metadata=metadata, verificado=True)


def _clases_modelo(modelo):
    clases = getattr(modelo, "classes_", None)
    if clases is not None:
        return np.asarray(clases)
    pasos = getattr(modelo, "named_steps", None)
    if isinstance(pasos, dict) and "model" in pasos:
        return np.asarray(getattr(pasos["model"], "classes_", []))
    return None


def obtener_risk_output(modelo, X) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (risk_probability, risk_score) orientados a clase 0.

    Con predict_proba localiza la columna donde classes_ == 0 y el score
    queda en NaN. Con solo decision_function orienta el signo según
    classes_ sin convertir a probabilidad. Sin ambos, todo NaN.
    Nunca asume [:, 1].
    """
    n = int(len(X))
    nan = np.full(n, np.nan, dtype=float)
    predict_proba = getattr(modelo, "predict_proba", None)
    if callable(predict_proba):
        try:
            proba = np.asarray(predict_proba(X))
            clases = _clases_modelo(modelo)
            indice = list(np.asarray(clases)).index(RISK_LABEL)
            return np.asarray(proba[:, indice], dtype=float), nan.copy()
        except Exception:
            pass
    decision_function = getattr(modelo, "decision_function", None)
    if callable(decision_function):
        try:
            scores = np.asarray(decision_function(X), dtype=float).ravel()
            clases = _clases_modelo(modelo)
            if clases is not None and len(np.asarray(clases)) == 2:
                clases = np.asarray(clases)
                if RISK_LABEL == clases[1]:
                    return nan.copy(), scores
                if RISK_LABEL == clases[0]:
                    return nan.copy(), -scores
            return nan.copy(), nan.copy()
        except Exception:
            return nan.copy(), nan.copy()
    return nan.copy(), nan.copy()


def predecir(modelo_o_artefacto, df_crudo: pd.DataFrame) -> ResultadoPrediccion:
    """Inferencia con el único camino de serving, sin target requerido.

    Acepta un Pipeline recién entrenado o un ArtefactoModelo cargado.
    Ignora Pago_atiempo y columnas extra, preserva el índice original
    y usa un timestamp UTC común al batch.

    Versiones: con Pipeline fitted usa model_version in-memory; con
    ArtefactoModelo usa la versión persistida. La validación fuerte de
    hash, sklearn, schema y versión persistente aplica al artefacto
    cargado; un Pipeline en memoria no tiene metadata persistida.
    """
    if isinstance(modelo_o_artefacto, ArtefactoModelo):
        modelo = modelo_o_artefacto.modelo
        version = str(modelo_o_artefacto.metadata.get("model_version", "desconocida"))
    else:
        modelo = modelo_o_artefacto
        version = str(getattr(modelo, "_model_version", "in-memory"))
    X = preparar_features_inferencia(df_crudo)
    y_pred = np.asarray(modelo.predict(X)).ravel()
    prob, score = obtener_risk_output(modelo, X)
    marca = _utc_now_iso()
    salida = pd.DataFrame(
        {
            "prediction": pd.Series(np.asarray(y_pred, dtype=int), index=df_crudo.index),
            "risk_probability": pd.Series(np.asarray(prob, dtype=float), index=df_crudo.index),
            "risk_score": pd.Series(np.asarray(score, dtype=float), index=df_crudo.index),
            "model_version": pd.Series([version] * len(df_crudo), index=df_crudo.index),
            "prediction_timestamp": pd.Series([marca] * len(df_crudo), index=df_crudo.index),
        },
        index=df_crudo.index,
    )
    return ResultadoPrediccion(dataframe=salida, model_version=version, prediction_timestamp=marca)


def main() -> None:
    """Demo académica temporal: entrena con train, guarda, carga y predice."""
    raiz = encontrar_raiz_proyecto()
    config = cargar_config(raiz / "mlops_pipeline" / "src" / "config.json")
    df_completo = cargar_datos(config)
    datos = preparar_datos(df_completo)
    resultado = entrenar(datos.X_train, datos.y_train, datos.preprocessor)
    df_demo = df_completo.head(5).copy()
    df_demo_sin_target = df_demo.drop(columns=[TARGET], errors="ignore")
    estimador = resultado.modelo_final_fitted
    pasos = getattr(estimador, "named_steps", {})
    clase_nombre = type(pasos.get("model", estimador)).__name__
    metadata = crear_metadata_modelo(
        nombre_modelo=str(resultado.nombre_modelo_final),
        clase_estimator=clase_nombre,
        hiperparametros_finales=dict(resultado.hiperparametros_finales),
        resultado_cv_modelo_final=resultado.resultado_cv_modelo_final,
        metricas_holdout_finales=None,
        model_version=MODEL_VERSION_DEFAULT,
        random_state=int(resultado.random_state),
        notas="Demo temporal de deployment, sin artefactos permanentes.",
    )
    with tempfile.TemporaryDirectory(prefix="demo_deploy_") as tmp:
        destino = Path(tmp) / str(metadata["model_version"])
        final = guardar_modelo(estimador, metadata, destino, overwrite=False)
        artefacto = cargar_modelo(destino)
        salida = predecir(artefacto, df_demo_sin_target)
    print("=== Deployment demo ===")
    print(f"model_version: {final['model_version']}")
    print(f"nombre_modelo: {final['nombre_modelo']}")
    print(f"artifact_sha256: {final['artifact_sha256']}")
    print(salida.dataframe.to_string())
    print(f"prediction_timestamp: {salida.prediction_timestamp}")


if __name__ == "__main__":
    main()
