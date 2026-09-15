"""Batch monitoring for the deployed risk model (Stage 3).

Scope is a time window or batch already served: data quality, feature
drift, prediction drift and performance drift only when labels exist.
It also checks model_version and feature_schema_version compatibility.

What this module never does outside the academic demo in main():
train, tune, select models, change thresholds, trigger retraining, use
holdout distributions as baseline, or duplicate feature engineering.
Serving always goes through preparar_features_inferencia, and the only
distribution baseline is X_train (post feature engineering, before the
ColumnTransformer). Aggregates only are persisted, never rows or PII.

Heuristic thresholds (psi_warning=0.10, psi_critical=0.25,
missing_delta_warning=0.05, min_batch_size=100) are descriptive and
configurable. They are recorded in every result and never tuned here.
"""

import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from mlops_pipeline.src.ft_engineering import (  # noqa: E402
    FEATURE_SCHEMA_VERSION,
    FEATURES_NOMINALES,
    FEATURES_NUMERICAS,
    FEATURES_V1,
    preparar_features_inferencia,
)

RISK_LABEL = 0
TARGET = "Pago_atiempo"
LABELS = [0, 1]

PSI_WARNING_DEFAULT = 0.10
PSI_CRITICAL_DEFAULT = 0.25
MISSING_DELTA_WARNING_DEFAULT = 0.05
MIN_BATCH_SIZE_DEFAULT = 100
N_BINS_DEFAULT = 10
EPSILON_DEFAULT = 1e-6

REFERENCE_SOURCE = "train"
BASELINE_FILENAME = "monitoring_baseline.json"

ALERT_LEVELS = ("info", "warning", "critical")
ALERT_TYPES = (
    "schema",
    "missingness",
    "feature_drift",
    "unseen_category",
    "prediction_drift",
    "performance",
)

# Raw columns known from Base_de_datos.csv. Anything else is reported as extra.
# Authority for serving is still preparar_features_inferencia, this list only
# supports descriptive raw-quality reporting.
COLUMNAS_CRUDAS_CONOCIDAS = [
    "tipo_credito",
    "fecha_prestamo",
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "tipo_laboral",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "puntaje",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "huella_consulta",
    "saldo_mora",
    "saldo_total",
    "saldo_principal",
    "saldo_mora_codeudor",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal",
    "promedio_ingresos_datacredito",
    "tendencia_ingresos",
    "Pago_atiempo",
]


@dataclass
class AlertaMonitoring:
    """Single descriptive finding. Data only, no delivery channel."""

    level: str
    type: str
    feature: str | None
    value: float | int | str | dict | None
    threshold: float | int | str | None
    message: str


@dataclass
class PerfilReferencia:
    """Frozen aggregates computed once from X_train and the frozen model."""

    model_version: str
    feature_schema_version: str
    reference_source: str
    n_reference: int
    created_at: str
    features: list
    numeric_profiles: dict
    categorical_profiles: dict
    missing_rates: dict
    prediction_baseline: dict
    holdout_reference: dict | None = None


@dataclass
class ResultadoMonitoring:
    """Full window report with descriptive alerts only."""

    window_start: str
    window_end: str
    model_version: str
    feature_schema_version: str
    n_rows: int
    generated_at: str
    thresholds: dict
    quality: dict
    feature_drift: dict
    unseen_categories: list
    prediction_drift: dict
    performance: dict | None
    alertas: list = field(default_factory=list)
    holdout_reference: dict | None = None
    batch_size_warning: bool = False


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _serializar(obj):
    """Convert numpy/pandas/path/datetime values into JSON-safe primitives."""
    if isinstance(obj, AlertaMonitoring):
        return {
            "level": str(obj.level),
            "type": str(obj.type),
            "feature": None if obj.feature is None else str(obj.feature),
            "value": _serializar(obj.value),
            "threshold": _serializar(obj.threshold),
            "message": str(obj.message),
        }
    if isinstance(obj, PerfilReferencia):
        return {
            "model_version": str(obj.model_version),
            "feature_schema_version": str(obj.feature_schema_version),
            "reference_source": str(obj.reference_source),
            "n_reference": int(obj.n_reference),
            "created_at": str(obj.created_at),
            "features": list(obj.features),
            "numeric_profiles": _serializar(obj.numeric_profiles),
            "categorical_profiles": _serializar(obj.categorical_profiles),
            "missing_rates": _serializar(obj.missing_rates),
            "prediction_baseline": _serializar(obj.prediction_baseline),
            "holdout_reference": _serializar(obj.holdout_reference),
        }
    if isinstance(obj, ResultadoMonitoring):
        return {
            "window_start": str(obj.window_start),
            "window_end": str(obj.window_end),
            "model_version": str(obj.model_version),
            "feature_schema_version": str(obj.feature_schema_version),
            "n_rows": int(obj.n_rows),
            "generated_at": str(obj.generated_at),
            "thresholds": _serializar(obj.thresholds),
            "quality": _serializar(obj.quality),
            "feature_drift": _serializar(obj.feature_drift),
            "unseen_categories": _serializar(obj.unseen_categories),
            "prediction_drift": _serializar(obj.prediction_drift),
            "performance": _serializar(obj.performance),
            "alertas": [_serializar(a) for a in obj.alertas],
            "holdout_reference": _serializar(obj.holdout_reference),
            "batch_size_warning": bool(obj.batch_size_warning),
        }
    if isinstance(obj, dict):
        return {str(k): _serializar(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serializar(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _serializar(obj.tolist())
    if isinstance(obj, (np.floating,)):
        valor = float(obj)
        if np.isnan(valor):
            return None
        if np.isinf(valor):
            return "inf" if valor > 0 else "-inf"
        return valor
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        try:
            return pd.Timestamp(obj).isoformat().replace("+00:00", "Z")
        except Exception:
            return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        if np.isnan(obj):
            return None
        if np.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return obj
    return obj


def _borde_desde_json(valor):
    """Decode a persisted bin edge, restoring infinite extremes."""
    if isinstance(valor, str):
        bajo = valor.strip().lower()
        if bajo in ("inf", "+inf", "infinity", "+infinity"):
            return float("inf")
        if bajo in ("-inf", "-infinity"):
            return -float("inf")
        return float(valor)
    if valor is None:
        return None
    return float(valor)


def _restaurar_bordes(perfil_dict: dict) -> dict:
    """Convert edge sentinels back to floats after JSON load."""
    for info in perfil_dict.get("numeric_profiles", {}).values():
        if isinstance(info, dict) and "edges" in info:
            info["edges"] = [_borde_desde_json(v) for v in info["edges"]]
    base = perfil_dict.get("prediction_baseline", {})
    if isinstance(base, dict):
        for clave in ("proba_edges", "score_edges"):
            if isinstance(base.get(clave), list):
                base[clave] = [_borde_desde_json(v) for v in base[clave]]
    return perfil_dict


def resultado_a_dict(resultado: ResultadoMonitoring) -> dict:
    """Return a JSON-serializable dict view of a monitoring result."""
    if not isinstance(resultado, ResultadoMonitoring):
        raise TypeError("resultado must be a ResultadoMonitoring.")
    return _serializar(resultado)


def _perfil_a_dict(perfil: PerfilReferencia) -> dict:
    if not isinstance(perfil, PerfilReferencia):
        raise TypeError("perfil must be a PerfilReferencia.")
    return _serializar(perfil)


def _bordes_cuantiles(serie: pd.Series, n_bins: int = N_BINS_DEFAULT):
    """Frozen bin edges from reference quantiles with infinite extremes.

    Uses finite non-null values only, keeps unique sorted cut points, then
    replaces the outer bounds with -inf/+inf so batch values outside the
    reference range fall into the extreme bins instead of crashing.
    Constant input yields a single-bin degenerate profile, never an error.
    """
    valores = pd.to_numeric(serie, errors="coerce")
    valores = valores.dropna()
    finitos = valores[np.isfinite(valores.to_numpy(dtype=float))]
    if len(finitos) < 2 or int(finitos.nunique()) < 2:
        return [-float("inf"), float("inf")], True
    cuantiles = np.quantile(finitos.to_numpy(dtype=float), np.linspace(0.0, 1.0, int(n_bins) + 1))
    unicos = sorted({float(v) for v in np.asarray(cuantiles).ravel() if np.isfinite(v)})
    if len(unicos) < 2:
        return [-float("inf"), float("inf")], True
    bordes = [-float("inf"), *unicos[1:-1], float("inf")]
    bordes = sorted(set(bordes))
    if len(bordes) < 2:
        return [-float("inf"), float("inf")], True
    return [float(b) for b in bordes], False


def _proporciones_histograma(valores, bordes, epsilon: float = EPSILON_DEFAULT):
    """Histogram proportions over frozen edges with smoothing to avoid zeros."""
    n_bins = len(bordes) - 1
    arr = pd.to_numeric(pd.Series(valores), errors="coerce")
    arr = arr.dropna()
    if len(arr) == 0:
        conteos = np.zeros(n_bins, dtype=float)
        n = 0
    else:
        conteos, _ = np.histogram(arr.to_numpy(dtype=float), bins=np.asarray(bordes))
        conteos = np.asarray(conteos, dtype=float)
        n = int(len(arr))
    if n > 0:
        props = conteos / float(n)
    else:
        props = np.zeros(n_bins, dtype=float)
    suavizadas = (props + float(epsilon)) / (1.0 + float(epsilon) * float(n_bins))
    total = float(suavizadas.sum())
    if total > 0:
        suavizadas = suavizadas / total
    return conteos, suavizadas, n


def _psi_desde_proporciones(esperadas, actuales) -> float:
    """PSI as sum((actual - expected) * log(actual / expected))."""
    esperadas = np.asarray(esperadas, dtype=float)
    actuales = np.asarray(actuales, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terminos = (actuales - esperadas) * np.log(actuales / esperadas)
    terminos = np.asarray(terminos, dtype=float)
    terminos[~np.isfinite(terminos)] = 0.0
    return float(terminos.sum())


def _nivel_psi(psi: float, psi_warning: float, psi_critical: float) -> str:
    if psi >= float(psi_critical):
        return "critical"
    if psi >= float(psi_warning):
        return "warning"
    return "info"


def _obtener_riesgo(modelo, X: pd.DataFrame):
    """Risk-oriented outputs without assuming column order.

    Uses the class-0 column of predict_proba located through classes_,
    never [:, 1]. Falls back to oriented decision_function, else NaN.
    Read-only: only predict-style calls, no model updates.
    """
    n = int(len(X))
    nan = np.full(n, np.nan, dtype=float)
    predict_proba = getattr(modelo, "predict_proba", None)
    if callable(predict_proba):
        try:
            proba = np.asarray(predict_proba(X))
            clases = getattr(modelo, "classes_", None)
            if clases is None:
                pasos = getattr(modelo, "named_steps", None)
                if isinstance(pasos, dict) and "model" in pasos:
                    clases = getattr(pasos["model"], "classes_", None)
            indice = list(np.asarray(clases)).index(RISK_LABEL)
            return np.asarray(proba[:, indice], dtype=float), nan.copy()
        except Exception:
            pass
    decision_function = getattr(modelo, "decision_function", None)
    if callable(decision_function):
        try:
            scores = np.asarray(decision_function(X), dtype=float).ravel()
            clases = getattr(modelo, "classes_", None)
            if clases is None:
                pasos = getattr(modelo, "named_steps", None)
                if isinstance(pasos, dict) and "model" in pasos:
                    clases = getattr(pasos["model"], "classes_", None)
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


def _resumen_numerico(serie: pd.Series) -> dict:
    valores = pd.to_numeric(serie, errors="coerce")
    n_total = int(len(valores))
    n_missing = int(valores.isna().sum())
    no_nulos = valores.dropna()
    arr = no_nulos.to_numpy(dtype=float) if len(no_nulos) else np.array([], dtype=float)
    finitos = arr[np.isfinite(arr)] if len(arr) else np.array([], dtype=float)
    n_inf = int(np.sum(~np.isfinite(arr))) if len(arr) else 0
    if len(finitos):
        resumen = {
            "missing_rate": float(n_missing / n_total) if n_total else 0.0,
            "n_non_null": int(len(no_nulos)),
            "n_missing": n_missing,
            "n_inf": n_inf,
            "min": float(np.min(finitos)),
            "max": float(np.max(finitos)),
            "median": float(np.median(finitos)),
            "p10": float(np.quantile(finitos, 0.10)),
            "p25": float(np.quantile(finitos, 0.25)),
            "p75": float(np.quantile(finitos, 0.75)),
            "p90": float(np.quantile(finitos, 0.90)),
        }
    else:
        resumen = {
            "missing_rate": float(n_missing / n_total) if n_total else 0.0,
            "n_non_null": int(len(no_nulos)),
            "n_missing": n_missing,
            "n_inf": n_inf,
            "min": None,
            "max": None,
            "median": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
        }
    return resumen


def _valor_referencia_numerica(serie) -> float | None:
    """Median of finite non-null values, or None when there is no baseline."""
    vals = pd.to_numeric(pd.Series(serie), errors="coerce").dropna()
    if len(vals) == 0:
        return None
    finitos = vals.to_numpy(dtype=float)
    finitos = finitos[np.isfinite(finitos)]
    if len(finitos) == 0:
        return None
    mediana = float(np.median(finitos))
    return mediana if np.isfinite(mediana) else None


def crear_perfil_referencia(X_reference, modelo_final, metadata, n_bins: int = 10):
    """Freeze reference aggregates from training features and the frozen model.

    X_reference must be post feature engineering and pre-ColumnTransformer
    with the 16 FEATURES_V1 columns (the train split in normal use).
    Prediction baseline is the frozen model applied to that same training
    population: it describes deployed-model behavior on train, not skill.

    Special treatment for degenerate numeric references: when the reference
    distribution has zero variance (or invalid/duplicated bin edges) the
    frozen bins collapse to a single [-inf, +inf] bin and the profile stores
    reference_value (median of finite non-null values). Drift is then
    measured as equal-vs-different share, never as silent PSI 0.
    """
    if not isinstance(X_reference, pd.DataFrame):
        raise TypeError("X_reference must be a pd.DataFrame.")
    if len(X_reference) == 0:
        raise ValueError("X_reference is empty.")
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dict.")
    faltantes = [c for c in FEATURES_V1 if c not in X_reference.columns]
    if faltantes:
        raise KeyError(f"X_reference misses v1 features: {faltantes}")
    if modelo_final is None or not callable(getattr(modelo_final, "predict", None)):
        raise TypeError("modelo_final must expose predict.")
    if "model_version" not in metadata:
        raise ValueError("metadata lacks model_version.")
    esquema_meta = metadata.get("feature_schema_version")
    if esquema_meta != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"metadata feature_schema_version {esquema_meta!r} != code {FEATURE_SCHEMA_VERSION!r}."
        )
    version = str(metadata["model_version"])
    holdout = metadata.get("metricas_holdout_finales")
    holdout_ref = dict(holdout) if isinstance(holdout, dict) else None

    X_ref = X_reference[list(FEATURES_V1)].copy()
    n_ref = int(len(X_ref))

    perfiles_num = {}
    perfiles_cat = {}
    missing_rates = {}
    for columna in FEATURES_V1:
        serie = X_ref[columna]
        tasa = float(serie.isna().mean()) if len(serie) else 0.0
        missing_rates[str(columna)] = tasa
        if columna in FEATURES_NUMERICAS:
            resumen = _resumen_numerico(serie)
            bordes, degenerado = _bordes_cuantiles(serie, n_bins=int(n_bins))
            _, props, _ = _proporciones_histograma(serie, bordes)
            valor_ref: float | None = None
            if bool(degenerado):
                valor_ref = _valor_referencia_numerica(serie)
            perfiles_num[str(columna)] = {
                **resumen,
                "missing_rate_reference": tasa,
                "edges": [float(b) for b in bordes],
                "proportions": [float(p) for p in np.asarray(props).ravel()],
                "degenerate": bool(degenerado),
                "n_reference_non_null": int(serie.dropna().shape[0]),
                "reference_value": valor_ref,
            }
        else:
            no_nulos = serie.dropna()
            if len(no_nulos):
                frec = no_nulos.astype("string").value_counts(dropna=True)
                total = float(frec.sum())
                props = {str(k): float(v) / total for k, v in frec.items()}
            else:
                props = {}
            perfiles_cat[str(columna)] = {
                "missing_rate_reference": tasa,
                "n_reference_non_null": int(len(no_nulos)),
                "categories": sorted(props),
                "proportions": dict(props),
            }

    y_ref = np.asarray(modelo_final.predict(X_ref)).ravel()
    n = int(len(y_ref))
    n_riesgo = int(np.sum(np.asarray(y_ref) == RISK_LABEL))
    prob_ref, score_ref = _obtener_riesgo(modelo_final, X_ref)
    prob_ref = np.asarray(prob_ref, dtype=float)
    score_ref = np.asarray(score_ref, dtype=float)
    prob_ok = prob_ref[np.isfinite(prob_ref)]
    score_ok = score_ref[np.isfinite(score_ref)]
    base_pred = {
        "n": n,
        "count_0": n_riesgo,
        "count_1": int(n - n_riesgo),
        "rate_0": float(n_riesgo / n) if n else 0.0,
        "rate_1": float((n - n_riesgo) / n) if n else 0.0,
        "has_proba": bool(len(prob_ok) > 0),
        "has_score": bool(len(score_ok) > 0),
        "proba_edges": None,
        "proba_proportions": None,
        "proba_median": None,
        "score_edges": None,
        "score_proportions": None,
        "score_median": None,
    }
    if len(prob_ok):
        bordes_p, _ = _bordes_cuantiles(pd.Series(prob_ok), n_bins=int(n_bins))
        _, props_p, _ = _proporciones_histograma(pd.Series(prob_ok), bordes_p)
        base_pred["proba_edges"] = [float(b) for b in bordes_p]
        base_pred["proba_proportions"] = [float(p) for p in np.asarray(props_p).ravel()]
        base_pred["proba_median"] = float(np.median(prob_ok))
    if len(score_ok):
        bordes_s, _ = _bordes_cuantiles(pd.Series(score_ok), n_bins=int(n_bins))
        _, props_s, _ = _proporciones_histograma(pd.Series(score_ok), bordes_s)
        base_pred["score_edges"] = [float(b) for b in bordes_s]
        base_pred["score_proportions"] = [float(p) for p in np.asarray(props_s).ravel()]
        base_pred["score_median"] = float(np.median(score_ok))

    return PerfilReferencia(
        model_version=version,
        feature_schema_version=str(FEATURE_SCHEMA_VERSION),
        reference_source=REFERENCE_SOURCE,
        n_reference=n_ref,
        created_at=_utc_now_iso(),
        features=list(FEATURES_V1),
        numeric_profiles=perfiles_num,
        categorical_profiles=perfiles_cat,
        missing_rates=dict(missing_rates),
        prediction_baseline=base_pred,
        holdout_reference=holdout_ref,
    )


def guardar_perfil_referencia(perfil, ruta) -> Path:
    """Persist aggregates only as JSON. Never rows, PII or salaries."""
    if not isinstance(perfil, PerfilReferencia):
        raise TypeError("perfil must be a PerfilReferencia.")
    if ruta is None or str(ruta).strip() == "":
        raise ValueError("ruta is required.")
    destino = Path(ruta)
    if destino.suffix.lower() != ".json":
        destino = destino / BASELINE_FILENAME
    destino.parent.mkdir(parents=True, exist_ok=True)
    datos = _perfil_a_dict(perfil)
    texto = json.dumps(datos, ensure_ascii=False, indent=2)
    if "salario_cliente=" in texto.lower():
        raise ValueError("Baseline must not contain row-level values.")
    with open(destino, "w", encoding="utf-8") as archivo:
        archivo.write(texto)
    return destino


def cargar_perfil_referencia(ruta) -> PerfilReferencia:
    """Load a baseline file created by guardar_perfil_referencia."""
    if ruta is None or str(ruta).strip() == "":
        raise ValueError("ruta is required.")
    origen = Path(ruta)
    if not origen.is_file():
        raise FileNotFoundError(f"Baseline file not found: {origen}")
    with open(origen, encoding="utf-8") as archivo:
        datos = json.load(archivo)
    datos = _restaurar_bordes(datos)
    requeridas = {
        "model_version",
        "feature_schema_version",
        "reference_source",
        "n_reference",
        "created_at",
        "features",
        "numeric_profiles",
        "categorical_profiles",
        "missing_rates",
        "prediction_baseline",
    }
    faltantes = requeridas - set(datos.keys())
    if faltantes:
        raise ValueError(f"Baseline file misses keys: {sorted(faltantes)}")
    return PerfilReferencia(
        model_version=str(datos["model_version"]),
        feature_schema_version=str(datos["feature_schema_version"]),
        reference_source=str(datos["reference_source"]),
        n_reference=int(datos["n_reference"]),
        created_at=str(datos["created_at"]),
        features=list(datos["features"]),
        numeric_profiles=dict(datos["numeric_profiles"]),
        categorical_profiles=dict(datos["categorical_profiles"]),
        missing_rates=dict(datos["missing_rates"]),
        prediction_baseline=dict(datos["prediction_baseline"]),
        holdout_reference=(
            dict(datos["holdout_reference"])
            if isinstance(datos.get("holdout_reference"), dict)
            else None
        ),
    )


def analizar_calidad_datos(df_crudo) -> dict:
    """Describe raw batch quality before feature engineering.

    Returns errors that block monitoring versus warnings that allow it to
    continue. The serving authority remains preparar_features_inferencia.
    """
    if not isinstance(df_crudo, pd.DataFrame):
        raise TypeError("df_crudo must be a pd.DataFrame.")
    n_rows = int(len(df_crudo))
    columnas = [str(c) for c in df_crudo.columns]
    errores: list = []
    avisos: list = []
    if n_rows == 0:
        errores.append("Empty batch: no rows to monitor.")
    n_dup = int(df_crudo.duplicated().sum()) if n_rows else 0
    if n_dup > 0:
        avisos.append(f"{n_dup} duplicated raw rows detected (kept, reported).")
    extras = [c for c in columnas if c not in COLUMNAS_CRUDAS_CONOCIDAS]
    if extras:
        avisos.append(f"Extra raw columns ignored by serving: {sorted(extras)}.")
    missing_raw = {}
    for columna in columnas:
        try:
            missing_raw[str(columna)] = float(df_crudo[columna].isna().mean())
        except Exception:
            missing_raw[str(columna)] = None
    estado_features = "not_checked"
    detalle_features = ""
    if n_rows > 0:
        try:
            probada = preparar_features_inferencia(df_crudo)
            if list(probada.columns) != list(FEATURES_V1):
                errores.append("Serving output column order mismatch.")
            else:
                estado_features = "ok"
        except Exception as exc:
            estado_features = "error"
            errores.append(f"Serving blocked by raw schema: {exc}")
            detalle_features = str(exc)
    estado = "ok"
    if errores:
        estado = "error"
    elif avisos:
        estado = "warning"
    return {
        "status": estado,
        "errors": list(errores),
        "warnings": list(avisos),
        "n_rows": n_rows,
        "n_columns": int(len(columnas)),
        "columns": columnas,
        "n_duplicated": n_dup,
        "duplicated_rate": float(n_dup / n_rows) if n_rows else 0.0,
        "missing_rate_raw": dict(missing_raw),
        "extra_columns": sorted(extras),
        "serving_check": estado_features,
        "serving_detail": detalle_features,
    }


def calcular_psi_numerico(serie_referencia, serie_batch, bordes, epsilon=1e-6) -> dict:
    """PSI for one numeric feature reusing frozen reference edges.

    Nulls are excluded from PSI (missingness is tracked separately).

    Special treatment for degenerate reference distributions: a constant
    reference collapses frozen edges to a single [-inf, +inf] bin. Instead
    of silent PSI 0, the median of the finite non-null reference values is
    used as reference_value and the batch equal-share versus
    different-share is scored with the same smoothed categorical PSI
    (epsilon plus renormalization). Identical batches give ~0, any real
    divergence gives finite >0 without inf/nan, and strong divergence can
    reach warning/critical. No alert is hardcoded for being constant.
    """
    bordes = [float(b) for b in np.asarray(bordes, dtype=float).ravel()]
    if len(bordes) < 2:
        raise ValueError("bordes need at least two edges.")
    degenerado = len(bordes) <= 2 and bordes[0] == -float("inf") and bordes[-1] == float("inf")
    _, esp_hist, n_ref = _proporciones_histograma(serie_referencia, bordes, epsilon=epsilon)
    _, act_hist, n_batch = _proporciones_histograma(serie_batch, bordes, epsilon=epsilon)
    if degenerado:
        valor_ref = _valor_referencia_numerica(serie_referencia)
        psi = _psi_degenerado_numerico(valor_ref, serie_batch, epsilon=epsilon)
        detalle = calcular_psi_categorico(
            {"equal": 1.0, "different": 0.0},
            _proporciones_igual_diferente(valor_ref, serie_batch),
            epsilon=epsilon,
        )
        esperado = [
            float(detalle["expected"].get("equal", 0.0)),
            float(detalle["expected"].get("different", 0.0)),
        ]
        actual = [
            float(detalle["actual"].get("equal", 0.0)),
            float(detalle["actual"].get("different", 0.0)),
        ]
        return {
            "psi": float(psi),
            "degenerate": True,
            "n_reference": int(n_ref),
            "n_batch": int(n_batch),
            "edges": list(bordes),
            "expected": esperado,
            "actual": actual,
        }
    psi = _psi_desde_proporciones(esp_hist, act_hist)
    if not np.isfinite(psi):
        psi = 0.0
    return {
        "psi": float(psi),
        "degenerate": bool(degenerado),
        "n_reference": int(n_ref),
        "n_batch": int(n_batch),
        "edges": list(bordes),
        "expected": [float(v) for v in np.asarray(esp_hist).ravel()],
        "actual": [float(v) for v in np.asarray(act_hist).ravel()],
    }


def _proporciones_igual_diferente(valor_referencia, serie_batch) -> dict:
    """Batch shares equal to the frozen constant versus different values.

    Nulls are excluded (missingness is tracked separately). Non-finite
    batch values count as different. Empty batches return equal shares
    so PSI stays 0 instead of failing.
    """
    if valor_referencia is None:
        return {"equal": 1.0, "different": 0.0}
    ref = float(valor_referencia)
    if not np.isfinite(ref):
        return {"equal": 1.0, "different": 0.0}
    vals = pd.to_numeric(pd.Series(serie_batch), errors="coerce").dropna()
    if len(vals) == 0:
        return {"equal": 1.0, "different": 0.0}
    arr = vals.to_numpy(dtype=float)
    total = int(len(arr))
    iguales = int(np.sum(arr == ref))
    p_equal = float(iguales / total) if total else 1.0
    return {"equal": p_equal, "different": 1.0 - p_equal}


def _psi_degenerado_numerico(valor_referencia, serie_batch, epsilon=1e-6) -> float:
    """PSI for a degenerate numeric reference via equal-vs-different shares.

    Special path for reference distributions with zero variance: the frozen
    reference is 100% equal to reference_value, the batch is scored by its
    observed equal/different proportions with the same smoothed categorical
    PSI (epsilon plus renormalization) used elsewhere. Returns finite PSI,
    0.0 when there is no baseline or no batch data. Never inf/nan.
    """
    detalle = calcular_psi_categorico(
        {"equal": 1.0, "different": 0.0},
        _proporciones_igual_diferente(valor_referencia, serie_batch),
        epsilon=epsilon,
    )
    psi = float(detalle.get("psi", 0.0))
    if not np.isfinite(psi):
        return 0.0
    return float(psi)


def calcular_psi_categorico(props_referencia, props_batch, epsilon=1e-6) -> dict:
    """PSI for one categorical feature over the union of categories.

    Nulls are excluded (missingness is separate). The string
    Ausente o invalida is a real category and does take part here.
    Smoothing plus renormalization keeps the formula free of inf/nan.
    """
    if isinstance(props_referencia, pd.Series):
        props_referencia = props_referencia.to_dict()
    if isinstance(props_batch, pd.Series):
        props_batch = props_batch.to_dict()
    if not isinstance(props_referencia, dict) or not isinstance(props_batch, dict):
        raise TypeError("props must be dicts or Series.")
    ref = {str(k): float(v) for k, v in props_referencia.items()}
    bat = {str(k): float(v) for k, v in props_batch.items()}
    union = sorted(set(ref) | set(bat))
    if not union:
        return {"psi": 0.0, "categories": [], "expected": {}, "actual": {}}
    suma_ref = sum(ref.values())
    suma_bat = sum(bat.values())
    if suma_ref > 0:
        ref = {k: v / suma_ref for k, v in ref.items()}
    if suma_bat > 0:
        bat = {k: v / suma_bat for k, v in bat.items()}
    n = len(union)
    esp = np.array([ref.get(c, 0.0) for c in union], dtype=float)
    act = np.array([bat.get(c, 0.0) for c in union], dtype=float)
    esp_s = (esp + float(epsilon)) / (1.0 + float(epsilon) * float(n))
    act_s = (act + float(epsilon)) / (1.0 + float(epsilon) * float(n))
    esp_s = esp_s / float(esp_s.sum())
    act_s = act_s / float(act_s.sum())
    psi = _psi_desde_proporciones(esp_s, act_s)
    if not np.isfinite(psi):
        psi = 0.0
    return {
        "psi": float(psi),
        "categories": list(union),
        "expected": {c: float(v) for c, v in zip(union, esp_s, strict=True)},
        "actual": {c: float(v) for c, v in zip(union, act_s, strict=True)},
    }


def analizar_feature_drift(
    perfil,
    X_batch,
    psi_warning: float = PSI_WARNING_DEFAULT,
    psi_critical: float = PSI_CRITICAL_DEFAULT,
    missing_delta_warning: float = MISSING_DELTA_WARNING_DEFAULT,
) -> dict:
    """Compare batch feature distributions against the frozen profile.

    Special treatment for degenerate numeric references: when the frozen
    profile marks a numeric feature as degenerate (constant reference with
    a stored reference_value), drift is measured as batch equal-share
    versus different-share with the same smoothed categorical PSI
    (epsilon plus renormalization). Identical batches give ~0 without a
    hardcoded critical alert; observed divergence drives warning/critical
    through the existing thresholds. Nulls stay in missingness, not in PSI.
    """
    if not isinstance(perfil, PerfilReferencia):
        raise TypeError("perfil must be a PerfilReferencia.")
    if not isinstance(X_batch, pd.DataFrame):
        raise TypeError("X_batch must be a pd.DataFrame.")
    if len(X_batch) == 0:
        raise ValueError("X_batch is empty.")
    faltantes = [c for c in FEATURES_V1 if c not in X_batch.columns]
    if faltantes:
        raise KeyError(f"X_batch misses v1 features: {faltantes}")
    Xb = X_batch[list(FEATURES_V1)].copy()
    por_feature: dict = {}
    nuevas: list = []
    for columna in FEATURES_V1:
        serie = Xb[columna]
        tasa_batch = float(serie.isna().mean()) if len(serie) else 0.0
        if columna in FEATURES_NUMERICAS:
            base = perfil.numeric_profiles.get(str(columna), {})
            tasa_ref = float(base.get("missing_rate_reference", 0.0))
            bordes = base.get("edges", [-float("inf"), float("inf")])
            if bool(base.get("degenerate", False)):
                valor_ref = base.get("reference_value", None)
                if valor_ref is None:
                    valor_ref = base.get("median", None)
                psi_val = float(_psi_degenerado_numerico(valor_ref, serie, epsilon=EPSILON_DEFAULT))
                if not np.isfinite(psi_val):
                    psi_val = 0.0
            else:
                # Reference histogram uses stored frozen proportions directly.
                esperadas = np.asarray(base.get("proportions", []), dtype=float)
                _, actuales, _ = _proporciones_histograma(serie, bordes)
                if len(esperadas) == len(actuales) and len(esperadas) > 0:
                    psi_val = _psi_desde_proporciones(esperadas, actuales)
                    if not np.isfinite(psi_val):
                        psi_val = 0.0
                else:
                    psi_val = 0.0
            resumen = _resumen_numerico(serie)
            fuera = 0
            minimo = base.get("min")
            maximo = base.get("max")
            vals = pd.to_numeric(serie, errors="coerce").dropna()
            if len(vals) and minimo is not None and maximo is not None:
                fuera = int(((vals < float(minimo)) | (vals > float(maximo))).sum())
                fuera += int(np.sum(~np.isfinite(vals.to_numpy(dtype=float))))
            nivel = _nivel_psi(psi_val, psi_warning, psi_critical)
            por_feature[str(columna)] = {
                "kind": "numeric",
                "psi": float(psi_val),
                "level": nivel,
                "degenerate": bool(base.get("degenerate", False)),
                "missing_rate_reference": float(tasa_ref),
                "missing_rate_batch": float(tasa_batch),
                "missing_delta": float(tasa_batch - tasa_ref),
                "batch_summary": resumen,
                "out_of_reference_range": int(fuera),
                "edges": [float(b) for b in bordes],
            }
        elif columna in FEATURES_NOMINALES:
            base = perfil.categorical_profiles.get(str(columna), {})
            tasa_ref = float(base.get("missing_rate_reference", 0.0))
            ref_props = dict(base.get("proportions", {}))
            no_nulos = serie.dropna()
            if len(no_nulos):
                frec = no_nulos.astype("string").value_counts(dropna=True)
                total = float(frec.sum())
                bat_props = {str(k): float(v) / total for k, v in frec.items()}
                frecuencias = {str(k): int(v) for k, v in frec.items()}
            else:
                bat_props = {}
                frecuencias = {}
            psi_res = calcular_psi_categorico(ref_props, bat_props)
            psi_val = float(psi_res["psi"])
            nivel = _nivel_psi(psi_val, psi_warning, psi_critical)
            base_cats = set(ref_props)
            for categoria, prop in bat_props.items():
                if categoria not in base_cats:
                    nuevas.append(
                        {
                            "feature": str(columna),
                            "category": str(categoria),
                            "count": int(frecuencias.get(categoria, 0)),
                            "proportion": float(prop),
                        }
                    )
            por_feature[str(columna)] = {
                "kind": "categorical",
                "psi": float(psi_val),
                "level": nivel,
                "missing_rate_reference": float(tasa_ref),
                "missing_rate_batch": float(tasa_batch),
                "missing_delta": float(tasa_batch - tasa_ref),
                "categories_reference": sorted(ref_props),
                "categories_batch": sorted(bat_props),
                "frequencies_batch": dict(frecuencias),
            }
        else:
            raise ValueError(f"Unknown v1 feature: {columna}")
    return {
        "features": por_feature,
        "unseen_categories": list(nuevas),
        "psi_warning": float(psi_warning),
        "psi_critical": float(psi_critical),
        "missing_delta_warning": float(missing_delta_warning),
    }


def analizar_prediction_drift(
    perfil,
    resultado_prediccion,
    psi_warning: float = PSI_WARNING_DEFAULT,
    psi_critical: float = PSI_CRITICAL_DEFAULT,
) -> dict:
    """Compare batch prediction behavior with the frozen train baseline."""
    if not isinstance(perfil, PerfilReferencia):
        raise TypeError("perfil must be a PerfilReferencia.")
    marco = getattr(resultado_prediccion, "dataframe", None)
    if not isinstance(marco, pd.DataFrame):
        raise TypeError("resultado_prediccion must expose a DataFrame.")
    if "prediction" not in marco.columns:
        raise KeyError("resultado_prediccion lacks prediction column.")
    if len(marco) == 0:
        raise ValueError("resultado_prediccion is empty.")
    base = dict(perfil.prediction_baseline)
    y_batch = np.asarray(marco["prediction"]).ravel()
    n = int(len(y_batch))
    c0 = int(np.sum(np.asarray(y_batch) == RISK_LABEL))
    tasa0 = float(c0 / n) if n else 0.0
    tasa1 = float((n - c0) / n) if n else 0.0
    ref_props = {
        str(RISK_LABEL): float(base.get("rate_0", 0.0)),
        "1": float(base.get("rate_1", 0.0)),
    }
    bat_props = {str(RISK_LABEL): tasa0, "1": tasa1}
    psi_lab = calcular_psi_categorico(ref_props, bat_props)["psi"]
    nivel_lab = _nivel_psi(float(psi_lab), psi_warning, psi_critical)
    salida: dict = {
        "n": n,
        "count_0_batch": c0,
        "count_1_batch": int(n - c0),
        "rate_0_batch": tasa0,
        "rate_1_batch": tasa1,
        "rate_0_reference": float(base.get("rate_0", 0.0)),
        "rate_1_reference": float(base.get("rate_1", 0.0)),
        "rate_0_delta": float(tasa0 - float(base.get("rate_0", 0.0))),
        "psi_labels": float(psi_lab),
        "level_labels": nivel_lab,
        "psi_warning": float(psi_warning),
        "psi_critical": float(psi_critical),
        "proba": None,
        "score": None,
    }
    if "risk_probability" in marco.columns and base.get("proba_edges"):
        probs = pd.to_numeric(marco["risk_probability"], errors="coerce")
        probs_ok = probs.dropna()
        finitos = probs_ok[np.isfinite(probs_ok.to_numpy(dtype=float))]
        bordes = [float(b) for b in base["proba_edges"]]
        esperadas = np.asarray(base.get("proba_proportions", []), dtype=float)
        _, actuales, _ = _proporciones_histograma(probs_ok, bordes)
        if len(esperadas) == len(actuales) and len(esperadas):
            psi_p = _psi_desde_proporciones(esperadas, actuales)
        else:
            psi_p = 0.0
        if not np.isfinite(psi_p):
            psi_p = 0.0
        salida["proba"] = {
            "psi": float(psi_p),
            "level": _nivel_psi(float(psi_p), psi_warning, psi_critical),
            "median_batch": float(np.median(finitos.to_numpy(dtype=float)))
            if len(finitos)
            else None,
            "median_reference": base.get("proba_median"),
            "n_finite": int(len(finitos)),
        }
    if "risk_score" in marco.columns and base.get("score_edges"):
        scores = pd.to_numeric(marco["risk_score"], errors="coerce")
        scores_ok = scores.dropna()
        finitos = scores_ok[np.isfinite(scores_ok.to_numpy(dtype=float))]
        bordes = [float(b) for b in base["score_edges"]]
        esperadas = np.asarray(base.get("score_proportions", []), dtype=float)
        _, actuales, _ = _proporciones_histograma(scores_ok, bordes)
        if len(esperadas) == len(actuales) and len(esperadas):
            psi_s = _psi_desde_proporciones(esperadas, actuales)
        else:
            psi_s = 0.0
        if not np.isfinite(psi_s):
            psi_s = 0.0
        salida["score"] = {
            "psi": float(psi_s),
            "level": _nivel_psi(float(psi_s), psi_warning, psi_critical),
            "median_batch": float(np.median(finitos.to_numpy(dtype=float)))
            if len(finitos)
            else None,
            "median_reference": base.get("score_median"),
            "n_finite": int(len(finitos)),
        }
    return salida


def analizar_performance(resultado_prediccion, y_true, metricas_holdout_finales=None):
    """Describe batch skill only when labels are available.

    Compares against frozen holdout metrics with batch-minus-holdout deltas.
    Single-class batches do not fail; they use zero division and a
    representativeness note. Never proposes model changes.
    """
    if y_true is None:
        return None
    marco = getattr(resultado_prediccion, "dataframe", None)
    if not isinstance(marco, pd.DataFrame):
        raise TypeError("resultado_prediccion must expose a DataFrame.")
    if "prediction" not in marco.columns:
        raise KeyError("resultado_prediccion lacks prediction column.")
    y_pred = np.asarray(marco["prediction"]).ravel()
    if isinstance(y_true, pd.Series):
        y_arr = y_true.to_numpy()
    else:
        y_arr = np.asarray(y_true)
    try:
        y_arr = np.asarray(y_arr, dtype=int).ravel()
    except Exception as exc:
        raise ValueError(f"y_true must be binary integers: {exc}") from exc
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    if len(y_arr) != len(y_pred):
        raise ValueError("y_true length must match predictions.")
    if set(np.unique(y_arr).tolist()) - {0, 1}:
        raise ValueError("y_true must be binary {0, 1}.")
    tp0 = int(np.sum((y_arr == 0) & (y_pred == 0)))
    fp0 = int(np.sum((y_arr == 1) & (y_pred == 0)))
    fn0 = int(np.sum((y_arr == 0) & (y_pred == 1)))
    tn1 = int(np.sum((y_arr == 1) & (y_pred == 1)))
    fp1 = int(np.sum((y_arr == 0) & (y_pred == 1)))
    fn1 = int(np.sum((y_arr == 1) & (y_pred == 0)))
    precision_0 = float(tp0 / (tp0 + fp0)) if (tp0 + fp0) > 0 else 0.0
    recall_0 = float(tp0 / (tp0 + fn0)) if (tp0 + fn0) > 0 else 0.0
    f1_0 = (
        float(2 * precision_0 * recall_0 / (precision_0 + recall_0))
        if (precision_0 + recall_0) > 0
        else 0.0
    )
    recall_1 = float(tn1 / (tn1 + fn1)) if (tn1 + fn1) > 0 else 0.0
    balanced = float((recall_0 + recall_1) / 2.0)
    support_0 = int(np.sum(y_arr == 0))
    support_1 = int(np.sum(y_arr == 1))
    matriz = np.array([[tp0, fn0], [fp1, tn1]], dtype=int)
    nombres = {
        "riesgo_real_predicho_riesgo": tp0,
        "riesgo_real_predicho_pago": fn0,
        "pago_real_predicho_riesgo": fp1,
        "pago_real_predicho_pago": tn1,
    }
    notas: list = []
    if support_0 == 0 or support_1 == 0:
        notas.append(
            "Single-class batch: metrics use zero division and are not "
            "representative of both outcomes."
        )
    deltas: dict = {}
    if isinstance(metricas_holdout_finales, dict):
        for clave in ("precision_0", "recall_0", "f1_0", "balanced_accuracy"):
            if clave in metricas_holdout_finales:
                try:
                    ref = float(metricas_holdout_finales[clave])
                except Exception:
                    continue
                actual = {
                    "precision_0": precision_0,
                    "recall_0": recall_0,
                    "f1_0": f1_0,
                    "balanced_accuracy": balanced,
                }[clave]
                deltas[clave] = float(actual - ref)
    return {
        "precision_0": float(precision_0),
        "recall_0": float(recall_0),
        "f1_0": float(f1_0),
        "balanced_accuracy": float(balanced),
        "support_0": support_0,
        "support_1": support_1,
        "confusion_matrix": matriz,
        "confusion_names": dict(nombres),
        "deltas_vs_holdout": dict(deltas),
        "notes": list(notas),
        "n": int(len(y_arr)),
    }


def _validar_alineacion(df_crudo_batch, resultado_prediccion, y_true):
    marco = getattr(resultado_prediccion, "dataframe", None)
    if not isinstance(marco, pd.DataFrame):
        raise TypeError("resultado_prediccion must expose a DataFrame.")
    if len(df_crudo_batch) != len(marco):
        raise ValueError("Batch and prediction lengths differ; batches must not mix.")
    if list(df_crudo_batch.index) != list(marco.index):
        raise ValueError("Batch and prediction indexes differ; explicit alignment needed.")
    if y_true is None:
        return
    if isinstance(y_true, pd.Series):
        if len(y_true) != len(marco):
            raise ValueError("y_true length must match the batch.")
        if list(y_true.index) != list(marco.index):
            raise ValueError("y_true index must match the batch index and order.")
    else:
        if len(np.asarray(y_true)) != len(marco):
            raise ValueError("y_true length must match the batch.")


def monitorizar(
    perfil,
    df_crudo_batch,
    resultado_prediccion,
    y_true=None,
    psi_warning: float = PSI_WARNING_DEFAULT,
    psi_critical: float = PSI_CRITICAL_DEFAULT,
    missing_delta_warning: float = MISSING_DELTA_WARNING_DEFAULT,
    min_batch_size: int = MIN_BATCH_SIZE_DEFAULT,
    window_start=None,
    window_end=None,
):
    """Run the full window check and return descriptive results plus alerts."""
    if not isinstance(perfil, PerfilReferencia):
        raise TypeError("perfil must be a PerfilReferencia.")
    if not isinstance(df_crudo_batch, pd.DataFrame):
        raise TypeError("df_crudo_batch must be a pd.DataFrame.")
    version_pred = str(getattr(resultado_prediccion, "model_version", ""))
    if version_pred != str(perfil.model_version):
        raise ValueError(
            f"model_version mismatch: profile {perfil.model_version!r} "
            f"!= prediction {version_pred!r}."
        )
    if str(perfil.feature_schema_version) != str(FEATURE_SCHEMA_VERSION):
        raise ValueError(
            f"feature_schema mismatch: profile {perfil.feature_schema_version!r} "
            f"!= code {FEATURE_SCHEMA_VERSION!r}."
        )
    _validar_alineacion(df_crudo_batch, resultado_prediccion, y_true)

    calidad = analizar_calidad_datos(df_crudo_batch)
    if calidad["status"] == "error":
        raise ValueError(f"Raw quality blocks monitoring: {calidad['errors']}")

    X_batch = preparar_features_inferencia(df_crudo_batch)
    n_rows = int(len(X_batch))
    generado = _utc_now_iso()
    inicio = str(window_start) if window_start is not None else generado
    fin = str(window_end) if window_end is not None else generado
    umbrales = {
        "psi_warning": float(psi_warning),
        "psi_critical": float(psi_critical),
        "missing_delta_warning": float(missing_delta_warning),
        "min_batch_size": int(min_batch_size),
    }

    aviso_tamano = n_rows < int(min_batch_size)
    deriva = analizar_feature_drift(
        perfil,
        X_batch,
        psi_warning=float(psi_warning),
        psi_critical=float(psi_critical),
        missing_delta_warning=float(missing_delta_warning),
    )
    pred_deriva = analizar_prediction_drift(
        perfil,
        resultado_prediccion,
        psi_warning=float(psi_warning),
        psi_critical=float(psi_critical),
    )
    holdout_ref = (
        dict(perfil.holdout_reference) if isinstance(perfil.holdout_reference, dict) else None
    )
    desempeno = analizar_performance(resultado_prediccion, y_true, holdout_ref)

    alertas: list = []
    if aviso_tamano:
        alertas.append(
            AlertaMonitoring(
                level="warning",
                type="schema",
                feature=None,
                value=n_rows,
                threshold=int(min_batch_size),
                message=(
                    f"Small batch n={n_rows} below min_batch_size={min_batch_size}: "
                    "drift estimates are unreliable."
                ),
            )
        )
    for aviso in calidad.get("warnings", []):
        alertas.append(
            AlertaMonitoring(
                level="info",
                type="schema",
                feature=None,
                value=None,
                threshold=None,
                message=str(aviso),
            )
        )
    for nombre, info in deriva.get("features", {}).items():
        delta = abs(float(info.get("missing_delta", 0.0)))
        if delta >= float(missing_delta_warning):
            alertas.append(
                AlertaMonitoring(
                    level="warning",
                    type="missingness",
                    feature=str(nombre),
                    value=float(info.get("missing_rate_batch", 0.0)),
                    threshold=float(missing_delta_warning),
                    message=(
                        f"Missingness shift on {nombre}: "
                        f"ref={info.get('missing_rate_reference', 0.0):.4f} "
                        f"batch={info.get('missing_rate_batch', 0.0):.4f}."
                    ),
                )
            )
        nivel = str(info.get("level", "info"))
        if nivel in ("warning", "critical"):
            alertas.append(
                AlertaMonitoring(
                    level=nivel,
                    type="feature_drift",
                    feature=str(nombre),
                    value=float(info.get("psi", 0.0)),
                    threshold=float(psi_critical if nivel == "critical" else psi_warning),
                    message=(
                        f"Feature drift on {nombre}: PSI={info.get('psi', 0.0):.4f} ({nivel})."
                    ),
                )
            )
    for nueva in deriva.get("unseen_categories", []):
        alertas.append(
            AlertaMonitoring(
                level="warning",
                type="unseen_category",
                feature=str(nueva.get("feature")),
                value=float(nueva.get("proportion", 0.0)),
                threshold=None,
                message=(
                    f"Unseen category {nueva.get('category')!r} on "
                    f"{nueva.get('feature')} "
                    f"(count={nueva.get('count')}, "
                    f"prop={nueva.get('proportion', 0.0):.4f}). "
                    "Encoder ignores it, distribution still shifted."
                ),
            )
        )
    nivel_pred = str(pred_deriva.get("level_labels", "info"))
    if nivel_pred in ("warning", "critical"):
        alertas.append(
            AlertaMonitoring(
                level=nivel_pred,
                type="prediction_drift",
                feature=None,
                value=float(pred_deriva.get("psi_labels", 0.0)),
                threshold=float(psi_critical if nivel_pred == "critical" else psi_warning),
                message=(
                    f"Prediction drift: PSI={pred_deriva.get('psi_labels', 0.0):.4f} "
                    f"({nivel_pred}), rate_0 "
                    f"{pred_deriva.get('rate_0_reference', 0.0):.4f}->"
                    f"{pred_deriva.get('rate_0_batch', 0.0):.4f}."
                ),
            )
        )
    for clave in ("proba", "score"):
        bloque = pred_deriva.get(clave)
        if isinstance(bloque, dict) and str(bloque.get("level")) in ("warning", "critical"):
            lvl = str(bloque.get("level"))
            alertas.append(
                AlertaMonitoring(
                    level=lvl,
                    type="prediction_drift",
                    feature=str(clave),
                    value=float(bloque.get("psi", 0.0)),
                    threshold=float(psi_critical if lvl == "critical" else psi_warning),
                    message=f"Risk {clave} distribution drift: PSI={bloque.get('psi', 0.0):.4f}.",
                )
            )
    if isinstance(desempeno, dict):
        for nota in desempeno.get("notes", []):
            alertas.append(
                AlertaMonitoring(
                    level="warning",
                    type="performance",
                    feature=None,
                    value=None,
                    threshold=None,
                    message=str(nota),
                )
            )

    return ResultadoMonitoring(
        window_start=inicio,
        window_end=fin,
        model_version=str(perfil.model_version),
        feature_schema_version=str(perfil.feature_schema_version),
        n_rows=n_rows,
        generated_at=generado,
        thresholds=dict(umbrales),
        quality=dict(calidad),
        feature_drift=dict(deriva),
        unseen_categories=list(deriva.get("unseen_categories", [])),
        prediction_drift=dict(pred_deriva),
        performance=None if desempeno is None else dict(desempeno),
        alertas=list(alertas),
        holdout_reference=None if holdout_ref is None else dict(holdout_ref),
        batch_size_warning=bool(aviso_tamano),
    )


def plot_psi_features(resultado):
    """Bar chart of PSI per feature with warning/critical guides."""
    import matplotlib.pyplot as plt

    datos = resultado if isinstance(resultado, dict) else resultado_a_dict(resultado)
    deriva = datos.get("feature_drift", {}).get("features", {})
    nombres = sorted(deriva)
    valores = [float(deriva[c].get("psi", 0.0)) for c in nombres]
    umbrales = datos.get("thresholds", {})
    aviso = float(umbrales.get("psi_warning", PSI_WARNING_DEFAULT))
    critico = float(umbrales.get("psi_critical", PSI_CRITICAL_DEFAULT))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(nombres, valores)
    ax.axhline(aviso, linestyle="--", label=f"warning={aviso}")
    ax.axhline(critico, linestyle=":", label=f"critical={critico}")
    ax.set_ylabel("PSI")
    ax.set_title("Feature PSI vs reference")
    ax.legend()
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return fig


def plot_missingness(resultado):
    """Grouped bars of reference versus batch missing rates."""
    import matplotlib.pyplot as plt

    datos = resultado if isinstance(resultado, dict) else resultado_a_dict(resultado)
    deriva = datos.get("feature_drift", {}).get("features", {})
    nombres = sorted(deriva)
    ref = [float(deriva[c].get("missing_rate_reference", 0.0)) for c in nombres]
    bat = [float(deriva[c].get("missing_rate_batch", 0.0)) for c in nombres]
    x = np.arange(len(nombres))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - 0.2, ref, width=0.4, label="reference")
    ax.bar(x + 0.2, bat, width=0.4, label="batch")
    ax.set_xticks(x, nombres)
    ax.set_ylabel("Missing rate")
    ax.set_title("Missingness reference vs batch")
    ax.legend()
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return fig


def plot_prediction_distribution(resultado):
    """Batch versus reference prediction rates."""
    import matplotlib.pyplot as plt

    datos = resultado if isinstance(resultado, dict) else resultado_a_dict(resultado)
    deriva = datos.get("prediction_drift", {})
    etiquetas = ["rate_0", "rate_1"]
    ref = [float(deriva.get("rate_0_reference", 0.0)), float(deriva.get("rate_1_reference", 0.0))]
    bat = [float(deriva.get("rate_0_batch", 0.0)), float(deriva.get("rate_1_batch", 0.0))]
    x = np.arange(len(etiquetas))
    fig, ax = plt.subplots()
    ax.bar(x - 0.2, ref, width=0.4, label="reference")
    ax.bar(x + 0.2, bat, width=0.4, label="batch")
    ax.set_xticks(x, etiquetas)
    ax.set_ylabel("Rate")
    ax.set_title("Prediction distribution")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_performance(resultado):
    """Batch metrics versus frozen holdout reference when both exist."""
    import matplotlib.pyplot as plt

    datos = resultado if isinstance(resultado, dict) else resultado_a_dict(resultado)
    desempeno = datos.get("performance") or {}
    referencia = datos.get("holdout_reference") or {}
    metricas = ["precision_0", "recall_0", "f1_0", "balanced_accuracy"]
    bat = [float(desempeno.get(m, 0.0)) for m in metricas]
    ref = [float(referencia.get(m, 0.0)) if m in referencia else 0.0 for m in metricas]
    x = np.arange(len(metricas))
    fig, ax = plt.subplots()
    ax.bar(x - 0.2, ref, width=0.4, label="holdout reference")
    ax.bar(x + 0.2, bat, width=0.4, label="batch")
    ax.set_xticks(x, metricas)
    ax.set_ylabel("Score")
    ax.set_title("Batch performance vs holdout")
    ax.legend()
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    return fig


def main() -> None:
    """Academic demo with stable and perturbed windows, labels unused.

    Standalone predictions come from an in-memory fitted Pipeline, so the
    reference profile uses model_version in-memory to stay coherent with
    model_deploy.predecir without mutating the Pipeline.
    """
    from mlops_pipeline.src.Cargar_datos import (
        cargar_config,
        cargar_datos,
        encontrar_raiz_proyecto,
        resolver_ruta_datos,
    )
    from mlops_pipeline.src.ft_engineering import preparar_datos
    from mlops_pipeline.src.model_deploy import crear_metadata_modelo, predecir
    from mlops_pipeline.src.model_training import entrenar

    raiz = encontrar_raiz_proyecto()
    config = cargar_config(raiz / "mlops_pipeline" / "src" / "config.json")
    df_completo = cargar_datos(resolver_ruta_datos(raiz, config), config)
    datos = preparar_datos(df_completo)
    resultado_ent = entrenar(datos.X_train, datos.y_train, datos.preprocessor)
    modelo = resultado_ent.modelo_final_fitted
    pasos = getattr(modelo, "named_steps", {})
    clase_nombre = type(pasos.get("model", modelo)).__name__
    metadata = crear_metadata_modelo(
        nombre_modelo=str(resultado_ent.nombre_modelo_final),
        clase_estimator=clase_nombre,
        hiperparametros_finales=dict(resultado_ent.hiperparametros_finales),
        resultado_cv_modelo_final=resultado_ent.resultado_cv_modelo_final,
        metricas_holdout_finales=None,
        model_version="in-memory",
        random_state=int(resultado_ent.random_state),
        notas="Monitoring academic demo without labels.",
    )
    perfil = crear_perfil_referencia(datos.X_train, modelo, metadata)

    idx_estable = list(datos.X_train.index[:300])
    df_estable = df_completo.loc[idx_estable].copy()
    if TARGET in df_estable.columns:
        df_estable = df_estable.drop(columns=[TARGET])
    salida_estable = predecir(modelo, df_estable)
    res_estable = monitorizar(perfil, df_estable, salida_estable)

    df_perturbada = df_estable.copy()
    capital = pd.to_numeric(df_perturbada["capital_prestado"], errors="coerce")
    df_perturbada["capital_prestado"] = (capital * 2.0).astype("string")
    n_pert = len(df_perturbada)
    rng = np.random.default_rng(7)
    idx_nan = rng.choice(n_pert, size=max(1, n_pert // 5), replace=False)
    col_sal = df_perturbada.columns.get_loc("salario_cliente")
    for i in list(idx_nan):
        df_perturbada.iat[int(i), int(col_sal)] = pd.NA
    idx_nueva = rng.choice(n_pert, size=max(1, n_pert // 10), replace=False)
    col_lab = df_perturbada.columns.get_loc("tipo_laboral")
    for i in list(idx_nueva):
        df_perturbada.iat[int(i), int(col_lab)] = "NuevaCategoriaDemo"
    salida_pert = predecir(modelo, df_perturbada)
    res_pert = monitorizar(perfil, df_perturbada, salida_pert)

    def _max_psi(res):
        vals = [v.get("psi", 0.0) for v in res.feature_drift.get("features", {}).values()]
        return float(max(vals)) if vals else 0.0

    print("=== Monitoring demo (academic simulation, no labels) ===")
    print(f"model_version: {perfil.model_version}")
    print(f"reference_source: {perfil.reference_source}")
    print(f"n_reference: {perfil.n_reference}")
    print(f"stable n={res_estable.n_rows} max_PSI={_max_psi(res_estable):.4f}")
    print(f"perturbed n={res_pert.n_rows} max_PSI={_max_psi(res_pert):.4f}")
    print(f"stable unseen={len(res_estable.unseen_categories)}")
    print(f"perturbed unseen={len(res_pert.unseen_categories)}")
    print(f"stable alerts={len(res_estable.alertas)}")
    print(f"perturbed alerts={len(res_pert.alertas)}")
    print(
        "stable rate_0 "
        f"{res_estable.prediction_drift.get('rate_0_batch', 0.0):.4f} "
        "perturbed rate_0 "
        f"{res_pert.prediction_drift.get('rate_0_batch', 0.0):.4f}"
    )


if __name__ == "__main__":
    main()
