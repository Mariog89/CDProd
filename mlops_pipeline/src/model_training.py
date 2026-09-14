"""Comparación CV-safe y selección de familia ML para riesgo crediticio.

Trabaja exclusivamente con X_train/y_train y un preprocessor UNFITTED:
cada pipeline clona el preprocessor para que cada fold ajuste imputación
y encoding solo con su train fold. El holdout X_eval/y_eval jamás se usa.

Métrica primaria: F1 de la clase de riesgo (Pago_atiempo = 0).
"""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from mlops_pipeline.src.ft_engineering import (
    FEATURES_NOMINALES,
    FEATURES_NUMERICAS,
    RISK_LABEL,
)
from mlops_pipeline.src.hueristic_model import HeuristicModel

RANDOM_STATE = 42
N_SPLITS = 5

FAMILIAS_ML = ("logistic_regression", "random_forest", "hist_gradient_boosting")

ESPACIOS_TUNING = {
    "logistic_regression": {"model__C": [0.1, 1.0, 10.0]},
    "random_forest": {
        "model__n_estimators": [200, 500],
        "model__max_depth": [None, 12],
        "model__min_samples_leaf": [1, 5],
    },
    "hist_gradient_boosting": {
        "model__max_iter": [200, 400],
        "model__max_leaf_nodes": [31, 63],
        "model__learning_rate": [0.05, 0.1],
        "model__l2_regularization": [0.0, 10.0],
    },
}

SCORERS = {
    "precision_0": make_scorer(precision_score, pos_label=RISK_LABEL, zero_division=0),
    "recall_0": make_scorer(recall_score, pos_label=RISK_LABEL, zero_division=0),
    "f1_0": make_scorer(f1_score, pos_label=RISK_LABEL, zero_division=0),
    "balanced_accuracy": "balanced_accuracy",
}


def crear_folds(X: pd.DataFrame, y: pd.Series) -> list:
    """Materializa UNA VEZ los 5 folds estratificados reproducibles."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return list(cv.split(X, y))


def crear_scorers() -> dict:
    """Scorers explícitos para la clase de riesgo 0."""
    return dict(SCORERS)


def clonar_preprocesador_con_escala(
    preprocessor: ColumnTransformer,
) -> ColumnTransformer:
    """Clona el preprocessor añadiendo StandardScaler solo a la rama numérica.

    No modifica el preprocessor original.
    """
    _ = preprocessor
    nuevo = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(FEATURES_NUMERICAS),
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                list(FEATURES_NOMINALES),
            ),
        ]
    )
    nuevo.set_output(transform="pandas")
    return nuevo


def crear_modelos_candidatos(preprocessor: ColumnTransformer) -> dict:
    """Construye los 3 pipelines ML base, cada uno con su clone."""
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", clonar_preprocesador_con_escala(preprocessor)),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        max_iter=2000,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    RandomForestClassifier(
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def crear_baselines() -> dict:
    """Heurístico de negocio + dummy técnico (fuera de la selección)."""
    return {
        "heuristic": HeuristicModel(),
        "dummy": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
    }


@dataclass
class ResultadoCV:
    """Métricas por fold + resumen de un candidato."""

    nombre: str
    metricas_por_fold: dict
    mean_precision_0: float
    std_precision_0: float
    mean_recall_0: float
    std_recall_0: float
    mean_f1_0: float
    std_f1_0: float
    mean_balanced_accuracy: float
    std_balanced_accuracy: float
    mean_cv_fit_time: float
    std_cv_fit_time: float
    mean_cv_score_time: float
    std_cv_score_time: float


def _resumir(nombre: str, cv_out: dict) -> ResultadoCV:
    def media(clave: str) -> float:
        return float(np.mean(cv_out[clave]))

    def desv(clave: str) -> float:
        return float(np.std(cv_out[clave]))

    return ResultadoCV(
        nombre=nombre,
        metricas_por_fold={
            "precision_0": [float(v) for v in cv_out["test_precision_0"]],
            "recall_0": [float(v) for v in cv_out["test_recall_0"]],
            "f1_0": [float(v) for v in cv_out["test_f1_0"]],
            "balanced_accuracy": [float(v) for v in cv_out["test_balanced_accuracy"]],
        },
        mean_precision_0=media("test_precision_0"),
        std_precision_0=desv("test_precision_0"),
        mean_recall_0=media("test_recall_0"),
        std_recall_0=desv("test_recall_0"),
        mean_f1_0=media("test_f1_0"),
        std_f1_0=desv("test_f1_0"),
        mean_balanced_accuracy=media("test_balanced_accuracy"),
        std_balanced_accuracy=desv("test_balanced_accuracy"),
        mean_cv_fit_time=media("fit_time"),
        std_cv_fit_time=desv("fit_time"),
        mean_cv_score_time=media("score_time"),
        std_cv_score_time=desv("score_time"),
    )


def evaluar_cv(
    nombre: str,
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list,
) -> ResultadoCV:
    """Ejecuta cross_validate con folds materializados y scorers de clase 0."""
    cv_out = cross_validate(
        estimator,
        X,
        y,
        cv=folds,
        scoring=crear_scorers(),
        n_jobs=1,
        return_train_score=False,
        error_score="raise",
    )
    return _resumir(nombre, cv_out)


def comparar_modelos(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    folds: list | None = None,
) -> tuple[dict, ResultadoCV]:
    """CV comparable de heurístico + 3 familias base + dummy aparte."""
    folds = folds if folds is not None else crear_folds(X_train, y_train)
    candidatos = crear_modelos_candidatos(preprocessor)
    comparativa = {}
    for nombre in [*candidatos, *crear_baselines()]:
        estimator = candidatos[nombre] if nombre in candidatos else crear_baselines()[nombre]
        comparativa[nombre] = evaluar_cv(nombre, estimator, X_train, y_train, folds)
    dummy = comparativa.pop("dummy")
    return comparativa, dummy


def seleccionar_modelo(comparativa: dict) -> tuple[str, str]:
    """Elige familia ML con F1 primario y desempate multidimensional auditable.

    1. El mejor mean_f1_0 marca la referencia.
    2. Heurística de comparabilidad práctica basada en la variabilidad
       observada entre folds: son comparables las familias cuya diferencia
       con la referencia cumple diferencia_f1 <= min(std_ref, std_candidata).
       No es prueba estadística ni criterio de equivalencia formal: no prueba
       igualdad estadística y no reemplaza ANOVA/Friedman posteriores; solo
       evita seleccionar automáticamente el máximo F1 cuando la diferencia
       está dentro del ruido CV observado.
    3. Entre comparables gana: mayor recall_0 de riesgo (Performance); luego
       menor std_f1_0 / estabilidad (Consistency); luego menor costo CV
       (fit_time / Scalability).
    Sin fórmula ponderada: la decisión final sigue siendo auditable mediante
    Performance + Consistency + Scalability y la justificación textual
    registra el criterio.
    """
    candidatas = {k: v for k, v in comparativa.items() if k in FAMILIAS_ML}
    if len(candidatas) != len(FAMILIAS_ML):
        raise ValueError("La comparativa debe contener las 3 familias ML.")
    referencia = max(candidatas, key=lambda k: candidatas[k].mean_f1_0)
    ref = candidatas[referencia]
    comparables = []
    for nombre, r in candidatas.items():
        ruido = min(ref.std_f1_0, r.std_f1_0)
        if (ref.mean_f1_0 - r.mean_f1_0) <= ruido:
            comparables.append(nombre)
    orden = sorted(
        comparables,
        key=lambda k: (
            -candidatas[k].mean_recall_0,
            candidatas[k].std_f1_0,
            candidatas[k].mean_cv_fit_time,
            -candidatas[k].mean_f1_0,
        ),
    )
    elegido = orden[0]
    lineas = [
        f"Referencia F1: {referencia}={ref.mean_f1_0:.4f}±{ref.std_f1_0:.4f}.",
        f"Comparables dentro del ruido CV: {sorted(comparables)}.",
        f"Elegido: {elegido} (recall_0, estabilidad y costo como desempate).",
    ]
    for nombre in sorted(candidatas, key=lambda k: candidatas[k].mean_f1_0, reverse=True):
        r = candidatas[nombre]
        lineas.append(
            f"{nombre}: f1={r.mean_f1_0:.4f}±{r.std_f1_0:.4f}, "
            f"recall_0={r.mean_recall_0:.4f}±{r.std_recall_0:.4f}, "
            f"fit={r.mean_cv_fit_time:.2f}s"
        )
    return elegido, "\n".join(lineas)


def _contar_combinaciones(espacio: dict) -> int:
    total = 1
    for valores in espacio.values():
        total *= len(valores)
    return total


def ajustar_hiperparametros(
    nombre: str,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    folds: list | None = None,
):
    """Tunea SOLO la familia elegida, exclusivamente con X_train."""
    if nombre not in FAMILIAS_ML:
        raise ValueError("Solo se tunea una de las 3 familias ML.")
    folds = folds if folds is not None else crear_folds(X_train, y_train)
    espacio = ESPACIOS_TUNING[nombre]
    base = crear_modelos_candidatos(preprocessor)[nombre]
    if _contar_combinaciones(espacio) <= 12:
        search = GridSearchCV(
            base,
            espacio,
            scoring=crear_scorers()["f1_0"],
            cv=folds,
            n_jobs=1,
            error_score="raise",
        )
    else:
        search = RandomizedSearchCV(
            base,
            espacio,
            n_iter=10,
            scoring=crear_scorers()["f1_0"],
            cv=folds,
            random_state=RANDOM_STATE,
            n_jobs=1,
            error_score="raise",
        )
    search.fit(X_train, y_train)
    return search


def _entrenar_cv_final(search, X_train, y_train, folds) -> ResultadoCV:
    return evaluar_cv("final_tuned", deepcopy(search.best_estimator_), X_train, y_train, folds)


@dataclass
class ResultadoEntrenamiento:
    """Paquete estructurado para model_evaluation.py."""

    comparativa_cv_familias: dict
    resultado_dummy: ResultadoCV
    resultado_cv_modelo_final: ResultadoCV
    modelo_final_fitted: object
    nombre_modelo_final: str
    hiperparametros_finales: dict
    justificacion_seleccion: str
    random_state: int = RANDOM_STATE
    n_splits: int = N_SPLITS


def entrenar(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
) -> ResultadoEntrenamiento:
    """Protocolo completo: compara, selecciona, tunea, valida CV y ajusta final."""
    X_train = X_train.copy()
    y_train = y_train.copy()
    folds = crear_folds(X_train, y_train)
    comparativa, dummy = comparar_modelos(X_train, y_train, preprocessor, folds)
    elegido, justificacion = seleccionar_modelo(comparativa)
    search = ajustar_hiperparametros(elegido, preprocessor, X_train, y_train, folds)
    resultado_final = _entrenar_cv_final(search, X_train, y_train, folds)
    modelo_final = deepcopy(search.best_estimator_)
    modelo_final.fit(X_train, y_train)
    check_is_fitted(modelo_final)
    return ResultadoEntrenamiento(
        comparativa_cv_familias=comparativa,
        resultado_dummy=dummy,
        resultado_cv_modelo_final=resultado_final,
        modelo_final_fitted=modelo_final,
        nombre_modelo_final=elegido,
        hiperparametros_finales=dict(search.best_params_),
        justificacion_seleccion=justificacion,
        random_state=RANDOM_STATE,
        n_splits=N_SPLITS,
    )
