"""Métricas de la Fase 2 calculadas igual para todos los enfoques.

La clase de riesgo es 0 (Pago_atiempo = 0). Las funciones reciben el target
real y un score de riesgo (probabilidad de clase 0), de modo que el modelo
de Python local, el de Vertex Pipelines y el de AutoML se miden con la misma
regla sobre el mismo holdout.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

RISK_LABEL = 0


def predecir_con_umbral(score_riesgo, umbral: float) -> np.ndarray:
    """Clase 0 (riesgo) si el score de riesgo >= umbral, si no clase 1."""
    return np.where(np.asarray(score_riesgo, dtype=float) >= umbral, 0, 1)


def metricas_clasificacion(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    kw = {"pos_label": RISK_LABEL, "zero_division": 0}
    return {
        "f1_0": float(f1_score(y_true, y_pred, **kw)),
        "recall_0": float(recall_score(y_true, y_pred, **kw)),
        "precision_0": float(precision_score(y_true, y_pred, **kw)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "predichos_riesgo": int((y_pred == RISK_LABEL).sum()),
    }


def metricas_probabilisticas(y_true, score_riesgo) -> dict:
    y_riesgo = (np.asarray(y_true).astype(int) == RISK_LABEL).astype(int)
    s = np.asarray(score_riesgo, dtype=float)
    return {
        "roc_auc_risk": float(roc_auc_score(y_riesgo, s)),
        "average_precision_risk": float(average_precision_score(y_riesgo, s)),
        # Equivalente a un RMSE de probabilidad (Brier) para la clase de riesgo.
        "rmse_prob_riesgo": float(np.sqrt(np.mean((s - y_riesgo) ** 2))),
    }


def umbral_max_f1(y_true, score_riesgo) -> float:
    """Umbral que maximiza F1 de la clase de riesgo (usar SOLO en validación)."""
    s = np.asarray(score_riesgo, dtype=float)
    candidatos = np.unique(np.quantile(s, np.linspace(0.0, 1.0, 201)))
    mejores = max(
        candidatos,
        key=lambda u: f1_score(
            np.asarray(y_true).astype(int),
            predecir_con_umbral(s, u),
            pos_label=RISK_LABEL,
            zero_division=0,
        ),
    )
    return float(mejores)


def evaluar_scores(y_true, score_riesgo, umbral: float) -> dict:
    """Métricas probabilísticas + de clasificación a un umbral dado."""
    salida = metricas_probabilisticas(y_true, score_riesgo)
    salida.update(metricas_clasificacion(y_true, predecir_con_umbral(score_riesgo, umbral)))
    salida["umbral"] = float(umbral)
    return salida
