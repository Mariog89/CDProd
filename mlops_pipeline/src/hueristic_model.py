"""Baseline heurístico de negocio: reglas fijas, deterministas y explicables.

Compara riesgo crediticio con tres señales congeladas del EDA:

- R1: 0 < puntaje_datacredito <= 750
- R2: capital_prestado > 3_400_000
- R3: tendencia_ingresos_limpia == "Decreciente"

Predice 0 (riesgo) si alguna regla dispara, 1 en caso contrario.
Las reglas están CONGELADAS: no se aprenden, no se tunean, no usan y.
score_riesgo es un conteo interpretable 0..3, NO una probabilidad.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

RISK_LABEL = 0

COLUMNAS_REQUERIDAS = [
    "puntaje_datacredito",
    "capital_prestado",
    "tendencia_ingresos_limpia",
]

LIMITE_PUNTAJE = 750
LIMITE_CAPITAL = 3_400_000
TENDENCIA_RIESGO = "Decreciente"

REGLAS = ("R1", "R2", "R3")


def _regla_puntaje(puntaje: pd.Series) -> pd.Series:
    valores = pd.to_numeric(puntaje, errors="coerce")
    return (valores > 0) & (valores <= LIMITE_PUNTAJE)


def _regla_capital(capital: pd.Series) -> pd.Series:
    valores = pd.to_numeric(capital, errors="coerce")
    return valores > LIMITE_CAPITAL


def _regla_tendencia(tendencia: pd.Series) -> pd.Series:
    return tendencia.astype("string") == TENDENCIA_RIESGO


def _validar_input(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X debe ser un pd.DataFrame.")
    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in X.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")
    return X


def predecir_heuristico(X: pd.DataFrame) -> pd.Series:
    """Aplica las reglas congeladas y devuelve labels {0, 1} como enteros.

    Funciona sin fit. NaN en una regla la abstiene (no dispara riesgo).
    No modifica X. Columnas adicionales se ignoran.
    """
    X = _validar_input(X)
    r1 = _regla_puntaje(X["puntaje_datacredito"]).fillna(False)
    r2 = _regla_capital(X["capital_prestado"]).fillna(False)
    r3 = _regla_tendencia(X["tendencia_ingresos_limpia"]).fillna(False)
    riesgo = (r1 | r2 | r3).to_numpy(dtype=bool)
    return pd.Series(np.where(riesgo, 0, 1), index=X.index, dtype="int64")


def explicar_heuristico(X: pd.DataFrame) -> pd.DataFrame:
    """Devuelve score_riesgo (conteo 0..3) y reglas_activadas por fila.

    score_riesgo NO es probabilidad ni score calibrado: es el número de
    reglas R1/R2/R3 que dispararon para cada fila.
    """
    X = _validar_input(X)
    r1 = _regla_puntaje(X["puntaje_datacredito"]).fillna(False)
    r2 = _regla_capital(X["capital_prestado"]).fillna(False)
    r3 = _regla_tendencia(X["tendencia_ingresos_limpia"]).fillna(False)
    activadas = pd.DataFrame({"R1": r1, "R2": r2, "R3": r3}, index=X.index)
    score = activadas.sum(axis=1).astype("int64")
    reglas = activadas.apply(lambda fila: [regla for regla in REGLAS if fila[regla]], axis=1)
    return pd.DataFrame({"score_riesgo": score, "reglas_activadas": reglas}, index=X.index)


class HeuristicModel(ClassifierMixin, BaseEstimator):
    """Estimator sklearn-compatible del baseline heurístico (sin aprendizaje)."""

    def fit(self, X: pd.DataFrame, y=None):
        """Valida columnas y fija metadata; no aprende thresholds ni usa y."""
        _validar_input(X)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Exige fit previo y delega en predecir_heuristico."""
        check_is_fitted(self)
        return predecir_heuristico(X).to_numpy()
