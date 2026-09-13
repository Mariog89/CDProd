"""Feature engineering reproducible del Stage 3.

Convierte el CSV crudo (dtype="string") en matrices listas para modelado,
sin fuga de información: las transformaciones deterministas fila a fila
pueden aplicarse antes del split; las que aprenden parámetros
(imputación, encoding) se ajustan SOLO con train.

La clase de riesgo es el 0 de Pago_atiempo (1 = pago oportuno).
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RISK_LABEL = 0
CLASES_BINARIAS = {0, 1}

TOKENS_NULOS = [
    "",
    " ",
    "NA",
    "N/A",
    "n/a",
    "NULL",
    "null",
    "None",
    "none",
    "NaN",
    "nan",
    "-",
    "--",
]

TENDENCIAS_VALIDAS = ["Creciente", "Estable", "Decreciente"]
TENDENCIA_AUSENTE = "Ausente o inválida"

COLUMNAS_NUMERICAS_BASE = [
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal",
    "promedio_ingresos_datacredito",
]

FEATURES_NUMERICAS = [*COLUMNAS_NUMERICAS_BASE, "ratio_cuota_ingreso"]
FEATURES_NOMINALES = ["tipo_credito", "tipo_laboral", "tendencia_ingresos_limpia"]
FEATURES_V1 = [*FEATURES_NUMERICAS, *FEATURES_NOMINALES]

# Nota: "puntaje" (interno, casi constante en el EDA) se excluye de FEATURES_V1.
# "puntaje_datacredito" (externo, con señal de riesgo) sí pertenece a FEATURES_V1.
# No confundirlas: son variables distintas con distinto origen y utilidad.


def _convertir_numerico(serie: pd.Series) -> pd.Series:
    """Convierte texto a número con coma decimal, tolerando errores como NaN."""
    return pd.to_numeric(
        serie.astype("string").str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )


def normalizar_tipos(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza un DataFrame crudo a tipos trabajables, sobre una copia.

    Reproduce la normalización del EDA: recorte de espacios, tokens de
    nulos, conversión numérica tolerante y fecha a datetime. No imputa,
    no recorta atípicos, no crea features.
    """
    df = df.copy()
    for columna in df.columns:
        if df[columna].dtype == "string" or df[columna].dtype == object:
            df[columna] = df[columna].astype("string").str.strip()
    df = df.replace(TOKENS_NULOS, pd.NA)
    for columna in [*COLUMNAS_NUMERICAS_BASE, "Pago_atiempo"]:
        if columna in df.columns:
            df[columna] = _convertir_numerico(df[columna])
    if "fecha_prestamo" in df.columns:
        df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"], dayfirst=True, errors="coerce")
    return df


def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea las features deterministas fila a fila, reutilizable en inferencia.

    - ratio_cuota_ingreso: cuota/salario si salario > 0, si no NaN (nunca inf).
    - tendencia_ingresos_limpia: dominio válido o "Ausente o inválida", sin NaN.

    La coerción numérica de salario/cuota es defensiva e intencional: aunque
    normalizar_tipos() ya las convierte, crear_features() debe seguir siendo
    robusta al reutilizarse en inferencia con inputs heterogéneos.
    """
    df = df.copy()
    salario = pd.to_numeric(df["salario_cliente"], errors="coerce")
    cuota = pd.to_numeric(df["cuota_pactada"], errors="coerce")
    denominador = salario.mask(salario.fillna(0) <= 0)
    df["ratio_cuota_ingreso"] = (cuota / denominador).astype(float)
    df["tendencia_ingresos_limpia"] = (
        df["tendencia_ingresos"]
        .astype("string")
        .where(df["tendencia_ingresos"].isin(TENDENCIAS_VALIDAS), TENDENCIA_AUSENTE)
    )
    return df


def separar_xy(df: pd.DataFrame, target: str = "Pago_atiempo") -> tuple[pd.DataFrame, pd.Series]:
    """Separa X (whitelist exacta v1) e y binaria {0, 1}."""
    if target not in df.columns:
        raise KeyError(f"Falta la variable objetivo: {target}")
    faltantes = [c for c in FEATURES_V1 if c not in df.columns]
    if faltantes:
        raise KeyError(f"Faltan features v1 (¿olvidaste crear_features?): {faltantes}")
    if df[target].isna().any():
        raise ValueError("El target contiene nulos.")
    valores = set(df[target].dropna().unique().tolist())
    if valores - CLASES_BINARIAS:
        raise ValueError(f"El target debe ser binario {{0, 1}}, se encontró: {valores}")
    X = df[FEATURES_V1].copy()
    y = df[target].astype(int)
    return X, y


def dividir_train_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split estratificado reproducible (preserva la tasa de clase 0)."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def crear_preprocesador() -> ColumnTransformer:
    """ColumnTransformer sin ajustar: mediana para numéricas, OHE para nominales."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), FEATURES_NUMERICAS),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                FEATURES_NOMINALES,
            ),
        ]
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor


@dataclass
class DatosProcesados:
    """Split estratificado + preprocessor SIN ajustar para model_training.

    X_train/X_eval contienen las 16 features v1 sin preprocessing aprendido.
    model_training debe clonar el preprocessor dentro de un Pipeline por
    modelo para que cross-validation ajuste imputación/encoding solo con
    cada fold de entrenamiento.
    """

    X_train: pd.DataFrame
    X_eval: pd.DataFrame
    y_train: pd.Series
    y_eval: pd.Series
    preprocessor: ColumnTransformer
    feature_names: list
    metadata: dict = field(default_factory=dict)


def preparar_datos(
    df_crudo: pd.DataFrame,
    target: str = "Pago_atiempo",
    test_size: float = 0.20,
    random_state: int = 42,
) -> DatosProcesados:
    """Orquesta hasta el split: normaliza, deriva, separa y crea el preprocessor.

    NO ajusta (fit) ni transforma: el preprocessor se devuelve SIN fitear
    para que model_training lo clone dentro de cada Pipeline y
    cross-validation no contamine folds. No serializa artefactos.
    """
    df = crear_features(normalizar_tipos(df_crudo))
    X, y = separar_xy(df, target=target)
    X_train, X_eval, y_train, y_eval = dividir_train_evaluation(
        X, y, test_size=test_size, random_state=random_state
    )
    preprocessor = crear_preprocesador()
    metadata: dict[str, Any] = {
        "risk_label": RISK_LABEL,
        "random_state": random_state,
        "test_size": test_size,
        "tasa_riesgo_train": float((y_train == RISK_LABEL).mean()),
        "tasa_riesgo_eval": float((y_eval == RISK_LABEL).mean()),
        "n_train": int(len(X_train)),
        "n_eval": int(len(X_eval)),
    }
    return DatosProcesados(
        X_train=X_train,
        X_eval=X_eval,
        y_train=y_train,
        y_eval=y_eval,
        preprocessor=preprocessor,
        feature_names=list(X.columns),
        metadata=metadata,
    )
