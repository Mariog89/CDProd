"""Pruebas de ft_engineering: features deterministas, split y preprocesador."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from mlops_pipeline.src.Cargar_datos import (
    cargar_config,
    cargar_datos,
    encontrar_raiz_proyecto,
    resolver_ruta_datos,
)
from mlops_pipeline.src.ft_engineering import (
    FEATURES_V1,
    RISK_LABEL,
    DatosProcesados,
    crear_features,
    crear_preprocesador,
    dividir_train_evaluation,
    normalizar_tipos,
    preparar_datos,
    separar_xy,
)

PROHIBIDAS = {
    "Pago_atiempo",
    "incumplio",
    "puntaje",
    "fecha_prestamo",
    "huella_consulta",
    "saldo_mora",
    "saldo_total",
    "saldo_principal",
    "saldo_mora_codeudor",
    "segmento_huella",
    "quintil_capital",
    "quintil_datacredito",
    "flag_edad_fuera_rango",
    "flag_salario_no_positivo",
    "flag_tendencia_invalida",
}


def _crudo_base(n=40, semilla=7):
    """CSV crudo sintético como strings, con todas las columnas necesarias."""
    rng = np.random.default_rng(semilla)
    df = pd.DataFrame(
        {
            "tipo_credito": rng.choice(["4", "9"], size=n).astype(str),
            "fecha_prestamo": ["15/01/2025 10:00"] * n,
            "capital_prestado": rng.integers(500000, 5000000, size=n).astype(str),
            "plazo_meses": rng.integers(6, 24, size=n).astype(str),
            "edad_cliente": rng.integers(25, 60, size=n).astype(str),
            "tipo_laboral": rng.choice(["Empleado", "Independiente"], size=n),
            "salario_cliente": rng.integers(1000000, 5000000, size=n).astype(str),
            "total_otros_prestamos": rng.integers(0, 2000000, size=n).astype(str),
            "cuota_pactada": rng.integers(100000, 600000, size=n).astype(str),
            "puntaje": ["95,227787"] * n,
            "puntaje_datacredito": ["791,5"] * n,
            "cant_creditosvigentes": rng.integers(0, 5, size=n).astype(str),
            "huella_consulta": rng.integers(0, 8, size=n).astype(str),
            "saldo_mora": ["0"] * n,
            "saldo_total": ["10000"] * n,
            "saldo_principal": ["9000"] * n,
            "saldo_mora_codeudor": ["0"] * n,
            "creditos_sectorFinanciero": rng.integers(0, 5, size=n).astype(str),
            "creditos_sectorCooperativo": rng.integers(0, 3, size=n).astype(str),
            "creditos_sectorReal": rng.integers(0, 3, size=n).astype(str),
            "promedio_ingresos_datacredito": rng.integers(1000000, 4000000, size=n).astype(str),
            "tendencia_ingresos": rng.choice(["Creciente", "Estable", "Decreciente"], size=n),
            "Pago_atiempo": rng.choice(["1", "1", "1", "0"], size=n),
        }
    )
    df["incumplio"] = (df["Pago_atiempo"] == "0").astype(int)
    return df


def _real_crudo():
    raiz = encontrar_raiz_proyecto()
    config = cargar_config(raiz / "mlops_pipeline" / "src" / "config.json")
    return cargar_datos(resolver_ruta_datos(raiz, config), config)


def test_normalizar_tipos_convierte():
    df = _crudo_base()
    out = normalizar_tipos(df)
    assert pd.api.types.is_numeric_dtype(out["salario_cliente"])
    assert pd.api.types.is_numeric_dtype(out["Pago_atiempo"])
    assert abs(out["puntaje_datacredito"].iloc[0] - 791.5) < 1e-9
    assert pd.api.types.is_datetime64_any_dtype(out["fecha_prestamo"])
    assert out["salario_cliente"].isna().sum() == 0
    # Opera sobre copia: el crudo queda intacto.
    assert not pd.api.types.is_numeric_dtype(df["salario_cliente"])


def test_ratio_correcto():
    df = crear_features(
        normalizar_tipos(_crudo_base().assign(salario_cliente="2000000", cuota_pactada="200000"))
    )
    assert (df["ratio_cuota_ingreso"] == 0.1).all()


def test_ratio_salario_no_positivo_sin_inf():
    df = _crudo_base(n=4).assign(
        salario_cliente=["0", "-500", "NA", "1000000"],
        cuota_pactada=["100000", "100000", "100000", "100000"],
    )
    out = crear_features(normalizar_tipos(df))
    ratio = out["ratio_cuota_ingreso"]
    assert ratio.isna().sum() == 3
    assert np.isfinite(ratio.dropna()).all()


def test_tendencia_valida_se_conserva():
    df = _crudo_base(n=4).assign(
        tendencia_ingresos=["Creciente", "Estable", "Decreciente", "Creciente"]
    )
    out = crear_features(normalizar_tipos(df))
    assert out["tendencia_ingresos_limpia"].tolist() == [
        "Creciente",
        "Estable",
        "Decreciente",
        "Creciente",
    ]


def test_tendencia_invalida_a_ausente():
    df = _crudo_base(n=4).assign(tendencia_ingresos=["8315", "", "NA", "Creciente"])
    out = crear_features(normalizar_tipos(df))
    assert out["tendencia_ingresos_limpia"].tolist() == [
        "Ausente o inválida",
        "Ausente o inválida",
        "Ausente o inválida",
        "Creciente",
    ]


def test_tendencia_limpia_nunca_nan():
    df = _crudo_base(n=5).assign(tendencia_ingresos=["8315", None, "NA", "", "Estable"])
    out = crear_features(normalizar_tipos(df))
    assert out["tendencia_ingresos_limpia"].isna().sum() == 0


def test_separar_xy_exactamente_16():
    X, y = separar_xy(crear_features(normalizar_tipos(_crudo_base())))
    assert len(FEATURES_V1) == 16
    assert set(X.columns) == set(FEATURES_V1)
    assert len(y) == len(X)


def test_columnas_prohibidas_fuera_de_X():
    X, _ = separar_xy(crear_features(normalizar_tipos(_crudo_base())))
    assert PROHIBIDAS.isdisjoint(set(X.columns))


def test_split_reproducible():
    X, y = separar_xy(crear_features(normalizar_tipos(_crudo_base(n=200))))
    a = dividir_train_evaluation(X, y, random_state=42)
    b = dividir_train_evaluation(X, y, random_state=42)
    for fa, fb in zip(a, b, strict=True):
        pd.testing.assert_frame_equal(
            fa.reset_index(drop=True), fb.reset_index(drop=True)
        ) if isinstance(fa, pd.DataFrame) else pd.testing.assert_series_equal(
            fa.reset_index(drop=True), fb.reset_index(drop=True)
        )


def test_split_mantiene_clases_y_proporcion():
    X, y = separar_xy(crear_features(normalizar_tipos(_crudo_base(n=200))))
    _, _, y_train, y_eval = dividir_train_evaluation(X, y, random_state=42)
    assert set(y_train.unique()) == {0, 1}
    assert set(y_eval.unique()) == {0, 1}
    assert abs((y_train == 0).mean() - (y_eval == 0).mean()) < 0.05


def test_imputer_aprende_solo_de_train():
    base = _crudo_base(n=12).assign(
        salario_cliente=["100", "200", "300"] * 4,
        tendencia_ingresos="Estable",
        tipo_credito="4",
        tipo_laboral="Empleado",
    )
    X, _ = separar_xy(crear_features(normalizar_tipos(base)))
    X_train = X.iloc[:9].copy()
    X_eval = X.iloc[9:].copy()
    X_eval.loc[:, "salario_cliente"] = np.nan
    prep = crear_preprocesador()
    prep.fit(X_train)
    out_eval = prep.transform(X_eval)
    assert (out_eval["num__salario_cliente"] == 200.0).all()


def test_categoria_desconocida_no_rompe():
    X, y = separar_xy(crear_features(normalizar_tipos(_crudo_base(n=60))))
    X_train, X_eval, _, _ = dividir_train_evaluation(X, y, random_state=42)
    X_eval = X_eval.copy()
    X_eval.loc[:, "tipo_laboral"] = "Otro"
    prep = crear_preprocesador()
    out_train = prep.fit_transform(X_train)
    out_eval = prep.transform(X_eval)
    assert list(out_train.columns) == list(out_eval.columns)
    assert len(out_eval) == len(X_eval)


def test_output_es_dataframe_con_mismas_columnas():
    datos = preparar_datos(_crudo_base(n=120))
    prep = crear_preprocesador()
    X_train_t = prep.fit_transform(datos.X_train)
    X_eval_t = prep.transform(datos.X_eval)
    assert isinstance(X_train_t, pd.DataFrame)
    assert isinstance(X_eval_t, pd.DataFrame)
    assert list(X_train_t.columns) == list(X_eval_t.columns)
    # Sintético (semilla fija): 13 num + 2 tipo_credito + 2 tipo_laboral
    # + 3 tendencia = 20. El conteo 25 con todas las categorías se verifica
    # con datos reales en test_real_sin_infinitos.
    assert X_train_t.shape[1] == 20
    assert all(c.startswith(("num__", "cat__")) for c in X_train_t.columns)


def test_preprocessor_retorna_sin_fitear():
    datos = preparar_datos(_crudo_base(n=120))
    assert isinstance(datos, DatosProcesados)
    assert list(datos.X_train.columns) == FEATURES_V1
    with pytest.raises(NotFittedError):
        datos.preprocessor.transform(datos.X_train)


def test_risk_label_cero():
    assert RISK_LABEL == 0


def test_metadata_risk_label_cero():
    datos = preparar_datos(_crudo_base(n=120))
    assert datos.metadata["risk_label"] == 0
    assert datos.metadata["random_state"] == 42
    assert datos.metadata["test_size"] == 0.20


def test_target_faltante_error():
    df = crear_features(normalizar_tipos(_crudo_base())).drop(columns=["Pago_atiempo"])
    with pytest.raises(KeyError, match="objetivo"):
        separar_xy(df)


def test_features_faltantes_error():
    df = crear_features(normalizar_tipos(_crudo_base())).drop(columns=["ratio_cuota_ingreso"])
    with pytest.raises(KeyError, match="features v1"):
        separar_xy(df)


def test_target_nulo_error():
    df = crear_features(normalizar_tipos(_crudo_base()))
    df.loc[df.index[:2], "Pago_atiempo"] = np.nan
    with pytest.raises(ValueError, match="nulos"):
        separar_xy(df)


def test_target_no_binario_error():
    df = crear_features(normalizar_tipos(_crudo_base()))
    df.loc[:, "Pago_atiempo"] = 2
    with pytest.raises(ValueError, match="binario"):
        separar_xy(df)


@pytest.mark.data_regression
def test_real_shapes_y_eventos():
    datos = preparar_datos(_real_crudo())
    assert datos.X_train.shape == (8610, 16)
    assert datos.X_eval.shape == (2153, 16)
    assert list(datos.X_train.columns) == datos.feature_names
    assert datos.y_train.value_counts().to_dict() == {1: 8201, 0: 409}
    assert datos.y_eval.value_counts().to_dict() == {1: 2051, 0: 102}
    total_riesgo = int((datos.y_train == 0).sum() + (datos.y_eval == 0).sum())
    assert total_riesgo == 511
    assert abs(datos.metadata["tasa_riesgo_train"] - 0.0475) < 0.005
    assert abs(datos.metadata["tasa_riesgo_eval"] - 0.0475) < 0.01


@pytest.mark.data_regression
def test_real_sin_infinitos():
    datos = preparar_datos(_real_crudo())
    prep = crear_preprocesador()
    X_train_t = prep.fit_transform(datos.X_train)
    X_eval_t = prep.transform(datos.X_eval)
    assert X_train_t.shape == (8610, 25)
    assert X_eval_t.shape == (2153, 25)
    for matriz in (X_train_t, X_eval_t):
        numericas = matriz.select_dtypes(include=[np.number])
        assert np.isfinite(numericas.to_numpy()).all()
