"""Pruebas del baseline heurístico congelado (reglas R1/R2/R3)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from mlops_pipeline.src.Cargar_datos import (
    cargar_config,
    cargar_datos,
    encontrar_raiz_proyecto,
)
from mlops_pipeline.src.ft_engineering import preparar_datos
from mlops_pipeline.src.hueristic_model import (
    HeuristicModel,
    explicar_heuristico,
    predecir_heuristico,
)


def _fila(**cambios):
    base = {
        "puntaje_datacredito": 800.0,
        "capital_prestado": 1_000_000.0,
        "tendencia_ingresos_limpia": "Estable",
    }
    base.update(cambios)
    return pd.DataFrame([base])


def test_funcion_sin_fit_solo_01_entero_indice_filas():
    X = pd.DataFrame(
        {
            "puntaje_datacredito": [700.0, 800.0],
            "capital_prestado": [1_000_000.0, 5_000_000.0],
            "tendencia_ingresos_limpia": ["Estable", "Decreciente"],
        },
        index=[7, 9],
    )
    out = predecir_heuristico(X)
    assert set(out.unique().tolist()) <= {0, 1}
    assert str(out.dtype).startswith("int")
    assert list(out.index) == [7, 9]
    assert len(out) == 2


def test_r1_dispara():
    assert predecir_heuristico(_fila(puntaje_datacredito=700.0)).iloc[0] == 0


def test_borde_750_dispara():
    assert predecir_heuristico(_fila(puntaje_datacredito=750.0)).iloc[0] == 0


def test_puntaje_no_positivo_no_dispara():
    for valor in [0.0, -7.0]:
        assert predecir_heuristico(_fila(puntaje_datacredito=valor)).iloc[0] == 1


def test_r2_dispara():
    assert predecir_heuristico(_fila(capital_prestado=5_000_000.0)).iloc[0] == 0
    assert predecir_heuristico(_fila(capital_prestado=3_400_000.0)).iloc[0] == 1


def test_r3_dispara():
    assert predecir_heuristico(_fila(tendencia_ingresos_limpia="Decreciente")).iloc[0] == 0


def test_sin_reglas_es_1():
    assert predecir_heuristico(_fila()).iloc[0] == 1


def test_cualquier_regla_basta_y_multiples_siguen_0():
    assert predecir_heuristico(_fila(puntaje_datacredito=700.0)).iloc[0] == 0
    assert predecir_heuristico(_fila(capital_prestado=5_000_000.0)).iloc[0] == 0
    assert (
        predecir_heuristico(_fila(puntaje_datacredito=700.0, capital_prestado=5_000_000.0)).iloc[0]
        == 0
    )


def test_nan_abstiene_por_regla_y_todos_nan_es_1():
    assert predecir_heuristico(_fila(puntaje_datacredito=np.nan)).iloc[0] == 1
    assert predecir_heuristico(_fila(capital_prestado=np.nan)).iloc[0] == 1
    assert predecir_heuristico(_fila(tendencia_ingresos_limpia=np.nan)).iloc[0] == 1
    X = pd.DataFrame(
        {
            "puntaje_datacredito": [np.nan],
            "capital_prestado": [np.nan],
            "tendencia_ingresos_limpia": [np.nan],
        }
    )
    assert predecir_heuristico(X).iloc[0] == 1


def test_falta_columna_error():
    with pytest.raises(ValueError, match="requeridas"):
        predecir_heuristico(_fila().drop(columns=["capital_prestado"]))
    with pytest.raises(TypeError):
        predecir_heuristico("no-df")


def test_input_no_modificado_y_extras_ignoradas():
    X = _fila()
    X["Pago_atiempo"] = 0
    X["incumplio"] = 1
    X["saldo_mora"] = 999.0
    X["huella_consulta"] = 9
    X["salario_cliente"] = 1.0
    antes = X.copy(deep=True)
    base = predecir_heuristico(_fila())
    assert (predecir_heuristico(X) == base).all()
    pd.testing.assert_frame_equal(X, antes)


def test_score_y_reglas_coherentes():
    X = pd.DataFrame(
        {
            "puntaje_datacredito": [700.0, 800.0, 700.0],
            "capital_prestado": [1_000_000.0, 5_000_000.0, 5_000_000.0],
            "tendencia_ingresos_limpia": ["Estable", "Estable", "Decreciente"],
        }
    )
    exp = explicar_heuristico(X)
    assert exp["score_riesgo"].tolist() == [1, 1, 3]
    assert exp["reglas_activadas"].tolist() == [["R1"], ["R2"], ["R1", "R2", "R3"]]
    assert exp["score_riesgo"].between(0, 3).all()


def test_predict_antes_de_fit_falla():
    with pytest.raises(NotFittedError):
        HeuristicModel().predict(_fila())


def test_fit_retorna_self_y_metadata():
    X = _fila()
    modelo = HeuristicModel()
    assert modelo.fit(X) is modelo
    assert modelo.classes_.tolist() == [0, 1]


def test_n_features_y_names_con_16_y_5():
    from mlops_pipeline.src.ft_engineering import FEATURES_V1

    X16 = pd.DataFrame(
        {c: [1] for c in FEATURES_V1}
        | {
            "puntaje_datacredito": [800.0],
            "capital_prestado": [1_000_000.0],
            "tendencia_ingresos_limpia": ["Estable"],
        }
    )
    m16 = HeuristicModel().fit(X16)
    assert m16.n_features_in_ == 16
    assert list(m16.feature_names_in_) == list(X16.columns)
    X5 = X16[
        [
            "puntaje_datacredito",
            "capital_prestado",
            "tendencia_ingresos_limpia",
            "edad_cliente",
            "salario_cliente",
        ]
    ]
    m5 = HeuristicModel().fit(X5)
    assert m5.n_features_in_ == 5
    assert (m16.predict(X16) == m5.predict(X5)).all()


def test_y_permutada_no_altera():
    X = pd.DataFrame(
        {
            "puntaje_datacredito": [700.0, 800.0, 600.0, 900.0],
            "capital_prestado": [1_000_000.0] * 4,
            "tendencia_ingresos_limpia": ["Estable"] * 4,
        }
    )
    a = HeuristicModel().fit(X, pd.Series([0, 1, 0, 1])).predict(X)
    b = HeuristicModel().fit(X, pd.Series([1, 0, 1, 0])).predict(X)
    assert (a == b).all()


def test_determinismo_y_paridad_funcion_estimator():
    X = _fila(puntaje_datacredito=700.0)
    assert (predecir_heuristico(X) == predecir_heuristico(X)).all()
    modelo = HeuristicModel().fit(X)
    assert (modelo.predict(X) == predecir_heuristico(X).to_numpy()).all()


def test_score_no_es_probabilidad_documentado():
    import inspect

    from mlops_pipeline.src import hueristic_model as modulo

    fuente = inspect.getsource(modulo.explicar_heuristico)
    assert "NO es probabilidad" in fuente


@pytest.mark.data_regression
def test_data_regression_heuristico_train():
    """Smoke de integración sobre X_train; el holdout sigue sellado.

    Las métricas son descriptivas de regresión de datos, no objetivos:
    ninguna regla se ajusta a partir de ellas.
    """
    raiz = encontrar_raiz_proyecto()
    config = cargar_config(raiz / "mlops_pipeline" / "src" / "config.json")
    df = cargar_datos(config)
    datos = preparar_datos(df)
    modelo = HeuristicModel().fit(datos.X_train)
    y_pred = pd.Series(modelo.predict(datos.X_train), index=datos.y_train.index)
    print(
        f"\nheuristico train: n={len(y_pred)} "
        f"pred0={(y_pred == 0).sum()} "
        f"pred1={(y_pred == 1).sum()} "
        f"precision_0={precision_score(datos.y_train, y_pred, pos_label=0, zero_division=0):.4f} "
        f"recall_0={recall_score(datos.y_train, y_pred, pos_label=0, zero_division=0):.4f} "
        f"f1_0={f1_score(datos.y_train, y_pred, pos_label=0, zero_division=0):.4f} "
        f"balAcc={balanced_accuracy_score(datos.y_train, y_pred):.4f}"
    )
    assert set(y_pred.unique().tolist()) <= {0, 1}
    assert len(y_pred) == len(datos.y_train) == 8610
    assert (y_pred == 0).sum() > 0
    assert (y_pred == 1).sum() > 0
