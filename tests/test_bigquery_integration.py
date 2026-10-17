"""Integración real con BigQuery (requiere ADC; se omite sin credenciales).

No se ejecuta en CI: valida dimensiones, esquema, tipos string,
distribución del target y compatibilidad con preparar_datos().
"""

import pandas as pd
import pytest

pytestmark = pytest.mark.bq_live

bq = pytest.importorskip("google.cloud.bigquery", reason="google-cloud-bigquery no instalado")

from mlops_pipeline.src.Cargar_datos import (  # noqa: E402
    COLUMNAS_CRUDAS,
    cargar_config,
    cargar_datos,
    encontrar_raiz_proyecto,
)
from mlops_pipeline.src.ft_engineering import preparar_datos  # noqa: E402

try:
    from google.auth.exceptions import DefaultCredentialsError

    _cred_error = DefaultCredentialsError
except ImportError:  # pragma: no cover
    _cred_error = Exception


def _requiere_adc():
    try:
        bq.Client(project="cdprod-upb-mg")
    except _cred_error as exc:
        pytest.skip(f"sin ADC: {exc}")
    except Exception as exc:  # pragma: no cover - red/experiencia
        pytest.skip(f"BigQuery no accesible: {type(exc).__name__}")


def test_bigquery_dimensiones_y_esquema():
    _requiere_adc()
    raiz = encontrar_raiz_proyecto()
    config = cargar_config(raiz / "mlops_pipeline" / "src" / "config.json")
    df = cargar_datos(config)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (10763, 23)
    assert list(df.columns) == list(COLUMNAS_CRUDAS)
    assert all(str(t) == "string" for t in df.dtypes)


def test_bigquery_target_y_nulos():
    _requiere_adc()
    raiz = encontrar_raiz_proyecto()
    config = cargar_config(raiz / "mlops_pipeline" / "src" / "config.json")
    df = cargar_datos(config)
    assert df["Pago_atiempo"].value_counts(dropna=False).to_dict() == {"1": 10252, "0": 511}
    assert int(df.isna().sum().sum()) == 7175
    assert df["Pago_atiempo"].isna().sum() == 0


def test_bigquery_compatible_con_pipeline():
    _requiere_adc()
    raiz = encontrar_raiz_proyecto()
    config = cargar_config(raiz / "mlops_pipeline" / "src" / "config.json")
    datos = preparar_datos(cargar_datos(config))
    assert datos.X_train.shape == (8610, 16)
    assert datos.X_eval.shape == (2153, 16)
    assert datos.y_train.value_counts().to_dict() == {1: 8201, 0: 409}
    assert datos.y_eval.value_counts().to_dict() == {1: 2051, 0: 102}
