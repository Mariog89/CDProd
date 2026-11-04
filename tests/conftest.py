"""Gates de entorno para tests que requieren BigQuery real.

Los tests marcados con `data_regression` o `bq_live` consultan la tabla
`novabank_dev.Creditos` y solo se ejecutan cuando CDPROD_BQ_LIVE=1.
Así `pytest -q` en CI (sin credenciales GCP) ejecuta únicamente unitarios
con mocks/sintéticos. Los errores de conexión NO se ocultan cuando el
gate está habilitado: el fallo se propaga.
"""

import os

import pytest

_LIVE = os.environ.get("CDPROD_BQ_LIVE") == "1"


def pytest_collection_modifyitems(items):
    for item in items:
        if "data_regression" in item.keywords or "bq_live" in item.keywords:
            item.add_marker(pytest.mark.skipif(not _LIVE, reason="requiere CDPROD_BQ_LIVE=1"))
