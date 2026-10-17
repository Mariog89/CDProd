"""Pruebas del loader BigQuery de Stage 3 (config + datos crudos)."""

import json

import pandas as pd
import pytest

from mlops_pipeline.src.Cargar_datos import (
    COLUMNAS_CRUDAS,
    cargar_config,
    cargar_datos,
    construir_consulta,
    construir_tabla,
    encontrar_raiz_proyecto,
)

FILAS_ESPERADAS = 10763
COLUMNAS_ESPERADAS = 23


def _raiz_y_config():
    raiz = encontrar_raiz_proyecto()
    config = cargar_config(raiz / "mlops_pipeline" / "src" / "config.json")
    return raiz, config


def test_raiz_contiene_config_stage3():
    raiz = encontrar_raiz_proyecto()
    assert (raiz / "mlops_pipeline" / "src" / "config.json").is_file()


def test_stage3_config_carga_y_claves_minimas():
    _, config = _raiz_y_config()
    assert {"bq_project", "bq_dataset", "bq_table", "target"}.issubset(config.keys())


def test_tabla_bigquery_bien_formada():
    _, config = _raiz_y_config()
    assert construir_tabla(config) == "cdprod-upb-mg.novabank_dev.Creditos"


def test_consulta_ordena_y_castea():
    _, config = _raiz_y_config()
    consulta = construir_consulta(config)
    assert "CAST(Pago_atiempo AS STRING)" in consulta
    assert "ORDER BY" in consulta
    assert consulta.count("CAST(") == COLUMNAS_ESPERADAS


class _FakeJob:
    def __init__(self, marco):
        self._marco = marco

    def to_dataframe(self):
        return self._marco


class _FakeClient:
    def __init__(self, marco):
        self._marco = marco
        self.consultas = []

    def query(self, consulta):
        self.consultas.append(consulta)
        return _FakeJob(self._marco)


def _crudo_sintetico(n=8):
    return pd.DataFrame({c: [str(i) for i in range(n)] for c in COLUMNAS_CRUDAS})


def test_cargar_datos_devuelve_dataframe_esperado():
    _, config = _raiz_y_config()
    cliente = _FakeClient(_crudo_sintetico())
    df = cargar_datos(config, client=cliente)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == list(COLUMNAS_CRUDAS)
    assert len(df) == 8
    assert len(cliente.consultas) == 1
    assert "cdprod-upb-mg.novabank_dev.Creditos" in cliente.consultas[0]


def test_cargar_datos_todo_string():
    _, config = _raiz_y_config()
    df = cargar_datos(config, client=_FakeClient(_crudo_sintetico()))
    assert all(str(t) == "string" for t in df.dtypes)


def test_config_inexistente_error_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="configuraci"):
        cargar_config(tmp_path / "noexiste.json")


def test_config_json_invalido_error_claro(tmp_path):
    ruta = tmp_path / "config.json"
    ruta.write_text("{json invalido", encoding="utf-8")
    with pytest.raises(ValueError, match="[Ii]nv"):
        cargar_config(ruta)


def test_config_sin_claves_minimas_error_claro(tmp_path):
    ruta = tmp_path / "config.json"
    ruta.write_text(json.dumps({"bq_project": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="[Ff]altantes"):
        cargar_config(ruta)


def test_config_no_objeto_error_claro(tmp_path):
    ruta = tmp_path / "config.json"
    ruta.write_text(json.dumps(["no", "objeto"]), encoding="utf-8")
    with pytest.raises(ValueError, match="[Oo]bjeto"):
        cargar_config(ruta)
