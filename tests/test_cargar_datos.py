"""Pruebas del loader reproducible de Stage 3 (config + datos crudos)."""

import json

import pandas as pd
import pytest

from mlops_pipeline.src.Cargar_datos import (
    cargar_config,
    cargar_datos,
    encontrar_raiz_proyecto,
    resolver_ruta_datos,
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
    assert {"data_file", "separator", "encoding", "target"}.issubset(config.keys())


def test_data_file_se_resuelve():
    raiz, config = _raiz_y_config()
    ruta = resolver_ruta_datos(raiz, config)
    assert ruta.is_file()
    assert ruta.name == config["data_file"]


def test_cargar_datos_devuelve_dataframe_esperado():
    raiz, config = _raiz_y_config()
    df = cargar_datos(resolver_ruta_datos(raiz, config), config)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (FILAS_ESPERADAS, COLUMNAS_ESPERADAS)


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
    ruta.write_text(json.dumps({"data_file": "x.csv"}), encoding="utf-8")
    with pytest.raises(ValueError, match="[Ff]altantes"):
        cargar_config(ruta)


def test_config_no_objeto_error_claro(tmp_path):
    ruta = tmp_path / "config.json"
    ruta.write_text(json.dumps(["no", "objeto"]), encoding="utf-8")
    with pytest.raises(ValueError, match="[Oo]bjeto"):
        cargar_config(ruta)


def test_data_file_inexistente_error_claro(tmp_path):
    _, config = _raiz_y_config()
    config = dict(config, data_file="inexistente.csv")
    with pytest.raises(FileNotFoundError, match="datos"):
        resolver_ruta_datos(tmp_path, config)
