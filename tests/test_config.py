"""Validación de mlops_pipeline/src/config.json como contrato de Stage 3."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "mlops_pipeline" / "src" / "config.json"

EXPECTED_KEYS = {"bq_project", "bq_dataset", "bq_table", "target"}


def _load_config():
    assert CONFIG_PATH.is_file(), f"no existe {CONFIG_PATH}"
    with open(CONFIG_PATH, encoding="utf-8") as archivo:
        config = json.load(archivo)
    assert isinstance(config, dict), "config.json debe ser un objeto JSON"
    return config


def test_config_exists_and_is_valid_json():
    config = _load_config()
    assert config, "config.json no debe estar vacío"


def test_config_contains_expected_keys():
    config = _load_config()
    assert EXPECTED_KEYS.issubset(config.keys())


def test_config_value_types():
    config = _load_config()
    assert isinstance(config["bq_project"], str)
    assert isinstance(config["bq_dataset"], str)
    assert isinstance(config["bq_table"], str)
    assert isinstance(config["target"], str)


def test_bigquery_table_identifier_is_complete():
    config = _load_config()
    tabla = f"{config['bq_project']}.{config['bq_dataset']}.{config['bq_table']}"
    assert tabla == "cdprod-upb-mg.novabank_dev.Creditos"


def test_target_is_not_empty():
    config = _load_config()
    assert config["target"].strip() != ""
