"""Validación de src/config.json como contrato de lectura del proyecto."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "src" / "config.json"

EXPECTED_KEYS = {"data_file", "separator", "encoding", "target"}


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
    assert isinstance(config["data_file"], str)
    assert isinstance(config["separator"], str)
    assert isinstance(config["encoding"], str)
    assert isinstance(config["target"], str)


def test_data_file_resolves_to_existing_file():
    config = _load_config()
    data_path = REPO_ROOT / config["data_file"]
    assert data_path.is_file(), f"no existe {data_path}"


def test_target_is_not_empty():
    config = _load_config()
    assert config["target"].strip() != ""
