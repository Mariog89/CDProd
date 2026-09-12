"""Carga reproducible de configuración y datos crudos del Stage 3.

Lee mlops_pipeline/src/config.json, resuelve la ruta del dataset
desde config["data_file"] y lo carga sin limpieza ni transformaciones.
"""

import json
from pathlib import Path

import pandas as pd

CONFIG_MARCADOR = "mlops_pipeline/src/config.json"
CLAVES_MINIMAS = {"data_file", "separator", "encoding", "target"}


def encontrar_raiz_proyecto(
    marcador: str = CONFIG_MARCADOR,
    inicio: Path | None = None,
) -> Path:
    """Localiza la raíz del repositorio buscando el marcador de configuración.

    Primero intenta derivarla desde la ubicación de este archivo; si no
    aplica, sube desde `inicio` (o el cwd) hasta encontrar el marcador.
    """
    candidato = Path(__file__).resolve().parents[2]
    if (candidato / marcador).is_file():
        return candidato

    ruta = (inicio or Path.cwd()).resolve()
    while True:
        if (ruta / marcador).is_file():
            return ruta
        if ruta == ruta.parent:
            raise FileNotFoundError(f"No se encontró {marcador} en la jerarquía del proyecto.")
        ruta = ruta.parent


def cargar_config(ruta_config: Path) -> dict:
    """Lee y valida el JSON de configuración sin cerrar el esquema a futuro."""
    if not ruta_config.is_file():
        raise FileNotFoundError(f"No existe el archivo de configuración: {ruta_config}")
    try:
        with open(ruta_config, encoding="utf-8") as archivo:
            config = json.load(archivo)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Configuración JSON inválida en {ruta_config}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"La configuración debe ser un objeto JSON: {ruta_config}")
    faltantes = CLAVES_MINIMAS - set(config.keys())
    if faltantes:
        raise ValueError(f"Claves faltantes en {ruta_config}: {sorted(faltantes)}")
    return config


def resolver_ruta_datos(raiz: Path, config: dict) -> Path:
    """Construye DATA_PATH como raíz / config["data_file"] y verifica que exista."""
    ruta_datos = raiz / config["data_file"]
    if not ruta_datos.is_file():
        raise FileNotFoundError(f"No existe el archivo de datos: {ruta_datos}")
    return ruta_datos


def cargar_datos(ruta_datos: Path, config: dict) -> pd.DataFrame:
    """Carga el CSV crudo con separator/encoding del config y todo como string."""
    return pd.read_csv(
        ruta_datos,
        sep=config["separator"],
        encoding=config["encoding"],
        dtype="string",
    )
