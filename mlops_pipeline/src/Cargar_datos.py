"""Carga reproducible de configuración y datos crudos del Stage 3.

Lee mlops_pipeline/src/config.json y obtiene los registros desde Google
BigQuery (config["bq_project/bq_dataset/bq_table"]) como pandas.DataFrame
con todo como string, sin limpieza ni transformaciones.

El orden es determinista (ORDER BY sobre las 23 columnas crudas, únicas como
tupla) pero NO reproduce el orden del antiguo Base_de_datos.csv: cualquier
métrica sensible al orden puede variar respecto a corridas históricas.
"""

import json
from pathlib import Path

import pandas as pd

CONFIG_MARCADOR = "mlops_pipeline/src/config.json"
CLAVES_MINIMAS = {"bq_project", "bq_dataset", "bq_table", "target"}

COLUMNAS_CRUDAS = (
    "tipo_credito",
    "fecha_prestamo",
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "tipo_laboral",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "puntaje",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "huella_consulta",
    "saldo_mora",
    "saldo_total",
    "saldo_principal",
    "saldo_mora_codeudor",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal",
    "promedio_ingresos_datacredito",
    "tendencia_ingresos",
    "Pago_atiempo",
)


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


def construir_tabla(config: dict) -> str:
    """Devuelve el identificador completo proyecto.dataset.tabla del config."""
    return f"{config['bq_project']}.{config['bq_dataset']}.{config['bq_table']}"


def construir_consulta(config: dict) -> str:
    """SELECT con CAST explícito a STRING y ORDER BY determinista.

    Las 23 tuplas crudas son únicas en la tabla, por lo que ordenar por
    todas las columnas produce un orden total reproducible. No reproduce
    el orden del antiguo CSV local.
    """
    tabla = construir_tabla(config)
    columnas = ", ".join(f"CAST({c} AS STRING) AS {c}" for c in COLUMNAS_CRUDAS)
    orden = ", ".join(COLUMNAS_CRUDAS)
    return f"SELECT {columnas} FROM {tabla} ORDER BY {orden}"


def cargar_datos(config: dict, client=None) -> pd.DataFrame:
    """Obtiene el crudo desde BigQuery como DataFrame todo-string.

    Usa Application Default Credentials salvo que se inyecte `client`
    (tests). La consulta vive solo en esta capa de carga.
    """
    if client is None:
        from google.cloud import bigquery

        client = bigquery.Client(project=config["bq_project"])
    marco = client.query(construir_consulta(config)).to_dataframe()
    return marco.astype("string")[list(COLUMNAS_CRUDAS)]
