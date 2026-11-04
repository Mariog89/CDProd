"""Crea la tabla BigQuery que consume AutoML Tabular.

- Mismas 15 variables RAW que usa el modelo del repositorio (RAW_FEATURES_V1),
  así AutoML no ve columnas excluidas por leakage (saldos, puntaje, huella...).
- Misma partición que el pipeline local: preparar_datos(random_state=42)
  sobre la carga determinista de Cargar_datos. El holdout local se marca TEST.
- Del train se separa un 10 % estratificado como VALIDATE (lo usa AutoML para
  su selección interna); el resto es TRAIN.

Uso:
    python -m vertex.preparar_tabla_vertex
"""

import pandas as pd
from google.cloud import bigquery
from sklearn.model_selection import train_test_split

from mlops_pipeline.src.Cargar_datos import cargar_config, cargar_datos, encontrar_raiz_proyecto
from mlops_pipeline.src.ft_engineering import RAW_FEATURES_V1, normalizar_tipos, preparar_datos
from vertex import comun

COLUMNAS_NOMINALES = ("tipo_credito", "tipo_laboral", "tendencia_ingresos")


def construir_tabla(df_crudo: pd.DataFrame) -> pd.DataFrame:
    """Devuelve RAW_FEATURES_V1 tipadas + target (string) + split + id de fila."""
    datos = preparar_datos(df_crudo, random_state=comun.SEMILLA_SPLIT)
    idx_train, idx_val = train_test_split(
        datos.X_train.index,
        test_size=comun.FRACCION_VALIDACION,
        random_state=comun.SEMILLA_SPLIT,
        stratify=datos.y_train,
    )
    tipado = normalizar_tipos(df_crudo)
    tabla = pd.DataFrame(index=df_crudo.index)
    for col in RAW_FEATURES_V1:
        if col in COLUMNAS_NOMINALES:
            tabla[col] = tipado[col].astype("string")
        else:
            tabla[col] = pd.to_numeric(tipado[col], errors="coerce").astype("Float64")
    tabla[comun.TARGET] = df_crudo[comun.TARGET].astype("string")
    tabla[comun.COLUMNA_SPLIT] = "TEST"
    tabla.loc[idx_train, comun.COLUMNA_SPLIT] = "TRAIN"
    tabla.loc[idx_val, comun.COLUMNA_SPLIT] = "VALIDATE"
    tabla.insert(0, comun.COLUMNA_ID, df_crudo.index.astype("int64"))
    return tabla.reset_index(drop=True)


def main() -> None:
    raiz = encontrar_raiz_proyecto()
    config = cargar_config(raiz / "mlops_pipeline" / "src" / "config.json")
    tabla = construir_tabla(cargar_datos(config))
    cliente = bigquery.Client(project=comun.PROJECT)
    trabajo = cliente.load_table_from_dataframe(
        tabla,
        comun.BQ_TABLA_VERTEX,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
    )
    trabajo.result()
    conteos = (
        tabla.groupby([comun.COLUMNA_SPLIT, comun.TARGET]).size().unstack(fill_value=0).to_dict()
    )
    resumen = {
        "tabla": comun.BQ_TABLA_VERTEX,
        "filas": len(tabla),
        "columnas": list(tabla.columns),
        "conteos_split_por_clase": conteos,
    }
    comun.guardar_json("tabla_vertex", resumen)
    print(resumen)


if __name__ == "__main__":
    main()
