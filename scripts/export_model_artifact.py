"""Exporta el artefacto operativo del mejor modelo para servir por batch.

Flujo exclusivo con train (sin holdout):

Base_de_datos.csv -> preparar_datos() -> entrenar(X_train, y_train)
-> crear_metadata_modelo(metricas_holdout_finales=None)
-> guardar_modelo(..., artifacts/1.0.0)

No usa X_eval/y_eval ni reejecuta evaluation. El artefacto resultante
contiene model.joblib + metadata.json y es consumido por app.py.
"""

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    _REPO = Path(__file__).resolve().parents[1]
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))

from mlops_pipeline.src.Cargar_datos import (  # noqa: E402
    cargar_config,
    cargar_datos,
    encontrar_raiz_proyecto,
    resolver_ruta_datos,
)
from mlops_pipeline.src.ft_engineering import preparar_datos  # noqa: E402
from mlops_pipeline.src.model_deploy import (  # noqa: E402
    MODEL_VERSION_DEFAULT,
    crear_metadata_modelo,
    guardar_modelo,
)
from mlops_pipeline.src.model_training import entrenar  # noqa: E402

DEFAULT_OUTPUT = Path("artifacts") / "1.0.0"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporta artefacto operativo train-only para el endpoint batch."
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Directorio destino del artefacto (ej. artifacts/1.0.0).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reemplaza model.joblib/metadata.json si ya existen.",
    )
    return parser.parse_args(argv)


def exportar(output: Path, overwrite: bool = False) -> dict:
    """Genera el artefacto usando solo X_train/y_train/preprocessor."""
    raiz = encontrar_raiz_proyecto()
    config = cargar_config(raiz / "mlops_pipeline" / "src" / "config.json")
    df = cargar_datos(resolver_ruta_datos(raiz, config), config)
    datos = preparar_datos(df)
    resultado = entrenar(datos.X_train, datos.y_train, datos.preprocessor)
    estimador = resultado.modelo_final_fitted
    pasos = getattr(estimador, "named_steps", {})
    clase_nombre = type(pasos.get("model", estimador)).__name__
    metadata = crear_metadata_modelo(
        nombre_modelo=str(resultado.nombre_modelo_final),
        clase_estimator=clase_nombre,
        hiperparametros_finales=dict(resultado.hiperparametros_finales),
        resultado_cv_modelo_final=resultado.resultado_cv_modelo_final,
        metricas_holdout_finales=None,
        model_version=MODEL_VERSION_DEFAULT,
        random_state=int(resultado.random_state),
        notas="Artefacto operativo train-only para endpoint batch.",
    )
    destino = output if Path(output).is_absolute() else (raiz / output)
    return guardar_modelo(estimador, metadata, destino, overwrite=overwrite)


def main(argv=None) -> None:
    args = parse_args(argv)
    final = exportar(Path(args.output), overwrite=bool(args.overwrite))
    print(f"model_version: {final['model_version']}")
    print(f"nombre_modelo: {final['nombre_modelo']}")
    print(f"artifact_sha256: {final['artifact_sha256']}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
