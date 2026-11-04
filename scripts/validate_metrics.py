"""Quality gate de métricas de evaluación para el pipeline de Jenkins.

Flujo (mismo código de Stage 3, sin lógica nueva de modelado):

BigQuery(Creditos) -> preparar_datos() -> entrenar(X_train, y_train)
-> evaluar(X_eval, y_eval) -> métricas planas -> comparación con umbrales

Escribe un JSON con métricas, umbrales y veredicto (y opcionalmente la
pestaña HTML de métricas) y termina con código 1 si alguna métrica del
modelo final sobre el holdout queda por debajo de su umbral mínimo.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

RAIZ_REPO = Path(__file__).resolve().parents[1]

if __package__ in {None, ""}:
    if str(RAIZ_REPO) not in sys.path:
        sys.path.insert(0, str(RAIZ_REPO))

# Umbrales mínimos sobre el holdout del modelo final (clase de riesgo 0).
# Calibrados con margen bajo los resultados BigQuery documentados en readme.md
# (F1_0 ≈ 0.12, Recall_0 ≈ 0.53, Balanced ≈ 0.58, ROC-AUC ≈ 0.60): detectan
# un modelo degradado o aleatorio sin bloquear la variación normal.
UMBRALES_DEFAULT = {
    "f1_0": 0.10,
    "recall_0": 0.40,
    "balanced_accuracy": 0.55,
    "roc_auc_risk": 0.55,
}


def extraer_metricas(resultado) -> dict:
    """Aplana un ResultadoEvaluacionCompleta en un dict serializable."""
    ml = resultado.holdout_modelo_final
    heu = resultado.holdout_heuristico
    comp = resultado.comparacion_cv_final_vs_holdout
    return {
        "modelo_final": resultado.nombre_modelo_final,
        "hiperparametros": {k: str(v) for k, v in resultado.hiperparametros_finales.items()},
        "n_eval": int(ml.n_eval),
        "f1_0": float(ml.metricas.f1_0),
        "recall_0": float(ml.metricas.recall_0),
        "precision_0": float(ml.metricas.precision_0),
        "balanced_accuracy": float(ml.metricas.balanced_accuracy),
        "accuracy": float(ml.metricas.accuracy_ctx),
        "roc_auc_risk": ml.probabilisticas.roc_auc_risk,
        "average_precision_risk": ml.probabilisticas.average_precision_risk,
        "cv_f1_0_mean": float(comp.f1_0.cv_mean),
        "cv_f1_0_std": float(comp.f1_0.cv_std),
        "heuristico_f1_0": float(heu.metricas.f1_0),
        "heuristico_recall_0": float(heu.metricas.recall_0),
        "heuristico_balanced_accuracy": float(heu.metricas.balanced_accuracy),
    }


def validar_umbrales(metricas: dict, umbrales: dict) -> list[str]:
    """Devuelve las violaciones de umbral (vacía si el modelo aprueba)."""
    violaciones = []
    for nombre, minimo in umbrales.items():
        valor = metricas.get(nombre)
        if valor is None:
            violaciones.append(f"{nombre}: no disponible (mínimo {minimo:.4f})")
        elif valor < minimo:
            violaciones.append(f"{nombre}: {valor:.4f} < mínimo {minimo:.4f}")
    return violaciones


def construir_reporte(metricas: dict, umbrales: dict) -> dict:
    violaciones = validar_umbrales(metricas, umbrales)
    return {
        "generado_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "aprobado": not violaciones,
        "violaciones": violaciones,
        "umbrales": umbrales,
        "metricas": metricas,
    }


def evaluar_modelo(html: Path | None = None):  # pragma: no cover - requiere BigQuery
    """Ejecuta carga -> features -> training -> evaluation del Stage 3."""
    from mlops_pipeline.src.Cargar_datos import (
        cargar_config,
        cargar_datos,
        encontrar_raiz_proyecto,
    )
    from mlops_pipeline.src.ft_engineering import preparar_datos
    from mlops_pipeline.src.model_evaluation import evaluar, generar_pestana_metricas
    from mlops_pipeline.src.model_training import entrenar

    raiz = encontrar_raiz_proyecto()
    config = cargar_config(raiz / "mlops_pipeline" / "src" / "config.json")
    datos = preparar_datos(cargar_datos(config))
    entrenamiento = entrenar(datos.X_train, datos.y_train, datos.preprocessor)
    resultado = evaluar(entrenamiento, datos.X_eval, datos.y_eval)
    if html is not None:
        generar_pestana_metricas(resultado, html)
    return resultado


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quality gate de métricas del modelo final.")
    parser.add_argument(
        "--output",
        default="reports/metrics.json",
        help="Ruta del JSON con métricas, umbrales y veredicto.",
    )
    parser.add_argument("--html", default=None, help="Ruta opcional de la pestaña HTML.")
    for nombre, minimo in UMBRALES_DEFAULT.items():
        parser.add_argument(
            f"--min-{nombre.replace('_', '-')}",
            dest=f"min_{nombre}",
            type=float,
            default=minimo,
            help=f"Mínimo aceptado para {nombre} (default {minimo}).",
        )
    return parser.parse_args(argv)


def resolver_salida(ruta: str, raiz: Path) -> Path:
    """Resuelve una ruta de salida y exige que quede dentro de `raiz`.

    Evita path traversal: `--output ../../x` o rutas absolutas fuera del
    repositorio se rechazan antes de escribir nada.
    """
    base = raiz.resolve()
    destino = (base / ruta).resolve()
    if not destino.is_relative_to(base):
        raise ValueError(f"Ruta de salida fuera del repositorio: {ruta}")
    return destino


def main(argv=None, evaluador=evaluar_modelo, raiz: Path = RAIZ_REPO) -> int:
    args = parse_args(argv)
    umbrales = {nombre: getattr(args, f"min_{nombre}") for nombre in UMBRALES_DEFAULT}
    destino = resolver_salida(args.output, raiz)
    html = resolver_salida(args.html, raiz) if args.html else None
    resultado = evaluador(html)
    reporte = construir_reporte(extraer_metricas(resultado), umbrales)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")

    m = reporte["metricas"]
    print(f"Modelo final: {m['modelo_final']} {m['hiperparametros']}")
    for nombre, minimo in umbrales.items():
        valor = m.get(nombre)
        texto = "n/d" if valor is None else f"{valor:.4f}"
        print(f"  {nombre:<18} {texto:>8}  (mínimo {minimo:.4f})")
    if reporte["violaciones"]:
        print("QUALITY GATE: RECHAZADO")
        for violacion in reporte["violaciones"]:
            print(f"  - {violacion}")
        return 1
    print("QUALITY GATE: APROBADO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
