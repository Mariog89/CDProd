"""Línea base "Python directo": el flujo del repo ejecutado en la máquina local.

Guarda métricas (calculadas con vertex.metricas, igual que AutoML), tiempos y
los scores de riesgo del holdout por fila_id para las curvas comparativas.

Uso:
    python -m vertex.python_local
"""

import os
import platform
import time

import pandas as pd

from mlops_pipeline.src.Cargar_datos import cargar_config, cargar_datos, encontrar_raiz_proyecto
from mlops_pipeline.src.ft_engineering import preparar_datos
from mlops_pipeline.src.model_evaluation import obtener_score_riesgo
from mlops_pipeline.src.model_training import entrenar
from vertex import comun, metricas


def main() -> None:
    raiz = encontrar_raiz_proyecto()
    config = cargar_config(raiz / "mlops_pipeline" / "src" / "config.json")
    t0 = time.perf_counter()
    df = cargar_datos(config)
    t_carga = time.perf_counter() - t0
    datos = preparar_datos(df, random_state=comun.SEMILLA_SPLIT)
    t1 = time.perf_counter()
    entrenamiento = entrenar(datos.X_train, datos.y_train, datos.preprocessor)
    t_entrenamiento = time.perf_counter() - t1
    modelo = entrenamiento.modelo_final_fitted
    t2 = time.perf_counter()
    score = obtener_score_riesgo(modelo, datos.X_eval)
    t_inferencia = time.perf_counter() - t2

    # predict() del LogisticRegression (class_weight=balanced) equivale a score >= 0.5.
    resultado = {
        "modelo": entrenamiento.nombre_modelo_final,
        "hiperparametros": {k: str(v) for k, v in entrenamiento.hiperparametros_finales.items()},
        "n_test": int(len(datos.y_eval)),
        "test_umbral_0_5": metricas.evaluar_scores(datos.y_eval, score, 0.5),
        "cv_f1_0_mean": float(entrenamiento.resultado_cv_modelo_final.mean_f1_0),
        "cv_f1_0_std": float(entrenamiento.resultado_cv_modelo_final.std_f1_0),
        "tiempo_carga_s": round(t_carga, 2),
        "tiempo_entrenamiento_s": round(t_entrenamiento, 2),
        "tiempo_inferencia_test_s": round(t_inferencia, 4),
        "cpu_count": os.cpu_count(),
        "procesador": platform.processor(),
        "python": platform.python_version(),
    }
    comun.guardar_json("python_local_metricas", resultado)
    pd.DataFrame(
        {comun.COLUMNA_ID: datos.X_eval.index, "y": datos.y_eval.values, "score_riesgo": score}
    ).to_csv(comun.RESULTADOS / "python_local_scores_test.csv", index=False)
    print(resultado)


if __name__ == "__main__":
    main()
