"""Cumplimiento literal PDF: training, pestana de metricas y monitoring (sin reentrenar)."""

import inspect
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import mlops_pipeline.src.model_evaluation as ev
import mlops_pipeline.src.model_monitoring as mm
import mlops_pipeline.src.model_training as mt
from mlops_pipeline.src.ft_engineering import (
    FEATURE_SCHEMA_VERSION,
    crear_preprocesador,
    preparar_features_inferencia,
)
from mlops_pipeline.src.hueristic_model import HeuristicModel


def _rcv(nombre, f1=0.13, std=0.01, recall=0.58, precision=0.07, bal=0.61, fit=0.06):
    return mt.ResultadoCV(
        nombre=nombre,
        metricas_por_fold={},
        mean_precision_0=precision,
        std_precision_0=0.005,
        mean_recall_0=recall,
        std_recall_0=0.04,
        mean_f1_0=f1,
        std_f1_0=std,
        mean_balanced_accuracy=bal,
        std_balanced_accuracy=0.02,
        mean_cv_fit_time=fit,
        std_cv_fit_time=0.01,
        mean_cv_score_time=0.02,
        std_cv_score_time=0.005,
    )


def _comparativa():
    return {
        "heuristic": _rcv("heuristic", 0.11, 0.01, 0.60, fit=0.02),
        "logistic_regression": _rcv("logistic_regression", 0.134, 0.009, 0.58, fit=0.06),
        "random_forest": _rcv("random_forest", 0.04, 0.02, 0.03, fit=0.5),
        "hist_gradient_boosting": _rcv("hist_gradient_boosting", 0.14, 0.012, 0.16, fit=1.0),
    }


def test_summarize_classification_delega(monkeypatch):
    import sklearn.pipeline

    pipe = sklearn.pipeline.Pipeline([("m", HeuristicModel())])
    visto = {}

    def _falso(nombre, estimator, x, y, folds):
        visto["args"] = (nombre, estimator)
        return _rcv(nombre)

    monkeypatch.setattr(mt, "evaluar_cv", _falso)
    monkeypatch.setattr(mt, "crear_folds", lambda x, y: ["f1"])
    x = pd.DataFrame({"a": [1, 2, 3, 4]})
    y = pd.Series([0, 1, 0, 1])
    resumen = mt.summarize_classification("heuristic", pipe, x, y)
    assert visto["args"][0] == "heuristic" and visto["args"][1] is pipe
    assert resumen["mean_f1_0"] == 0.13 and resumen["nombre"] == "heuristic"
    assert "metricas_por_fold" in resumen


def test_build_model_reutiliza_candidatos():
    prep = crear_preprocesador()
    for nombre in ("logistic_regression", "random_forest", "hist_gradient_boosting"):
        modelo = mt.build_model(nombre, prep)
        esperado = mt.crear_modelos_candidatos(prep)[nombre]
        assert type(modelo.named_steps["model"]) is type(esperado.named_steps["model"])
    try:
        mt.build_model("inexistente", prep)
        raise AssertionError("debió fallar")
    except ValueError:
        pass
    fuente = inspect.getsource(mt.build_model)
    assert "crear_modelos_candidatos" in fuente


def test_tabla_resumen_contiene_criterios():
    tabla = mt.tabla_resumen_modelos(_comparativa())
    assert list(tabla["modelo"]) == [
        "heuristic",
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    ]
    for col in (
        "mean_f1_0",
        "std_f1_0",
        "mean_recall_0",
        "mean_precision_0",
        "mean_balanced_accuracy",
        "mean_cv_fit_time",
        "Performance",
        "Consistency",
        "Scalability",
    ):
        assert col in tabla.columns
    fila = tabla.set_index("modelo").loc["logistic_regression"]
    assert fila["mean_f1_0"] == 0.134 and fila["Consistency"] == 0.009
    assert fila["Scalability"] == 0.06 and "recall_0" in fila["Performance"]


def test_plot_comparativo_devuelve_figura():
    import matplotlib.pyplot as plt

    fig = mt.plot_comparacion_modelos(_comparativa())
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)
    fuente = inspect.getsource(mt.plot_comparacion_modelos)
    assert "show()" not in fuente and "savefig" not in fuente


def _evaluacion_sintetica():
    metricas = ev.MetricasRiesgo(
        precision_0=0.0801,
        recall_0=0.6569,
        f1_0=0.1429,
        balanced_accuracy=0.6410,
        accuracy_ctx=0.6266,
        support_0=102,
        support_1=2051,
    )
    holdout = ev.ResultadoHoldout(
        nombre="logistic_regression",
        metricas=metricas,
        matriz_nombres={
            "riesgo_real_predicho_riesgo": 67,
            "riesgo_real_predicho_pago": 35,
            "pago_real_predicho_riesgo": 769,
            "pago_real_predicho_pago": 1282,
        },
        matriz_array=np.array([[67, 35], [769, 1282]]),
        reporte_clasificacion={},
        probabilisticas=ev.MetricasProbabilisticas(
            roc_auc_risk=0.6786, average_precision_risk=0.1075
        ),
        n_eval=2153,
    )
    heu = ev.ResultadoHoldout(
        nombre="heuristic",
        metricas=ev.MetricasRiesgo(0.0707, 0.6471, 0.1274, 0.6119, 0.6, 102, 2051),
        matriz_nombres={},
        matriz_array=np.zeros((2, 2), dtype=int),
        reporte_clasificacion={},
        probabilisticas=ev.MetricasProbabilisticas(None, None, motivo="sin score"),
        n_eval=2153,
    )
    comp = ev.ComparativaCVHoldout(
        f1_0=ev.ComparativaItem(0.1429, 0.1322, 0.0084, 0.0106),
        recall_0=ev.ComparativaItem(0.6569, 0.5891, 0.0458, 0.0677),
        precision_0=ev.ComparativaItem(0.0801, 0.0745, 0.0047, 0.0057),
        balanced_accuracy=ev.ComparativaItem(0.6410, 0.6121, 0.0201, 0.0289),
    )
    tukey = SimpleNamespace(
        statistic=np.zeros((4, 4)),
        pvalue=np.full((4, 4), 0.5),
        confidence_interval=lambda confidence_level=0.95: SimpleNamespace(
            low=np.zeros((4, 4)), high=np.zeros((4, 4))
        ),
    )
    est = ev.ResultadoEstadistico(
        modelos=["heuristic", "logistic_regression", "random_forest", "hist_gradient_boosting"],
        n_folds=5,
        metrica="f1_0",
        anova_statistic=21.2083,
        anova_pvalue=8.042e-06,
        anova_advertencia="ANOVA exploratorio.",
        tukey=tukey,
        tukey_mensaje="Tukey ejecutado.",
        friedman_statistic=10.68,
        friedman_pvalue=0.01359,
        friedman_advertencia="Friedman complementario.",
    )
    timing = ev.InferenceTiming(10.0, 10.5, 0.6, 0.005, 198371.9, 2153, 10)
    return ev.ResultadoEvaluacionCompleta(
        nombre_modelo_final="logistic_regression",
        hiperparametros_finales={"model__C": 0.1},
        justificacion_seleccion="sintética",
        holdout_modelo_final=holdout,
        holdout_heuristico=heu,
        comparacion_cv_final_vs_holdout=comp,
        analisis_estadistico_familias=est,
        inference_timing=timing,
        advertencias_metodologicas=["Holdout de uso único."],
    )


def test_pestana_metricas_crea_html(tmp_path):
    destino = tmp_path / "metrics.html"
    ruta = ev.generar_pestana_metricas(_evaluacion_sintetica(), destino)
    assert ruta == destino and destino.is_file()
    texto = destino.read_text(encoding="utf-8")
    for esperado in (
        "logistic_regression",
        "model__C",
        "0.1429",
        "0.6569",
        "0.0801",
        "0.6410",
        "67",
        "1282",
        "ANOVA",
        "8.042e-06",
        "Friedman",
        "0.01359",
        "Tukey",
        "Holdout de uso único",
        "0.6786",
    ):
        assert esperado in texto


def test_pestana_no_recalcula():
    fuente = inspect.getsource(ev.generar_pestana_metricas)
    assert ".fit(" not in fuente
    assert "entrenar(" not in fuente
    assert "evaluar(" not in fuente
    assert "plot_matriz_confusion" in fuente and "plot_comparacion_holdout" in fuente


def _crudo_monitor(n=120, semilla=5):
    rng = np.random.default_rng(semilla)
    return pd.DataFrame(
        {
            "capital_prestado": rng.integers(500000, 5000000, size=n).astype(str),
            "plazo_meses": rng.integers(6, 24, size=n).astype(str),
            "edad_cliente": rng.integers(25, 60, size=n).astype(str),
            "salario_cliente": rng.integers(1000000, 5000000, size=n).astype(str),
            "total_otros_prestamos": rng.integers(0, 2000000, size=n).astype(str),
            "cuota_pactada": rng.integers(100000, 600000, size=n).astype(str),
            "puntaje_datacredito": rng.integers(600, 950, size=n).astype(str),
            "cant_creditosvigentes": rng.integers(0, 5, size=n).astype(str),
            "creditos_sectorFinanciero": rng.integers(0, 5, size=n).astype(str),
            "creditos_sectorCooperativo": rng.integers(0, 3, size=n).astype(str),
            "creditos_sectorReal": rng.integers(0, 3, size=n).astype(str),
            "promedio_ingresos_datacredito": rng.integers(1000000, 4000000, size=n).astype(str),
            "tipo_credito": rng.choice(["4", "9"], size=n),
            "tipo_laboral": rng.choice(["Empleado", "Independiente"], size=n),
            "tendencia_ingresos": rng.choice(["Creciente", "Estable", "Decreciente"], size=n),
        }
    )


def _perfil_log():
    crudo = _crudo_monitor()
    equis = preparar_features_inferencia(crudo)
    y = pd.Series([0, 1] * (len(equis) // 2))
    modelo = HeuristicModel().fit(equis, y)
    meta = {"model_version": "t", "feature_schema_version": FEATURE_SCHEMA_VERSION}
    perfil = mm.crear_perfil_referencia(equis, modelo, meta)
    return perfil, crudo


def _log_csv(ruta, crudo, version="t"):
    preds = (
        HeuristicModel()
        .fit(preparar_features_inferencia(crudo), pd.Series([0] * len(crudo)))
        .predict(preparar_features_inferencia(crudo))
    )
    with open(ruta, "w", encoding="utf-8", newline="") as archivo:
        import csv

        cols = [
            "request_id",
            "prediction_timestamp",
            "model_version",
            "prediction",
            "risk_probability",
            "risk_score",
        ]
        raw_cols = [f"raw_{c}" for c in crudo.columns]
        escritor = csv.DictWriter(archivo, fieldnames=[*cols, *raw_cols])
        escritor.writeheader()
        for i in range(len(crudo)):
            fila = {
                "request_id": "r1",
                "prediction_timestamp": "2026-09-16T00:00:00Z",
                "model_version": version,
                "prediction": int(preds[i]),
                "risk_probability": 0.5,
                "risk_score": "",
            }
            for c in crudo.columns:
                fila[f"raw_{c}"] = crudo.iloc[i][c]
            escritor.writerow(fila)


def test_cargar_tabla_y_columnas(tmp_path):
    perfil, crudo = _perfil_log()
    assert perfil is not None
    ruta = tmp_path / "log.csv"
    _log_csv(ruta, crudo)
    tabla = mm.cargar_tabla_predicciones(ruta)
    assert len(tabla) == len(crudo)
    incompleta = tmp_path / "mala.csv"
    tabla.drop(columns=["prediction"]).to_csv(incompleta, index=False)
    try:
        mm.cargar_tabla_predicciones(incompleta)
        raise AssertionError("debió fallar")
    except ValueError as exc:
        assert "prediction" in str(exc)
    try:
        mm.cargar_tabla_predicciones(tmp_path / "nope.csv")
        raise AssertionError("debió fallar")
    except FileNotFoundError:
        pass


def test_muestreo_reproducible_y_copia():
    marco = pd.DataFrame({"a": range(100)})
    antes = marco.copy(deep=True)
    m1 = mm.muestrear_batch(marco, 20)
    m2 = mm.muestrear_batch(marco, 20)
    pd.testing.assert_frame_equal(m1, m2)
    assert len(m1) == 20
    pd.testing.assert_frame_equal(marco, antes)
    completa = mm.muestrear_batch(marco, 200)
    assert len(completa) == 100 and completa is not marco


def test_monitoring_period_weekly():
    assert mm.MONITORING_PERIOD == "weekly"


def test_monitorizar_desde_log_reutiliza(tmp_path):
    perfil, crudo = _perfil_log()
    ruta = tmp_path / "log.csv"
    _log_csv(ruta, crudo)
    res = mm.monitorizar_desde_log(perfil, ruta)
    assert res.n_total == len(crudo) and res.n_sampled == len(crudo)
    assert "features" in res.feature_drift
    res2 = mm.monitorizar_desde_log(perfil, ruta, n_sample=30)
    assert res2.n_total == len(crudo) and res2.n_sampled == 30
    res3 = mm.monitorizar_desde_log(perfil, ruta, n_sample=30)
    assert res2.n_sampled == res3.n_sampled == 30
    fuente = inspect.getsource(mm.monitorizar_desde_log)
    assert "monitorizar(perfil" in fuente
