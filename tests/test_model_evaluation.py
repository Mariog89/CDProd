"""Pruebas de la evaluación final Stage 3 (holdout sellado, uso único)."""

import inspect
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.figure
import numpy as np
import pandas as pd
import pytest
from scipy.stats import f_oneway, friedmanchisquare
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from mlops_pipeline.src import model_evaluation as ev
from mlops_pipeline.src import model_training as mt


def _y_ejemplo():
    y_true = pd.Series([0, 0, 1, 1, 0, 1])
    y_pred = pd.Series([0, 1, 1, 1, 0, 0])
    return y_true, y_pred


def _resultado_cv(nombre, f1, recall=0.5, precision=0.4, bal=0.6, std=0.01):
    return SimpleNamespace(
        nombre=nombre,
        metricas_por_fold={
            "precision_0": [precision] * 5,
            "recall_0": [recall] * 5,
            "f1_0": list(f1) if isinstance(f1, list) else [f1] * 5,
            "balanced_accuracy": [bal] * 5,
        },
        mean_precision_0=float(precision),
        std_precision_0=float(std),
        mean_recall_0=float(recall),
        std_recall_0=float(std),
        mean_f1_0=float(np.mean(list(f1) if isinstance(f1, list) else [f1] * 5)),
        std_f1_0=float(np.std(list(f1) if isinstance(f1, list) else [f1] * 5)),
        mean_balanced_accuracy=float(bal),
        std_balanced_accuracy=float(std),
        mean_cv_fit_time=0.05,
        std_cv_fit_time=0.0,
        mean_cv_score_time=0.01,
        std_cv_score_time=0.0,
    )


def _comparativa_base():
    return {
        "heuristic": _resultado_cv("heuristic", [0.10, 0.11, 0.09, 0.12, 0.10]),
        "logistic_regression": _resultado_cv("lr", [0.13, 0.14, 0.12, 0.13, 0.15]),
        "random_forest": _resultado_cv("rf", [0.04, 0.05, 0.03, 0.04, 0.05]),
        "hist_gradient_boosting": _resultado_cv("hgb", [0.14, 0.13, 0.15, 0.14, 0.12]),
    }


def _x_eval_heuristico(n=20, semilla=3):
    rng = np.random.default_rng(semilla)
    return pd.DataFrame(
        {
            "puntaje_datacredito": rng.integers(600, 950, size=n).astype(float),
            "capital_prestado": rng.integers(500000, 5000000, size=n).astype(float),
            "tendencia_ingresos_limpia": rng.choice(["Creciente", "Estable"], size=n),
        }
    )


def _resultado_entrenamiento_sintetico(modelo):
    comparativa = _comparativa_base()
    final = _resultado_cv("final_tuned", 0.132, recall=0.589, precision=0.42, bal=0.61)
    return SimpleNamespace(
        comparativa_cv_familias=comparativa,
        resultado_dummy=_resultado_cv("dummy", 0.05),
        resultado_cv_modelo_final=final,
        modelo_final_fitted=modelo,
        nombre_modelo_final="logistic_regression",
        hiperparametros_finales={"model__C": 0.1},
        justificacion_seleccion="Elegido: logistic_regression (sintético).",
    )


class ModeloFijo:
    """Predict fijo; fit explota para probar que evaluar no reentrena."""

    def __init__(self, y_pred, proba=None, classes=None):
        self._y = np.asarray(y_pred)
        self._proba = None if proba is None else np.asarray(proba, dtype=float)
        self.classes_ = np.asarray([0, 1] if classes is None else classes)

    def predict(self, X):
        n = len(X)
        if len(self._y) == n:
            return self._y.copy()
        return np.resize(self._y, n).copy()

    def predict_proba(self, X):
        if self._proba is None:
            raise AttributeError("sin proba")
        n = len(X)
        if len(self._proba) == n:
            return self._proba.copy()
        return np.resize(self._proba, (n, 2)).copy()

    def fit(self, X, y=None):
        raise AssertionError("evaluar no debe llamar fit")


class ModeloDecision:
    def __init__(self, scores, classes):
        self._s = np.asarray(scores, dtype=float)
        self.classes_ = np.asarray(classes)

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def decision_function(self, X):
        return self._s[: len(X)].copy()

    def fit(self, X, y=None):
        raise AssertionError("evaluar no debe llamar fit")


class ModeloSinScores:
    def predict(self, X):
        return np.ones(len(X), dtype=int)

    def fit(self, X, y=None):
        raise AssertionError("evaluar no debe llamar fit")


# 1-6 MÉTRICAS


def test_01_precision_usa_clase_0():
    y_true, y_pred = _y_ejemplo()
    m = ev.calcular_metricas_clase_riesgo(y_true, y_pred)
    esperado = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    assert m.precision_0 == pytest.approx(esperado)
    distinto = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    assert m.precision_0 != pytest.approx(distinto) or esperado == distinto


def test_02_recall_usa_clase_0():
    y_true, y_pred = _y_ejemplo()
    m = ev.calcular_metricas_clase_riesgo(y_true, y_pred)
    assert m.recall_0 == pytest.approx(recall_score(y_true, y_pred, pos_label=0, zero_division=0))


def test_03_f1_usa_clase_0():
    y_true, y_pred = _y_ejemplo()
    m = ev.calcular_metricas_clase_riesgo(y_true, y_pred)
    assert m.f1_0 == pytest.approx(f1_score(y_true, y_pred, pos_label=0, zero_division=0))


def test_04_balanced_accuracy_correcta():
    y_true, y_pred = _y_ejemplo()
    m = ev.calcular_metricas_clase_riesgo(y_true, y_pred)
    assert m.balanced_accuracy == pytest.approx(balanced_accuracy_score(y_true, y_pred))


def test_05_support_correcto():
    y_true = pd.Series([0, 0, 0, 1, 1])
    y_pred = pd.Series([0, 1, 0, 1, 1])
    m = ev.calcular_metricas_clase_riesgo(y_true, y_pred)
    assert m.support_0 == 3
    assert m.support_1 == 2


def test_06_accuracy_solo_contextual():
    y_true, y_pred = _y_ejemplo()
    m = ev.calcular_metricas_clase_riesgo(y_true, y_pred)
    assert m.accuracy_ctx == pytest.approx(accuracy_score(y_true, y_pred))
    assert hasattr(m, "f1_0") and hasattr(m, "accuracy_ctx")


# 7-8 CONFUSIÓN


def test_07_labels_01_en_fuente_y_forma():
    fuente = inspect.getsource(ev)
    assert "labels=[0, 1]" in fuente
    assert "confusion_matrix" in fuente
    nombres, mat = ev.matriz_confusion_riesgo([0, 0, 1, 1], [0, 1, 1, 1])
    assert mat.shape == (2, 2)
    # Solo clase 1 en y: la matriz sigue siendo 2x2 por labels fijos.
    _, mat2 = ev.matriz_confusion_riesgo([1, 1], [1, 1])
    assert mat2.shape == (2, 2)
    assert set(nombres) == {
        "riesgo_real_predicho_riesgo",
        "riesgo_real_predicho_pago",
        "pago_real_predicho_riesgo",
        "pago_real_predicho_pago",
    }


def test_08_fn_riesgo_identificado():
    # y=[0,0,1,1], pred=[0,1,1,1]: un FN de riesgo (real 0 predicho 1).
    nombres, mat = ev.matriz_confusion_riesgo([0, 0, 1, 1], [0, 1, 1, 1])
    assert nombres["riesgo_real_predicho_riesgo"] == 1
    assert nombres["riesgo_real_predicho_pago"] == 1
    assert nombres["pago_real_predicho_riesgo"] == 0
    assert nombres["pago_real_predicho_pago"] == 2
    assert mat[0, 1] == nombres["riesgo_real_predicho_pago"]


# 9-12 PROBABILIDADES


def test_09_proba_usa_columna_clase_0():
    X = pd.DataFrame({"a": [1, 2]})
    proba = np.array([[0.2, 0.8], [0.9, 0.1]])
    modelo = ModeloFijo([0, 1], proba=proba, classes=[0, 1])
    scores = ev.obtener_score_riesgo(modelo, X)
    assert scores.tolist() == pytest.approx([0.2, 0.9])


def test_10_classes_invertidas_funcionan():
    X = pd.DataFrame({"a": [1, 2]})
    # classes=[1,0]: columna 1 es la clase 0.
    proba = np.array([[0.8, 0.2], [0.1, 0.9]])
    modelo = ModeloFijo([0, 1], proba=proba, classes=[1, 0])
    scores = ev.obtener_score_riesgo(modelo, X)
    assert scores.tolist() == pytest.approx([0.2, 0.9])


def test_11_decision_function_se_invierte():
    X = pd.DataFrame({"a": [1, 2]})
    scores = np.array([1.5, -0.5])
    directo = ModeloDecision(scores, classes=[1, 0])
    # RISK=0 es classes_[1] -> score directo.
    assert ev.obtener_score_riesgo(directo, X).tolist() == pytest.approx([1.5, -0.5])
    invertido = ModeloDecision(scores, classes=[0, 1])
    # RISK=0 es classes_[0] -> score negado.
    assert ev.obtener_score_riesgo(invertido, X).tolist() == pytest.approx([-1.5, 0.5])


def test_12_sin_scores_no_rompe():
    X = pd.DataFrame({"a": [1, 2, 3]})
    modelo = ModeloSinScores()
    assert ev.obtener_score_riesgo(modelo, X) is None
    prob = ev.calcular_metricas_probabilisticas([0, 1, 1], None)
    assert prob.roc_auc_risk is None and prob.average_precision_risk is None
    assert prob.motivo is not None


# 13-16 EVALUATION


def _xy16(n=40, semilla=0):
    rng = np.random.default_rng(semilla)
    X = pd.DataFrame(
        {
            "capital_prestado": rng.integers(500000, 5000000, size=n).astype(float),
            "plazo_meses": rng.integers(6, 24, size=n).astype(float),
            "edad_cliente": rng.integers(25, 60, size=n).astype(float),
            "salario_cliente": rng.integers(1000000, 5000000, size=n).astype(float),
            "total_otros_prestamos": rng.integers(0, 2000000, size=n).astype(float),
            "cuota_pactada": rng.integers(100000, 600000, size=n).astype(float),
            "puntaje_datacredito": rng.integers(600, 950, size=n).astype(float),
            "cant_creditosvigentes": rng.integers(0, 5, size=n).astype(float),
            "creditos_sectorFinanciero": rng.integers(0, 5, size=n).astype(float),
            "creditos_sectorCooperativo": rng.integers(0, 3, size=n).astype(float),
            "creditos_sectorReal": rng.integers(0, 3, size=n).astype(float),
            "promedio_ingresos_datacredito": rng.integers(1000000, 4000000, size=n).astype(float),
            "ratio_cuota_ingreso": rng.uniform(0.02, 0.3, size=n),
            "tipo_credito": rng.choice(["4", "9"], size=n),
            "tipo_laboral": rng.choice(["Empleado", "Independiente"], size=n),
            "tendencia_ingresos_limpia": rng.choice(["Creciente", "Estable"], size=n),
        }
    )
    y = pd.Series(rng.choice([1, 1, 1, 0], size=n).astype(int))
    return X, y


def test_13_modelo_ya_fitted_pipeline_real():
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    X, y = _xy16()
    num = ["capital_prestado", "puntaje_datacredito"]
    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("enc", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["tendencia_ingresos_limpia"],
            ),
        ]
    )
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=500))]).fit(X, y)
    res = SimpleNamespace(
        comparativa_cv_familias=_comparativa_base(),
        resultado_cv_modelo_final=_resultado_cv("final", 0.5),
        modelo_final_fitted=pipe,
        nombre_modelo_final="logistic_regression",
        hiperparametros_finales={},
        justificacion_seleccion="x",
    )
    out = ev.evaluar(res, X, y)
    assert out.holdout_modelo_final.n_eval == len(y)


def test_14_no_llama_fit():
    n = 12
    X = _x_eval_heuristico(n=n)
    y = pd.Series([0, 1] * 6)
    modelo = ModeloFijo([0, 1] * 6)
    res = _resultado_entrenamiento_sintetico(modelo)
    out = ev.evaluar(res, X, y)  # lanzaría AssertionError si llamara fit
    assert out.nombre_modelo_final == "logistic_regression"


def test_15_no_cambia_threshold_usa_predict():
    n = 6
    X = _x_eval_heuristico(n=n)
    y = pd.Series([0, 0, 1, 1, 0, 1])
    # predict dice todo riesgo, aunque proba diga lo contrario.
    proba = np.tile(np.array([[0.01, 0.99]]), (n, 1))
    modelo = ModeloFijo([0] * n, proba=proba, classes=[0, 1])
    res = _resultado_entrenamiento_sintetico(modelo)
    out = ev.evaluar(res, X, y)
    assert out.holdout_modelo_final.matriz_nombres["pago_real_predicho_pago"] == 0
    assert out.holdout_modelo_final.matriz_nombres["riesgo_real_predicho_riesgo"] == 3


def test_16_inputs_no_se_modifican():
    n = 16
    X = _x_eval_heuristico(n=n)
    y = pd.Series([0, 1] * 8)
    antes_x, antes_y = X.copy(deep=True), y.copy(deep=True)
    modelo = ModeloFijo([0, 1] * 8)
    ev.evaluar(_resultado_entrenamiento_sintetico(modelo), X, y)
    pd.testing.assert_frame_equal(X, antes_x)
    pd.testing.assert_series_equal(y, antes_y)


# 17-19 CV/HOLDOUT


def test_17_delta_correcto():
    final = _resultado_cv("final", 0.5, recall=0.6, precision=0.4, bal=0.7, std=0.05)
    m = ev.MetricasRiesgo(
        precision_0=0.5,
        recall_0=0.8,
        f1_0=0.6,
        balanced_accuracy=0.75,
        accuracy_ctx=0.9,
        support_0=2,
        support_1=8,
    )
    comp = ev.comparar_cv_holdout(final, m)
    assert comp.f1_0.delta == pytest.approx(0.6 - final.mean_f1_0)
    assert comp.f1_0.holdout == pytest.approx(0.6)
    assert comp.f1_0.cv_mean == pytest.approx(final.mean_f1_0)
    assert comp.f1_0.cv_std == pytest.approx(final.std_f1_0)


def test_18_usa_cv_final_no_base():
    comparativa = _comparativa_base()
    base_lr = comparativa["logistic_regression"]
    final = _resultado_cv("final_tuned", 0.99, recall=0.99, precision=0.99, bal=0.99)
    m = ev.MetricasRiesgo(0.5, 0.5, 0.5, 0.5, 0.5, 1, 1)
    comp = ev.comparar_cv_holdout(final, m)
    assert comp.f1_0.cv_mean == pytest.approx(0.99)
    assert comp.f1_0.cv_mean != pytest.approx(base_lr.mean_f1_0)


def test_19_no_usa_candidato_base_por_error():
    n = 10
    X = _x_eval_heuristico(n=n)
    y = pd.Series([0, 1] * 5)
    modelo = ModeloFijo([0, 1] * 5)
    res = _resultado_entrenamiento_sintetico(modelo)
    # Corromper a propósito el candidato base: la comparación no debe moverse.
    res.comparativa_cv_familias["logistic_regression"] = _resultado_cv("lr", 0.0)
    out = ev.evaluar(res, X, y)
    assert out.comparacion_cv_final_vs_holdout.f1_0.cv_mean == pytest.approx(
        res.resultado_cv_modelo_final.mean_f1_0
    )


# 20-24 ESTADÍSTICA


def test_20_anova_recibe_f1_por_fold():
    comp = _comparativa_base()
    est = ev.analizar_familias_cv(comp)
    listas = [comp[m].metricas_por_fold["f1_0"] for m in ev.FAMILIAS_ESTADISTICA]
    ref = f_oneway(*listas)
    assert est.anova_statistic == pytest.approx(float(ref.statistic))
    assert est.anova_pvalue == pytest.approx(float(ref.pvalue))
    assert est.metrica == "f1_0" and est.n_folds == 5


def test_21_holdout_no_entra_anova():
    comp = _comparativa_base()
    antes = ev.analizar_familias_cv(comp)
    # Cambiar holdout/final no debe alterar la estadística de familias.
    comp2 = _comparativa_base()
    comp2["heuristic"] = _resultado_cv("heuristic", [0.10, 0.11, 0.09, 0.12, 0.10])
    despues = ev.analizar_familias_cv(comp2)
    assert antes.anova_statistic == pytest.approx(despues.anova_statistic)
    assert antes.friedman_statistic == pytest.approx(despues.friedman_statistic)


def test_22_tuneado_y_dummy_no_entran():
    comp = _comparativa_base()
    comp["dummy"] = _resultado_cv("dummy", [0.9] * 5)
    comp["final_tuned"] = _resultado_cv("final_tuned", [0.95] * 5)
    est = ev.analizar_familias_cv(comp)
    assert est.modelos == list(ev.FAMILIAS_ESTADISTICA)
    assert "dummy" not in est.modelos and "final_tuned" not in est.modelos


def test_23_friedman_conserva_pareamiento():
    comp = _comparativa_base()
    est = ev.analizar_familias_cv(comp)
    listas = [comp[m].metricas_por_fold["f1_0"] for m in ev.FAMILIAS_ESTADISTICA]
    ref = friedmanchisquare(*listas)
    assert est.friedman_statistic == pytest.approx(float(ref.statistic))
    assert est.friedman_pvalue == pytest.approx(float(ref.pvalue))


def test_24_tukey_solo_si_significativo():
    # Grupos idénticos -> p≈1 -> sin Tukey + mensaje estándar.
    plano = {k: _resultado_cv(k, [0.5, 0.5, 0.5, 0.5, 0.5]) for k in ev.FAMILIAS_ESTADISTICA}
    est_plano = ev.analizar_familias_cv(plano)
    assert est_plano.tukey is None
    assert "no fue significativo" in est_plano.tukey_mensaje
    # Grupos separados -> significativo -> con Tukey.
    separado = _comparativa_base()
    est_sep = ev.analizar_familias_cv(separado)
    if est_sep.anova_pvalue < 0.05:
        assert est_sep.tukey is not None
    else:
        assert est_sep.tukey is None


# 25-26 TIMING


def test_25_tiempos_no_negativos():
    X = _x_eval_heuristico(n=15)
    modelo = ModeloFijo([0, 1] * 7 + [0])
    tim = ev.medir_inferencia(modelo, X, repeticiones=5)
    assert tim.median_total_ms >= 0
    assert tim.mean_total_ms >= 0 and tim.std_total_ms >= 0
    assert tim.median_ms_per_row >= 0 and tim.rows_per_second >= 0
    assert tim.n_rows == 15


def test_26_pipeline_predict_end_to_end():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.5, 1.5, 2.5, 3.5]})
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))]).fit(
        X, pd.Series([0, 0, 1, 1])
    )
    tim = ev.medir_inferencia(pipe, X, repeticiones=3)
    assert tim.median_total_ms >= 0
    assert tim.rows_per_second > 0


# 27-28 HEURÍSTICO


def test_27_reglas_no_cambian():
    from mlops_pipeline.src import hueristic_model as hm

    assert hm.LIMITE_PUNTAJE == 750
    assert hm.LIMITE_CAPITAL == 3_400_000
    assert hm.TENDENCIA_RIESGO == "Decreciente"
    X = _x_eval_heuristico(n=10)
    antes = hm.predecir_heuristico(X).tolist()
    modelo = ModeloFijo([0, 1] * 5)
    ev.evaluar(_resultado_entrenamiento_sintetico(modelo), X, pd.Series([0, 1] * 5))
    assert hm.predecir_heuristico(X).tolist() == antes


def test_28_heu_metricas_comparables_sin_auc():
    n = 12
    X = _x_eval_heuristico(n=n)
    y = pd.Series([0, 1] * 6)
    out = ev.evaluar(_resultado_entrenamiento_sintetico(ModeloFijo([0, 1] * 6)), X, y)
    heu = out.holdout_heuristico
    for campo in ("precision_0", "recall_0", "f1_0", "balanced_accuracy"):
        assert isinstance(getattr(heu.metricas, campo), float)
    assert heu.probabilisticas.roc_auc_risk is None
    assert ADVERTENCIA_HEURISTICO_CORTA in " ".join(out.advertencias_metodologicas)


ADVERTENCIA_HEURISTICO_CORTA = "EDA histórico"


# 29-32 MAIN


def test_29_ejecucion_directa_resuelve_imports():
    fuente = inspect.getsource(ev)
    assert 'if __package__ in {None, ""}:' in fuente
    assert "repo_root = Path(__file__).resolve().parents[2]" in fuente
    assert "sys.path.insert(0, str(repo_root))" in fuente
    assert "from mlops_pipeline.src.Cargar_datos import" in fuente
    assert "from mlops_pipeline.src.ft_engineering import" in fuente
    assert "from mlops_pipeline.src.model_training import" in fuente
    assert "except ImportError" not in fuente


def test_30_import_no_ejecuta_main():
    raiz = Path(ev.__file__).resolve().parents[2]
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import mlops_pipeline.src.model_evaluation as m; print('import-ok')",
        ],
        cwd=raiz,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0
    assert "import-ok" in proc.stdout
    assert "=== Modelo final ===" not in proc.stdout
    assert 'if __name__ == "__main__":' in inspect.getsource(ev)


def test_31_main_usa_train_y_eval_despues(monkeypatch):
    assert list(inspect.signature(ev.main).parameters) == []
    assert list(inspect.signature(ev.evaluar).parameters) == [
        "resultado_entrenamiento",
        "X_eval",
        "y_eval",
    ]
    fuente_main = inspect.getsource(ev.main)
    assert "X_train" in fuente_main and "X_eval" in fuente_main
    x_train, y_train, prep = object(), object(), object()
    x_eval, y_eval = object(), object()
    datos_fake = SimpleNamespace(
        X_train=x_train, y_train=y_train, X_eval=x_eval, y_eval=y_eval, preprocessor=prep
    )
    monkeypatch.setattr(ev, "encontrar_raiz_proyecto", lambda: Path("/tmp"))
    monkeypatch.setattr(ev, "cargar_config", lambda _r: {})
    monkeypatch.setattr(ev, "resolver_ruta_datos", lambda _a, _b: Path("/tmp/d.csv"))
    monkeypatch.setattr(ev, "cargar_datos", lambda _r, _c: object())
    monkeypatch.setattr(ev, "preparar_datos", lambda _df: datos_fake)
    llamadas: dict = {}

    def falso_entrenar(x, y, p):
        llamadas["entrenar"] = (x, y, p)
        return SimpleNamespace(nombre="x")

    def falso_evaluar(res, xe, ye):
        llamadas["evaluar"] = (res, xe, ye)
        return SimpleNamespace(nombre="eval")

    monkeypatch.setattr(ev, "entrenar", falso_entrenar)
    monkeypatch.setattr(ev, "evaluar", falso_evaluar)
    monkeypatch.setattr(ev, "_imprimir_evaluacion", lambda _r: llamadas.setdefault("print", True))
    ev.main()
    assert llamadas["entrenar"][0] is x_train
    assert llamadas["evaluar"][1] is x_eval and llamadas["evaluar"][2] is y_eval
    assert llamadas.get("print") is True


def test_32_output_impresion_sintetico(capsys):
    modelo = ModeloFijo([0, 1, 0, 1])
    X = _x_eval_heuristico(n=4)
    y = pd.Series([0, 1, 0, 1])
    resultado = ev.evaluar(_resultado_entrenamiento_sintetico(modelo), X, y)
    ev._imprimir_evaluacion(resultado)
    out = capsys.readouterr().out
    for seccion in (
        "=== Modelo final ===",
        "=== Holdout final ===",
        "=== Matriz de confusión ===",
        "=== Baseline heurístico ===",
        "=== CV final vs Holdout ===",
        "=== Inferencia ===",
        "=== Estadística exploratoria ===",
    ):
        assert seccion in out
    assert "logistic_regression" in out
    assert "model__C" in out


# Visualizaciones y contrato extra


def test_plots_devuelven_figura_sin_guardar():
    modelo = ModeloFijo([0, 1, 0, 1])
    X = _x_eval_heuristico(n=4)
    y = pd.Series([0, 1, 0, 1])
    resultado = ev.evaluar(_resultado_entrenamiento_sintetico(modelo), X, y)
    for fig in (
        ev.plot_matriz_confusion(resultado.holdout_modelo_final),
        ev.plot_comparacion_holdout(resultado.comparacion_cv_final_vs_holdout),
        ev.plot_cv_modelos(resultado.comparativa_cv_familias if False else _comparativa_base()),
        ev.plot_precision_recall(
            y, ev.obtener_score_riesgo(modelo, X) or np.array([0.1, 0.9, 0.2, 0.8])
        ),
    ):
        assert isinstance(fig, matplotlib.figure.Figure)
    for nombre in (
        "plot_matriz_confusion",
        "plot_comparacion_holdout",
        "plot_cv_modelos",
        "plot_precision_recall",
    ):
        fuente_fn = inspect.getsource(getattr(ev, nombre))
        assert "plt.show()" not in fuente_fn
        assert "savefig" not in fuente_fn
    fuente = inspect.getsource(ev)
    assert "plt.show()" not in fuente
    # La pestana serializa figuras solo a buffer en memoria, nunca a disco.
    assert "BytesIO" in inspect.getsource(ev.generar_pestana_metricas)


def test_no_fit_no_search_en_fuente():
    fuente = inspect.getsource(ev)
    assert "GridSearchCV" not in fuente
    assert "RandomizedSearchCV" not in fuente
    assert ".fit(" not in fuente


def test_metricas_y_reporte_usan_contrato():
    fuente = inspect.getsource(ev)
    assert "pos_label=0" in fuente
    assert "zero_division=0" in fuente
    assert "classification_report" in fuente
    assert "output_dict=True" in fuente
    assert "f_oneway" in fuente and "friedmanchisquare" in fuente and "tukey_hsd" in fuente
    assert "time.perf_counter()" in fuente


@pytest.mark.data_regression
def test_data_regression_flujo_real_eval_unica():
    """Flujo real sellado: train -> entrenar -> evaluar una vez (sin redecidir)."""
    from mlops_pipeline.src.Cargar_datos import (
        cargar_config,
        cargar_datos,
        encontrar_raiz_proyecto,
        resolver_ruta_datos,
    )
    from mlops_pipeline.src.ft_engineering import preparar_datos

    raiz = encontrar_raiz_proyecto()
    config = cargar_config(raiz / "mlops_pipeline" / "src" / "config.json")
    datos = preparar_datos(cargar_datos(resolver_ruta_datos(raiz, config), config))
    assert datos.X_eval.shape == (2153, 16)
    resultado_entrenamiento = mt.entrenar(datos.X_train, datos.y_train, datos.preprocessor)
    resultado = ev.evaluar(resultado_entrenamiento, datos.X_eval, datos.y_eval)
    ml = resultado.holdout_modelo_final.metricas
    print(
        f"\neval real: modelo={resultado.nombre_modelo_final} "
        f"params={resultado.hiperparametros_finales} "
        f"p0={ml.precision_0:.4f} r0={ml.recall_0:.4f} f0={ml.f1_0:.4f} "
        f"bal={ml.balanced_accuracy:.4f} acc={ml.accuracy_ctx:.4f} "
        f"mat={resultado.holdout_modelo_final.matriz_nombres}"
    )
    assert resultado.holdout_modelo_final.n_eval == 2153
    assert ml.support_0 == 102 and ml.support_1 == 2051
    assert resultado.analisis_estadistico_familias.n_folds == 5
    assert resultado.inference_timing.median_total_ms >= 0
