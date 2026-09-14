"""Pruebas del protocolo experimental CV-safe de model_training."""

import inspect
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from mlops_pipeline.src import model_training as mt
from mlops_pipeline.src.ft_engineering import crear_preprocesador


def _xy(n=120, semilla=11):
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
            "tendencia_ingresos_limpia": rng.choice(
                ["Creciente", "Estable", "Decreciente", "Ausente o inválida"], size=n
            ),
        }
    )
    y = pd.Series(rng.choice([1, 1, 1, 0], size=n).astype(int))
    return X, y


def test_folds_cinco_y_reproducibles():
    X, y = _xy()
    a = mt.crear_folds(X, y)
    b = mt.crear_folds(X, y)
    assert len(a) == 5
    for (tr_a, va_a), (tr_b, va_b) in zip(a, b, strict=True):
        assert (tr_a == tr_b).all()
        assert (va_a == va_b).all()


def test_folds_cobertura_y_sin_solape_y_estratificados():
    X, y = _xy(n=200)
    folds = mt.crear_folds(X, y)
    vistos = np.concatenate([va for _, va in folds])
    assert sorted(vistos.tolist()) == list(range(len(X)))
    for tr, va in folds:
        assert set(tr).isdisjoint(set(va))
        assert abs((y.iloc[va] == 0).mean() - (y == 0).mean()) < 0.05


def test_scorers_clase_0():
    from sklearn.metrics import f1_score, precision_score, recall_score

    scorers = mt.crear_scorers()
    assert set(scorers) == {"precision_0", "recall_0", "f1_0", "balanced_accuracy"}
    esperados = {
        "precision_0": precision_score,
        "recall_0": recall_score,
        "f1_0": f1_score,
    }
    for nombre, funcion in esperados.items():
        assert scorers[nombre]._score_func is funcion
        assert scorers[nombre]._kwargs.get("pos_label") == 0
        assert scorers[nombre]._kwargs.get("zero_division") == 0
    # Antisimétrico: con pos_label=1 este caso daría 2/3, con 0 da 1.0.
    y_true = pd.Series([0, 0, 1, 1])
    y_pred = pd.Series([0, 1, 1, 1])
    assert abs(precision_score(y_true, y_pred, pos_label=1) - 2 / 3) < 1e-9
    assert abs(precision_score(y_true, y_pred, pos_label=0) - 1.0) < 1e-9
    assert abs(recall_score(y_true, y_pred, pos_label=0) - 0.5) < 1e-9
    assert abs(f1_score(y_true, y_pred, pos_label=0) - 2 / 3) < 1e-9


def test_preprocessor_original_unfitted_y_clones():
    prep = crear_preprocesador()
    candidatos = mt.crear_modelos_candidatos(prep)
    assert set(candidatos) == set(mt.FAMILIAS_ML)
    with pytest.raises(NotFittedError):
        prep.transform(_xy(n=10)[0])
    for cand in candidatos.values():
        assert cand.named_steps["preprocessor"] is not prep


def test_lr_escala_solo_numerica():
    prep = crear_preprocesador()
    lr = mt.crear_modelos_candidatos(prep)["logistic_regression"]
    num_pipe = lr.named_steps["preprocessor"].transformers[0][1]
    cat_pipe = lr.named_steps["preprocessor"].transformers[1][1]
    assert [n for n, _ in num_pipe.steps] == ["imputer", "scaler"]
    assert [n for n, _ in cat_pipe.steps] == ["imputer", "encoder"]
    assert "StandardScaler" not in repr(cat_pipe)


def test_rf_hgb_sin_scaler_y_configurados():
    prep = crear_preprocesador()
    candidatos = mt.crear_modelos_candidatos(prep)
    for nombre in ("random_forest", "hist_gradient_boosting"):
        assert "scaler" not in repr(candidatos[nombre]).lower()
    rf = candidatos["random_forest"].named_steps["model"]
    hgb = candidatos["hist_gradient_boosting"].named_steps["model"]
    lr = candidatos["logistic_regression"].named_steps["model"]
    assert rf.class_weight == "balanced" and rf.n_jobs == 1
    assert rf.random_state == 42
    assert hgb.class_weight == "balanced" and hgb.random_state == 42
    assert lr.class_weight == "balanced" and lr.random_state == 42


def test_heuristico_sin_transformer_y_dummy_aparte():
    prep = crear_preprocesador()
    X, y = _xy()
    folds = mt.crear_folds(X, y)
    comparativa, dummy = mt.comparar_modelos(X, y, prep, folds)
    assert "heuristic" in comparativa and dummy.nombre == "dummy"
    assert not hasattr(mt.HeuristicModel(), "named_steps")


def test_resultadocv_forma_y_tiempos():
    X, y = _xy()
    res = mt.evaluar_cv("dummy", mt.crear_baselines()["dummy"], X, y, mt.crear_folds(X, y))
    for metrica in ("precision_0", "recall_0", "f1_0", "balanced_accuracy"):
        assert len(res.metricas_por_fold[metrica]) == 5
    assert abs(res.mean_f1_0 - np.mean(res.metricas_por_fold["f1_0"])) < 1e-12
    assert res.std_f1_0 >= 0
    assert res.mean_cv_fit_time >= 0 and res.mean_cv_score_time >= 0


def _resultado(nombre, f1, std, recall, fit=0.1):
    return mt.ResultadoCV(
        nombre=nombre,
        metricas_por_fold={},
        mean_precision_0=0.1,
        std_precision_0=0,
        mean_recall_0=recall,
        std_recall_0=0,
        mean_f1_0=f1,
        std_f1_0=std,
        mean_balanced_accuracy=0.6,
        std_balanced_accuracy=0,
        mean_cv_fit_time=fit,
        std_cv_fit_time=0,
        mean_cv_score_time=0.1,
        std_cv_score_time=0,
    )


def test_seleccion_solo_ml_con_justificacion():
    comparativa = {
        "heuristic": _resultado("heuristic", 0.99, 0, 0.9),
        "dummy": _resultado("dummy", 0.99, 0, 0.9),
        "logistic_regression": _resultado("logistic_regression", 0.15, 0.01, 0.5),
        "random_forest": _resultado("random_forest", 0.25, 0.02, 0.4, fit=1.0),
        "hist_gradient_boosting": _resultado("hist_gradient_boosting", 0.20, 0.01, 0.45, fit=0.5),
    }
    elegido, justificacion = mt.seleccionar_modelo(comparativa)
    assert elegido == "random_forest"
    assert "recall_0" in justificacion and "Comparables" in justificacion
    with pytest.raises(ValueError, match="3 familias ML"):
        mt.seleccionar_modelo({"random_forest": comparativa["random_forest"]})


def test_seleccion_prefiere_recall_estabilidad_costo_en_empate():
    comparativa = {
        "logistic_regression": _resultado("logistic_regression", 0.134, 0.009, 0.58, fit=0.06),
        "random_forest": _resultado("random_forest", 0.04, 0.02, 0.03, fit=0.5),
        "hist_gradient_boosting": _resultado("hist_gradient_boosting", 0.14, 0.012, 0.16, fit=1.0),
    }
    elegido, justificacion = mt.seleccionar_modelo(comparativa)
    assert elegido == "logistic_regression"
    assert "logistic_regression" in justificacion


def test_tuning_solo_elegido_y_con_folds():
    prep = crear_preprocesador()
    X, y = _xy(n=150)
    folds = mt.crear_folds(X, y)
    with pytest.raises(ValueError, match="familias ML"):
        mt.ajustar_hiperparametros("heuristic", prep, X, y, folds)
    search = mt.ajustar_hiperparametros("logistic_regression", prep, X, y, folds)
    check_is_fitted(search)
    assert set(search.best_params_) == {"model__C"}


def test_entrenar_firma_sin_eval_no_modifica_y_reproducible():
    params = list(inspect.signature(mt.entrenar).parameters)
    assert params == ["X_train", "y_train", "preprocessor"]
    assert not any("eval" in p.lower() for p in params)
    prep = crear_preprocesador()
    X, y = _xy(n=150)
    antes_x, antes_y = X.copy(deep=True), y.copy(deep=True)
    res = mt.entrenar(X, y, prep)
    pd.testing.assert_frame_equal(X, antes_x)
    pd.testing.assert_series_equal(y, antes_y)
    check_is_fitted(res.modelo_final_fitted)
    assert isinstance(res.modelo_final_fitted, Pipeline)
    assert res.resultado_cv_modelo_final is not res.comparativa_cv_familias.get(
        res.nombre_modelo_final
    )
    res2 = mt.entrenar(X, y, crear_preprocesador())
    assert res.nombre_modelo_final == res2.nombre_modelo_final
    assert res.resultado_cv_modelo_final.mean_f1_0 == res2.resultado_cv_modelo_final.mean_f1_0
    with pytest.raises(NotFittedError):
        clone(prep).transform(X)


def test_hgb_search_usa_random_state():
    prep = crear_preprocesador()
    X, y = _xy(n=150)
    search = mt.ajustar_hiperparametros("hist_gradient_boosting", prep, X, y, mt.crear_folds(X, y))
    assert search.random_state == 42


def _resultado_entrenamiento_sintetico():
    comparativa = {
        "heuristic": _resultado("heuristic", 0.11, 0.01, 0.60, fit=0.02),
        "logistic_regression": _resultado("logistic_regression", 0.134, 0.009, 0.58, fit=0.06),
        "random_forest": _resultado("random_forest", 0.04, 0.02, 0.03, fit=0.5),
        "hist_gradient_boosting": _resultado("hist_gradient_boosting", 0.14, 0.012, 0.16, fit=1.0),
    }
    dummy = _resultado("dummy", 0.05, 0.005, 0.10, fit=0.02)
    final = _resultado("final_tuned", 0.132, 0.008, 0.589, fit=0.07)
    return mt.ResultadoEntrenamiento(
        comparativa_cv_familias=comparativa,
        resultado_dummy=dummy,
        resultado_cv_modelo_final=final,
        modelo_final_fitted=object(),
        nombre_modelo_final="logistic_regression",
        hiperparametros_finales={"model__C": 0.1},
        justificacion_seleccion="Elegido: logistic_regression (sintético).",
    )


def test_import_no_ejecuta_main():
    raiz = Path(mt.__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "-c", "import mlops_pipeline.src.model_training as m; print('import-ok')"],
        cwd=raiz,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0
    assert "import-ok" in proc.stdout
    assert "=== Comparación CV de modelos ===" not in proc.stdout
    fuente = inspect.getsource(mt)
    assert 'if __name__ == "__main__":' in fuente


def test_ejecucion_directa_resuelve_imports_absolutos():
    fuente = inspect.getsource(mt)
    assert 'if __package__ in {None, ""}:' in fuente
    assert "repo_root = Path(__file__).resolve().parents[2]" in fuente
    assert "sys.path.insert(0, str(repo_root))" in fuente
    assert "from mlops_pipeline.src.Cargar_datos import" in fuente
    assert "from mlops_pipeline.src.ft_engineering import" in fuente
    assert "from mlops_pipeline.src.hueristic_model import" in fuente
    assert "except ImportError" not in fuente
    assert "from ft_engineering import" not in fuente
    assert "from Cargar_datos import" not in fuente
    assert "from hueristic_model import" not in fuente


def test_bootstrap_no_duplica_repo_root():
    fuente = inspect.getsource(mt)
    assert "if str(repo_root) not in sys.path:" in fuente
    repo_root = str(Path(mt.__file__).resolve().parents[2])
    simulado: list = []
    for _ in range(2):
        if repo_root not in simulado:
            simulado.insert(0, repo_root)
    assert simulado.count(repo_root) == 1
    assert sys.path.count(repo_root) <= 1


def test_imprimir_resultados_con_sintetico(capsys):
    resultado = _resultado_entrenamiento_sintetico()
    mt._imprimir_resultados(resultado)
    salida = capsys.readouterr().out
    assert "=== Comparación CV de modelos ===" in salida
    assert "Heuristic" in salida
    assert "LogisticRegression" in salida
    assert "RandomForest" in salida
    assert "HistGradientBoosting" in salida
    assert "=== Baseline técnico ===" in salida
    assert "Dummy" in salida
    assert "=== Selección ===" in salida
    assert "logistic_regression" in salida
    assert "=== Hiperparámetros ===" in salida
    assert "model__C" in salida
    assert "=== CV configuración final ===" in salida
    assert (
        "Todas las métricas anteriores se calcularon exclusivamente sobre X_train/y_train."
        in salida
    )
    assert "X_eval/y_eval permanecen sellados para model_evaluation.py." in salida


def test_main_usa_solo_train_y_preprocessor(monkeypatch, capsys):
    assert list(inspect.signature(mt.main).parameters) == []
    fuente_main = inspect.getsource(mt.main)
    assert "X_eval" not in fuente_main
    assert "y_eval" not in fuente_main
    x_train, y_train, preprocessor = object(), object(), object()
    datos_fake = SimpleNamespace(X_train=x_train, y_train=y_train, preprocessor=preprocessor)
    monkeypatch.setattr(mt, "encontrar_raiz_proyecto", lambda: Path("/tmp"))
    monkeypatch.setattr(mt, "cargar_config", lambda _ruta: {})
    monkeypatch.setattr(mt, "resolver_ruta_datos", lambda _raiz, _cfg: Path("/tmp/d.csv"))
    monkeypatch.setattr(mt, "cargar_datos", lambda _ruta, _cfg: object())
    monkeypatch.setattr(mt, "preparar_datos", lambda _df: datos_fake)
    llamadas: dict = {}

    def falso_entrenar(x, y, prep):
        llamadas["args"] = (x, y, prep)
        return _resultado_entrenamiento_sintetico()

    monkeypatch.setattr(mt, "entrenar", falso_entrenar)
    mt.main()
    assert llamadas["args"][0] is x_train
    assert llamadas["args"][1] is y_train
    assert llamadas["args"][2] is preprocessor
    salida = capsys.readouterr().out
    assert "=== Selección ===" in salida
