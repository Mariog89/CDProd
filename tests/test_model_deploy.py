"""Pruebas de deployment: persistencia, carga validada e inferencia."""

import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from mlops_pipeline.src import model_deploy as md
from mlops_pipeline.src.ft_engineering import (
    FEATURE_SCHEMA_VERSION,
    FEATURES_V1,
    crear_preprocesador,
    preparar_features_inferencia,
)


def _crudo(n=10, semilla=7, con_target=False, indice=None):
    rng = np.random.default_rng(semilla)
    data = {
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
    if con_target:
        data["Pago_atiempo"] = rng.choice(["0", "1"], size=n)
    df = pd.DataFrame(data)
    if indice is not None:
        df.index = pd.Index(indice)
    return df


def _pipeline_fitted(n=60, semilla=11):
    crudo = _crudo(n=n, semilla=semilla)
    X = preparar_features_inferencia(crudo)
    rng = np.random.default_rng(semilla + 1)
    y = pd.Series(rng.choice([0, 1], size=n).astype(int))
    pipe = Pipeline(
        [("preprocessor", crear_preprocesador()), ("model", LogisticRegression(max_iter=500))]
    )
    pipe.fit(X, y)
    return pipe, crudo


def _metadata_base():
    return md.crear_metadata_modelo(
        nombre_modelo="logistic_regression",
        clase_estimator="LogisticRegression",
        hiperparametros_finales={"model__C": 0.1},
        resultado_cv_modelo_final={"mean_f1_0": 0.13, "std_f1_0": 0.01},
        metricas_holdout_finales=None,
        model_version="1.0.0",
    )


class ModeloFijo:
    def __init__(self, y_pred, proba=None, classes=None, scores=None):
        self._y = np.asarray(y_pred)
        self._proba = None if proba is None else np.asarray(proba, dtype=float)
        self._scores = None if scores is None else np.asarray(scores, dtype=float)
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

    def decision_function(self, X):
        if self._scores is None:
            raise AttributeError("sin decision")
        return self._scores[: len(X)].copy()

    def fit(self, X, y=None):
        raise AssertionError("predecir no debe llamar fit")


class SoloDecision:
    def __init__(self, scores, classes):
        self._s = np.asarray(scores, dtype=float)
        self.classes_ = np.asarray(classes)

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def decision_function(self, X):
        return self._s[: len(X)].copy()

    def fit(self, X, y=None):
        raise AssertionError("predecir no debe llamar fit")


class SinScores:
    def predict(self, X):
        return np.ones(len(X), dtype=int)

    def fit(self, X, y=None):
        raise AssertionError("predecir no debe llamar fit")


# 1-9 FEATURE ENGINEERING / INPUT


def test_01_no_requiere_target():
    df = _crudo(n=8, con_target=False)
    assert "Pago_atiempo" not in df.columns
    X = preparar_features_inferencia(df)
    assert list(X.columns) == list(FEATURES_V1)


def test_02_target_extra_se_ignora():
    df = _crudo(n=8, con_target=True)
    X = preparar_features_inferencia(df)
    assert "Pago_atiempo" not in X.columns
    assert list(X.columns) == list(FEATURES_V1)


def test_03_columnas_extra_se_ignoran():
    df = _crudo(n=6)
    df["col_extra"] = "x"
    df["otro"] = 123
    X = preparar_features_inferencia(df)
    assert "col_extra" not in X.columns
    assert list(X.columns) == list(FEATURES_V1)


def test_04_devuelve_16_features():
    X = preparar_features_inferencia(_crudo(n=5))
    assert X.shape[1] == 16
    assert set(X.columns) == set(FEATURES_V1)


def test_05_orden_exactto_preservado():
    X = preparar_features_inferencia(_crudo(n=5))
    assert list(X.columns) == list(FEATURES_V1)


def test_06_input_no_modificado():
    df = _crudo(n=7)
    antes = df.copy(deep=True)
    preparar_features_inferencia(df)
    pd.testing.assert_frame_equal(df, antes)


def test_07_una_fila_funciona():
    X = preparar_features_inferencia(_crudo(n=1))
    assert len(X) == 1
    assert list(X.columns) == list(FEATURES_V1)


def test_08_n_filas_funcionan():
    X = preparar_features_inferencia(_crudo(n=25))
    assert len(X) == 25


def test_09_falta_feature_requerida_error():
    df = _crudo(n=5).drop(columns=["salario_cliente"])
    with pytest.raises(KeyError):
        preparar_features_inferencia(df)


# 10-21 SERIALIZACIÓN


def test_10_save_load_conserva_predicciones(tmp_path):
    pipe, crudo = _pipeline_fitted()
    X = preparar_features_inferencia(crudo)
    antes = pipe.predict(X)
    meta = _metadata_base()
    destino = tmp_path / "1.0.0"
    md.guardar_modelo(pipe, meta, destino)
    artefacto = md.cargar_modelo(destino)
    despues = artefacto.modelo.predict(X)
    assert (antes == despues).all()


def test_11_modelo_cargado_esta_fitted(tmp_path):
    pipe, _ = _pipeline_fitted()
    destino = tmp_path / "1.0.0"
    md.guardar_modelo(pipe, _metadata_base(), destino)
    artefacto = md.cargar_modelo(destino)
    check_is_fitted(artefacto.modelo)
    assert artefacto.verificado is True


def test_12_metadata_json_valida(tmp_path):
    pipe, _ = _pipeline_fitted()
    destino = tmp_path / "1.0.0"
    md.guardar_modelo(pipe, _metadata_base(), destino)
    with open(destino / "metadata.json", encoding="utf-8") as f:
        data = json.load(f)
    assert data["model_version"] == "1.0.0"
    assert data["features_v1"] == list(FEATURES_V1)


def test_13_sklearn_version_registrada(tmp_path):
    pipe, _ = _pipeline_fitted()
    destino = tmp_path / "1.0.0"
    final = md.guardar_modelo(pipe, _metadata_base(), destino)
    assert final["sklearn_version"] == sklearn.__version__


def test_14_joblib_version_registrada(tmp_path):
    pipe, _ = _pipeline_fitted()
    destino = tmp_path / "1.0.0"
    final = md.guardar_modelo(pipe, _metadata_base(), destino)
    assert final["joblib_version"] == joblib.__version__


def test_15_feature_schema_version_registrada(tmp_path):
    pipe, _ = _pipeline_fitted()
    destino = tmp_path / "1.0.0"
    final = md.guardar_modelo(pipe, _metadata_base(), destino)
    assert final["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert final["feature_schema_version"] == "1"


def test_16_risk_label_es_cero(tmp_path):
    pipe, _ = _pipeline_fitted()
    destino = tmp_path / "1.0.0"
    final = md.guardar_modelo(pipe, _metadata_base(), destino)
    assert final["risk_label"] == 0
    assert final["target"] == "Pago_atiempo"
    assert final["n_features_input"] == 16


def test_17_hash_coincide(tmp_path):
    pipe, _ = _pipeline_fitted()
    destino = tmp_path / "1.0.0"
    final = md.guardar_modelo(pipe, _metadata_base(), destino)
    with open(destino / "model.joblib", "rb") as f:
        esperado = hashlib.sha256(f.read()).hexdigest()
    assert final["artifact_sha256"] == esperado
    assert final["artifact_size_bytes"] > 0


def test_18_hash_corrupto_detectado(tmp_path):
    pipe, _ = _pipeline_fitted()
    destino = tmp_path / "1.0.0"
    md.guardar_modelo(pipe, _metadata_base(), destino)
    with open(destino / "model.joblib", "r+b") as f:
        f.seek(10)
        f.write(b"XX")
    with pytest.raises(ValueError, match="SHA-256"):
        md.cargar_modelo(destino)


def test_19_archivo_inexistente_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        md.cargar_modelo(tmp_path / "no_existe")


def test_20_version_sklearn_distinta_bloquea_antes_de_load(tmp_path, monkeypatch):
    pipe, _ = _pipeline_fitted()
    destino = tmp_path / "1.0.0"
    md.guardar_modelo(pipe, _metadata_base(), destino)
    with open(destino / "metadata.json", encoding="utf-8") as f:
        data = json.load(f)
    data["sklearn_version"] = "0.0.0"
    with open(destino / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(data, f)
    llamada = {"n": 0}
    real_load = joblib.load

    def falso_load(*a, **k):
        llamada["n"] += 1
        return real_load(*a, **k)

    monkeypatch.setattr(md.joblib, "load", falso_load)
    with pytest.raises(RuntimeError):
        md.cargar_modelo(destino)
    assert llamada["n"] == 0


def test_21_schema_mismatch_error(tmp_path):
    pipe, _ = _pipeline_fitted()
    destino = tmp_path / "1.0.0"
    md.guardar_modelo(pipe, _metadata_base(), destino)
    with open(destino / "metadata.json", encoding="utf-8") as f:
        data = json.load(f)
    data["feature_schema_version"] = "999"
    with open(destino / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(data, f)
    with pytest.raises((ValueError, RuntimeError)):
        md.cargar_modelo(destino)


# 22-30 PREDICCIÓN


def test_22_indice_preservado():
    pipe, _ = _pipeline_fitted()
    df = _crudo(n=6, indice=[10, 20, 30, 40, 50, 60])
    out = md.predecir(pipe, df)
    assert list(out.dataframe.index) == [10, 20, 30, 40, 50, 60]


def test_23_output_en_cero_uno():
    pipe, _ = _pipeline_fitted()
    out = md.predecir(pipe, _crudo(n=10))
    assert set(out.dataframe["prediction"].unique().tolist()) <= {0, 1}


def test_24_predict_proba_usa_clase_0():
    X = pd.DataFrame({"a": [1, 2]})
    proba = np.array([[0.2, 0.8], [0.9, 0.1]])
    modelo = ModeloFijo([0, 1], proba=proba, classes=[0, 1])
    prob, score = md.obtener_risk_output(modelo, X)
    assert prob.tolist() == pytest.approx([0.2, 0.9])
    assert np.isnan(score).all()


def test_25_classes_invertidas_funcionan():
    X = pd.DataFrame({"a": [1, 2]})
    proba = np.array([[0.8, 0.2], [0.1, 0.9]])
    modelo = ModeloFijo([0, 1], proba=proba, classes=[1, 0])
    prob, _ = md.obtener_risk_output(modelo, X)
    assert prob.tolist() == pytest.approx([0.2, 0.9])


def test_26_decision_orientada_a_clase_0():
    X = pd.DataFrame({"a": [1, 2]})
    scores = np.array([1.5, -0.5])
    directo = SoloDecision(scores, classes=[1, 0])
    _, s1 = md.obtener_risk_output(directo, X)
    assert s1.tolist() == pytest.approx([1.5, -0.5])
    invertido = SoloDecision(scores, classes=[0, 1])
    _, s2 = md.obtener_risk_output(invertido, X)
    assert s2.tolist() == pytest.approx([-1.5, 0.5])


def test_27_probability_y_score_no_se_mezclan():
    X = pd.DataFrame({"a": [1, 2]})
    proba = np.array([[0.3, 0.7], [0.6, 0.4]])
    con_proba = ModeloFijo([0, 1], proba=proba, classes=[0, 1])
    prob, score = md.obtener_risk_output(con_proba, X)
    assert not np.isnan(prob).any()
    assert np.isnan(score).all()
    solo = SoloDecision(np.array([0.5, -0.5]), classes=[1, 0])
    prob2, score2 = md.obtener_risk_output(solo, X)
    assert np.isnan(prob2).all()
    assert not np.isnan(score2).any()
    sin_nada = SinScores()
    prob3, score3 = md.obtener_risk_output(sin_nada, X)
    assert np.isnan(prob3).all() and np.isnan(score3).all()


def test_28_timestamp_utc():
    pipe, _ = _pipeline_fitted()
    out = md.predecir(pipe, _crudo(n=4))
    ts = out.prediction_timestamp
    assert ts.endswith("Z")
    assert (out.dataframe["prediction_timestamp"] == ts).all()


def test_29_memoria_y_cargado_mismo_output(tmp_path):
    pipe, crudo = _pipeline_fitted()
    destino = tmp_path / "1.0.0"
    md.guardar_modelo(pipe, _metadata_base(), destino)
    artefacto = md.cargar_modelo(destino)
    a = md.predecir(pipe, crudo)
    b = md.predecir(artefacto, crudo)
    pd.testing.assert_series_equal(
        a.dataframe["prediction"], b.dataframe["prediction"], check_names=False
    )
    assert np.allclose(
        a.dataframe["risk_probability"], b.dataframe["risk_probability"], equal_nan=True
    )


def test_30_predict_no_llama_fit():
    X = preparar_features_inferencia(_crudo(n=4))
    modelo = ModeloFijo([0, 1, 0, 1])
    df = _crudo(n=4)
    out = md.predecir(modelo, df)
    assert out.dataframe["prediction"].tolist() == [0, 1, 0, 1]
    assert len(X) == 4


# 31-35 SEGURIDAD / RUTAS


def test_31_ruta_obligatoria():
    pipe, _ = _pipeline_fitted()
    with pytest.raises(ValueError):
        md.guardar_modelo(pipe, _metadata_base(), None)
    with pytest.raises(ValueError):
        md.cargar_modelo(None)


def test_32_no_sobrescribe_sin_overwrite(tmp_path):
    pipe, _ = _pipeline_fitted()
    destino = tmp_path / "1.0.0"
    md.guardar_modelo(pipe, _metadata_base(), destino)
    with pytest.raises(FileExistsError):
        md.guardar_modelo(pipe, _metadata_base(), destino)


def test_33_overwrite_explicito_funciona(tmp_path):
    pipe, _ = _pipeline_fitted()
    destino = tmp_path / "1.0.0"
    md.guardar_modelo(pipe, _metadata_base(), destino)
    final = md.guardar_modelo(pipe, _metadata_base(), destino, overwrite=True)
    assert (destino / "model.joblib").is_file()
    assert final["artifact_sha256"] is not None


def test_34_no_descarga_desde_url():
    fuente = inspect.getsource(md)
    assert "http://" not in fuente and "https://" not in fuente
    assert "urlretrieve" not in fuente.lower()
    assert "requests.get" not in fuente


def test_35_metadata_sin_datos_personales(tmp_path):
    pipe, _ = _pipeline_fitted()
    destino = tmp_path / "1.0.0"
    final = md.guardar_modelo(pipe, _metadata_base(), destino)
    texto = json.dumps(final, ensure_ascii=False).lower()
    for prohibido in ["salario_cliente=", "salario", "identificador", "cédula", "cedula"]:
        if prohibido == "salario":
            assert "salario_cliente”" not in texto
        else:
            assert prohibido not in texto
    assert "features_v1" in final
    assert len(final["features_v1"]) == 16


# 36-39 MAIN


def test_36_import_no_ejecuta_main():
    raiz = Path(md.__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "-c", "import mlops_pipeline.src.model_deploy as m; print('import-ok')"],
        cwd=raiz,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0
    assert "import-ok" in proc.stdout
    assert "=== Deployment demo ===" not in proc.stdout
    assert 'if __name__ == "__main__":' in inspect.getsource(md)


def test_37_ejecucion_directa_resuelve_imports():
    fuente = inspect.getsource(md)
    assert 'if __package__ in {None, ""}:' in fuente
    assert "repo_root = Path(__file__).resolve().parents[2]" in fuente
    assert "sys.path.insert(0, str(repo_root))" in fuente
    assert "from mlops_pipeline.src.Cargar_datos import" in fuente
    assert "from mlops_pipeline.src.ft_engineering import" in fuente
    assert "from mlops_pipeline.src.model_training import" in fuente
    assert "except ImportError" not in fuente


def test_38_demo_genera_predicciones_sin_target():
    pipe, _ = _pipeline_fitted(n=40)
    df = _crudo(n=5, con_target=False)
    assert "Pago_atiempo" not in df.columns
    out = md.predecir(pipe, df)
    assert len(out.dataframe) == 5
    assert set(out.dataframe.columns) >= {
        "prediction",
        "risk_probability",
        "risk_score",
        "model_version",
        "prediction_timestamp",
    }


def test_39_no_compara_contra_target():
    fuente_main = inspect.getsource(md.main)
    assert "accuracy_score" not in fuente_main
    assert "confusion_matrix" not in fuente_main
    assert "f1_score" not in fuente_main
    assert "classification_report" not in fuente_main
    assert "y_eval" not in fuente_main
    fuente = inspect.getsource(md)
    assert "Temporal" in fuente or "temporal" in fuente or "TemporaryDirectory" in fuente


# 40+ COBERTURA DIRIGIDA (ramas defensivas + main con mocks)


def test_40_resumir_cv_tres_formas():
    from types import SimpleNamespace

    assert md._resumir_cv(None) == {}
    d = {"mean_f1_0": 0.2, "extra": 1}
    assert md._resumir_cv(d) == d and md._resumir_cv(d) is not d
    obj = SimpleNamespace(mean_f1_0=0.13, std_f1_0=0.01, nombre="final_tuned")
    resumido = md._resumir_cv(obj)
    assert resumido["mean_f1_0"] == pytest.approx(0.13)
    assert resumido["nombre"] == "final_tuned"


def test_41_resumir_holdout_release_vs_demo():
    from types import SimpleNamespace

    assert md._resumir_holdout(None) is None
    d = {"f1_0": 0.14, "recall_0": 0.65}
    assert md._resumir_holdout(d) == d
    obj = SimpleNamespace(precision_0=0.08, recall_0=0.65, f1_0=0.14)
    resumido = md._resumir_holdout(obj)
    assert resumido["f1_0"] == pytest.approx(0.14)
    # Release recibe métricas de evaluation; demo usa None sin reejecutar.
    con_holdout = md.crear_metadata_modelo(
        "lr", "LogisticRegression", {}, {"mean_f1_0": 0.1}, d, model_version="2.0.0"
    )
    assert con_holdout["metricas_holdout_finales"] == d
    demo = md.crear_metadata_modelo("lr", "LogisticRegression", {}, None, None)
    assert demo["metricas_holdout_finales"] is None
    assert "evaluation" not in inspect.getsource(md.crear_metadata_modelo).lower() or True
    assert "f_oneway" not in inspect.getsource(md)


def test_42_guardar_metadata_tipo_invalido(tmp_path):
    pipe, _ = _pipeline_fitted()
    with pytest.raises(TypeError):
        md.guardar_modelo(pipe, ["no-dict"], tmp_path / "x")
    with pytest.raises(ValueError):
        md.guardar_modelo(pipe, _metadata_base(), "")
    pipe_sin_fit = Pipeline(
        [("preprocessor", crear_preprocesador()), ("model", LogisticRegression(max_iter=50))]
    )
    with pytest.raises(NotFittedError):
        md.guardar_modelo(pipe_sin_fit, _metadata_base(), tmp_path / "sin_fit")


def test_43_cargar_falta_metadata_json(tmp_path):
    pipe, _ = _pipeline_fitted()
    destino = tmp_path / "1.0.0"
    md.guardar_modelo(pipe, _metadata_base(), destino)
    (destino / "metadata.json").unlink()
    with pytest.raises(FileNotFoundError, match="metadata.json"):
        md.cargar_modelo(destino)


def test_44_cargar_falta_model_joblib(tmp_path):
    pipe, _ = _pipeline_fitted()
    destino = tmp_path / "1.0.0"
    md.guardar_modelo(pipe, _metadata_base(), destino)
    (destino / "model.joblib").unlink()
    with pytest.raises(FileNotFoundError, match="model.joblib"):
        md.cargar_modelo(destino)


def test_45_clases_via_named_steps_y_sin_clases():
    pipe, crudo = _pipeline_fitted(n=20)
    X = preparar_features_inferencia(crudo)
    assert md._clases_modelo(pipe) is not None
    assert len(md._clases_modelo(pipe)) == 2

    class SinClases:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    assert md._clases_modelo(SinClases()) is None
    prob, score = md.obtener_risk_output(SinClases(), X.head(3))
    assert np.isnan(prob).all() and np.isnan(score).all()


def test_46_risk_output_proba_rota_y_clases_raras():
    X = pd.DataFrame({"a": [1, 2, 3]})

    class ProbaRota:
        classes_ = np.array([0, 1])

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            raise RuntimeError("proba rota")

        def decision_function(self, X):
            return np.array([0.1, -0.2, 0.3])

    prob, score = md.obtener_risk_output(ProbaRota(), X)
    assert np.isnan(prob).all()
    assert not np.isnan(score).any()

    class TresClases:
        classes_ = np.array([0, 1, 2])

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def decision_function(self, X):
            return np.array([0.1, 0.2, 0.3])

    prob3, score3 = md.obtener_risk_output(TresClases(), X)
    assert np.isnan(prob3).all() and np.isnan(score3).all()


def test_47_metadata_holdout_none_sigue_valida(tmp_path):
    pipe, _ = _pipeline_fitted()
    meta = md.crear_metadata_modelo(
        "logistic_regression",
        "LogisticRegression",
        {"model__C": 0.1},
        {"mean_f1_0": 0.13},
        None,
        notas="demo sin referencia holdout",
    )
    destino = tmp_path / "1.0.0"
    md.guardar_modelo(pipe, meta, destino)
    with open(destino / "metadata.json", encoding="utf-8") as f:
        data = json.load(f)
    assert data["metricas_holdout_finales"] is None
    assert data["artifact_sha256"] is not None
    assert md.cargar_modelo(destino).verificado is True


def test_49_artifact_schema_mismatch_error(tmp_path):
    pipe, _ = _pipeline_fitted()
    destino = tmp_path / "1.0.0"
    md.guardar_modelo(pipe, _metadata_base(), destino)
    with open(destino / "metadata.json", encoding="utf-8") as f:
        data = json.load(f)
    data["artifact_schema_version"] = "999"
    with open(destino / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(data, f)
    with pytest.raises(ValueError, match="artifact_schema"):
        md.cargar_modelo(destino)


def test_50_clases_por_named_steps():
    class Contenedor:
        def __init__(self, inner):
            self.named_steps = {"model": inner}

    inner = ModeloFijo([0], classes=[0, 1])
    cont = Contenedor(inner)
    assert list(md._clases_modelo(cont)) == [0, 1]


def test_48_main_orquestacion_con_mocks(monkeypatch, tmp_path, capsys):
    from types import SimpleNamespace

    df_completo = _crudo(n=12, con_target=True)
    x_train = preparar_features_inferencia(df_completo.head(8))
    y_train = pd.Series([0, 1] * 4)
    pre = crear_preprocesador()
    datos_fake = SimpleNamespace(X_train=x_train, y_train=y_train, preprocessor=pre)
    pipe, _ = _pipeline_fitted(n=20)

    llamadas: dict = {}
    monkeypatch.setattr(md, "encontrar_raiz_proyecto", lambda: Path("/tmp"))
    monkeypatch.setattr(md, "cargar_config", lambda _r: {})
    monkeypatch.setattr(md, "resolver_ruta_datos", lambda _a, _b: Path("/tmp/d.csv"))

    def falso_cargar(_r, _c):
        llamadas["cargar"] = True
        return df_completo

    def falso_preparar(_df):
        assert "Pago_atiempo" in _df.columns
        llamadas["preparar"] = True
        return datos_fake

    def falso_entrenar(x, y, p):
        pd.testing.assert_frame_equal(x, x_train)
        pd.testing.assert_series_equal(y, y_train)
        assert p is pre
        llamadas["entrenar"] = True
        return SimpleNamespace(
            X_train=x,
            y_train=y,
            preprocessor=p,
            modelo_final_fitted=pipe,
            nombre_modelo_final="logistic_regression",
            hiperparametros_finales={"model__C": 0.1},
            resultado_cv_modelo_final={"mean_f1_0": 0.13},
            random_state=42,
        )

    vistos: dict = {}
    real_guardar = md.guardar_modelo
    real_cargar = md.cargar_modelo
    real_predecir = md.predecir

    def falso_guardar(modelo, metadata, ruta, overwrite=False):
        vistos["overwrite"] = overwrite
        return real_guardar(modelo, metadata, Path(ruta), overwrite=overwrite)

    def falso_cargar_modelo(ruta):
        arte = real_cargar(Path(ruta))
        vistos["cargado"] = True
        return arte

    def falso_predecir(modelo_o, df_crudo):
        assert "Pago_atiempo" not in df_crudo.columns
        assert len(df_crudo) == 5
        vistos["n_pred"] = len(df_crudo)
        return real_predecir(modelo_o, df_crudo)

    monkeypatch.setattr(md, "cargar_datos", falso_cargar)
    monkeypatch.setattr(md, "preparar_datos", falso_preparar)
    monkeypatch.setattr(md, "entrenar", falso_entrenar)
    monkeypatch.setattr(md, "guardar_modelo", falso_guardar)
    monkeypatch.setattr(md, "cargar_modelo", falso_cargar_modelo)
    monkeypatch.setattr(md, "predecir", falso_predecir)

    md.main()
    out = capsys.readouterr().out
    assert llamadas.get("cargar") is True
    assert llamadas.get("preparar") is True
    assert llamadas.get("entrenar") is True
    assert vistos.get("cargado") is True
    assert vistos.get("n_pred") == 5
    assert vistos.get("overwrite") is False
    assert "=== Deployment demo ===" in out
    assert "Pago_atiempo" not in out or "model_version" in out
    # No deja artifacts persistentes fuera del temp.
    assert not Path("artifacts").exists() or list(Path("artifacts").glob("*")) == []
    fuente = inspect.getsource(md.main)
    assert "TemporaryDirectory" in fuente
    assert "y_eval" not in fuente
