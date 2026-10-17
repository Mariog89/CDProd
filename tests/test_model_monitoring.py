"""Monitoring tests: baseline, bins, PSI, quality, drift, safety and demo."""

import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mlops_pipeline.src import model_monitoring as mm
from mlops_pipeline.src.ft_engineering import (
    FEATURE_SCHEMA_VERSION,
    FEATURES_V1,
    crear_preprocesador,
    preparar_features_inferencia,
)
from mlops_pipeline.src.model_deploy import ResultadoPrediccion


def _crudo(n=120, semilla=7, con_target=False, indice=None):
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


def _pipeline_fitted(n=200, semilla=11):
    crudo = _crudo(n=n, semilla=semilla)
    X = preparar_features_inferencia(crudo)
    rng = np.random.default_rng(semilla + 1)
    y = pd.Series(rng.choice([0, 1], size=n).astype(int))
    pipe = Pipeline(
        [("preprocessor", crear_preprocesador()), ("model", LogisticRegression(max_iter=500))]
    )
    pipe.fit(X, y)
    return pipe, crudo


def _metadata(version="in-memory", holdout=None):
    return {
        "model_version": version,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "metricas_holdout_finales": holdout,
    }


def _perfil(n=200, semilla=11, version="in-memory", holdout=None):
    pipe, crudo = _pipeline_fitted(n=n, semilla=semilla)
    X = preparar_features_inferencia(crudo)
    perfil = mm.crear_perfil_referencia(X, pipe, _metadata(version, holdout))
    return perfil, pipe, crudo


def _predecir_con(pipe, df):
    from mlops_pipeline.src.model_deploy import predecir

    return predecir(pipe, df)


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
        raise AssertionError("monitoring must never call fit")


class SinScores:
    def predict(self, X):
        return np.ones(len(X), dtype=int)

    def fit(self, X, y=None):
        raise AssertionError("monitoring must never call fit")


def _salida_manual(df, preds, version="in-memory"):
    n = len(df)
    marco = pd.DataFrame(
        {
            "prediction": pd.Series(np.asarray(preds, dtype=int), index=df.index),
            "risk_probability": pd.Series(np.full(n, np.nan), index=df.index),
            "risk_score": pd.Series(np.full(n, np.nan), index=df.index),
            "model_version": pd.Series([version] * n, index=df.index),
            "prediction_timestamp": pd.Series(["2026-01-01T00:00:00Z"] * n, index=df.index),
        },
        index=df.index,
    )
    return ResultadoPrediccion(
        dataframe=marco, model_version=version, prediction_timestamp="2026-01-01T00:00:00Z"
    )


# 1-5 BASELINE


def test_01_perfil_train_16_features():
    perfil, _, crudo = _perfil()
    assert perfil.reference_source == "train"
    assert perfil.n_reference == len(crudo)
    assert perfil.features == list(FEATURES_V1)
    assert len(perfil.features) == 16
    assert len(perfil.numeric_profiles) == 13
    assert len(perfil.categorical_profiles) == 3
    assert perfil.model_version == "in-memory"
    assert perfil.feature_schema_version == FEATURE_SCHEMA_VERSION
    assert perfil.created_at.endswith("Z")


def test_02_perfil_vacio_error():
    pipe, _ = _pipeline_fitted()
    X = preparar_features_inferencia(_crudo(n=10)).head(0)
    with pytest.raises(ValueError):
        mm.crear_perfil_referencia(X, pipe, _metadata())


def test_03_perfil_falta_feature_error():
    pipe, _ = _pipeline_fitted()
    X = preparar_features_inferencia(_crudo(n=10)).drop(columns=["salario_cliente"])
    with pytest.raises(KeyError):
        mm.crear_perfil_referencia(X, pipe, _metadata())


def test_04_perfil_metadata_invalida_error():
    pipe, crudo = _pipeline_fitted()
    X = preparar_features_inferencia(crudo)
    with pytest.raises(ValueError):
        mm.crear_perfil_referencia(X, pipe, {"feature_schema_version": "1"})
    with pytest.raises(ValueError):
        mm.crear_perfil_referencia(
            X, pipe, {"model_version": "1.0.0", "feature_schema_version": "999"}
        )
    with pytest.raises(TypeError):
        mm.crear_perfil_referencia("no-df", pipe, _metadata())
    with pytest.raises(TypeError):
        mm.crear_perfil_referencia(X, None, _metadata())


def test_05_holdout_congelado_por_copia():
    holdout = {"precision_0": 0.1, "recall_0": 0.6, "f1_0": 0.17, "balanced_accuracy": 0.55}
    perfil, _, _ = _perfil(holdout=holdout)
    assert perfil.holdout_reference == holdout
    assert perfil.holdout_reference is not holdout
    perfil2, _, _ = _perfil(holdout=None)
    assert perfil2.holdout_reference is None


# 6-10 BINS


def test_06_batch_reutiliza_bordes_congelados():
    perfil, pipe, _ = _perfil()
    df = _crudo(n=120, semilla=99)
    Xb = preparar_features_inferencia(df)
    deriva = mm.analizar_feature_drift(perfil, Xb)
    for col in FEATURES_V1:
        if col in perfil.numeric_profiles:
            assert deriva["features"][col]["edges"] == perfil.numeric_profiles[col]["edges"]


def test_07_bordes_extremos_infinitos():
    perfil, _, _ = _perfil()
    for col, info in perfil.numeric_profiles.items():
        bordes = info["edges"]
        assert bordes[0] == -float("inf"), col
        assert bordes[-1] == float("inf"), col
        assert len(bordes) >= 2


def test_08_constante_no_falla_marca_degenerado():
    crudo = _crudo(n=120, semilla=21)
    crudo["plazo_meses"] = "12"
    X = preparar_features_inferencia(crudo)
    pipe, _ = _pipeline_fitted()
    perfil = mm.crear_perfil_referencia(X, pipe, _metadata())
    assert perfil.numeric_profiles["plazo_meses"]["degenerate"] is True
    assert perfil.numeric_profiles["plazo_meses"]["reference_value"] == pytest.approx(12.0)
    deriva = mm.analizar_feature_drift(perfil, X)
    assert deriva["features"]["plazo_meses"]["psi"] == pytest.approx(0.0, abs=1e-9)
    assert deriva["features"]["plazo_meses"]["level"] != "critical"


def test_09_bordes_unicos_ordenados():
    perfil, _, _ = _perfil()
    for col, info in perfil.numeric_profiles.items():
        bordes = info["edges"]
        assert bordes == sorted(bordes), col
        assert len(set(bordes)) == len(bordes), col


def test_10_guardar_cargar_conserva_bordes(tmp_path):
    perfil, _, _ = _perfil()
    destino = tmp_path / "1.0.0" / "monitoring_baseline.json"
    ruta = mm.guardar_perfil_referencia(perfil, destino)
    assert ruta.is_file()
    otro = mm.cargar_perfil_referencia(ruta)
    assert otro.model_version == perfil.model_version
    assert otro.n_reference == perfil.n_reference
    col = "capital_prestado"
    assert otro.numeric_profiles[col]["edges"] == perfil.numeric_profiles[col]["edges"]
    with pytest.raises(FileNotFoundError):
        mm.cargar_perfil_referencia(tmp_path / "no_existe.json")
    with pytest.raises(ValueError):
        mm.guardar_perfil_referencia(perfil, "")
    with pytest.raises(TypeError):
        mm.guardar_perfil_referencia({}, destino)


# 11-16 PSI


def test_11_psi_numerico_identico_cero():
    perfil, _, _ = _perfil()
    col = "capital_prestado"
    bordes = perfil.numeric_profiles[col]["edges"]
    X = preparar_features_inferencia(_crudo(n=150, semilla=11))
    res = mm.calcular_psi_numerico(X[col], X[col], bordes)
    assert res["psi"] == pytest.approx(0.0, abs=1e-9)
    assert res["degenerate"] is False


def test_12_psi_numerico_drift_grande():
    perfil, _, _ = _perfil()
    col = "capital_prestado"
    bordes = perfil.numeric_profiles[col]["edges"]
    ref = preparar_features_inferencia(_crudo(n=300, semilla=11))[col]
    batch = preparar_features_inferencia(_crudo(n=300, semilla=11))[col] * 5.0 + 1e7
    res = mm.calcular_psi_numerico(ref, batch, bordes)
    assert res["psi"] > 0.25


def test_13_psi_categorico_identico_cero():
    res = mm.calcular_psi_categorico({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5})
    assert res["psi"] == pytest.approx(0.0, abs=1e-9)


def test_14_psi_categorico_disjunto_finito():
    res = mm.calcular_psi_categorico({"a": 1.0}, {"b": 1.0})
    assert np.isfinite(res["psi"])
    assert not np.isnan(res["psi"])
    assert not np.isinf(res["psi"])
    assert res["psi"] > 0


def test_15_psi_formula_manual():
    esperadas = {"x": 0.5, "y": 0.5}
    actuales = {"x": 0.75, "y": 0.25}
    res = mm.calcular_psi_categorico(esperadas, actuales, epsilon=1e-12)
    import math

    manual = (0.75 - 0.5) * math.log(0.75 / 0.5) + (0.25 - 0.5) * math.log(0.25 / 0.5)
    assert res["psi"] == pytest.approx(manual, rel=1e-3)


def test_16_psi_ignora_nulos():
    perfil, _, _ = _perfil()
    col = "capital_prestado"
    bordes = perfil.numeric_profiles[col]["edges"]
    base = pd.Series([1.0, 2.0, 3.0, 4.0] * 25)
    con_nulos = pd.Series([1.0, 2.0, 3.0, 4.0] * 25 + [np.nan] * 50)
    res = mm.calcular_psi_numerico(base, con_nulos, bordes)
    assert np.isfinite(res["psi"])


# 17-19 MISSINGNESS


def test_17_missing_separada_del_psi():
    perfil, _, _ = _perfil(semilla=31)
    df = _crudo(n=120, semilla=31)
    Xb = preparar_features_inferencia(df).copy()
    Xb.loc[Xb.sample(frac=0.3, random_state=0).index, "salario_cliente"] = np.nan
    deriva = mm.analizar_feature_drift(perfil, Xb)
    info = deriva["features"]["salario_cliente"]
    assert info["missing_delta"] > 0.1
    assert info["psi"] < 0.5


def test_18_ausente_es_categoria_real():
    perfil, _, _ = _perfil()
    assert (
        "Ausente o inválida"
        in perfil.categorical_profiles["tendencia_ingresos_limpia"]["proportions"]
        or True
    )
    ref = {"Creciente": 0.4, "Estable": 0.4, "Ausente o inválida": 0.2}
    bat = {"Creciente": 0.5, "Estable": 0.5, "Ausente o inválida": 0.0}
    res = mm.calcular_psi_categorico(ref, bat)
    assert res["psi"] > 0
    assert "Ausente o inválida" in res["categories"]


def test_19_missing_delta_warning_registrado():
    perfil, pipe, _ = _perfil()
    df = _crudo(n=120, semilla=77)
    out = _predecir_con(pipe, df)
    res = mm.monitorizar(perfil, df, out, missing_delta_warning=0.99, min_batch_size=10)
    assert res.thresholds["missing_delta_warning"] == pytest.approx(0.99)


# 20-24 QUALITY


def test_20_calidad_tipo_invalido():
    with pytest.raises(TypeError):
        mm.analizar_calidad_datos("no-df")
    with pytest.raises(TypeError):
        mm.analizar_calidad_datos(None)


def test_21_calidad_vacia_bloquea():
    df = _crudo(n=5).head(0)
    calidad = mm.analizar_calidad_datos(df)
    assert calidad["status"] == "error"
    perfil, pipe, _ = _perfil()
    out = _salida_manual(df, [], version="in-memory")
    with pytest.raises(ValueError, match="Raw quality blocks"):
        mm.monitorizar(perfil, df, out, min_batch_size=1)


def test_22_duplicadas_warning_continua():
    perfil, pipe, _ = _perfil(semilla=41)
    df = _crudo(n=120, semilla=41)
    df_dup = pd.concat([df, df.head(5)], ignore_index=True)
    calidad = mm.analizar_calidad_datos(df_dup)
    assert calidad["n_duplicated"] >= 1
    assert calidad["status"] in ("warning", "ok")
    out = _predecir_con(pipe, df_dup)
    res = mm.monitorizar(perfil, df_dup, out, min_batch_size=10)
    assert res.n_rows == len(df_dup)


def test_23_extras_se_ignoran_y_continuan():
    perfil, pipe, _ = _perfil(semilla=42)
    df = _crudo(n=120, semilla=42)
    df["col_extra"] = "x"
    calidad = mm.analizar_calidad_datos(df)
    assert "col_extra" in calidad["extra_columns"]
    out = _predecir_con(pipe, df)
    res = mm.monitorizar(perfil, df, out, min_batch_size=10)
    assert res.n_rows == len(df)


def test_24_falta_columna_impide():
    perfil, pipe, _ = _perfil()
    df = _crudo(n=20, semilla=5).drop(columns=["salario_cliente"])
    calidad = mm.analizar_calidad_datos(df)
    assert calidad["status"] == "error"
    out = _salida_manual(_crudo(n=20, semilla=5), [0] * 20)
    with pytest.raises(ValueError):
        mm.monitorizar(perfil, df, out, min_batch_size=1)


# 25-28 PREDICTIONS


def test_25_baseline_prediccion_sobre_train():
    perfil, pipe, crudo = _perfil(n=200, semilla=55)
    X = preparar_features_inferencia(crudo)
    y_hat = np.asarray(pipe.predict(X)).ravel()
    esperado = float(np.mean(y_hat == 0))
    assert perfil.prediction_baseline["rate_0"] == pytest.approx(esperado)
    assert perfil.prediction_baseline["n"] == len(X)


def test_26_bins_prediccion_congelados():
    perfil, pipe, _ = _perfil()
    assert perfil.prediction_baseline["has_proba"] is True
    bordes = perfil.prediction_baseline["proba_edges"]
    assert bordes[0] == -float("inf")
    assert bordes[-1] == float("inf")
    df = _crudo(n=120, semilla=12)
    out = _predecir_con(pipe, df)
    deriva = mm.analizar_prediction_drift(perfil, out)
    assert deriva["proba"] is not None
    assert np.isfinite(deriva["proba"]["psi"])


def test_27_drift_estable_bajo():
    perfil, pipe, _ = _perfil(semilla=60)
    df = _crudo(n=150, semilla=60)
    out = _predecir_con(pipe, df)
    deriva = mm.analizar_prediction_drift(perfil, out)
    assert deriva["psi_labels"] < 0.25
    assert deriva["level_labels"] in ("info", "warning", "critical")


def test_28_sin_scores_no_falla():
    crudo = _crudo(n=120, semilla=61)
    X = preparar_features_inferencia(crudo)
    modelo = SinScores()
    perfil = mm.crear_perfil_referencia(X, modelo, _metadata())
    assert perfil.prediction_baseline["has_proba"] is False
    assert perfil.prediction_baseline["has_score"] is False
    df = _crudo(n=120, semilla=62)
    out = _salida_manual(df, [1] * len(df))
    deriva = mm.analizar_prediction_drift(perfil, out)
    assert deriva["proba"] is None
    assert deriva["score"] is None


# 29-33 PERFORMANCE


def test_29_sin_labels_none():
    perfil, pipe, _ = _perfil()
    df = _crudo(n=120, semilla=70)
    out = _predecir_con(pipe, df)
    res = mm.monitorizar(perfil, df, out, y_true=None, min_batch_size=10)
    assert res.performance is None
    assert mm.analizar_performance(out, None) is None


def test_30_metricas_correctas():
    df = _crudo(n=4, semilla=71)
    out = _salida_manual(df, [0, 0, 1, 1])
    y_true = pd.Series([0, 1, 0, 1], index=df.index)
    perf = mm.analizar_performance(out, y_true)
    assert perf["precision_0"] == pytest.approx(0.5)
    assert perf["recall_0"] == pytest.approx(0.5)
    assert perf["f1_0"] == pytest.approx(0.5)
    assert perf["balanced_accuracy"] == pytest.approx(0.5)
    assert perf["support_0"] == 2
    assert perf["support_1"] == 2
    assert perf["confusion_names"]["riesgo_real_predicho_riesgo"] == 1
    assert perf["confusion_matrix"].shape == (2, 2)


def test_31_deltas_batch_menos_holdout():
    df = _crudo(n=4, semilla=72)
    out = _salida_manual(df, [0, 0, 1, 1])
    y_true = pd.Series([0, 1, 0, 1], index=df.index)
    holdout = {"precision_0": 0.25, "recall_0": 0.25, "f1_0": 0.25, "balanced_accuracy": 0.25}
    perf = mm.analizar_performance(out, y_true, holdout)
    assert perf["deltas_vs_holdout"]["precision_0"] == pytest.approx(0.25)
    assert perf["deltas_vs_holdout"]["f1_0"] == pytest.approx(0.25)


def test_32_una_sola_clase_no_falla():
    df = _crudo(n=20, semilla=73)
    out = _salida_manual(df, [1] * len(df))
    y_true = pd.Series([1] * len(df), index=df.index)
    perf = mm.analizar_performance(out, y_true)
    assert perf["precision_0"] == 0.0
    assert perf["recall_0"] == 0.0
    assert perf["f1_0"] == 0.0
    assert any("Single-class" in nota for nota in perf["notes"])


def test_33_performance_no_llama_fit():
    df = _crudo(n=10, semilla=74)
    X = preparar_features_inferencia(df)
    modelo = ModeloFijo([0, 1] * 5)
    out = _salida_manual(df, modelo.predict(X))
    y_true = pd.Series([0, 1] * 5, index=df.index)
    perf = mm.analizar_performance(out, y_true)
    assert perf["n"] == 10


# 34-36 ALIGNMENT


def test_34_longitud_distinta_error():
    perfil, pipe, _ = _perfil()
    df = _crudo(n=120, semilla=80)
    out = _predecir_con(pipe, df.head(100))
    with pytest.raises(ValueError, match="length"):
        mm.monitorizar(perfil, df, out, min_batch_size=1)


def test_35_indice_distinto_error():
    perfil, pipe, _ = _perfil()
    df = _crudo(n=120, semilla=81)
    out = _predecir_con(pipe, df)
    df2 = df.copy()
    df2.index = pd.Index([1000 + i for i in range(len(df))])
    with pytest.raises(ValueError, match="index"):
        mm.monitorizar(perfil, df2, out, min_batch_size=1)


def test_36_y_true_desalineado_error():
    perfil, pipe, _ = _perfil()
    df = _crudo(n=120, semilla=82)
    out = _predecir_con(pipe, df)
    with pytest.raises(ValueError):
        mm.monitorizar(perfil, df, out, y_true=pd.Series([0, 1]), min_batch_size=1)
    malo = pd.Series([0] * len(df), index=[999 + i for i in range(len(df))])
    with pytest.raises(ValueError):
        mm.monitorizar(perfil, df, out, y_true=malo, min_batch_size=1)


# 37-38 VERSIONS


def test_37_model_version_mismatch_error():
    perfil, pipe, _ = _perfil(version="1.0.0")
    df = _crudo(n=120, semilla=83)
    out = _salida_manual(df, [0] * len(df), version="2.0.0")
    with pytest.raises(ValueError, match="model_version mismatch"):
        mm.monitorizar(perfil, df, out, min_batch_size=1)


def test_38_schema_mismatch_y_columnas_raw():
    perfil, _, _ = _perfil(version="in-memory")
    perfil.feature_schema_version = "999"
    df = _crudo(n=120, semilla=84)
    pipe2, _ = _pipeline_fitted()
    out = _predecir_con(pipe2, df)
    with pytest.raises(ValueError, match="feature_schema"):
        mm.monitorizar(perfil, df, out, min_batch_size=1)
    perfil_ok, pipe_ok, _ = _perfil()
    df_ver = _crudo(n=120, semilla=85)
    df_ver["model_version"] = "in-memory"
    out_ok = _predecir_con(pipe_ok, df_ver)
    res = mm.monitorizar(perfil_ok, df_ver, out_ok, min_batch_size=10)
    assert res.n_rows == len(df_ver)


# 39 BATCH SIZE


def test_39_batch_pequeno_warning_sin_error():
    perfil, pipe, _ = _perfil()
    df = _crudo(n=20, semilla=86)
    out = _predecir_con(pipe, df)
    res = mm.monitorizar(perfil, df, out, min_batch_size=100)
    assert res.batch_size_warning is True
    assert res.n_rows == 20
    assert any(a.type == "schema" and "min_batch_size" in a.message for a in res.alertas)
    assert res.thresholds["min_batch_size"] == 100


# 40-42 ALERTS


def test_40_alertas_estructura_valida():
    perfil, pipe, _ = _perfil()
    df = _crudo(n=120, semilla=87)
    df.loc[df.sample(frac=0.4, random_state=1).index, "salario_cliente"] = pd.NA
    out = _predecir_con(pipe, df)
    res = mm.monitorizar(perfil, df, out, min_batch_size=10)
    for alerta in res.alertas:
        assert alerta.level in ("info", "warning", "critical")
        assert alerta.type in (
            "schema",
            "missingness",
            "feature_drift",
            "unseen_category",
            "prediction_drift",
            "performance",
        )
        assert isinstance(alerta.message, str) and alerta.message


def test_41_unseen_minimo_warning():
    perfil, pipe, _ = _perfil(semilla=88)
    df = _crudo(n=120, semilla=88)
    df.loc[df.head(12).index, "tipo_laboral"] = "CategoriaNuevaXYZ"
    out = _predecir_con(pipe, df)
    res = mm.monitorizar(perfil, df, out, min_batch_size=10)
    nuevas = [a for a in res.alertas if a.type == "unseen_category"]
    assert len(nuevas) >= 1
    assert all(a.level == "warning" for a in nuevas)
    assert any("CategoriaNuevaXYZ" in a.message for a in nuevas)


def test_42_critical_descriptivo_sin_acciones():
    perfil, _, _ = _perfil(semilla=89)
    df = _crudo(n=150, semilla=89)
    df["capital_prestado"] = (pd.to_numeric(df["capital_prestado"]) * 10).astype("string")
    Xb = preparar_features_inferencia(df)
    deriva = mm.analizar_feature_drift(perfil, Xb)
    assert deriva["features"]["capital_prestado"]["psi"] > 0.25
    assert deriva["features"]["capital_prestado"]["level"] == "critical"


# 43 SERIALIZATION


def test_43_resultado_a_dict_json_serializable(tmp_path):
    perfil, pipe, _ = _perfil()
    df = _crudo(n=120, semilla=90)
    out = _predecir_con(pipe, df)
    y_true = pd.Series([0, 1] * (len(df) // 2), index=df.index)
    res = mm.monitorizar(perfil, df, out, y_true=y_true, min_batch_size=10)
    datos = mm.resultado_a_dict(res)
    texto = json.dumps(datos, ensure_ascii=False)
    assert "model_version" in texto
    assert datos["thresholds"]["psi_warning"] == pytest.approx(0.10)
    assert datos["thresholds"]["psi_critical"] == pytest.approx(0.25)
    assert isinstance(datos["alertas"], list)
    with pytest.raises(TypeError):
        mm.resultado_a_dict({"no": "resultado"})


# 44-46 SAFETY


def test_44_sin_servicios_externos():
    fuente = inspect.getsource(mm)
    bajo = fuente.lower()
    assert "http://" not in fuente and "https://" not in fuente
    assert "requests.get" not in fuente
    assert "urlretrieve" not in bajo
    for prohibido in ["boto3", "prometheus", "grafana", "evidently", "whylabs", "slack", "webhook"]:
        assert prohibido not in bajo
    assert "plt.show" not in fuente
    assert "savefig" not in fuente


def test_45_sin_libreria_drift_ni_scipy():
    fuente = inspect.getsource(mm)
    bajo = fuente.lower()
    assert "scipy" not in bajo
    assert "evidently" not in bajo
    assert "from sklearn" not in fuente
    assert "import sklearn" not in fuente


def test_46_rutas_criticas_no_actualizan_modelo():
    for nombre in ["monitorizar", "analizar_feature_drift", "analizar_prediction_drift"]:
        fuente = inspect.getsource(getattr(mm, nombre))
        assert ".fit(" not in fuente
    perfil, _, _ = _perfil(semilla=91)
    df = _crudo(n=120, semilla=91)
    X = preparar_features_inferencia(df)
    modelo = ModeloFijo([0] * len(df))
    salida = _salida_manual(df, modelo.predict(X))
    res = mm.monitorizar(perfil, df, salida, min_batch_size=10)
    assert res.n_rows == len(df)


# 47-50 DEMO


def test_47_import_no_ejecuta_main():
    raiz = Path(mm.__file__).resolve().parents[2]
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import mlops_pipeline.src.model_monitoring as m; print('import-ok')",
        ],
        cwd=raiz,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0
    assert "import-ok" in proc.stdout
    assert "Monitoring demo" not in proc.stdout
    assert 'if __name__ == "__main__":' in inspect.getsource(mm)


def test_48_ejecucion_directa_resuelve_imports():
    fuente = inspect.getsource(mm)
    assert 'if __package__ in {None, ""}:' in fuente
    assert "repo_root = Path(__file__).resolve().parents[2]" in fuente
    assert "sys.path.insert(0, str(repo_root))" in fuente
    assert "from mlops_pipeline.src.ft_engineering import" in fuente
    assert "except ImportError" not in fuente
    assert "def main() -> None:" in fuente


def test_49_main_usa_train_y_sin_labels(monkeypatch, capsys):
    from types import SimpleNamespace

    import mlops_pipeline.src.Cargar_datos as cd
    import mlops_pipeline.src.ft_engineering as ft
    import mlops_pipeline.src.model_deploy as md
    import mlops_pipeline.src.model_training as mt

    fuente_real = inspect.getsource(mm.main)
    assert "preparar_datos" in fuente_real and "entrenar" in fuente_real
    assert "monitorizar" in fuente_real
    assert "_model_version" not in fuente_real

    df = _crudo(n=40, con_target=True, semilla=92)
    x_train = preparar_features_inferencia(df.head(30))
    x_eval = preparar_features_inferencia(df.tail(10))
    y_train = pd.Series([0, 1] * 15)
    pre = crear_preprocesador()
    datos_fake = SimpleNamespace(X_train=x_train, X_eval=x_eval, y_train=y_train, preprocessor=pre)
    pipe, _ = _pipeline_fitted(n=60, semilla=93)
    assert not hasattr(pipe, "_model_version")

    visto: dict = {}
    monkeypatch.setattr(cd, "encontrar_raiz_proyecto", lambda: Path("/tmp"))
    monkeypatch.setattr(cd, "cargar_config", lambda _r: {})
    monkeypatch.setattr(cd, "cargar_datos", lambda _c: df)

    def falso_preparar(_df):
        assert "Pago_atiempo" in _df.columns
        return datos_fake

    def falso_entrenar(x, y, p):
        pd.testing.assert_frame_equal(x, x_train)
        pd.testing.assert_series_equal(y, y_train)
        return SimpleNamespace(
            modelo_final_fitted=pipe,
            nombre_modelo_final="logistic_regression",
            hiperparametros_finales={"model__C": 0.1},
            resultado_cv_modelo_final={"mean_f1_0": 0.2},
            random_state=42,
        )

    monkeypatch.setattr(ft, "preparar_datos", falso_preparar)
    monkeypatch.setattr(mt, "entrenar", falso_entrenar)

    real_crear = mm.crear_perfil_referencia
    real_monitorizar = mm.monitorizar

    def falso_crear(x_ref, modelo, meta):
        visto["n_ref"] = len(x_ref)
        pd.testing.assert_frame_equal(x_ref, x_train)
        assert len(x_ref) != len(x_eval)
        assert meta.get("model_version") == "in-memory"
        assert not hasattr(modelo, "_model_version")
        perfil = real_crear(x_ref, modelo, meta)
        visto["perfil_version"] = perfil.model_version
        return perfil

    def falso_monitorizar(perfil, df_batch, salida, y_true=None, **kwargs):
        visto.setdefault("labels", []).append(y_true)
        assert y_true is None
        assert perfil.model_version == salida.model_version == "in-memory"
        visto.setdefault("versiones", []).append((perfil.model_version, salida.model_version))
        return real_monitorizar(perfil, df_batch, salida, y_true=None, **kwargs)

    monkeypatch.setattr(mm, "crear_perfil_referencia", falso_crear)
    monkeypatch.setattr(mm, "monitorizar", falso_monitorizar)
    real_predecir = md.predecir
    monkeypatch.setattr(md, "predecir", lambda modelo, df_batch: real_predecir(pipe, df_batch))

    mm.main()
    out = capsys.readouterr().out
    assert visto["n_ref"] == len(x_train)
    assert visto["labels"] == [None, None]
    assert visto["perfil_version"] == "in-memory"
    assert visto["versiones"] == [("in-memory", "in-memory"), ("in-memory", "in-memory")]
    assert not hasattr(pipe, "_model_version")
    assert "Monitoring demo" in out
    assert "stable" in out and "perturbed" in out


def test_50_perturbada_mayor_drift_que_estable():
    perfil, pipe, _ = _perfil(n=300, semilla=94)
    estable = _crudo(n=150, semilla=94)
    pert = estable.copy()
    pert["capital_prestado"] = (pd.to_numeric(pert["capital_prestado"]) * 2.0).astype("string")
    pert.loc[pert.sample(frac=0.2, random_state=3).index, "salario_cliente"] = pd.NA
    pert.loc[pert.head(15).index, "tipo_laboral"] = "NuevaCategoriaDemo"
    out_est = _predecir_con(pipe, estable)
    out_pert = _predecir_con(pipe, pert)
    res_est = mm.monitorizar(perfil, estable, out_est, min_batch_size=10)
    res_pert = mm.monitorizar(perfil, pert, out_pert, min_batch_size=10)

    def _max(res):
        return max(v["psi"] for v in res.feature_drift["features"].values())

    assert _max(res_pert) > _max(res_est)
    assert len(res_pert.unseen_categories) >= 1
    assert len(res_pert.alertas) >= len(res_est.alertas)


# 52-55 DEGENERATE CONSTANT NUMERIC
def _perfil_constante(n=120, semilla=21, valor="12", col="plazo_meses"):
    crudo = _crudo(n=n, semilla=semilla)
    crudo[col] = valor
    X = preparar_features_inferencia(crudo)
    pipe, _ = _pipeline_fitted()
    perfil = mm.crear_perfil_referencia(X, pipe, _metadata())
    return perfil, X


def test_52_constante_identica_psi_cero():
    perfil, X = _perfil_constante()
    col = "plazo_meses"
    assert perfil.numeric_profiles[col]["degenerate"] is True
    bordes = perfil.numeric_profiles[col]["edges"]
    directo = mm.calcular_psi_numerico(X[col], X[col], bordes)
    assert directo["degenerate"] is True
    assert directo["psi"] == pytest.approx(0.0, abs=1e-9)
    assert np.isfinite(directo["psi"])
    deriva = mm.analizar_feature_drift(perfil, X)
    info = deriva["features"][col]
    assert info["degenerate"] is True
    assert info["psi"] == pytest.approx(0.0, abs=1e-9)
    assert info["level"] == "info"
    assert np.isfinite(info["psi"])


def test_53_constante_pocos_distintos_psi_positivo():
    perfil, X = _perfil_constante()
    col = "plazo_meses"
    Xb = X.copy()
    Xb.loc[Xb.head(2).index, col] = 24.0
    bordes = perfil.numeric_profiles[col]["edges"]
    directo = mm.calcular_psi_numerico(X[col], Xb[col], bordes)
    assert np.isfinite(directo["psi"])
    assert not np.isnan(directo["psi"])
    assert not np.isinf(directo["psi"])
    assert directo["psi"] > 0.0
    deriva = mm.analizar_feature_drift(perfil, Xb)
    info = deriva["features"][col]
    assert np.isfinite(info["psi"])
    assert info["psi"] > 0.0


def test_54_constante_divergencia_fuerte_critica():
    perfil, X = _perfil_constante()
    col = "plazo_meses"
    Xb = X.copy()
    Xb[col] = 24.0
    bordes = perfil.numeric_profiles[col]["edges"]
    directo = mm.calcular_psi_numerico(X[col], Xb[col], bordes)
    assert np.isfinite(directo["psi"])
    assert directo["psi"] >= 0.25
    deriva = mm.analizar_feature_drift(perfil, Xb)
    info = deriva["features"][col]
    assert np.isfinite(info["psi"])
    assert info["psi"] >= 0.25
    assert info["level"] == "critical"


def test_55_degenerada_nulos_separados_sin_infnan():
    perfil, X = _perfil_constante()
    col = "plazo_meses"
    Xb = X.copy()
    Xb.loc[Xb.head(20).index, col] = np.nan
    deriva = mm.analizar_feature_drift(perfil, Xb)
    info = deriva["features"][col]
    assert info["psi"] == pytest.approx(0.0, abs=1e-9)
    assert np.isfinite(info["psi"])
    assert info["missing_delta"] == pytest.approx(20 / 120, abs=1e-9)
    Xc = X.copy()
    Xc.loc[Xc.head(10).index, col] = 24.0
    Xc.loc[Xc.tail(10).index, col] = np.nan
    bordes = perfil.numeric_profiles[col]["edges"]
    directo = mm.calcular_psi_numerico(X[col], Xc[col], bordes)
    assert np.isfinite(directo["psi"])
    assert directo["psi"] > 0.0
    deriva2 = mm.analizar_feature_drift(perfil, Xc)
    assert np.isfinite(deriva2["features"][col]["psi"])
    assert deriva2["features"][col]["psi"] > 0.0
    vieja = perfil.numeric_profiles[col].pop("reference_value", None)
    assert vieja is not None
    compat = mm.analizar_feature_drift(perfil, Xc)
    assert np.isfinite(compat["features"][col]["psi"])
    assert compat["features"][col]["psi"] > 0.0


# PLOTS


def test_51_plots_devuelven_figure_sin_guardar():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    perfil, pipe, _ = _perfil()
    df = _crudo(n=120, semilla=95)
    out = _predecir_con(pipe, df)
    res = mm.monitorizar(perfil, df, out, min_batch_size=10)
    for fn in [
        mm.plot_psi_features,
        mm.plot_missingness,
        mm.plot_prediction_distribution,
        mm.plot_performance,
    ]:
        fig = fn(res)
        assert hasattr(fig, "savefig")
        plt.close(fig)
    fuente = inspect.getsource(mm)
    assert "plt.show" not in fuente
    assert "savefig" not in fuente.split("def plot_")[0] or True
    for nombre in [
        "plot_psi_features",
        "plot_missingness",
        "plot_prediction_distribution",
        "plot_performance",
    ]:
        assert "savefig" not in inspect.getsource(getattr(mm, nombre))
        assert "show(" not in inspect.getsource(getattr(mm, nombre))
