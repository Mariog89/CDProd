"""Pruebas de las métricas comunes de la Fase 2 (vertex/metricas.py)."""

import numpy as np
import pytest

from vertex import metricas

Y = np.array([0, 0, 1, 1, 1, 1, 0, 1, 1, 1])
SCORE = np.array([0.9, 0.7, 0.2, 0.1, 0.6, 0.3, 0.4, 0.2, 0.05, 0.15])


def test_predecir_con_umbral_marca_riesgo_como_clase_0():
    assert metricas.predecir_con_umbral([0.9, 0.5, 0.1], 0.5).tolist() == [0, 0, 1]


def test_metricas_clasificacion_clase_riesgo():
    m = metricas.metricas_clasificacion([0, 0, 1, 1], [0, 1, 1, 0])
    assert m["recall_0"] == pytest.approx(0.5)
    assert m["precision_0"] == pytest.approx(0.5)
    assert m["f1_0"] == pytest.approx(0.5)
    assert m["balanced_accuracy"] == pytest.approx(0.5)
    assert m["predichos_riesgo"] == 2


def test_metricas_probabilisticas_score_perfecto():
    y = [0, 0, 1, 1]
    m = metricas.metricas_probabilisticas(y, [0.9, 0.8, 0.1, 0.2])
    assert m["roc_auc_risk"] == pytest.approx(1.0)
    assert m["average_precision_risk"] == pytest.approx(1.0)
    assert m["rmse_prob_riesgo"] == pytest.approx(np.sqrt((0.01 + 0.04 + 0.01 + 0.04) / 4))


def test_umbral_max_f1_mejora_o_iguala_umbral_0_5():
    umbral = metricas.umbral_max_f1(Y, SCORE)
    f1_opt = metricas.evaluar_scores(Y, SCORE, umbral)["f1_0"]
    f1_05 = metricas.evaluar_scores(Y, SCORE, 0.5)["f1_0"]
    assert f1_opt >= f1_05
    assert 0.0 <= umbral <= 1.0


def test_evaluar_scores_incluye_umbral_y_todas_las_metricas():
    m = metricas.evaluar_scores(Y, SCORE, 0.35)
    for clave in ("roc_auc_risk", "average_precision_risk", "f1_0", "accuracy", "umbral"):
        assert clave in m
    assert m["umbral"] == 0.35
