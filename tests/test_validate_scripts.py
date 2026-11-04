"""Pruebas de los scripts de validación usados por el pipeline de Jenkins."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import validate_metrics as vm
from scripts import validate_structure as vs

REPO_ROOT = Path(__file__).resolve().parents[1]


def _crear_estructura(raiz: Path) -> None:
    for rel in vs.DIRECTORIOS_REQUERIDOS:
        (raiz / rel).mkdir(parents=True, exist_ok=True)
    for rel in vs.ARCHIVOS_REQUERIDOS:
        (raiz / rel).parent.mkdir(parents=True, exist_ok=True)
        (raiz / rel).write_text("", encoding="utf-8")
    (raiz / "tests" / "test_dummy.py").write_text("", encoding="utf-8")


def test_estructura_real_del_repositorio_es_valida():
    assert vs.validar_estructura(REPO_ROOT) == []


def test_estructura_completa_sin_problemas(tmp_path):
    _crear_estructura(tmp_path)
    assert vs.validar_estructura(tmp_path, versionados=["app.py"]) == []


def test_estructura_detecta_faltantes(tmp_path):
    _crear_estructura(tmp_path)
    (tmp_path / "Jenkinsfile").unlink()
    (tmp_path / "tests" / "test_dummy.py").unlink()
    problemas = vs.validar_estructura(tmp_path)
    assert "falta archivo: Jenkinsfile" in problemas
    assert "tests/ no contiene archivos test_*.py" in problemas


def test_estructura_detecta_archivos_versionados_prohibidos(tmp_path):
    _crear_estructura(tmp_path)
    problemas = vs.validar_estructura(
        tmp_path, versionados=["Base_de_datos.csv", "artifacts/1.0.0/model.joblib", "app.py"]
    )
    assert len(problemas) == 2
    assert all("no debería estar en Git" in p for p in problemas)


def test_validate_structure_main_codigos(tmp_path, capsys):
    _crear_estructura(tmp_path)
    assert vs.main(["--root", str(tmp_path)]) == 0
    (tmp_path / "Dockerfile").unlink()
    assert vs.main(["--root", str(tmp_path)]) == 1
    assert "falta archivo: Dockerfile" in capsys.readouterr().out


def test_archivos_versionados_sin_git(tmp_path):
    assert vs.archivos_versionados(tmp_path / "no-existe") is None


def _resultado_falso(f1=0.12, recall=0.53, balanced=0.58, auc=0.60):
    metricas = SimpleNamespace(
        f1_0=f1,
        recall_0=recall,
        precision_0=0.07,
        balanced_accuracy=balanced,
        accuracy_ctx=0.63,
    )
    item = SimpleNamespace(cv_mean=0.136, cv_std=0.016)
    return SimpleNamespace(
        nombre_modelo_final="logistic_regression",
        hiperparametros_finales={"model__C": 10.0},
        holdout_modelo_final=SimpleNamespace(
            n_eval=2153,
            metricas=metricas,
            probabilisticas=SimpleNamespace(roc_auc_risk=auc, average_precision_risk=0.08),
        ),
        holdout_heuristico=SimpleNamespace(
            metricas=SimpleNamespace(f1_0=0.10, recall_0=0.60, balanced_accuracy=0.57)
        ),
        comparacion_cv_final_vs_holdout=SimpleNamespace(f1_0=item),
    )


def test_extraer_metricas_aplana_resultado():
    metricas = vm.extraer_metricas(_resultado_falso())
    assert metricas["modelo_final"] == "logistic_regression"
    assert metricas["hiperparametros"] == {"model__C": "10.0"}
    assert metricas["roc_auc_risk"] == 0.60
    assert metricas["heuristico_f1_0"] == 0.10
    json.dumps(metricas)


def test_validar_umbrales_aprueba_y_rechaza():
    umbrales = vm.UMBRALES_DEFAULT
    assert vm.validar_umbrales(vm.extraer_metricas(_resultado_falso()), umbrales) == []
    malas = vm.extraer_metricas(_resultado_falso(recall=0.2, auc=None))
    violaciones = vm.validar_umbrales(malas, umbrales)
    assert any(v.startswith("recall_0: 0.2000 <") for v in violaciones)
    assert any(v.startswith("roc_auc_risk: no disponible") for v in violaciones)


def test_validate_metrics_main_escribe_reporte(tmp_path, capsys):
    codigo = vm.main(
        ["--output", "reports/metrics.json"],
        evaluador=lambda html: _resultado_falso(),
        raiz=tmp_path,
    )
    assert codigo == 0
    reporte = json.loads((tmp_path / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert reporte["aprobado"] is True and reporte["violaciones"] == []
    assert "QUALITY GATE: APROBADO" in capsys.readouterr().out


def test_validate_metrics_main_rechaza_con_umbral_personalizado(tmp_path, capsys):
    salida = tmp_path / "metrics.json"
    codigo = vm.main(
        ["--output", "metrics.json", "--min-f1-0", "0.5"],
        evaluador=lambda html: _resultado_falso(),
        raiz=tmp_path,
    )
    assert codigo == 1
    reporte = json.loads(salida.read_text(encoding="utf-8"))
    assert reporte["aprobado"] is False
    assert reporte["umbrales"]["f1_0"] == 0.5
    assert "QUALITY GATE: RECHAZADO" in capsys.readouterr().out


def test_resolver_salida_rechaza_rutas_fuera_del_repositorio(tmp_path):
    raiz = tmp_path / "repo"
    raiz.mkdir()
    assert vm.resolver_salida("reports/m.json", raiz) == (raiz / "reports" / "m.json").resolve()
    for ruta in ("../fuera.json", str(tmp_path / "fuera.json"), "reports/../../x.json"):
        with pytest.raises(ValueError, match="fuera del repositorio"):
            vm.resolver_salida(ruta, raiz)


def test_validate_metrics_main_no_evalua_si_la_salida_es_invalida(tmp_path):
    llamado = []
    with pytest.raises(ValueError):
        vm.main(
            ["--output", "../escape.json"],
            evaluador=lambda html: llamado.append(html),
            raiz=tmp_path,
        )
    assert llamado == []
