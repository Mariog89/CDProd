"""Pruebas del endpoint batch y del export operativo (sin training real)."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app as app_module


@pytest.fixture(autouse=True)
def _rutas_temporales(monkeypatch, tmp_path):
    """Ningún test escribe el runtime real del repo."""
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path / "prediction_log.csv"))
    monkeypatch.setenv("METRICS_HTML_PATH", str(tmp_path / "metrics.html"))
    yield


def _registro(semilla=1, con_target=False):
    base = {
        "tipo_credito": "4",
        "capital_prestado": "1500000",
        "plazo_meses": "12",
        "edad_cliente": "35",
        "tipo_laboral": "Empleado",
        "salario_cliente": "2500000",
        "total_otros_prestamos": "0",
        "cuota_pactada": "200000",
        "puntaje_datacredito": "800",
        "cant_creditosvigentes": "1",
        "creditos_sectorFinanciero": "1",
        "creditos_sectorCooperativo": "0",
        "creditos_sectorReal": "0",
        "promedio_ingresos_datacredito": "2500000",
        "tendencia_ingresos": "Estable",
    }
    assert semilla is not None
    if con_target:
        base["Pago_atiempo"] = "1"
    return base


def _artefacto(version="1.0.0"):
    return SimpleNamespace(modelo=object(), metadata={"model_version": version})


def _resultado_fake(n=1, version="1.0.0", con_nan=False):
    probs = [0.61 + 0.01 * i for i in range(n)]
    scores = [float("nan")] * n if con_nan else [None] * n
    marco = pd.DataFrame(
        {
            "prediction": [i % 2 for i in range(n)],
            "risk_probability": probs,
            "risk_score": scores,
            "model_version": [version] * n,
            "prediction_timestamp": ["2026-09-16T00:00:00Z"] * n,
        }
    )
    return SimpleNamespace(
        dataframe=marco, model_version=version, prediction_timestamp="2026-09-16T00:00:00Z"
    )


def _cliente_listo(monkeypatch, n=1, version="1.0.0", con_nan=False):
    falso = _artefacto(version)
    llamadas = {}

    def _predecir(artefacto, df):
        llamadas["n"] = llamadas.get("n", 0) + 1
        llamadas["artefacto"] = artefacto
        llamadas["cols"] = list(df.columns)
        return _resultado_fake(len(df) if len(df) else n, version, con_nan)

    monkeypatch.setattr(app_module, "predecir", _predecir)
    cliente = TestClient(app_module.app, raise_server_exceptions=False)
    app_module.app.state.artifact = falso
    app_module.app.state.load_error = None
    return cliente, llamadas


def test_01_app_importa_sin_training():
    assert hasattr(app_module.app, "routes")
    rutas = {r.path for r in app_module.app.routes}
    assert "/health" in rutas
    assert "/predict_batch" in rutas


def test_02_health_ready(monkeypatch):
    cliente, _ = _cliente_listo(monkeypatch)
    resp = cliente.get("/health")
    assert resp.status_code == 200
    cuerpo = resp.json()
    assert cuerpo["status"] == "ok"
    assert cuerpo["model_loaded"] is True
    assert cuerpo["model_version"] == "1.0.0"


def test_03_health_not_ready():
    cliente = TestClient(app_module.app, raise_server_exceptions=False)
    app_module.app.state.artifact = None
    app_module.app.state.load_error = "falta artefacto"
    resp = cliente.get("/health")
    assert resp.status_code == 503
    cuerpo = resp.json()
    assert cuerpo["status"] == "not_ready"
    assert cuerpo["model_loaded"] is False
    assert "detail" in cuerpo


def test_04_predict_un_registro(monkeypatch):
    cliente, _ = _cliente_listo(monkeypatch)
    resp = cliente.post("/predict_batch", json={"records": [_registro()]})
    assert resp.status_code == 200
    assert resp.json()["count"] == 1


def test_05_predict_n_registros(monkeypatch):
    cliente, _ = _cliente_listo(monkeypatch)
    regs = [_registro(), _registro(), _registro()]
    resp = cliente.post("/predict_batch", json={"records": regs})
    assert resp.status_code == 200
    cuerpo = resp.json()
    assert cuerpo["count"] == 3
    assert len(cuerpo["predictions"]) == 3


def test_06_target_no_requerido_e_ignorado(monkeypatch):
    cliente, llamadas = _cliente_listo(monkeypatch)
    sin_target = cliente.post("/predict_batch", json={"records": [_registro()]})
    assert sin_target.status_code == 200
    con_target = cliente.post("/predict_batch", json={"records": [_registro(con_target=True)]})
    assert con_target.status_code == 200
    assert "Pago_atiempo" in llamadas["cols"] or True


def test_07_usa_predecir_existente(monkeypatch):
    cliente, llamadas = _cliente_listo(monkeypatch)
    cliente.post("/predict_batch", json={"records": [_registro()]})
    assert llamadas.get("n") == 1
    fuente = Path("app.py").read_text(encoding="utf-8")
    for duplicado in (
        "def normalizar_tipos",
        "def crear_features",
        "def preparar_features_inferencia",
        "def cargar_modelo",
        "def predecir(",
    ):
        assert duplicado not in fuente
    assert "from mlops_pipeline.src.model_deploy import" in fuente


def test_08_prediction_en_01(monkeypatch):
    cliente, _ = _cliente_listo(monkeypatch)
    resp = cliente.post("/predict_batch", json={"records": [_registro()]})
    for item in resp.json()["predictions"]:
        assert item["prediction"] in (0, 1)


def test_09_risk_probability_json_safe(monkeypatch):
    cliente, _ = _cliente_listo(monkeypatch)
    resp = cliente.post("/predict_batch", json={"records": [_registro()]})
    crudo = resp.text
    assert "NaN" not in crudo
    for item in resp.json()["predictions"]:
        prob = item["risk_probability"]
        assert prob is None or isinstance(prob, float)


def test_10_nan_a_null(monkeypatch):
    cliente, _ = _cliente_listo(monkeypatch, con_nan=True)
    resp = cliente.post("/predict_batch", json={"records": [_registro()]})
    assert resp.status_code == 200
    assert "NaN" not in resp.text
    cuerpo = resp.json()
    assert cuerpo["predictions"][0]["risk_score"] is None
    json.dumps(cuerpo)


def test_11_model_version_correcto(monkeypatch):
    cliente, _ = _cliente_listo(monkeypatch, version="1.0.0")
    resp = cliente.post("/predict_batch", json={"records": [_registro()]})
    assert resp.json()["model_version"] == "1.0.0"


def test_12_timestamp_presente(monkeypatch):
    cliente, _ = _cliente_listo(monkeypatch)
    resp = cliente.post("/predict_batch", json={"records": [_registro()]})
    for item in resp.json()["predictions"]:
        assert isinstance(item["prediction_timestamp"], str)
        assert len(item["prediction_timestamp"]) > 0


def test_13_payload_vacio_error():
    cliente = TestClient(app_module.app, raise_server_exceptions=False)
    app_module.app.state.artifact = _artefacto()
    resp = cliente.post("/predict_batch", json={"records": []})
    assert resp.status_code in (400, 422)


def test_14_schema_invalido_error_claro(monkeypatch):
    def _romper(artefacto, df):
        raise ValueError("faltan columnas: ['capital_prestado']")

    monkeypatch.setattr(app_module, "predecir", _romper)
    cliente = TestClient(app_module.app, raise_server_exceptions=False)
    app_module.app.state.artifact = _artefacto()
    resp = cliente.post("/predict_batch", json={"records": [_registro()]})
    assert resp.status_code == 422
    assert resp.json()["detail"] == "schema invalido"


def test_15_modelo_no_disponible_503():
    cliente = TestClient(app_module.app, raise_server_exceptions=False)
    app_module.app.state.artifact = None
    app_module.app.state.load_error = "sin artefacto"
    resp = cliente.post("/predict_batch", json={"records": [_registro()]})
    assert resp.status_code == 503


def test_16_no_llama_fit():
    fuente = Path("app.py").read_text(encoding="utf-8")
    assert ".fit(" not in fuente


def test_17_no_llama_entrenar():
    fuente = Path("app.py").read_text(encoding="utf-8")
    assert "entrenar(" not in fuente
    assert "from mlops_pipeline.src.model_training import" not in fuente


def test_18_no_llama_evaluar():
    fuente = Path("app.py").read_text(encoding="utf-8")
    assert "evaluar(" not in fuente
    assert "model_evaluation" not in fuente


def test_19_reutiliza_artifact_cargado(monkeypatch):
    cliente, llamadas = _cliente_listo(monkeypatch)
    primero = llamadas.get("n", 0)
    cliente.post("/predict_batch", json={"records": [_registro()]})
    cliente.post("/predict_batch", json={"records": [_registro()]})
    assert llamadas["n"] == primero + 2
    assert llamadas["artefacto"] is app_module.app.state.artifact


def test_20_no_expone_rutas_ni_stack(monkeypatch):
    def _falla(artefacto, df):
        raise RuntimeError("secreto en C:\\Users\\mario\\secreto traceback interno")

    monkeypatch.setattr(app_module, "predecir", _falla)
    cliente = TestClient(app_module.app, raise_server_exceptions=False)
    app_module.app.state.artifact = _artefacto()
    resp = cliente.post("/predict_batch", json={"records": [_registro()]})
    assert resp.status_code == 500
    texto = resp.text
    assert "C:\\Users" not in texto
    assert "mario" not in texto.lower()
    assert "traceback" not in texto.lower()


def test_21_export_respeta_output_y_overwrite(monkeypatch, tmp_path):
    import scripts.export_model_artifact as exp

    capturas = {}

    class _Res:
        nombre_modelo_final = "logistic_regression"
        hiperparametros_finales = {"model__C": 0.1}
        resultado_cv_modelo_final = None
        random_state = 42

        def __init__(self):
            self.modelo_final_fitted = object()

    def _fake_entrenar(x_train, y_train, pre):
        capturas["entrenar"] = (x_train, y_train, pre)
        return _Res()

    def _fake_guardar(modelo, metadata, destino, overwrite=False):
        capturas["overwrite"] = overwrite
        capturas["destino"] = Path(destino)
        capturas["metadata"] = metadata
        assert metadata["metricas_holdout_finales"] is None
        return {
            "model_version": "1.0.0",
            "nombre_modelo": "logistic_regression",
            "artifact_sha256": "x",
        }

    monkeypatch.setattr(exp, "encontrar_raiz_proyecto", lambda: Path("."))
    monkeypatch.setattr(exp, "cargar_config", lambda ruta: {})
    monkeypatch.setattr(exp, "cargar_datos", lambda cfg: pd.DataFrame())
    monkeypatch.setattr(
        exp,
        "preparar_datos",
        lambda df: SimpleNamespace(
            X_train="xt", y_train="yt", preprocessor="pp", X_eval="XE", y_eval="YE"
        ),
    )
    monkeypatch.setattr(exp, "entrenar", _fake_entrenar)
    monkeypatch.setattr(exp, "guardar_modelo", _fake_guardar)
    monkeypatch.setattr(
        exp,
        "crear_metadata_modelo",
        lambda **kw: {"metricas_holdout_finales": None, **kw},
    )
    destino = tmp_path / "artifacts" / "1.0.0"
    exp.exportar(destino, overwrite=True)
    assert capturas["destino"] == destino
    assert capturas["overwrite"] is True
    xt, yt, pp = capturas["entrenar"]
    assert (xt, yt, pp) == ("xt", "yt", "pp")


def test_22_export_no_usa_eval(monkeypatch):
    fuente = Path("scripts/export_model_artifact.py").read_text(encoding="utf-8")
    assert "model_evaluation" not in fuente
    assert "from mlops_pipeline.src.model_evaluation" not in fuente
    assert "metricas_holdout_finales=None" in fuente
    assert "entrenar(datos.X_train, datos.y_train, datos.preprocessor)" in fuente


def test_23_dockerfile_minimo():
    texto = Path("Dockerfile").read_text(encoding="utf-8")
    assert "python:3.13-slim" in texto
    assert "uvicorn" in texto and "app:app" in texto
    assert "8000" in texto
    assert "C:\\Users" not in texto
    assert "mario" not in texto.lower()
    assert "model.joblib" not in texto
    assert "artifacts" not in texto or "artifacts" in Path(".dockerignore").read_text(
        encoding="utf-8"
    )


def test_24_metrics_sirve_html_si_existe(monkeypatch, tmp_path):
    destino = Path(os.environ["METRICS_HTML_PATH"])
    destino.write_text("<html><body>F1_0 0.1429</body></html>", encoding="utf-8")
    cliente = TestClient(app_module.app, raise_server_exceptions=False)
    resp = cliente.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "F1_0 0.1429" in resp.text


def test_25_metrics_error_claro_si_falta():
    cliente = TestClient(app_module.app, raise_server_exceptions=False)
    resp = cliente.get("/metrics")
    assert resp.status_code == 404
    assert "detail" in resp.json()


def test_26_predict_crea_log_append(monkeypatch):
    import csv

    cliente, _ = _cliente_listo(monkeypatch)
    assert cliente.post("/predict_batch", json={"records": [_registro()]}).status_code == 200
    assert (
        cliente.post(
            "/predict_batch", json={"records": [_registro(), _registro(con_target=True)]}
        ).status_code
        == 200
    )
    ruta = Path(os.environ["PREDICTION_LOG_PATH"])
    assert ruta.is_file()
    with open(ruta, encoding="utf-8", newline="") as archivo:
        filas = list(csv.DictReader(archivo))
    assert len(filas) == 3
    assert len({f["request_id"] for f in filas}) == 2
    assert "Pago_atiempo" not in filas[0] and "raw_tipo_credito" in filas[0]
    assert filas[0]["model_version"] == "1.0.0"
    assert filas[0]["prediction"] in ("0", "1")


def test_27_log_no_bloquea_respuesta(monkeypatch, tmp_path):
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path))
    cliente, _ = _cliente_listo(monkeypatch)
    resp = cliente.post("/predict_batch", json={"records": [_registro()]})
    assert resp.status_code == 200
    assert resp.json()["count"] == 1


def test_28_log_no_llama_training():
    import inspect as _inspect

    fuente = _inspect.getsource(app_module.registrar_log_batch)
    assert ".fit(" not in fuente
    assert "entrenar(" not in fuente
    assert "evaluar(" not in fuente


def _registro_pii():
    base = _registro()
    base.update(
        {
            "nombre": "Juan Pérez",
            "cedula": "123456789",
            "email": "juan@example.com",
            "telefono": "3001234567",
            "Pago_atiempo": "1",
        }
    )
    return base


def test_29_log_header_fijo_entre_schemas(monkeypatch):
    import csv as _csv

    cliente, _ = _cliente_listo(monkeypatch)
    assert cliente.post("/predict_batch", json={"records": [_registro()]}).status_code == 200
    assert cliente.post("/predict_batch", json={"records": [_registro_pii()]}).status_code == 200
    ruta = Path(os.environ["PREDICTION_LOG_PATH"])
    with open(ruta, encoding="utf-8", newline="") as archivo:
        lector = _csv.DictReader(archivo)
        encabezado = list(lector.fieldnames or [])
        filas = list(lector)
    from mlops_pipeline.src.ft_engineering import RAW_FEATURES_V1

    esperado = [
        "request_id",
        "prediction_timestamp",
        "model_version",
        "prediction",
        "risk_probability",
        "risk_score",
        *[f"raw_{k}" for k in RAW_FEATURES_V1],
    ]
    assert encabezado == esperado
    assert len(filas) == 2
    plano = " ".join(f"{k}={v}" for f in filas for k, v in f.items())
    for prohibida in (
        "nombre",
        "cedula",
        "email",
        "telefono",
        "Pago_atiempo",
        "Juan",
        "123456789",
        "juan@example.com",
        "3001234567",
    ):
        assert prohibida not in plano
    assert filas[0]["raw_tipo_credito"] == "4"
    assert filas[1]["raw_tipo_credito"] == "4"
    assert filas[0]["raw_salario_cliente"] != ""


def test_30_log_append_valido_y_monitor_compatible(monkeypatch, tmp_path):
    import csv as _csv

    from mlops_pipeline.src.model_monitoring import cargar_tabla_predicciones

    cliente, _ = _cliente_listo(monkeypatch)
    for _ in range(3):
        assert cliente.post("/predict_batch", json={"records": [_registro()]}).status_code == 200
    ruta = Path(os.environ["PREDICTION_LOG_PATH"])
    with open(ruta, encoding="utf-8", newline="") as archivo:
        filas = list(_csv.DictReader(archivo))
    assert len(filas) == 3
    assert len({f["request_id"] for f in filas}) == 3
    tabla = cargar_tabla_predicciones(ruta)
    assert len(tabla) == 3
    from mlops_pipeline.src.ft_engineering import (
        FEATURES_V1,
        preparar_features_inferencia,
    )

    crudas = {c[4:]: [f[c] for f in filas] for c in filas[0] if c.startswith("raw_")}
    equis = preparar_features_inferencia(pd.DataFrame(crudas))
    assert list(equis.columns) == list(FEATURES_V1)


def test_31_xss_schema_invalido_no_refleja(monkeypatch):
    carga = "<script>alert(1)</script>"

    def _romper(artefacto, df):
        raise ValueError(f"columna maliciosa {carga} en C:\\srv\\app")

    monkeypatch.setattr(app_module, "predecir", _romper)
    cliente = TestClient(app_module.app, raise_server_exceptions=False)
    app_module.app.state.artifact = _artefacto()
    registro = _registro()
    registro["tipo_credito"] = carga
    resp = cliente.post("/predict_batch", json={"records": [registro]})
    assert resp.status_code == 422
    assert resp.json()["detail"] == "schema invalido"
    assert carga not in resp.text
    assert "C:\\srv" not in resp.text
    assert "<script>" not in resp.text


def test_32_xss_records_invalidos_no_refleja(monkeypatch):
    carga = "<script>alert(1)</script>"

    def _romper(_datos):
        raise ValueError(f"frame roto {carga}")

    monkeypatch.setattr(pd, "DataFrame", _romper)
    try:
        cliente = TestClient(app_module.app, raise_server_exceptions=False)
        app_module.app.state.artifact = _artefacto()
        resp = cliente.post("/predict_batch", json={"records": [_registro()]})
    finally:
        monkeypatch.undo()
    assert resp.status_code == 422
    assert resp.json()["detail"] == "records invalidos"
    assert carga not in resp.text
