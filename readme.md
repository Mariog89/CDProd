# CDProd — Análisis y modelado de riesgo crediticio

## Objetivo

Analizar la información disponible sobre créditos y clientes con el fin de identificar problemas de calidad de datos, caracterizar el comportamiento de la cartera y encontrar factores asociados con el cumplimiento o incumplimiento oportuno de los pagos; construir y evaluar un modelo de riesgo crediticio y exponerlo mediante un endpoint batch con monitoreo de deriva.

## Estado actual

Disponible actualmente:

- Datos en Google BigQuery (`cdprod-upb-mg.novabank_dev.Creditos`): 10.763 registros sintéticos, 23 columnas.
- Pipeline de Stage 3 en `mlops_pipeline/src/`.
- `Cargar_datos.py`: carga reproducible desde BigQuery vía ADC (todo como string, orden determinista).
- `ft_engineering.py`: feature engineering reproducible sin leakage (16 features v1, split 80/20 estratificado).
- `hueristic_model.py`: baseline heurístico de negocio fijo y explicable (reglas R1/R2/R3 congeladas).
- `model_training.py`: comparación CV-safe de familias ML, selección, tuning y fit final sobre train.
- `model_evaluation.py`: evaluación final sobre holdout sellado (métricas de riesgo, matriz, CV-vs-holdout, inferencia y estadística exploratoria).
- `model_deploy.py`: persistencia versionada del Pipeline final, carga validada con SHA-256 e inferencia reutilizable sin target.
- `model_monitoring.py`: monitoreo batch por ventana (calidad cruda, feature drift con PSI, prediction drift y performance solo con labels, sin retraining).
- Configuración canónica de Stage 3 en `mlops_pipeline/src/config.json`.
- Setup del entorno en Windows mediante `set_up.bat`.
- Dependencias directas reproducibles declaradas en `requirements.txt`.
- `comprension_eda.ipynb`: notebook de comprensión/EDA existente, migrado desde el análisis exploratorio previo, ubicado en `mlops_pipeline/src/comprension_eda.ipynb`.
- Calidad/CI con pytest, Ruff, GitHub Actions y SonarQube Cloud (cobertura y análisis sobre `mlops_pipeline/src`, `app.py` y `scripts`).

Stage 3 implementado con 8 archivos: `Cargar_datos.py`, `ft_engineering.py`, `hueristic_model.py`, `model_training.py`, `model_evaluation.py`, `model_deploy.py`, `model_monitoring.py` y `config.json`.

## Estructura del proyecto

```text
CDProd/
├── mlops_pipeline/
│   └── src/
│       ├── Cargar_datos.py
│       ├── comprension_eda.ipynb
│       ├── ft_engineering.py
│       ├── hueristic_model.py
│       ├── model_training.py
│       ├── model_deploy.py
│       ├── model_evaluation.py
│       ├── model_monitoring.py
│       └── config.json
├── scripts/
│   └── export_model_artifact.py
├── tests/
├── .github/
├── app.py
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── sonar-project.properties
├── .gitignore
├── readme.md
└── set_up.bat
```

`artifacts/` y `runtime/` son directorios generados localmente e ignorados por Git: no forman parte de los archivos versionados.

El entorno virtual local `codigoproyecto-venv` no forma parte del repositorio: está ignorado por Git y se genera localmente con `set_up.bat`.

El análisis exploratorio histórico que originó el diseño actual se conserva únicamente en el historial Git; ya no forma parte de la estructura activa.

## Requisitos del entorno

- El proyecto ha sido validado con Python 3.13.
- Windows para el script automatizado `set_up.bat`.
- Las dependencias directas están declaradas con versiones exactas en `requirements.txt`.

## Setup en Windows

Flujo recomendado desde un clon limpio:

```bat
git clone https://github.com/Mariog89/CDProd.git
cd CDProd
.\set_up.bat
```

`set_up.bat`:

- Crea `codigoproyecto-venv` si no existe y lo reutiliza si ya existe.
- Instala las dependencias desde `requirements.txt`.
- Registra el kernel de Jupyter como "Python CDProd".

## Ejecución de pruebas y calidad

```bat
.\codigoproyecto-venv\Scripts\python.exe -m pytest -q
.\codigoproyecto-venv\Scripts\python.exe -m ruff check .
.\codigoproyecto-venv\Scripts\python.exe -m ruff format --check .
.\codigoproyecto-venv\Scripts\python.exe -m pip check
```

Para reproducir el entorno validado del proyecto, seleccione el kernel `Python CDProd`.

## Configuración

`mlops_pipeline/src/config.json` contiene la configuración BigQuery y la variable objetivo:

- `bq_project`: proyecto GCP (`cdprod-upb-mg`).
- `bq_dataset`: dataset (`novabank_dev`).
- `bq_table`: tabla (`Creditos`).
- `target`: variable objetivo (`Pago_atiempo`).

## Datos

La tabla `cdprod-upb-mg.novabank_dev.Creditos` contiene registros históricos sintéticos de créditos con información sobre:

- Características del crédito y el cliente.
- Historial crediticio y puntajes.
- Saldos, cuotas y cumplimiento de pagos.

Los datos permanecen en BigQuery: el repositorio solo contiene código y configuración. La carga usa Application Default Credentials (ADC); en CI las pruebas usan mocks y DataFrames sintéticos, sin credenciales GCP. Los tests de integración reales (`data_regression`, `bq_live`) solo se ejecutan con `CDPROD_BQ_LIVE=1`.

La consulta centralizada en `Cargar_datos.py` aplica `CAST(col AS STRING)` a las 23 columnas y `ORDER BY` sobre las 23 columnas crudas. Las 10.763 tuplas son únicas como multiconjunto, por lo que el orden es total y determinista entre ejecuciones. No reproduce el orden del antiguo CSV local: el split 80/20 conserva los conteos (train 8610: 8201/409; eval 2153: 2051/102) pero la composición de cada partición puede diferir.

## Resultados del modelo por fuente de datos

Históricos (CSV local `Base_de_datos.csv`, partición por orden de archivo):

- Modelo: `LogisticRegression`, `model__C = 0.1`.
- CV: F1_0 ≈ 0.1322 ± 0.0084, Recall_0 ≈ 0.5891.
- Holdout: F1_0 ≈ 0.1429, Recall_0 ≈ 0.6569, Precision_0 ≈ 0.0801, Balanced Accuracy ≈ 0.6410, ROC-AUC ≈ 0.6786.

Experimentales (BigQuery `novabank_dev.Creditos`, partición por orden determinista BQ, mismo código sin cambios):

- Modelo: `LogisticRegression`, `model__C = 10.0`.
- CV: F1_0 ≈ 0.1364 ± 0.0164, Recall_0 ≈ 0.5917.
- Holdout: F1_0 ≈ 0.1196, Recall_0 ≈ 0.5294, Precision_0 ≈ 0.0674, Balanced Accuracy ≈ 0.5826, ROC-AUC ≈ 0.5995.

Los holdouts son particiones diferentes, por lo que estas cifras no deben interpretarse como mejora o degradación del modelo. El artefacto servido por FastAPI (`artifacts/1.0.0`) conserva su trazabilidad en `metadata.json` (`model_version`, `random_state`, hiperparámetros); un artefacto reexportado desde BigQuery quedará registrado con sus propios hiperparámetros y métricas CV.

## Endpoint de predicción batch

La API batch reutiliza `model_deploy.cargar_modelo()` y `model_deploy.predecir()` sin reentrenar ni recargar el modelo por request. Clase `0` = riesgo, `1` = pago oportuno. El input no requiere `Pago_atiempo`; si llega como columna extra se ignora.

1. Generar el artefacto operativo (solo train, sin holdout):

```bat
.\codigoproyecto-venv\Scripts\python.exe scripts/export_model_artifact.py --output artifacts/1.0.0
```

Genera `artifacts/1.0.0/model.joblib` + `metadata.json` (`model_version: 1.0.0`). `artifacts/` está ignorado por Git y se monta como volumen, no se hornea en la imagen.

2. Iniciar la API local:

```bat
.\codigoproyecto-venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Variable opcional: `MODEL_ARTIFACT_DIR` (default `artifacts/1.0.0`).

3. Documentación interactiva: `http://127.0.0.1:8000/docs`. Salud: `GET /health` responde `{"status":"ok","model_loaded":true,"model_version":"1.0.0"}`.

4. Ejemplo `POST /predict_batch`:

```json
{"records": [{"tipo_credito": "4", "capital_prestado": "1852560", "plazo_meses": "12", "edad_cliente": "32", "tipo_laboral": "Empleado", "salario_cliente": "3500000", "total_otros_prestamos": "1000000", "cuota_pactada": "128650", "puntaje_datacredito": "795.0", "cant_creditosvigentes": "0", "creditos_sectorFinanciero": "0", "creditos_sectorCooperativo": "0", "creditos_sectorReal": "0", "promedio_ingresos_datacredito": "916148", "tendencia_ingresos": "Creciente"}]}
```

Respuesta conceptual:

```json
{"model_version": "1.0.0", "count": 1, "predictions": [{"prediction": 1, "risk_probability": 0.4499, "risk_score": null, "prediction_timestamp": "2026-09-16T02:21:20Z"}]}
```

5. Build Docker:

```powershell
docker build -t cdprod-api .
```

6. Run Docker (el artefacto se monta, usar placeholder para la ruta local):

```powershell
docker run --rm `
  -p 8000:8000 `
  -v "<RUTA_ABSOLUTA_ARTIFACTS>:/app/artifacts" `
  -e MODEL_ARTIFACT_DIR=/app/artifacts/1.0.0 `
  cdprod-api
```

## Training literal (PDF)

`model_training.py` expone los wrappers exigidos sin duplicar lógica:

- `summarize_classification(...)` delega en `evaluar_cv`.
- `build_model(...)` reutiliza `crear_modelos_candidatos`.
- `tabla_resumen_modelos(comparativa)` presenta por modelo `mean_f1_0/std_f1_0/mean_recall_0/mean_precision_0/mean_balanced_accuracy/mean_cv_fit_time` más las columnas semánticas `Performance` (F1_0/recall_0), `Consistency` (std_f1_0) y `Scalability` (fit time), sin scores ponderados.
- `plot_comparacion_modelos(comparativa)` grafica F1_0 medio ± std para heuristic, logistic_regression, random_forest y hist_gradient_boosting.
- `main()` imprime además `=== Tabla resumen Performance / Consistency / Scalability ===`. Resultados y selección intactos.

## Pestaña de métricas

`model_evaluation.generar_pestana_metricas(resultado, ruta)` genera `runtime/metrics.html` (ignorada por Git) solo con datos existentes del `ResultadoEvaluacionCompleta`: modelo, best params, precision/recall/F1/balanced/accuracy, matriz, CV vs holdout, ROC-AUC/AP, ANOVA/Tukey/Friedman, advertencias e imágenes de `plot_*`. No recalcula nada.

```bat
.\codigoproyecto-venv\Scripts\python.exe -c "from mlops_pipeline.src.Cargar_datos import cargar_config,cargar_datos,encontrar_raiz_proyecto; from mlops_pipeline.src.ft_engineering import preparar_datos; from mlops_pipeline.src.model_training import entrenar; from mlops_pipeline.src.model_evaluation import evaluar,generar_pestana_metricas; r=encontrar_raiz_proyecto(); c=cargar_config(r/'mlops_pipeline/src/config.json'); d=preparar_datos(cargar_datos(c)); print(generar_pestana_metricas(evaluar(entrenar(d.X_train,d.y_train,d.preprocessor),d.X_eval,d.y_eval)))"
```

Servida por la API sin reevaluar: `GET /metrics` (ruta configurable con `METRICS_HTML_PATH`, default `runtime/metrics.html`; 404 claro si aún no se generó).

## Monitoring operativo

- Cada `POST /predict_batch` agrega al log append-only `runtime/prediction_log.csv` (configurable con `PREDICTION_LOG_PATH`): `request_id` común al batch, timestamp, versión, predicción, probabilidades e inputs raw con prefijo `raw_`. El esquema es fijo (`RAW_FEATURES_V1`: los 15 campos raw necesarios para reconstruir FEATURES_V1); columnas extra del request —incluido `Pago_atiempo` o datos personales arbitrarios— se aceptan en inferencia pero nunca se persisten. Nunca guarda `Pago_atiempo`.
- `model_monitoring.cargar_tabla_predicciones(ruta)` lee ese log validando columnas mínimas.
- `model_monitoring.muestrear_batch(df, n, random_state=42)` muestrea de forma reproducible sin modificar el input.
- `model_monitoring.monitorizar_desde_log(perfil, ruta_log, n_sample=...)` encadena carga → muestreo → `monitorizar(...)` y registra `n_total/n_sampled`. Baseline train, bins PSI y thresholds intactos.
- Periodicidad: `MONITORING_PERIOD = "weekly"`. Sin scheduler embebido; ejecutar semanalmente con cron, Windows Task Scheduler o scheduler externo:

```bat
.\codigoproyecto-venv\Scripts\python.exe -m mlops_pipeline.src.model_monitoring
```

## Workflow Git

- `main` representa la rama estable.
- `develop` es la rama de integración.
- Los cambios se realizan mediante ramas `feature/*` y Pull Requests hacia `develop`.

## Próximos pasos

Stage 3 completado: monitoreo batch descriptivo implementado en `model_monitoring.py` (baseline solo train, PSI congelado, sin retraining ni servicios externos).

Ya implementados: `Cargar_datos.py`, `comprension_eda.ipynb`, `ft_engineering.py`, `hueristic_model.py`, `model_training.py`, `model_evaluation.py`, `model_deploy.py` y `model_monitoring.py`.

## Licencia

Uso educativo / interno.
