"""Evaluación final del Stage 3 sobre el holdout sellado.

Único módulo downstream autorizado para consumir X_eval/y_eval con fines
de métricas. Recibe un ResultadoEntrenamiento ya ajustado, evalúa el
pipeline ML final y el heurístico congelado, compara CV final contra
holdout, mide inferencia end-to-end y ejecuta estadística exploratoria
(ANOVA/Tukey/Friedman) sobre F1_0 por fold de las familias base.

No reentrena, no tunea, no cambia thresholds, no selecciona otro modelo
y no modifica features. El holdout es de uso único por protocolo.
"""

import html
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import (
    f_oneway,
    friedmanchisquare,
    tukey_hsd,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from mlops_pipeline.src.Cargar_datos import (  # noqa: E402
    cargar_config,
    cargar_datos,
    encontrar_raiz_proyecto,
)
from mlops_pipeline.src.ft_engineering import preparar_datos  # noqa: E402
from mlops_pipeline.src.hueristic_model import predecir_heuristico  # noqa: E402
from mlops_pipeline.src.model_training import entrenar  # noqa: E402

RISK_LABEL = 0
LABELS = [0, 1]

ADVERTENCIA_HEURISTICO = (
    "Las reglas heurísticas se inspiraron en un EDA histórico realizado sobre el "
    "dataset completo, por lo que sus métricas de holdout se reportan descriptivamente "
    "como baseline de negocio y no representan independencia perfecta."
)

ADVERTENCIA_ANOVA = (
    "ANOVA EXPLORATORIO: n=5 folds, folds no completamente independientes "
    "y datos pareados por fold. Interpretar solo como señal descriptiva."
)

ADVERTENCIA_FRIEDMAN = (
    "Friedman es complementario y más coherente con el diseño pareado, "
    "pero con solo 5 folds tiene baja potencia. No se agrega Nemenyi/Wilcoxon."
)

ADVERTENCIA_CONSISTENCY = (
    "Consistency proviene de CV (std entre folds). El holdout es un único "
    "punto final: no se recalcula consistency ni z-score formal sobre holdout."
)

ADVERTENCIA_THRESHOLD = (
    "La evaluación usa predict tal cual, sin buscar threshold óptimo ni "
    "maximizar F1 sobre X_eval. Curvas ROC/PR descriptivas únicamente."
)

ADVERTENCIA_USO_UNICO = (
    "Holdout de uso único por protocolo/documentación. Después de ver el holdout "
    "no hay vuelta automática a training ni cambio de decisiones."
)

FAMILIAS_ESTADISTICA = (
    "heuristic",
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
)


@dataclass
class MetricasRiesgo:
    """Métricas orientadas a la clase de riesgo 0."""

    precision_0: float
    recall_0: float
    f1_0: float
    balanced_accuracy: float
    accuracy_ctx: float
    support_0: int
    support_1: int


@dataclass
class MetricasProbabilisticas:
    """AUC/AP orientados a riesgo; None + motivo si no hay score adecuado."""

    roc_auc_risk: float | None
    average_precision_risk: float | None
    motivo: str | None = None


@dataclass
class ResultadoHoldout:
    """Evaluación de un modelo sobre X_eval/y_eval."""

    nombre: str
    metricas: MetricasRiesgo
    matriz_nombres: dict
    matriz_array: np.ndarray
    reporte_clasificacion: dict
    probabilisticas: MetricasProbabilisticas
    n_eval: int


@dataclass
class ComparativaItem:
    """Comparación de una métrica: holdout contra media/desviación de CV."""

    holdout: float
    cv_mean: float
    cv_std: float
    delta: float


@dataclass
class ComparativaCVHoldout:
    """CV final (modelo tuneado) contra holdout para 4 métricas."""

    f1_0: ComparativaItem
    recall_0: ComparativaItem
    precision_0: ComparativaItem
    balanced_accuracy: ComparativaItem


@dataclass
class ResultadoEstadistico:
    """ANOVA/Tukey/Friedman exploratorios sobre F1_0 por fold."""

    modelos: list
    n_folds: int
    metrica: str
    anova_statistic: float
    anova_pvalue: float
    anova_advertencia: str
    tukey: object | None
    tukey_mensaje: str
    friedman_statistic: float
    friedman_pvalue: float
    friedman_advertencia: str


@dataclass
class InferenceTiming:
    """Tiempo puro end-to-end de predict incluyendo preprocessing."""

    median_total_ms: float
    mean_total_ms: float
    std_total_ms: float
    median_ms_per_row: float
    rows_per_second: float
    n_rows: int
    n_repeticiones: int


@dataclass
class ResultadoEvaluacionCompleta:
    """Paquete estructurado devuelto por evaluar()."""

    nombre_modelo_final: str
    hiperparametros_finales: dict
    justificacion_seleccion: str
    holdout_modelo_final: ResultadoHoldout
    holdout_heuristico: ResultadoHoldout
    comparacion_cv_final_vs_holdout: ComparativaCVHoldout
    analisis_estadistico_familias: ResultadoEstadistico
    inference_timing: InferenceTiming
    advertencias_metodologicas: list = field(default_factory=list)


def calcular_metricas_clase_riesgo(y_true, y_pred) -> MetricasRiesgo:
    """Calcula precisión/recall/F1 con clase 0 como positiva de negocio."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    precision_0 = float(precision_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0))
    recall_0 = float(recall_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0))
    f1_0 = float(f1_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0))
    balanced_accuracy = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
    accuracy_ctx = float(accuracy_score(y_true_arr, y_pred_arr))
    support_0 = int(np.sum(y_true_arr == 0))
    support_1 = int(np.sum(y_true_arr == 1))
    return MetricasRiesgo(
        precision_0=precision_0,
        recall_0=recall_0,
        f1_0=f1_0,
        balanced_accuracy=balanced_accuracy,
        accuracy_ctx=accuracy_ctx,
        support_0=support_0,
        support_1=support_1,
    )


def matriz_confusion_riesgo(y_true, y_pred) -> tuple[dict, np.ndarray]:
    """Matriz con labels=[0,1] y nombres explícitos de negocio."""
    mat = confusion_matrix(y_true, y_pred, labels=[0, 1])
    nombres = {
        "riesgo_real_predicho_riesgo": int(mat[0, 0]),
        "riesgo_real_predicho_pago": int(mat[0, 1]),
        "pago_real_predicho_riesgo": int(mat[1, 0]),
        "pago_real_predicho_pago": int(mat[1, 1]),
    }
    return nombres, mat


def reporte_clasificacion_riesgo(y_true, y_pred) -> dict:
    """Classification report estructurado como complemento, no única salida."""
    return classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        output_dict=True,
        zero_division=0,
    )


def obtener_score_riesgo(modelo, X):
    """Devuelve score orientado a riesgo (clase 0) o None si no hay.

    Si existe predict_proba localiza la columna de la clase 0 mediante
    modelo.classes_ y nunca asume [:, 1]. Si solo hay decision_function
    binaria, orienta el signo según classes_ sin convertir a probabilidad.
    """
    predict_proba = getattr(modelo, "predict_proba", None)
    if callable(predict_proba):
        try:
            proba = np.asarray(predict_proba(X))
            classes = np.asarray(modelo.classes_)
            indice = list(classes).index(RISK_LABEL)
            return np.asarray(proba[:, indice], dtype=float)
        except Exception:
            pass
    decision_function = getattr(modelo, "decision_function", None)
    if callable(decision_function):
        try:
            scores = np.asarray(decision_function(X), dtype=float)
            if scores.ndim > 1:
                scores = scores.ravel()
            classes = getattr(modelo, "classes_", None)
            if classes is not None:
                classes = np.asarray(classes)
            if classes is not None and len(classes) == 2:
                if RISK_LABEL == classes[1]:
                    return scores
                if RISK_LABEL == classes[0]:
                    return -scores
            return None
        except Exception:
            return None
    return None


def calcular_metricas_probabilisticas(y_true, risk_score) -> MetricasProbabilisticas:
    """Calcula ROC-AUC/AP de riesgo solo si existe score adecuado."""
    if risk_score is None:
        return MetricasProbabilisticas(
            roc_auc_risk=None,
            average_precision_risk=None,
            motivo="Modelo sin predict_proba ni decision_function utilizable.",
        )
    try:
        y_arr = np.asarray(y_true)
        s_arr = np.asarray(risk_score, dtype=float)
        if len(s_arr) != len(y_arr) or not np.all(np.isfinite(s_arr)):
            return MetricasProbabilisticas(
                roc_auc_risk=None,
                average_precision_risk=None,
                motivo="Score de riesgo con longitud inválida o valores no finitos.",
            )
        y_risk = (y_arr == 0).astype(int)
        if len(np.unique(y_risk)) < 2:
            return MetricasProbabilisticas(
                roc_auc_risk=None,
                average_precision_risk=None,
                motivo="y_true contiene una sola clase: AUC/AP no definidos.",
            )
        roc = float(roc_auc_score(y_risk, s_arr))
        ap = float(average_precision_score(y_risk, s_arr))
        return MetricasProbabilisticas(roc_auc_risk=roc, average_precision_risk=ap, motivo=None)
    except Exception as exc:
        return MetricasProbabilisticas(
            roc_auc_risk=None,
            average_precision_risk=None,
            motivo=f"No se pudo calcular AUC/AP: {exc}",
        )


def _evaluar_holdout_modelo(nombre: str, modelo, X_eval, y_eval) -> ResultadoHoldout:
    """Evalúa un estimator ya ajustado con predict tal cual, sin reentrenar."""
    y_pred = np.asarray(modelo.predict(X_eval))
    y_true_arr = np.asarray(y_eval)
    metricas = calcular_metricas_clase_riesgo(y_true_arr, y_pred)
    nombres, mat = matriz_confusion_riesgo(y_true_arr, y_pred)
    reporte = reporte_clasificacion_riesgo(y_true_arr, y_pred)
    risk_score = obtener_score_riesgo(modelo, X_eval)
    probabilisticas = calcular_metricas_probabilisticas(y_true_arr, risk_score)
    return ResultadoHoldout(
        nombre=nombre,
        metricas=metricas,
        matriz_nombres=nombres,
        matriz_array=mat,
        reporte_clasificacion=reporte,
        probabilisticas=probabilisticas,
        n_eval=int(len(y_true_arr)),
    )


def _evaluar_holdout_heuristico(X_eval, y_eval) -> ResultadoHoldout:
    """Evalúa reglas congeladas con las mismas 4 métricas comparables."""
    y_pred = predecir_heuristico(X_eval).to_numpy()
    y_true_arr = np.asarray(y_eval)
    metricas = calcular_metricas_clase_riesgo(y_true_arr, y_pred)
    nombres, mat = matriz_confusion_riesgo(y_true_arr, y_pred)
    reporte = reporte_clasificacion_riesgo(y_true_arr, y_pred)
    probabilisticas = MetricasProbabilisticas(
        roc_auc_risk=None,
        average_precision_risk=None,
        motivo="Heurístico sin AUC equivalente: score_riesgo es conteo 0..3, no probabilidad.",
    )
    return ResultadoHoldout(
        nombre="heuristic",
        metricas=metricas,
        matriz_nombres=nombres,
        matriz_array=mat,
        reporte_clasificacion=reporte,
        probabilisticas=probabilisticas,
        n_eval=int(len(y_true_arr)),
    )


def comparar_cv_holdout(resultado_cv_modelo_final, metricas_holdout) -> ComparativaCVHoldout:
    """Compara CV del modelo final tuneado contra holdout (descriptivo)."""

    def _item(holdout: float, cv_mean: float, cv_std: float) -> ComparativaItem:
        return ComparativaItem(
            holdout=float(holdout),
            cv_mean=float(cv_mean),
            cv_std=float(cv_std),
            delta=float(holdout - cv_mean),
        )

    return ComparativaCVHoldout(
        f1_0=_item(
            metricas_holdout.f1_0,
            resultado_cv_modelo_final.mean_f1_0,
            resultado_cv_modelo_final.std_f1_0,
        ),
        recall_0=_item(
            metricas_holdout.recall_0,
            resultado_cv_modelo_final.mean_recall_0,
            resultado_cv_modelo_final.std_recall_0,
        ),
        precision_0=_item(
            metricas_holdout.precision_0,
            resultado_cv_modelo_final.mean_precision_0,
            resultado_cv_modelo_final.std_precision_0,
        ),
        balanced_accuracy=_item(
            metricas_holdout.balanced_accuracy,
            resultado_cv_modelo_final.mean_balanced_accuracy,
            resultado_cv_modelo_final.std_balanced_accuracy,
        ),
    )


def medir_inferencia(modelo, X_eval, repeticiones: int = 10) -> InferenceTiming:
    """Mide predict end-to-end con warm-up, mediana principal y media/std."""
    modelo.predict(X_eval)
    tiempos_ms: list[float] = []
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        modelo.predict(X_eval)
        fin = time.perf_counter()
        tiempos_ms.append((fin - inicio) * 1000.0)
    arr = np.asarray(tiempos_ms, dtype=float)
    mediana = float(np.median(arr))
    media = float(np.mean(arr))
    desv = float(np.std(arr))
    n_rows = int(len(X_eval))
    mediana_por_fila = float(mediana / n_rows) if n_rows > 0 else 0.0
    filas_por_seg = float(n_rows / (mediana / 1000.0)) if mediana > 0 else 0.0
    return InferenceTiming(
        median_total_ms=mediana,
        mean_total_ms=media,
        std_total_ms=desv,
        median_ms_per_row=mediana_por_fila,
        rows_per_second=filas_por_seg,
        n_rows=n_rows,
        n_repeticiones=int(repeticiones),
    )


def analizar_familias_cv(comparativa_cv_familias) -> ResultadoEstadistico:
    """ANOVA/Tukey/Friedman exploratorios sobre F1_0 por fold de 4 familias."""
    modelos = [m for m in FAMILIAS_ESTADISTICA if m in comparativa_cv_familias]
    if len(modelos) != len(FAMILIAS_ESTADISTICA):
        raise ValueError("La estadística requiere las 4 familias: heurístico + 3 ML base.")
    muestras = [
        np.asarray(comparativa_cv_familias[m].metricas_por_fold["f1_0"], dtype=float)
        for m in modelos
    ]
    n_folds = int(len(muestras[0]))
    anova_res = f_oneway(*muestras)
    anova_stat = float(anova_res.statistic)
    anova_p = float(anova_res.pvalue)
    if anova_p < 0.05:
        tukey_res = tukey_hsd(*muestras)
        tukey_msg = "Post-hoc Tukey ejecutado (ANOVA significativo)."
    else:
        tukey_res = None
        tukey_msg = "Post-hoc no ejecutado porque el test omnibus no fue significativo."
    fried_res = friedmanchisquare(*muestras)
    return ResultadoEstadistico(
        modelos=list(modelos),
        n_folds=n_folds,
        metrica="f1_0",
        anova_statistic=anova_stat,
        anova_pvalue=anova_p,
        anova_advertencia=ADVERTENCIA_ANOVA,
        tukey=tukey_res,
        tukey_mensaje=tukey_msg,
        friedman_statistic=float(fried_res.statistic),
        friedman_pvalue=float(fried_res.pvalue),
        friedman_advertencia=ADVERTENCIA_FRIEDMAN,
    )


def evaluar(resultado_entrenamiento, X_eval, y_eval) -> ResultadoEvaluacionCompleta:
    """Evalúa el pipeline final y el heurístico sobre el holdout sellado."""
    modelo_final = resultado_entrenamiento.modelo_final_fitted
    nombre_final = str(resultado_entrenamiento.nombre_modelo_final)
    holdout_ml = _evaluar_holdout_modelo(nombre_final, modelo_final, X_eval, y_eval)
    holdout_h = _evaluar_holdout_heuristico(X_eval, y_eval)
    comparacion = comparar_cv_holdout(
        resultado_entrenamiento.resultado_cv_modelo_final, holdout_ml.metricas
    )
    estadistica = analizar_familias_cv(resultado_entrenamiento.comparativa_cv_familias)
    timing = medir_inferencia(modelo_final, X_eval)
    advertencias = [
        ADVERTENCIA_HEURISTICO,
        ADVERTENCIA_ANOVA,
        ADVERTENCIA_FRIEDMAN,
        ADVERTENCIA_CONSISTENCY,
        ADVERTENCIA_THRESHOLD,
        ADVERTENCIA_USO_UNICO,
        "Accuracy solo como contexto; la decisión usa F1/recall/precision de riesgo.",
    ]
    return ResultadoEvaluacionCompleta(
        nombre_modelo_final=nombre_final,
        hiperparametros_finales=dict(resultado_entrenamiento.hiperparametros_finales),
        justificacion_seleccion=str(resultado_entrenamiento.justificacion_seleccion),
        holdout_modelo_final=holdout_ml,
        holdout_heuristico=holdout_h,
        comparacion_cv_final_vs_holdout=comparacion,
        analisis_estadistico_familias=estadistica,
        inference_timing=timing,
        advertencias_metodologicas=advertencias,
    )


def plot_matriz_confusion(resultado_holdout: ResultadoHoldout):
    """Dibuja la matriz con semántica filas=real, columnas=predicción."""
    import matplotlib.pyplot as plt

    mat = np.asarray(resultado_holdout.matriz_array)
    fig, ax = plt.subplots()
    im = ax.imshow(mat, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1], labels=["pred_riesgo(0)", "pred_pago(1)"])
    ax.set_yticks([0, 1], labels=["real_riesgo(0)", "real_pago(1)"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión — {resultado_holdout.nombre}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(mat[i, j])), ha="center", va="center")
    fig.tight_layout()
    return fig


def plot_comparacion_holdout(comparacion: ComparativaCVHoldout):
    """Barras CV mean±std contra holdout para las 4 métricas."""
    import matplotlib.pyplot as plt

    metricas = ["f1_0", "recall_0", "precision_0", "balanced_accuracy"]
    items = [
        comparacion.f1_0,
        comparacion.recall_0,
        comparacion.precision_0,
        comparacion.balanced_accuracy,
    ]
    cv_means = [it.cv_mean for it in items]
    cv_stds = [it.cv_std for it in items]
    holds = [it.holdout for it in items]
    x = np.arange(len(metricas))
    fig, ax = plt.subplots()
    ax.bar(x - 0.2, cv_means, yerr=cv_stds, width=0.4, label="CV mean±std", capsize=4)
    ax.bar(x + 0.2, holds, width=0.4, label="Holdout")
    ax.set_xticks(x, metricas)
    ax.set_ylabel("Score")
    ax.set_title("CV final vs Holdout")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_cv_modelos(comparativa_cv_familias):
    """Medias F1_0 por familia con desviación entre folds."""
    import matplotlib.pyplot as plt

    modelos = [m for m in FAMILIAS_ESTADISTICA if m in comparativa_cv_familias]
    means = [float(comparativa_cv_familias[m].mean_f1_0) for m in modelos]
    stds = [float(comparativa_cv_familias[m].std_f1_0) for m in modelos]
    fig, ax = plt.subplots()
    ax.bar(modelos, means, yerr=stds, capsize=4)
    ax.set_ylabel("F1_0 medio")
    ax.set_title("F1_0 CV por familia (5 folds)")
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    return fig


def plot_precision_recall(y_true, risk_score):
    """Curva PR descriptiva; figura con mensaje si no hay score."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    if risk_score is None:
        ax.text(0.5, 0.5, "Sin score probabilístico", ha="center", va="center")
        ax.set_title("Precision-Recall (no disponible)")
        return fig
    y_arr = np.asarray(y_true)
    s_arr = np.asarray(risk_score, dtype=float)
    y_risk = (y_arr == 0).astype(int)
    precision, recall, _ = precision_recall_curve(y_risk, s_arr)
    ax.plot(recall, precision)
    ax.set_xlabel("Recall riesgo")
    ax.set_ylabel("Precision riesgo")
    ax.set_title("Curva Precision-Recall (descriptiva)")
    fig.tight_layout()
    return fig


def _imprimir_tukey(est: ResultadoEstadistico) -> None:
    """Presenta la tabla Tukey ya calculada, sin recalcular la prueba."""
    modelos = list(est.modelos)
    estadistico = np.asarray(est.tukey.statistic, dtype=float)
    pvalores = np.asarray(est.tukey.pvalue, dtype=float)
    intervalo = est.tukey.confidence_interval(confidence_level=0.95)
    bajos = np.asarray(intervalo.low, dtype=float)
    altos = np.asarray(intervalo.high, dtype=float)
    print("--- Tukey HSD (F1_0, IC 95%) ---")
    for i in range(len(modelos)):
        for j in range(i + 1, len(modelos)):
            diff = float(estadistico[i, j])
            p = float(pvalores[i, j])
            bajo = float(bajos[i, j])
            alto = float(altos[i, j])
            significativo = "sí" if p < 0.05 else "no"
            print(
                f"{modelos[i]} vs {modelos[j]}: "
                f"diff={diff:.4f} p={p:.4g} "
                f"IC95%=[{bajo:.4f}, {alto:.4f}] "
                f"significativo={significativo}"
            )


def generar_pestana_metricas(
    resultado: ResultadoEvaluacionCompleta,
    ruta_salida=None,
    titulo: str = "CDProd — Pestaña de métricas",
) -> Path:
    """Genera la pestaña de métricas como HTML estático.

    Usa exclusivamente los datos ya calculados en resultado; no recalcula
    métricas ni reejecuta evaluación. Las imágenes provienen de las
    funciones plot_* existentes.
    """
    import base64
    import io

    import matplotlib

    try:
        matplotlib.use("Agg")
    except Exception:
        pass

    raiz = encontrar_raiz_proyecto()
    destino = Path(ruta_salida) if ruta_salida is not None else (raiz / "runtime" / "metrics.html")
    if not destino.is_absolute():
        destino = raiz / destino
    ml = resultado.holdout_modelo_final
    heu = resultado.holdout_heuristico
    comp = resultado.comparacion_cv_final_vs_holdout
    est = resultado.analisis_estadistico_familias
    tim = resultado.inference_timing
    prob = ml.probabilisticas

    def _f(num, nd=4):
        return "n/a" if num is None else f"{float(num):.{nd}f}"

    imagenes = []
    for nombre_fig, fig in (
        ("Matriz de confusión", plot_matriz_confusion(ml)),
        ("CV final vs Holdout", plot_comparacion_holdout(comp)),
    ):
        try:
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png")
            imagenes.append((nombre_fig, base64.b64encode(buffer.getvalue()).decode("ascii")))
        except Exception:
            imagenes.append((nombre_fig, None))
        finally:
            try:
                import matplotlib.pyplot as plt

                plt.close(fig)
            except Exception:
                pass

    filas_tukey = ""
    if est.tukey is not None:
        try:
            modelos = list(est.modelos)
            estadistico = np.asarray(est.tukey.statistic, dtype=float)
            pvalores = np.asarray(est.tukey.pvalue, dtype=float)
            intervalo = est.tukey.confidence_interval(confidence_level=0.95)
            bajos = np.asarray(intervalo.low, dtype=float)
            altos = np.asarray(intervalo.high, dtype=float)
            for i in range(len(modelos)):
                for j in range(i + 1, len(modelos)):
                    p = float(pvalores[i, j])
                    sig = "sí" if p < 0.05 else "no"
                    filas_tukey += (
                        f"<tr><td>{html.escape(modelos[i])} vs "
                        f"{html.escape(modelos[j])}</td>"
                        f"<td>{float(estadistico[i, j]):.4f}</td>"
                        f"<td>{p:.4g}</td>"
                        f"<td>[{float(bajos[i, j]):.4f}, "
                        f"{float(altos[i, j]):.4f}]</td>"
                        f"<td>{sig}</td></tr>"
                    )
        except Exception:
            filas_tukey = ""
    if filas_tukey == "":
        filas_tukey = "<tr><td colspan='5'>Tukey no disponible</td></tr>"

    if prob.roc_auc_risk is not None:
        prob_html = (
            f"<p>ROC-AUC riesgo: {_f(prob.roc_auc_risk)}<br>"
            f"Average Precision riesgo: {_f(prob.average_precision_risk)}</p>"
        )
    else:
        prob_html = f"<p>Probabilísticas no disponibles: {html.escape(str(prob.motivo))}</p>"
    advertencias = "".join(
        f"<li>{html.escape(str(a))}</li>" for a in resultado.advertencias_metodologicas
    )
    imagenes_html = "".join(
        f"<h3>{html.escape(n)}</h3><img src='data:image/png;base64,{b}' alt='{html.escape(n)}'/>"
        if b is not None
        else f"<h3>{html.escape(n)}</h3><p>Imagen no disponible</p>"
        for n, b in imagenes
    )
    comp_filas = "".join(
        f"<tr><td>{n}</td><td>{it.cv_mean:.4f} ± {it.cv_std:.4f}</td>"
        f"<td>{it.holdout:.4f}</td><td>{it.delta:+.4f}</td></tr>"
        for n, it in (
            ("F1_0", comp.f1_0),
            ("Recall_0", comp.recall_0),
            ("Precision_0", comp.precision_0),
            ("BalancedAccuracy", comp.balanced_accuracy),
        )
    )
    html_doc = f"""<!DOCTYPE html>
<html lang="es">
<head><meta charset="utf-8"><title>{html.escape(titulo)}</title></head>
<body>
<h1>{html.escape(titulo)}</h1>
<h2>Modelo final</h2>
<p>Modelo: {html.escape(str(resultado.nombre_modelo_final))}<br>
Best params: {html.escape(str(resultado.hiperparametros_finales))}</p>
<h2>Métricas holdout (riesgo = clase 0)</h2>
<table border="1">
<tr><th>Métrica</th><th>Modelo final</th><th>Heurístico</th></tr>
<tr><td>Precision_0</td><td>{_f(ml.metricas.precision_0)}</td>
<td>{_f(heu.metricas.precision_0)}</td></tr>
<tr><td>Recall_0</td><td>{_f(ml.metricas.recall_0)}</td>
<td>{_f(heu.metricas.recall_0)}</td></tr>
<tr><td>F1_0</td><td>{_f(ml.metricas.f1_0)}</td>
<td>{_f(heu.metricas.f1_0)}</td></tr>
<tr><td>BalancedAccuracy</td><td>{_f(ml.metricas.balanced_accuracy)}</td>
<td>{_f(heu.metricas.balanced_accuracy)}</td></tr>
<tr><td>Accuracy contextual</td><td>{_f(ml.metricas.accuracy_ctx)}</td><td>n/a</td></tr>
</table>
<h2>Matriz de confusión</h2>
<table border="1">
<tr><th></th><th>pred_riesgo(0)</th><th>pred_pago(1)</th></tr>
<tr><td>real_riesgo(0)</td>
<td>{ml.matriz_nombres["riesgo_real_predicho_riesgo"]}</td>
<td>{ml.matriz_nombres["riesgo_real_predicho_pago"]}</td></tr>
<tr><td>real_pago(1)</td>
<td>{ml.matriz_nombres["pago_real_predicho_riesgo"]}</td>
<td>{ml.matriz_nombres["pago_real_predicho_pago"]}</td></tr>
</table>
<h2>CV final vs Holdout</h2>
<table border="1">
<tr><th>Métrica</th><th>CV mean ± std</th><th>Holdout</th><th>Delta</th></tr>
{comp_filas}</table>
<h2>Probabilísticas (riesgo)</h2>
{prob_html}
<h2>Inferencia</h2>
<p>mediana total ms: {tim.median_total_ms:.3f}<br>
rows/s: {tim.rows_per_second:.1f}</p>
<h2>Estadística exploratoria</h2>
<p>ANOVA F={est.anova_statistic:.4f} p={est.anova_pvalue:.4g}<br>
{html.escape(est.anova_advertencia)}<br>
{html.escape(est.tukey_mensaje)}<br>
Friedman chi2={est.friedman_statistic:.4f} p={est.friedman_pvalue:.4g}<br>
{html.escape(est.friedman_advertencia)}</p>
<table border="1">
<tr><th>Par</th><th>diff</th><th>p</th><th>IC95%</th><th>Significativo</th></tr>
{filas_tukey}</table>
<h2>Advertencias metodológicas</h2>
<ul>{advertencias}</ul>
{imagenes_html}
</body>
</html>
"""
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(html_doc, encoding="utf-8")
    return destino


def _imprimir_evaluacion(resultado: ResultadoEvaluacionCompleta) -> None:
    """Imprime el reporte final de terminal sin valores hardcodeados."""
    ml = resultado.holdout_modelo_final
    print("=== Modelo final ===")
    print(resultado.nombre_modelo_final)
    print(f"Best params: {resultado.hiperparametros_finales}")
    print("=== Holdout final ===")
    print(f"F1_0: {ml.metricas.f1_0:.4f}")
    print(f"Recall_0: {ml.metricas.recall_0:.4f}")
    print(f"Precision_0: {ml.metricas.precision_0:.4f}")
    print(f"Balanced Accuracy: {ml.metricas.balanced_accuracy:.4f}")
    print(f"Accuracy contextual: {ml.metricas.accuracy_ctx:.4f}")
    print("=== Matriz de confusión ===")
    print(f"riesgo -> riesgo: {ml.matriz_nombres['riesgo_real_predicho_riesgo']}")
    print(f"riesgo -> pago: {ml.matriz_nombres['riesgo_real_predicho_pago']}")
    print(f"pago -> riesgo: {ml.matriz_nombres['pago_real_predicho_riesgo']}")
    print(f"pago -> pago: {ml.matriz_nombres['pago_real_predicho_pago']}")
    print("=== Baseline heurístico ===")
    heu = resultado.holdout_heuristico
    print(f"F1_0: {heu.metricas.f1_0:.4f}")
    print(f"Recall_0: {heu.metricas.recall_0:.4f}")
    print(f"Precision_0: {heu.metricas.precision_0:.4f}")
    print(f"Balanced Accuracy: {heu.metricas.balanced_accuracy:.4f}")
    print("=== CV final vs Holdout ===")
    comp = resultado.comparacion_cv_final_vs_holdout
    for nombre, item in (
        ("F1_0", comp.f1_0),
        ("Recall_0", comp.recall_0),
        ("Precision_0", comp.precision_0),
        ("BalancedAccuracy", comp.balanced_accuracy),
    ):
        print(f"{nombre} CV mean ± std: {item.cv_mean:.4f} ± {item.cv_std:.4f}")
        print(f"{nombre} holdout: {item.holdout:.4f}")
        print(f"{nombre} delta: {item.delta:.4f}")
    print("=== Probabilísticas (riesgo) ===")
    prob = ml.probabilisticas
    if prob.roc_auc_risk is not None:
        print(f"ROC-AUC riesgo: {prob.roc_auc_risk:.4f}")
        print(f"Average Precision riesgo: {prob.average_precision_risk:.4f}")
    else:
        print(f"Probabilísticas no disponibles: {prob.motivo}")
    print("=== Inferencia ===")
    tim = resultado.inference_timing
    print(f"mediana total ms: {tim.median_total_ms:.3f}")
    print(f"media total ms: {tim.mean_total_ms:.3f} ± {tim.std_total_ms:.3f}")
    print(f"ms/row: {tim.median_ms_per_row:.4f}")
    print(f"rows/s: {tim.rows_per_second:.1f}")
    print("=== Estadística exploratoria ===")
    est = resultado.analisis_estadistico_familias
    print(f"ANOVA F={est.anova_statistic:.4f} p={est.anova_pvalue:.4g}")
    print(est.anova_advertencia)
    if est.tukey is not None:
        print("Tukey ejecutado (ver objeto ResultadoEstadistico.tukey).")
        _imprimir_tukey(est)
    print(est.tukey_mensaje)
    print(f"Friedman chi2={est.friedman_statistic:.4f} p={est.friedman_pvalue:.4g}")
    print(est.friedman_advertencia)
    print("=== Advertencias metodológicas ===")
    for adv in resultado.advertencias_metodologicas:
        print(f"- {adv}")


def main() -> None:
    """Carga datos, entrena solo con train y evalúa una vez sobre X_eval/y_eval."""
    raiz = encontrar_raiz_proyecto()
    config = cargar_config(raiz / "mlops_pipeline" / "src" / "config.json")
    df = cargar_datos(config)
    datos = preparar_datos(df)
    resultado_entrenamiento = entrenar(datos.X_train, datos.y_train, datos.preprocessor)
    resultado = evaluar(resultado_entrenamiento, datos.X_eval, datos.y_eval)
    _imprimir_evaluacion(resultado)


if __name__ == "__main__":
    main()
