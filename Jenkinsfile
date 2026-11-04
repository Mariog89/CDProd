// Pipeline de aprobación de CDProd (riesgo crediticio).
//
// Clonar repo -> entorno Python 3.13 -> estructura -> calidad (Ruff)
// -> pruebas unitarias -> integración BigQuery -> quality gate de métricas
// -> imagen Docker de la API -> aprobación manual -> publicar versión aprobada.
//
// Trigger: Poll SCM cada ~2 minutos sobre main/develop (un merge dispara el flujo).
// Notificación: correo HTML al terminar, con estado, duración y métricas.

def resumenMetricas = 'Sin métricas (el quality gate no llegó a ejecutarse).'

pipeline {
    agent any

    options {
        skipDefaultCheckout(true)
        disableConcurrentBuilds()
        timestamps()
        buildDiscarder(logRotator(numToKeepStr: '20'))
        timeout(time: 90, unit: 'MINUTES')
    }

    triggers {
        pollSCM('H/2 * * * *')
    }

    environment {
        VENV = '.venv-ci'
        PY = '.venv-ci/bin/python'
        MPLBACKEND = 'Agg'
        PYTHONUNBUFFERED = '1'
        IMAGE = 'cdprod-api'
    }

    stages {
        stage('Clonar repositorio') {
            steps {
                cleanWs()
                script {
                    def scmVars = checkout scm
                    env.GIT_BRANCH = scmVars.GIT_BRANCH
                    env.GIT_COMMIT = scmVars.GIT_COMMIT
                }
                sh 'git log -1 --pretty=format:"Commit %h (%an): %s%n"'
            }
        }

        stage('Preparar entorno Python') {
            steps {
                sh '''
                    uv venv --python 3.13 "$VENV"
                    uv pip install --python "$PY" -r requirements.txt -r requirements-dev.txt
                    uv pip check --python "$PY"
                '''
            }
        }

        stage('Validar estructura de carpetas') {
            steps {
                sh '"$PY" scripts/validate_structure.py'
            }
        }

        stage('Calidad de código') {
            parallel {
                stage('Ruff lint') {
                    steps { sh '"$PY" -m ruff check .' }
                }
                stage('Ruff format') {
                    steps { sh '"$PY" -m ruff format --check .' }
                }
            }
        }

        stage('Pruebas unitarias') {
            steps {
                sh '''
                    mkdir -p reports
                    "$PY" -m pytest -q \
                        --junitxml=reports/junit-unit.xml \
                        --cov=mlops_pipeline/src --cov=app --cov=scripts \
                        --cov-report=term --cov-report=xml:reports/coverage.xml
                '''
            }
            post {
                always { junit allowEmptyResults: true, testResults: 'reports/junit-unit.xml' }
            }
        }

        stage('Integración BigQuery') {
            environment { CDPROD_BQ_LIVE = '1' }
            steps {
                sh '''
                    "$PY" -m pytest -q -m "bq_live or data_regression" \
                        --junitxml=reports/junit-bigquery.xml
                '''
            }
            post {
                always { junit allowEmptyResults: true, testResults: 'reports/junit-bigquery.xml' }
            }
        }

        stage('Validar métricas de evaluación') {
            steps {
                sh '"$PY" scripts/validate_metrics.py --output reports/metrics.json --html reports/metrics.html'
            }
            post {
                always {
                    script {
                        if (fileExists('reports/metrics.json')) {
                            def r = readJSON file: 'reports/metrics.json'
                            def m = r.metricas
                            def f = { v -> v == null ? 'n/d' : String.format('%.4f', v as double) }
                            resumenMetricas = """
                                <b>Modelo:</b> ${m.modelo_final} ${m.hiperparametros}<br>
                                <b>Quality gate:</b> ${r.aprobado ? 'APROBADO' : 'RECHAZADO'}<br>
                                F1_0=${f(m.f1_0)}, Recall_0=${f(m.recall_0)}, Precision_0=${f(m.precision_0)},
                                Balanced=${f(m.balanced_accuracy)}, ROC-AUC=${f(m.roc_auc_risk)}<br>
                                ${r.violaciones ? 'Violaciones: ' + r.violaciones.join('; ') : ''}
                            """.stripIndent()
                        }
                    }
                    archiveArtifacts artifacts: 'reports/metrics.*', allowEmptyArchive: true
                    publishHTML(target: [
                        reportName : 'Métricas del modelo',
                        reportDir  : 'reports',
                        reportFiles: 'metrics.html',
                        keepAll    : true,
                        allowMissing: true,
                        alwaysLinkToLastBuild: true,
                    ])
                }
            }
        }

        stage('Construir imagen Docker') {
            steps {
                sh 'docker build -t "$IMAGE:$BUILD_NUMBER" .'
            }
        }

        stage('Aprobación') {
            steps {
                timeout(time: 30, unit: 'MINUTES') {
                    input message: "¿Aprobar ${env.IMAGE}:${env.BUILD_NUMBER} (quality gate superado)?",
                          ok: 'Aprobar versión'
                }
            }
        }

        stage('Publicar versión aprobada') {
            steps {
                sh 'docker tag "$IMAGE:$BUILD_NUMBER" "$IMAGE:aprobado"'
                echo "Imagen ${env.IMAGE}:${env.BUILD_NUMBER} etiquetada como ${env.IMAGE}:aprobado"
            }
        }
    }

    post {
        always {
            script {
                def estado = currentBuild.currentResult
                def destinatario = env.NOTIFY_EMAIL ?: ''
                def cuerpo = """
                    <h3>CDProd · Pipeline de aprobación: ${estado}</h3>
                    <p><b>Job:</b> ${env.JOB_NAME} #${env.BUILD_NUMBER}<br>
                    <b>Rama:</b> ${env.GIT_BRANCH ?: 'n/d'}<br>
                    <b>Duración:</b> ${currentBuild.durationString.replace(' and counting', '')}<br>
                    <b>Detalle:</b> <a href="${env.BUILD_URL}">${env.BUILD_URL}</a></p>
                    <p>${resumenMetricas}</p>
                """.stripIndent()
                if (destinatario) {
                    try {
                        mail to: destinatario,
                             subject: "[CDProd] ${estado}: ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                             body: cuerpo,
                             mimeType: 'text/html',
                             charset: 'UTF-8'
                    } catch (err) {
                        echo "No se pudo enviar la notificación por correo: ${err}"
                    }
                } else {
                    echo 'NOTIFY_EMAIL no configurado: se omite el correo.'
                }
                echo "Notificación (${estado}): ${resumenMetricas.replaceAll('<[^>]+>', ' ').trim()}"
            }
        }
    }
}
