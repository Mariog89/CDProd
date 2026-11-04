# Fase 3 – Investigación comparativa AWS vs Azure vs GCP (MLOps tabular)

**Caso de uso:** clasificación tabular de riesgo crediticio (~10 000 filas, 15 variables).
**Fecha de consulta de fuentes:** 2026-11-04.
**Regiones de referencia:** AWS `us-east-1` (N. Virginia), Azure `East US`, GCP `us-central1` (Iowa).
**Moneda:** USD, precio de lista on-demand (sin descuentos, sin impuestos).

> **Método de obtención de precios.** Las páginas públicas de precios de AWS y Azure cargan las cifras con JavaScript, y una lectura automática no las muestra (Azure muestra "$-", AWS deja las celdas vacías). Por eso los precios de AWS se tomaron del **AWS Price List Bulk API** (CSV oficial `pricing.us-east-1.amazonaws.com`, publicado el 2026-11-04T01:42Z) y los de Azure del **Azure Retail Prices API** (`prices.azure.com`), que son los endpoints oficiales de cada proveedor. Los precios de GCP están en el HTML estático de `cloud.google.com`. En las tablas se indica el origen de cada cifra.

> **Cambios de nombre importantes (2024–2026).**
> - **GCP:** el 22-abr-2026 Vertex AI pasó a formar parte de **Gemini Enterprise Agent Platform**. Equivalencias oficiales: Vertex AI Pipelines → *Agent Platform Pipelines*; Vertex AI AutoML (Vision/Video/Tables) → *Agent Platform AutoML*; Model Registry → *Agent Platform Model Registry*; Model Monitoring → *Model Monitoring on Agent Platform*; Experiments → *Experiments on Agent Platform*; Vertex AI Prediction → *Agent Platform Inference*. La página de precios se titula ahora "Gemini platform pricing". En este documento se mantiene el nombre "Vertex AI" porque es el que usan el proyecto y el código. Fuente: https://docs.cloud.google.com/gemini-enterprise-agent-platform/release-notes
> - **AWS:** la parte de ML de SageMaker se denomina **Amazon SageMaker AI** (las URL oficiales están bajo `/sagemaker/ai/`). La interfaz de **Autopilot migró a SageMaker Canvas** desde el 30-nov-2023. La API de AutoML (`CreateAutoMLJob` / `CreateAutoMLJobV2`) sigue documentada. Fuente: https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-automate-model-development.html
> - **Azure:** el **SDK v1** de Azure ML quedó obsoleto el 31-mar-2025 y **dejó de tener soporte el 30-jun-2026**. Los componentes *classic prebuilt (v1)* de Designer siguen soportados, pero ya no se añaden nuevos y no admiten managed online endpoints (v2). Fuentes: https://learn.microsoft.com/en-us/azure/machine-learning/concept-automated-ml?view=azureml-api-1 , https://learn.microsoft.com/en-us/azure/machine-learning/concept-designer?view=azureml-api-2

---

## 1. Mapeo de servicios equivalentes

| Capacidad | AWS | Azure | GCP |
|---|---|---|---|
| AutoML tabular | SageMaker **Canvas** (no-code; ejecuta Autopilot por debajo). API AutoML (Autopilot) con SDK/CLI | Azure ML **automated ML** (studio, CLI v2, SDK v2 `azure-ai-ml`) | **Vertex AI AutoML Tabular** (*Agent Platform AutoML*). También **Tabular Workflow End-to-End AutoML** (GA) |
| Pipelines / orquestación | **SageMaker Pipelines** (serverless) | **Azure ML pipelines** (SDK/CLI v2 + Designer v2 con componentes custom) | **Vertex AI Pipelines** (KFP v2) |
| Almacenamiento de objetos | **S3 Standard** | **Blob Storage Hot LRS** | **Cloud Storage Standard** (regional) |
| Data warehouse (opcional) | **Redshift Serverless** | **Synapse serverless SQL** / Microsoft Fabric | **BigQuery** |
| Entrenamiento custom | SageMaker Training jobs | Azure ML compute clusters / serverless compute | Vertex AI custom training (*Managed Training*) |
| Endpoint online | SageMaker real-time endpoint (también serverless y async) | **Managed online endpoint** | Vertex AI **Endpoint** (*Agent Platform Online Inference*) |
| Predicción batch | **Batch Transform** | **Batch endpoint** | Vertex AI **Batch prediction** |
| Registro de modelos | SageMaker **Model Registry** | Azure ML **model registry** (por workspace) + **registries** (entre workspaces) | Vertex AI **Model Registry** |
| Monitorización | SageMaker **Model Monitor** | Azure ML **model monitoring** (data drift, prediction drift, data quality, feature attribution drift en preview) | Vertex AI **Model Monitoring** (v2: solo modelos tabulares) |
| Explicabilidad / IA responsable | SageMaker **Clarify** | **Responsible AI dashboard** | Vertex **Explainable AI** (Shapley) |
| Experimentos / metadatos | SageMaker Experiments / MLflow gestionado | MLflow integrado en el workspace | Vertex **Experiments** + **ML Metadata** + TensorBoard |
| CI/CD nativo | CodePipeline + CodeBuild | Azure DevOps (Azure Pipelines) | Cloud Build |
| GitHub Actions | Sí (CLI/SDK con OIDC) | Sí (documentado por Microsoft para Azure ML) | Sí (CLI/SDK con Workload Identity Federation) |
| Jenkins | Acción nativa de Jenkins en CodePipeline (plugin "CodePipeline Plugin for Jenkins") | Los plugins de Jenkins para Azure se retiraron el 29-feb-2024. Microsoft recomienda usar Azure CLI desde Jenkins | Sin integración propia verificada. Se usa con `gcloud` y el SDK, autenticando con una cuenta de servicio (es como lo usa este proyecto) |
| Identidad y acceso | IAM (roles, políticas) + IAM Identity Center | Microsoft Entra ID + Azure RBAC (p. ej. *AzureML Data Scientist*, *AzureML Compute Operator*) | Cloud IAM (p. ej. `roles/aiplatform.user`, `roles/aiplatform.admin`) |
| Auditoría | CloudTrail | Activity Log (Azure Monitor) | Cloud Audit Logs |
| Políticas / compliance | AWS Organizations SCPs, AWS Config | Azure Policy, management groups | Organization Policy, folders, VPC Service Controls |
| Unidad de trabajo multi-rol | SageMaker AI **domain** + user profiles / spaces | Azure ML **workspace** (+ hub workspaces, registries) | **Project** (+ folders / organization) |

---

## 2. Tabla de precios (lista on-demand, USD)

### 2.1 Cómputo de referencia (CPU de 4 vCPU y GPU T4)

| Proveedor / servicio | Instancia | Precio | Unidad | Región | Fuente |
|---|---|---|---|---|---|
| AWS SageMaker – Training | ml.m5.xlarge (4 vCPU, 16 GiB) | **$0.23** | instancia-hora | us-east-1 | Price List CSV: https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonSageMaker/current/us-east-1/index.csv |
| AWS SageMaker – Real-time endpoint (Hosting) | ml.m5.xlarge | **$0.23** | instancia-hora | us-east-1 | ídem |
| AWS SageMaker – Batch Transform | ml.m5.xlarge | **$0.23** | instancia-hora | us-east-1 | ídem |
| AWS SageMaker – Processing / Model Monitor / Clarify | ml.m5.xlarge | **$0.23** | instancia-hora | us-east-1 | ídem ("Processing, Debugger and Model Monitor ml.m5.xlarge") |
| AWS SageMaker – Studio JupyterLab | ml.t3.medium | **$0.05** | instancia-hora | us-east-1 | ídem |
| AWS SageMaker – GPU (training/endpoint/batch) | ml.g4dn.xlarge (1× T4) | **$0.736** | instancia-hora | us-east-1 | ídem |
| Azure VM (compute de Azure ML) | Standard_DS3_v2 (4 vCPU, 14 GiB), Linux | **$0.293** | VM-hora | East US | Retail Prices API: `https://prices.azure.com/api/retail/prices?$filter=armRegionName eq 'eastus' and armSkuName eq 'Standard_DS3_v2'` |
| Azure VM (alternativa actual) | Standard_D4s_v5 (4 vCPU, 16 GiB), Linux | **$0.192** | VM-hora | East US | Retail Prices API (`armSkuName eq 'Standard_D4s_v5'`) |
| Azure VM (alternativa) | Standard_D4s_v3, Linux | **$0.192** | VM-hora | East US | Retail Prices API (`armSkuName eq 'Standard_D4s_v3'`) |
| Azure VM GPU | Standard_NC4as_T4_v3 (1× T4), Linux | **$0.526** | VM-hora | East US | Retail Prices API (`armSkuName eq 'Standard_NC4as_T4_v3'`) |
| Azure ML – recargo del servicio | — | **$0** ("Standard vCPU Surcharge") | hora | East US | Retail Prices API (`serviceName eq 'Azure Machine Learning'`). Página: https://azure.microsoft.com/en-us/pricing/details/machine-learning/ ("no additional charge to use Azure Machine Learning") |
| GCP Vertex – Custom training | e2-standard-4 | **$0.154126276** | hora | us-central1 | https://cloud.google.com/vertex-ai/pricing |
| GCP Vertex – Custom training | n1-standard-4 | **$0.21849885** | hora | us-central1 | ídem |
| GCP Vertex – Online/batch prediction | n1-standard-4 | **$0.219** | node-hour | us-central1 | ídem (sección "Prediction and explanation") |
| GCP Vertex – Online/batch prediction | e2-standard-4 | **$0.1541128** | node-hour | us-central1 | ídem |
| GCP Vertex – GPU (training) | NVIDIA_TESLA_T4 (solo el acelerador, se suma a la VM) | **$0.4025** | hora | us-central1 | ídem |

Notas:
- En Azure ML se paga solo la VM subyacente (recargo $0). Los managed online endpoints y los batch endpoints se facturan por VM-hora.
- La serie DSv2 (incluida Standard_DS3_v2) tiene **retiro anunciado para el 1-may-2028**. Microsoft recomienda Dsv5/Dsv6. Fuente: https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/lifecycle/retirements-and-capacity-restrictions
- En Vertex, la página indica que la facturación de training y prediction va **en incrementos de 30 s, sin mínimo**. Un endpoint con modelo desplegado factura aunque no reciba tráfico, y no admite *scale-to-zero*.

### 2.2 AutoML tabular

| Servicio | Precio | Unidad | Free tier | Fuente |
|---|---|---|---|---|
| **GCP Vertex AutoML Tabular** (clasificación/regresión) – training | **$21.252** | node-hour | No hay free tier específico (solo los créditos de prueba) | https://cloud.google.com/vertex-ai/pricing (sección "Tabular data") |
| GCP Vertex AutoML Tabular – inferencia | "Same price as inference for custom-trained models" (p. ej. n1-standard-4 = $0.219/node-hour) | node-hour | — | ídem |
| **AWS SageMaker Canvas** – sesión de workspace | **$1.90** | session-hour (mientras el usuario tiene sesión abierta) | 160 h/mes de sesión durante 2 meses | Price List CSV ("$1.9 per Hrs for Canvas:Workspace Instance (Session-Hrs)") + https://aws.amazon.com/sagemaker/ai/canvas/pricing/ |
| AWS Canvas – entrenamiento (standard build) | Estimación oficial **$2.3–$9.2** por modelo para datasets < 100 MB (se factura por instance-hours de training) | instancia-hora (estimación por tamaño) | — | https://aws.amazon.com/sagemaker/ai/canvas/pricing/ |
| AWS Autopilot (API) | Sin tarifa propia: "pay only for the underlying compute and storage" (p. ej. ml.m5.xlarge $0.23/h) | instancia-hora | Free tier de SageMaker (training: 50 h m5.xlarge/mes, 2 meses) | https://aws.amazon.com/sagemaker/ai/pricing/ |
| **Azure automated ML** | Sin recargo; se paga el compute cluster (p. ej. DS3_v2 $0.293/h, D4s_v5 $0.192/h) | VM-hora | Créditos de cuenta gratuita | https://azure.microsoft.com/en-us/pricing/details/machine-learning/ + Retail Prices API |

### 2.3 Pipelines, almacenamiento, data warehouse y CI/CD

| Servicio | Precio | Unidad | Free tier | Fuente |
|---|---|---|---|---|
| GCP Vertex Pipelines | **$0.03** por ejecución + cómputo de cada componente a tarifa de training | pipeline run | — | https://cloud.google.com/vertex-ai/pricing |
| AWS SageMaker Pipelines | Sin tarifa propia publicada. Se paga el cómputo de cada paso (Processing/Training) | instancia-hora de cada paso | — | https://aws.amazon.com/sagemaker/ai/pricing/ (no lista Pipelines por separado) – **ver salvedades** |
| Azure ML pipelines | Sin recargo. Se paga el compute | VM-hora | — | https://azure.microsoft.com/en-us/pricing/details/machine-learning/ |
| **S3 Standard** | **$0.023** (primeros 50 TB); PUT/LIST $0.005 por 1 000; GET $0.0004 por 1 000 | GB-mes | (Créditos de AWS Free Tier) | Price List CSV: https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonS3/current/us-east-1/index.csv |
| **Azure Blob Hot LRS** | **$0.0208** (primeros 50 TB); escrituras $0.05 por 10 000; lecturas $0.004 por 10 000 | GB-mes | (Créditos de cuenta gratuita) | Retail Prices API (`productName eq 'Blob Storage' and skuName eq 'Hot LRS'`, eastus) |
| **Cloud Storage Standard** | **$0.000027397/GiB-hora ≈ $0.020/GiB-mes**; clase A $0.005 por 1 000, clase B $0.0004 por 1 000 | GiB-mes | Always Free: 5 GB-mes Standard en us-central1/us-east1/us-west1, 5 000 ops clase A, 50 000 clase B | https://cloud.google.com/storage/pricing |
| Redshift Serverless | **$0.375** por RPU-hora; almacenamiento gestionado $0.024/GB-mes | RPU-hora | — | Price List CSV: https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonRedshift/current/us-east-1/index.csv |
| Azure Synapse serverless SQL | **$5** por TB procesado | TB | — | Retail Prices API (`serviceName eq 'Azure Synapse Analytics'`, eastus) |
| Microsoft Fabric (capacidad) | $0.18 por CU-hora | CU-hora | — | Retail Prices API (`serviceName eq 'Microsoft Fabric'`, eastus) |
| **BigQuery** on-demand | **$6.25** por TiB consultado; almacenamiento activo lógico $0.000031507/GiB-h (≈ $0.023/GiB-mes) | TiB / GiB-mes | 1 TiB de consultas y 10 GiB de almacenamiento gratis al mes | https://cloud.google.com/bigquery/pricing |
| AWS CodePipeline (V2) | **$0.002** por minuto de ejecución de acción; V1: $1 por pipeline activo/mes | minuto | 100 min/mes; primer pipeline V1 gratis | Price List CSV: https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AWSCodePipeline/current/us-east-1/index.csv |
| Azure Pipelines (Azure DevOps) | **$40**/mes por job paralelo Microsoft-hosted adicional; **$15**/mes self-hosted | job paralelo-mes | 1 job Microsoft-hosted con 1 800 min/mes; primeros 5 usuarios Basic gratis | https://azure.microsoft.com/en-us/pricing/details/devops/azure-devops-services/ |
| Cloud Build | **$0.006** por build-minute (e2-standard-2) | minuto | 2 500 build-min/mes por billing account | https://cloud.google.com/build/pricing |

### 2.4 Monitorización, registro y metadatos

| Servicio | Precio | Unidad | Fuente |
|---|---|---|---|
| SageMaker Model Monitor / Clarify | Se ejecutan como jobs de Processing (ml.m5.xlarge $0.23/h). Model Monitor: "up to 30 hours of monitoring at no charge" en reglas built-in. Clarify con evaluación humana: $0.21 por tarea | instancia-hora | https://aws.amazon.com/sagemaker/ai/pricing/ + Price List CSV |
| SageMaker Model Registry | Sin línea de precio en la página oficial (**no confirmado**) | — | https://aws.amazon.com/sagemaker/ai/pricing/ |
| Azure ML model monitoring | Se ejecuta en serverless Spark compute (tarifa **no confirmada** en este estudio) | vCore-hora | https://learn.microsoft.com/en-us/azure/machine-learning/how-to-monitor-model-performance?view=azureml-api-2 |
| Azure ML registry / Responsible AI dashboard | Sin recargo propio (solo almacenamiento y compute usados) | — | https://azure.microsoft.com/en-us/pricing/details/machine-learning/ |
| Vertex Model Registry | **Gratis** ("There is no cost associated with having your models in the Model Registry") | — | https://cloud.google.com/vertex-ai/pricing |
| Vertex Model Monitoring | **$3.50 por GB** analizado (training + predicciones). Solo en us-central1, europe-west4, asia-east1 y asia-southeast1 | GB | ídem |
| Vertex ML Metadata (Experiments) | **$10 por GiB-mes** | GiB-mes | ídem |
| Vertex TensorBoard | $10 por GiB-mes de almacenamiento | GiB-mes | ídem |

### 2.5 Free tier / créditos para cuentas nuevas

| Proveedor | Oferta | Fuente |
|---|---|---|
| AWS | $100 en créditos al registrarse y hasta $100 más por explorar servicios (máx. $200). El *Free plan* dura 6 meses o hasta agotar créditos. SageMaker AI está disponible en ambos planes. Free tier propio de SageMaker durante los 2 primeros meses: 250 h ml.t3.medium (notebooks), 50 h m4/m5.xlarge (training), 125 h m4/m5.xlarge (real-time inference), 150 000 s de serverless inference, 160 h/mes de Canvas | https://aws.amazon.com/free/ , https://aws.amazon.com/sagemaker/ai/pricing/ |
| Azure | $200 de crédito durante 30 días, más servicios gratuitos 12 meses y "65+ always-free services". Azure ML no aparece como servicio gratuito, pero no tiene recargo | https://azure.microsoft.com/en-us/pricing/details/machine-learning/ , https://azure.microsoft.com/en-us/pricing/free-services |
| GCP | $300 de crédito válido 90 días. Durante la prueba no se pueden **añadir GPUs** ni **pedir aumentos de cuota**. Always Free: 5 GB-mes de GCS, 1 TiB de consultas BigQuery, 2 500 min de Cloud Build | https://cloud.google.com/free/docs/free-cloud-features |

### 2.6 Estimación ilustrativa para nuestro caso (a partir de los precios de lista anteriores)

| Concepto | AWS | Azure | GCP |
|---|---|---|---|
| Endpoint online 24×7 (730 h), 1 nodo de 4 vCPU | ml.m5.xlarge: 0.23×730 = **$167.90/mes** | DS3_v2: 0.293×730 = **$213.89/mes** (D4s_v5: **$140.16/mes**) | n1-standard-4: 0.219×730 = **$159.87/mes** (e2-standard-4: **$112.50/mes**) |
| AutoML con ~1 h de entrenamiento | Canvas: $1.90/h de sesión + estimación oficial de $2.3–9.2 por build (< 100 MB) | Compute cluster: ~$0.19–0.29 por nodo-hora (las horas dependen de iteraciones y concurrencia) | **$21.252** por node-hour de presupuesto |
| Almacenamiento del dataset (< 1 GB) | ≈ $0.023/mes | ≈ $0.021/mes | ≈ $0.020/mes (dentro del Always Free) |

**Datos medidos en GCP (referencia del proyecto):**
- **Vertex AutoML Tabular, presupuesto de 1 node-hour:** 2 h 41 min de extremo a extremo (de reloj). El coste esperado de training es de ≈ **$21.25** (1 node-hour × $21.252). El tiempo de reloj incluye aprovisionamiento, preprocesado y evaluación, así que no equivale a las horas facturadas. Conviene confirmarlo en el informe de Billing (ver salvedades).
- **Vertex Pipelines, 5 tareas paralelas en e2-standard-4, ~11 min:** cota superior ≈ 5 × $0.154126 × (11/60) h + $0.03 por ejecución ≈ **$0.17** por run, suponiendo que las 5 tareas duran los 11 min completos.
- **Batch prediction en n1-standard-4:** $0.219 por node-hour, en incrementos de 30 s.

---

## 3. Gobierno, seguridad y CI/CD

| Aspecto | AWS | Azure | GCP |
|---|---|---|---|
| Identidad | IAM (usuarios, roles, políticas) + IAM Identity Center. El *quick setup* de SageMaker crea un rol de ejecución con `AmazonSageMakerFullAccess` (+ políticas de Canvas y `AmazonS3FullAccess`) | Microsoft Entra ID + Azure RBAC. Roles integrados de ML: **AzureML Data Scientist** (todo dentro del workspace salvo crear o borrar compute y modificar el workspace) y **AzureML Compute Operator**. Permite asignar roles a grupos de Entra | Cloud IAM: `roles/aiplatform.user` (suficiente para la mayoría), `roles/aiplatform.admin`. Workbench: `roles/notebooks.admin` + `roles/iam.serviceAccountUser`. Para habilitar APIs: `roles/serviceusage.serviceUsageAdmin` |
| Auditoría | **CloudTrail**: 90 días de *event history* gratis y la primera copia de management events a S3 gratis. Copias adicionales: $2.00 por 100 000 management events; data events: $0.10 por 100 000 | **Activity Log**: activado por defecto y retenido **90 días sin coste**. Para más tiempo se exporta con diagnostic settings a Log Analytics (hasta 12 años), Storage o Event Hubs. La ingesta de activity logs no tiene coste | **Cloud Audit Logs**: Admin Activity y System Event siempre activos (no se pueden desactivar) y se guardan en el bucket `_Required` (**400 días, sin coste**). Data Access desactivados por defecto (excepto BigQuery) y con coste de Logging si se activan |
| Políticas / compliance | AWS Organizations (SCPs), AWS Config | Azure Policy, management groups | Organization Policy, jerarquía organization → folders → projects |
| Aislamiento multi-equipo | SageMaker AI **domain** (user profiles, spaces, VPC, EFS opcional). Quick setup para un usuario o *Set up for organizations* | **Workspace** por proyecto (Microsoft recomienda "a workspace per project" para imputar costes). **Hub workspaces** para gobierno centralizado (mismo tipo de recurso que los hubs de Microsoft Foundry). **Registries** para compartir modelos, componentes y entornos entre workspaces | **Project** como unidad de facturación, IAM y cuotas; folders y organization para heredar políticas |
| Registro de modelos / versionado | SageMaker Model Registry (grupos de paquetes de modelo, estado de aprobación) | Model registry del workspace + registries cross-workspace (MLflow) | Vertex Model Registry (versiones, alias, gratis) |
| Monitorización | Model Monitor (data/model quality, bias y feature attribution drift con Clarify) | Model monitoring: data drift, prediction drift, data quality incluidos por defecto; feature attribution drift en **preview**; ejecuta en serverless Spark | Model Monitoring v2 (solo tabular): drift de features de entrada (L-infinity para categóricas, Jensen-Shannon para numéricas), drift de salida y atribuciones (SHAP). SDK en `vertexai.resources.preview.ml_monitoring` (preview) |
| Lineage / experimentos | SageMaker Experiments / MLflow gestionado | MLflow nativo, lineage de jobs ("Show lineage") | Vertex ML Metadata + Experiments ($10/GiB-mes) |
| CI/CD nativo | CodePipeline (+ CodeBuild). Integración nativa con Jenkins como acción de build/test (requiere "CodePipeline Plugin for Jenkins" en cada instancia de Jenkins) | Azure Pipelines (Azure DevOps). Plugins de Jenkins para Azure **retirados el 29-feb-2024**; la recomendación es invocar Azure CLI desde Jenkins | Cloud Build (2 500 min gratis/mes). Desde Jenkins o GitHub Actions se usa `gcloud` o el SDK con una cuenta de servicio o Workload Identity Federation |
| GitHub Actions | Compatible (acciones oficiales de credenciales AWS, OIDC) | Compatible (documentado para Azure ML) | Compatible (acciones oficiales `google-github-actions`) |

Fuentes de la tabla 3:
- https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-quick-start.html
- https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-onboard.html
- https://aws.amazon.com/cloudtrail/pricing/
- https://docs.aws.amazon.com/codepipeline/latest/userguide/tutorials-four-stage-pipeline.html
- https://docs.aws.amazon.com/codepipeline/latest/userguide/actions-valid-providers.html
- https://learn.microsoft.com/en-us/azure/machine-learning/concept-workspace?view=azureml-api-2
- https://learn.microsoft.com/en-us/azure/machine-learning/how-to-assign-roles?view=azureml-api-2
- https://learn.microsoft.com/en-us/azure/role-based-access-control/built-in-roles/ai-machine-learning
- https://learn.microsoft.com/en-us/azure/azure-monitor/platform/activity-log
- https://learn.microsoft.com/en-us/azure/machine-learning/how-to-monitor-model-performance?view=azureml-api-2
- https://azure.microsoft.com/updates/jenkins-plugins-for-azure-are-being-retired-on-29-february-2024
- https://docs.cloud.google.com/logging/docs/audit
- https://cloud.google.com/stackdriver/pricing
- https://docs.cloud.google.com/vertex-ai/docs/start/cloud-environment
- https://docs.cloud.google.com/vertex-ai/docs/model-monitoring/set-up-model-monitoring

---

## 4. Notas cualitativas (con fuente)

### Facilidad de configuración inicial
- **AWS:** hace falta crear un **SageMaker AI domain** para usar Studio y Canvas. El *quick setup* lo crea automáticamente con un rol IAM, un bucket S3 `sagemaker-studio-*` y la VPC por defecto. Para equipos está *Set up for organizations*, que pide configurar IAM Identity Center, VPC y EFS. Las cuotas por tipo de instancia se consultan y amplían en Service Quotas, y una organización puede usar plantillas que piden aumentos automáticamente al crear cuentas. Fuentes: https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-quick-start.html , https://docs.aws.amazon.com/sagemaker/latest/dg/regions-quotas.html
- **Azure:** se crea un **workspace** que depende de una Storage account, Key Vault, Application Insights y, cuando se construyen imágenes, Azure Container Registry. Si no se aportan, se crean automáticamente. Puede crearse desde studio, portal, CLI v2, SDK v2, Bicep o Terraform. Un workspace no se puede mover a otra suscripción. Fuente: https://learn.microsoft.com/en-us/azure/machine-learning/concept-workspace?view=azureml-api-2
- **GCP:** basta un **project** con facturación y habilitar las APIs de Agent Platform/Vertex (requiere `serviceusage.serviceUsageAdmin`), asignar `roles/aiplatform.user` e instalar `gcloud`. No hay un recurso "workspace" equivalente que crear. En la prueba gratuita no se pueden pedir aumentos de cuota ni usar GPUs, lo que puede bloquear cargas mayores. Fuentes: https://docs.cloud.google.com/vertex-ai/docs/start/cloud-environment , https://cloud.google.com/free/docs/free-cloud-features

### Consola / GUI
- **AWS:** SageMaker Studio (JupyterLab, Code Editor) y Canvas como no-code. La UI de Autopilot está ahora en Canvas, y Studio Classic queda oculto por defecto en los dominios nuevos. Fuente: https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-quick-start.html
- **Azure:** Azure ML studio (`ml.azure.com`) con automated ML, Designer v2 (drag-and-drop con componentes custom) y vistas de lineage. La gestión de accesos se hace en el portal de Azure, no en studio. Fuente: https://learn.microsoft.com/en-us/azure/machine-learning/concept-designer?view=azureml-api-2
- **GCP:** consola de Google Cloud con secciones de datasets, training, pipelines (grafo de KFP), Model Registry y endpoints. Desde abr-2026 la documentación y la consola usan el nombre *Agent Platform*. Fuente: https://docs.cloud.google.com/gemini-enterprise-agent-platform/release-notes

### Curva de aprendizaje
Es una valoración basada en el número de recursos previos y conceptos que exige cada plataforma, según las fuentes anteriores:
- **AWS** exige entender domains, user profiles, roles de ejecución IAM y VPC antes de entrenar.
- **Azure** exige entender workspaces, sus recursos asociados, compute targets y la diferencia v1/v2, que es relevante porque gran parte de la documentación v1 sigue publicada como archivada.
- **GCP** tiene menos piezas previas (project + API + IAM), pero el cambio de marca de 2026 mezcla en la documentación los nombres Vertex y Agent Platform.

### Madurez de los SDK
- **AWS:** SageMaker Python SDK **v3** es modular (`sagemaker-core`, `sagemaker-train`, `sagemaker-serve`, `sagemaker-mlops`) y tiene cambios incompatibles con v2: `Estimator`/`Model`/`Predictor` se sustituyen por `ModelTrainer`/`ModelBuilder`. Última versión vista: v3.20.0 (ago-2026). v2 emite avisos de migración. Fuentes: https://sagemaker.readthedocs.io/ , https://github.com/aws/sagemaker-python-sdk/releases
- **Azure:** SDK v2 `azure-ai-ml` y CLI `ml` v2 son los actuales. El SDK v1 está sin soporte desde el 30-jun-2026 (los flujos existentes siguen funcionando, pero con riesgo de seguridad o de cambios incompatibles). Fuente: https://learn.microsoft.com/en-us/azure/machine-learning/concept-automated-ml?view=azureml-api-1
- **GCP:** `google-cloud-aiplatform` (módulo `vertexai`, versión 1.153.x en la referencia) + KFP v2 + Google Cloud Pipeline Components. Parte de Model Monitoring v2 sigue en el namespace `preview`. Las versiones anteriores de los Tabular Workflows se conservan fijando (pinning) la versión del SDK. Fuentes: https://docs.cloud.google.com/python/docs/reference/vertexai/latest/vertexai.resources.preview.ml_monitoring.ModelMonitor , https://docs.cloud.google.com/vertex-ai/docs/tabular-data/tabular-workflows/overview

### Estado de los servicios AutoML
- **Vertex AutoML Tabular:** vigente, con precio publicado ($21.252/node-hour). **Tabular Workflows:** End-to-End AutoML en **GA**, Feature Transform Engine y Forecasting en **Public Preview**. No se encontraron avisos de deprecación. Fuente: https://docs.cloud.google.com/vertex-ai/docs/tabular-data/tabular-workflows/overview
- **SageMaker Autopilot:** la UI está en Canvas desde el 30-nov-2023. Los usuarios de Studio Classic conservan la UI antigua y la API sigue disponible; algunas tareas solo existen en AutoML API v2. No se encontró fecha de fin de vida de la API. Fuente: https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-automate-model-development.html
- **Azure automated ML:** vigente en v2 (studio, CLI, SDK). La versión v1 está archivada. Fuente: https://learn.microsoft.com/en-us/azure/machine-learning/concept-automated-ml?view=azureml-api-1

---

## 5. Salvedades y puntos no verificados

1. **Origen de los precios de AWS y Azure:** las páginas de precios de AWS y Azure no muestran las cifras sin ejecutar JavaScript. Se usaron sus APIs oficiales de precios (AWS Price List Bulk API, CSV del 2026-11-04; Azure Retail Prices API). Los valores deberían coincidir con la calculadora oficial, pero no se compararon visualmente con la página.
2. **SageMaker Pipelines y Model Registry:** la página oficial de precios no incluye una línea propia para ninguno de los dos. La afirmación de "sin coste adicional, solo cómputo" no está confirmada textualmente en aws.amazon.com.
3. **Canvas, precio por celdas:** el CSV de AWS contiene líneas de "$30/$15/$7 per million cells" que la página de Canvas no menciona (podrían corresponder a Redshift ML o a un esquema antiguo). No se usaron. La página de Canvas solo da estimaciones de coste por tamaño de dataset.
4. **Vertex AutoML Tabular, horas facturadas vs tiempo de reloj:** la medición de 2 h 41 min es de extremo a extremo. El coste real de un presupuesto de 1 node-hour (≈ $21.25) debe confirmarse en el informe de Billing del proyecto.
5. **Vertex AutoML Tabular, batch prediction:** la página de precios indica que la inferencia batch tabular usa "40 n1-highmem-8 machines". Nuestra ejecución usó n1-standard-4. Hay que verificar en la factura qué máquina se cobró realmente.
6. **Azure ML model monitoring:** se factura como serverless Spark compute, pero su tarifa por vCore-hora no se confirmó en este estudio.
7. **CloudTrail:** la página de precios no especifica región para $2.00 y $0.10 por 100 000 eventos. Se asumen tarifas generales.
8. **Fin de soporte de la CLI v1 de Azure ML (30-sep-2025):** se obtuvo de resultados de búsqueda de learn.microsoft.com, sin leer la página directamente. El fin de soporte del SDK v1 (30-jun-2026) sí se leyó en la página oficial.
9. **Jenkins en GCP:** no se encontró en la documentación oficial una integración "de primera parte" equivalente a la acción de Jenkins de CodePipeline. La compatibilidad se basa en `gcloud` y el SDK, que es como lo usa este proyecto.
10. **Cambio de marca de GCP:** las URL de documentación siguen bajo `/vertex-ai/`, pero los títulos dicen "Agent Platform" y la página de precios dice "Gemini platform". Los nombres pueden seguir cambiando.
11. **Free tiers y créditos:** cambian a menudo. El de SageMaker (2 meses) empieza al crear el primer recurso de SageMaker AI. Hay que revisar las condiciones al abrir la cuenta.
12. **Estimaciones de la sección 2.6:** son cálculos propios a partir de los precios de lista. No incluyen impuestos, red de salida, discos, logging ni descuentos (Savings Plans, CUDs, reservas).
