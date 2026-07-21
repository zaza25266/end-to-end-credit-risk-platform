# Credit Risk Production System
*(repo: `end-to-end-credit-risk-platform`)*

An automated, enterprise-grade MLOps ecosystem built to predict the probability of 2-year borrower default risk using the Kaggle **Give Me Some Credit** dataset.

This system is architected around a **14-layer decoupled framework** to eliminate train/serve skew, enforce complete data reproducibility, manage systematic model governance, provide a user-facing dashboard, and run a fully automated self-healing continuous monitoring/retraining feedback loop.

---

## 1. Core Problem Definition & ML Architecture

### The Business Objective

In consumer lending, a **False Negative** error (classifying a borrower who will default as low-risk) results in direct catastrophic capital loss for a financial institution. Conversely, a **False Positive** error (incorrectly flagging a creditworthy borrower as high-risk) causes missed interest revenue and user churn.

This system treats raw tabular features as an untrusted stream of incoming data, validating metrics and routing payloads through a strict pipeline optimized to catch financial defaults.

### Core Architecture Placeholder Specifications

- **PROJECT_NAME:** `credit_risk_production_system`
- **TARGET_COLUMN:** `SeriousDlqin2yrs` — Binary classification (`1` = Defaulted/90+ days past due, `0` = Stable)
- **PRIMARY_METRIC:** Recall (prioritized to catch defaults), paired with a strict **precision floor of 0.70** to prevent model degeneration into over-flagging
- **ID_COLUMN:** `Id`
- **DATA_SOURCE:** Tabular schema comprising 150,000 historical rows with severe missingness, non-linear variables, and heavy class imbalance (~6.7% default rate)

---

## 2. Key Findings & Model Performance

> Fill this section in as you complete each layer — this is the part of the README that actually proves the analytical work happened, not just the tooling.

### EDA Highlights
- [Fill in: e.g. "MonthlyIncome missingness (~20%) tested MAR — correlated with self-employment status, imputed via group-by rather than global median"]
- [Fill in: e.g. "DebtRatio contained X outliers >10,000, classified as data-entry errors and capped rather than dropped"]
- [Fill in: e.g. "Top 3 predictive features by bivariate significance: RevolvingUtilizationOfUnsecuredLines, NumberOfTimes90DaysLate, age"]

### Model Comparison

| Model | CV Recall | CV Precision | ROC-AUC | Overfit Gap |
|---|---|---|---|---|
| Logistic Regression | [fill] | [fill] | [fill] | [fill] |
| Random Forest | [fill] | [fill] | [fill] | [fill] |
| XGBoost | [fill] | [fill] | [fill] | [fill] |

**Selected model:** [name] — chosen because [statistical significance / simplicity / overfit gap reasoning]
**Tuned decision threshold:** [value] (vs. default 0.5)
**Final held-out test set performance:** Recall = [X], Precision = [X]

### Top Features (SHAP)
[Fill in: e.g. "RevolvingUtilizationOfUnsecuredLines and NumberOfTimes90DaysLate together account for ~60% of model decision weight"]

### How to Reproduce
- EDA findings: `notebooks/01_eda.ipynb`
- Final metrics: `notebooks/07_final_model.ipynb`

---

## 3. Technical Stack & Tool Ecosystem

The system intentionally segregates workloads across specialized tooling rather than clustering dependencies. The final API serving layer is configured for deployment on **Render**.

| Ecosystem Layer | Dedicated Technology | Operational Purpose |
|---|---|---|
| Data & Core Engine | DVC + Pandera + Scipy | Deterministic data version tracking, schema assertions, and missingness mechanism analysis |
| Feature Processing | Scikit-Learn Pipelines | Custom stateful transformers fit strictly on isolated training data |
| Model Lifecycle & Registry | XGBoost + LightGBM + MLflow + SHAP | Cross-validation, hyperparameter searches, paired t-tests, Model Registry staging, and post-hoc explainability |
| Serving API Layer | FastAPI + Pydantic + SlowAPI | Asynchronous high-performance execution, input sanitization, and endpoint rate-limiting |
| Telemetry Storage | PostgreSQL | Connection-pooled row storage for raw feature logs and delayed ground-truth outcomes |
| Observability & Monitoring | Prometheus + Grafana + Evidently AI | Scraping endpoint latency, visualizing operational dashboards, and calculating statistical data/concept drift |
| Automation & CI/CD | Pytest + GitHub Actions | Multistage regression suites, Docker compilation, and secure vulnerability scanning |
| Closed-Loop Core | Apache Airflow | Orchestrated task graphs (DAGs) triggering automated CT (Continuous Training) feedback loops |
| Infrastructure | Terraform + Docker Compose | Idempotent multi-stage environments and infrastructure as code tracking |

---

## 4. Layer Map (for consistent numbering throughout this document)

| # | Layer | Core Artifact |
|---|---|---|
| 1 | Data | DVC-tracked dataset, Pandera schema |
| 2 | EDA | Diagnostic reports (skew, drift risk, outliers) |
| 3 | Feature Processing | Fitted `sklearn.Pipeline` |
| 4 | Model Training | Trained candidate models |
| 5 | Experimentation & Registry | MLflow tracking + comparison |
| 6 | Governance | Approval gate, Staging/Production transitions |
| 7 | Serving (API) | FastAPI endpoint |
| 8 | Telemetry Storage | PostgreSQL prediction logs |
| 9 | CI | GitHub Actions test suites |
| 10 | CD | Docker build/push/deploy |
| 11 | CT | Airflow retraining DAG |
| 12 | Monitoring | Prometheus/Grafana/Evidently |
| 13 | Infrastructure as Code | Terraform |
| 14 | Frontend | User-facing dashboard |

---

## 5. Comprehensive File Structure

The project layout isolates concerns across micro-modules, ensuring that training engines can be safely imported by orchestrators without side-effect executions:

```
end-to-end-credit-risk-platform/
├── .github/
│   └── workflows/
│       ├── ci.yml               # Layer 9: Multi-stage lint, format, check, and unit validation
│       └── cd.yml               # Layer 10: Compilation, GHCR tagging, and Render webhook triggers
├── .dvc/                        # Layer 1: Data Version Control internal states
├── dags/
│   └── retraining_pipeline.py   # Layer 11: Apache Airflow DAG orchestrating the CT automation
├── data/
│   ├── raw/                     # Pristine DVC-tracked CSV datasets (Gitignored)
│   ├── processed/               # Local cache for engineering validation steps
│   └── external/                # Domain specific reference constraints
├── frontend/                    # Layer 14: User Interface
│   ├── public/                  # Static assets
│   ├── src/                     # UI application logic and components
│   └── package.json             # Frontend dependencies
├── infra/                       # Layer 13: Infrastructure as Code (IaC)
│   ├── modules/
│   │   ├── compute/             # ECS Fargate definitions / Task definitions
│   │   ├── database/            # Scoped Private PostgreSQL Subnets
│   │   └── networking/          # Isolated VPC and security groups
│   ├── main.tf                  # State configurations tied to S3 bucket & DynamoDB lock
│   ├── variables.tf             # Declared deployment parameters
│   └── outputs.tf               # Inferred serving URL endpoints
├── mlflow/                      # Layer 5 & 6: Tracking & Registry configuration
│   └── mlflow.db                # SQLite backend for local MLflow tracking & Model Registry
├── monitoring/                  # Layer 12: Observability infrastructure
│   ├── grafana/                 # Grafana dashboard JSON provisioning files
│   ├── prometheus/              # prometheus.yml scraping target configurations
│   └── evidently/               # Evidently AI HTML drift reports and JSON snapshots
├── notebooks/                   # R&D Sandbox (Excluded from production image)
│   ├── 01_eda.ipynb                    # Exploratory analysis: univariate/bivariate stats, missingness patterns, outlier checks
│   ├── 02_preprocessing.ipynb          # Imputation strategy testing (MCAR/MAR/MNAR), schema validation drafts
│   ├── 03_feature_engineering.ipynb    # Domain feature drafts (debt-to-income ratios, WOE/IV binning) before promoting to src/features/
│   ├── 04_model_baseline.ipynb         # Logistic Regression / Random Forest / XGBoost candidate comparison under identical CV folds
│   ├── 05_hyperparameter_tuning.ipynb  # RandomizedSearchCV sweeps and threshold calibration experiments on the winning candidate
│   ├── 06_explainability.ipynb         # SHAP/feature-importance analysis to justify model decisions for governance/stakeholder review
│   └── 07_final_model.ipynb            # End-to-end reproducible run tying 01-06 together, output matches what gets promoted to src/
├── scripts/
│   └── download_data.sh         # Layer 1: Scripted Kaggle CLI download into data/raw/, fails loudly if kaggle.json is missing
├── src/
│   ├── api/                     # Layer 7: Serving Layer Core
│   │   ├── routes/
│   │   │   ├── health.py        # Real operational lifecycle health check
│   │   │   └── predict.py       # Async endpoint running telemetry writes via BackgroundTasks
│   │   ├── dependencies.py      # Pre-loading models to memory via lru_cache
│   │   ├── main.py              # Application lifecycle entrypoint with CORS limits
│   │   └── schemas.py           # Pydantic data schemas enforcing strict range thresholds
│   ├── data/                    # Layer 1: Data Pipeline Core
│   │   ├── load_and_validate.py # Pipeline coordinator
│   │   └── validation.py        # Pandera specifications & missingness statistical tests
│   ├── eda/                     # Layer 2: Diagnostic Verification (Automated scripts)
│   │   ├── univariate.py        # Skewness, Kurtosis, and target rate validations
│   │   ├── bivariate.py         # Pearson vs Spearman nonlinearity, Simpson's Paradox tests
│   │   └── outliers.py          # Isolation Forest multivariate validation
│   ├── features/                # Layer 3: Reusable Processing Pipelines
│   │   ├── split.py             # Stratified segmentation scripts
│   │   ├── engineering.py       # Domain feature injection (Debt-to-Income weights)
│   │   └── transformers.py      # Custom GroupBy, MNARFlag, and Power Transformers
│   ├── models/                  # Layer 4 & 5: Training & Experiment Lifecycle
│   │   ├── candidates.py        # Pipeline generation for baseline models
│   │   ├── tuning.py            # RandomizedSearchCV metric sweeps
│   │   └── threshold.py         # Precision-Recall Curve threshold calibration
│   ├── governance/              # Layer 6: Promotion Gates
│   │   ├── approval_gate.py     # Hard threshold evaluation matrix
│   │   └── transitions.py       # Model Registry stage lifecycle managers
│   └── monitoring/              # Layer 12: Telemetry Analytics (Python logic)
│       ├── drift_detection.py   # Evidently AI Python integration and metric calculation
│       ├── concept_drift.py     # Stored target evaluation matrices
│       └── instrumentation.py   # Prometheus python client counters and gauge configurations
├── tests/                       # Layer 9: Testing Framework
│   ├── test_data_validation.py  # Script checking live dataset assertions
│   ├── test_transformers.py     # Stateful validation checks on data mutation boundaries
│   ├── test_preprocessing_pipeline.py # Validates transformation logic behavior
│   └── test_api.py              # Validates FastAPI endpoints and Pydantic constraints
├── .dockerignore
├── .gitignore
├── Dockerfile                   # Multi-stage secure python runtime image
├── docker-compose.yml           # Unified stack coordination engine (API, Postgres, MLflow, Prometheus, Grafana)
├── params.yaml                  # Unified configuration parameter dictionary
├── README.md                    # Project overview, architecture, tech stack, layer map (this document)
└── requirements.txt             # Locked framework application specifications
```

---

## 6. Getting Started

```bash
# 1. Clone
git clone https://github.com/<your-username>/end-to-end-credit-risk-platform.git
cd end-to-end-credit-risk-platform

# 2. Scaffold any missing directories/files (safe to re-run)
chmod +x scaffold_project.sh && ./scaffold_project.sh

# 3. Download data (requires ~/.kaggle/kaggle.json)
chmod +x scripts/download_data.sh && ./scripts/download_data.sh

# 4. Bring up the full local stack
docker compose up -d

# 5. Run tests
pytest tests/ -v
```