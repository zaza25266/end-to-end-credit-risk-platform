# End-to-End Credit Risk Prediction Platform

## Introduction

This project is an end-to-end Machine Learning platform for credit risk prediction, based on the **"Give Me Some Credit"** dataset. The goal of the platform is to assess the creditworthiness of loan applicants by calculating their probability of default. 

The system leverages a production-grade machine learning pipeline to engineer features, train a robust ensemble of gradient boosting models, and serve predictions asynchronously via a high-performance REST API. It includes a traditional, banking-style web interface for loan officers to submit applicant data and receive real-time, automated credit decisions.

The platform is designed with MLOps best practices in mind, featuring model telemetry, data validation, experiment tracking, and a governance gate to ensure only high-quality models reach production.

## Tech Stack

**Core ML & Data Science**
- **Scikit-Learn**: Custom feature engineering pipelines, imputation (KNN), and `VotingClassifier` ensemble.
- **CatBoost, XGBoost, LightGBM**: State-of-the-art gradient boosting algorithms for tabular data classification.
- **Pandas & NumPy**: Data manipulation and numerical operations.
- **Joblib**: Model artifact serialization.

**MLOps & Monitoring**
- **MLflow**: Experiment tracking, metric logging, and model registry.
- **Pandera**: Schema-based data validation for incoming raw datasets.
- **Evidently AI**: Automated data and target drift detection in production telemetry.
- **Prometheus**: Real-time API metrics and telemetry monitoring.
- **SHAP**: Explainable AI (XAI) for model interpretability and auditing.

**Backend API & Database**
- **FastAPI**: Asynchronous, high-performance RESTful API serving.
- **Pydantic (v2)**: Strict payload validation and type coercion.
- **SQLAlchemy**: ORM for logging real-time inference telemetry.
- **PostgreSQL / SQLite**: Telemetry database (PostgreSQL for production via Docker, SQLite fallback for local dev).
- **SlowAPI**: Rate limiting for API endpoints.

**Frontend & Deployment**
- **HTML/CSS/JS (Vanilla)**: A clean, formal, institutional web interface for loan officers.
- **Docker & Docker Compose**: Containerization for reproducible, isolated deployments.

## Project Structure

```text
.
├── README.md                      # Project documentation
├── Dockerfile                     # Docker configuration for the API
├── docker-compose.yml             # Orchestration for API, DB, and monitoring
├── requirements.txt               # Python dependencies
├── mlflow.db                      # Local MLflow tracking database (SQLite)
├── config/
│   ├── params.yaml                # Centralized hyperparameters and configurations
│   └── params_all.yaml
├── scripts/
│   └── download_data.sh           # Utility script to fetch raw data
├── notebooks/                     # Jupyter notebooks for EDA and modeling experiments
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_baseline.ipynb
│   ├── 05_hyperparameter_tuning.ipynb
│   ├── 6_WOE.ipynb
│   ├── 7.final_models.ipynb
│   └── 8.final_ensamble.ipynb
├── src/                           # Application Source Code
│   ├── api/                       # FastAPI Backend
│   │   ├── database.py            # SQLAlchemy setup and Telemetry model
│   │   ├── dependencies.py        # Cached artifact loaders (Model & Transformer)
│   │   ├── main.py                # FastAPI application and endpoints
│   │   └── schemas.py             # Pydantic validation schemas
│   ├── features/                  # Feature Engineering
│   │   └── transformers.py        # Custom scikit-learn transformers
│   ├── data/                      # Data Processing
│   │   ├── load_and_validate.py   # Raw data loading and Pandera validation
│   │   └── validation.py          # Pandera schema definitions
│   ├── models/                    # Modeling and Training
│   │   ├── candidates.py          # CatBoost, XGBoost, LightGBM initializers
│   │   ├── explainability.py      # SHAP integration
│   │   └── train.py               # E2E training pipeline and MLflow tracking
│   ├── governance/                # Deployment Governance
│   │   └── approval_gate.py       # Metrics threshold checks for model promotion
│   ├── monitoring/                # Production Monitoring
│   │   ├── drift_detection.py     # Evidently AI drift analysis
│   │   ├── instrumentation.py     # Prometheus metrics middleware
│   │   └── prometheus/            # Prometheus configuration
│   └── utils/
│       └── config.py              # YAML config loader
└── tests/                         # Pytest Suite
    ├── test_api.py                # End-to-end API tests
    ├── test_config.py             # Configuration loader tests
    ├── test_models.py             # Model training and artifact tests
    ├── test_transformers.py       # Feature engineering unit tests
    └── test_validation.py         # Data validation tests
```
