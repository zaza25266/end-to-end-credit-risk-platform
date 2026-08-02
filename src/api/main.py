import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from src.api.schemas import LoanApplicationRequest, PredictionResponse
from src.api.dependencies import get_production_artifacts
from src.api.database import init_db, SessionLocal, PredictionTelemetry

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="End-to-End Credit Risk API",
    description="Production-grade asynchronous scoring engine with model governance & telemetry.",
    version="1.0.0"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.on_event("startup")
def startup_event():
    init_db()

@app.get("/health", tags=["Lifecycle"])
@limiter.limit("10/minute")
def health_check(request: Request):
    artifacts = get_production_artifacts()
    status = "healthy" if artifacts["model"] is not None else "degraded_missing_model"
    return {
        "status": status,
        "threshold": artifacts["threshold"],
        "model_loaded": artifacts["model"] is not None
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
@limiter.limit("30/minute")
def predict_credit_risk(payload: LoanApplicationRequest, request: Request):
    artifacts = get_production_artifacts()
    model = artifacts["model"]
    transformer = artifacts["transformer"]
    threshold = artifacts["threshold"]

    if model is None or transformer is None:
        raise HTTPException(
            status_code=503,
            detail="Production model or transformer artifacts not found in memory. Please run training pipeline first."
        )

    # Convert request to DataFrame for transformer pipeline consumption
    input_data = pd.DataFrame([payload.dict(by_alias=True)])
    
    try:
        transformed_data = transformer.transform(input_data)
        probability = float(model.predict_proba(transformed_data)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference execution error: {str(e)}")

    is_default = probability >= threshold
    decision = "Flagged for Default Risk" if is_default else "Approved"

    # Log telemetry asynchronously/synchronously to database session
    try:
        db = SessionLocal()
        telemetry_record = PredictionTelemetry(
            revolving_utilization=payload.RevolvingUtilizationOfUnsecuredLines,
            age=payload.age,
            debt_ratio=payload.DebtRatio,
            monthly_income=payload.MonthlyIncome,
            default_probability=probability,
            prediction_decision=decision
        )
        db.add(telemetry_record)
        db.commit()
        db.close()
    except Exception:
        pass # Non-blocking failure for telemetry logging

    return PredictionResponse(
        default_probability=round(probability, 4),
        decision_threshold=threshold,
        prediction=decision,
        risk_score_percentage=round(probability * 100, 2)
    )