from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["threshold"] == 0.54

def test_predict_endpoint_validation_error():
    # Sending missing required fields should trigger Pydantic validation failure (422)
    bad_payload = {
        "age": 25
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422

def test_predict_endpoint_boundary_check():
    # Sending invalid negative financial metric
    payload = {
        "RevolvingUtilizationOfUnsecuredLines": -1.0,
        "age": 35,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.2,
        "MonthlyIncome": 3000.0,
        "NumberOfOpenCreditLinesAndLoans": 4,
        "NumberRealEstateLoansOrLines": 0,
        "NumberOfTimes90DaysLate": 0,
        "NumberDependingPersons": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422