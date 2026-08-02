from pydantic import BaseModel, Field, conint, confloat

class LoanApplicationRequest(BaseModel):
    """
    Validation schema for incoming loan applicant payloads.
    Enforces logical boundaries matching the credit risk feature set.
    """
    RevolvingUtilizationOfUnsecuredLines: confloat(ge=0.0) = Field(
        ..., description="Total balance on credit cards divided by the sum of credit limits"
    )
    age: conint(ge=18, le=120) = Field(..., description="Age of borrower in years")
    NumberOfTime30_59DaysPastDueNotWorse: conint(ge=0) = Field(
        ..., alias="NumberOfTime30-59DaysPastDueNotWorse", description="Number of times past due 30-59 days"
    )
    DebtRatio: confloat(ge=0.0) = Field(..., description="Monthly debt payments, alimony, and living costs divided by monthly gross income")
    MonthlyIncome: confloat(ge=0.0) = Field(..., description="Monthly income")
    NumberOfOpenCreditLinesAndLoans: conint(ge=0) = Field(..., description="Number of open loans and credit lines")
    NumberRealEstateLoansOrLines: conint(ge=0) = Field(..., description="Number of mortgage and real estate loans")
    NumberOfTimes90DaysLate: conint(ge=0) = Field(..., description="Number of times 90 days late or more")
    NumberDependingPersons: confloat(ge=0.0) = Field(None, description="Number of dependents in family excluding themselves")

    class Config:
        populate_by_name = True

class PredictionResponse(BaseModel):
    """
    Structured response payload returned to the frontend or loan officer.
    """
    default_probability: float
    decision_threshold: float
    prediction: str = Field(..., description="'Approved' or 'Flagged for Default Risk'")
    risk_score_percentage: float