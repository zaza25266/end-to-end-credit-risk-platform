import os
import logging
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/credit_risk_db")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionTelemetry(Base):
    """
    Database model to log live inference requests, feature payloads,
    and model outputs for monitoring and continuous retraining pipelines.
    """
    __tablename__ = "prediction_telemetry"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    revolving_utilization = Column(Float)
    age = Column(Integer)
    debt_ratio = Column(Float)
    monthly_income = Column(Float)
    default_probability = Column(Float)
    prediction_decision = Column(String(50))

def init_db():
    """Creates telemetry database tables if they do not exist."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("PostgreSQL telemetry database schema initialized successfully.")
    except Exception as e:
        logger.warning(f"Database initialization skipped or deferred: {e}")