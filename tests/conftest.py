import os

os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["ENVIRONMENT"] = "test"
os.environ["CORS_ORIGINS"] = "http://testserver"
os.environ["ALERT_MIN_TIER"] = "CRITICAL"

from collections.abc import Generator
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.database import Base, SessionLocal, engine, get_db
from app.ml.risk_mapping import PredictionResult
from app.models import Alert, Report, RiskPrediction  # noqa: F401
from app.main import app


@pytest.fixture
def db_session() -> Generator:
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(db_session) -> Generator[TestClient, None, None]:
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


CRITICAL_FEATURES = {
    "elevation_m": 1200.0,
    "slope_degrees": 38.0,
    "aspect_degrees": 135.0,
    "rainfall_1d_before": 110.0,
    "rainfall_3d_before": 220.0,
    "rainfall_7d_before": 320.0,
    "rainfall_14d_before": 480.0,
    "rainfall_30d_before": 750.0,
    "rainfall_7d_max1d": 130.0,
    "rainfall_3d_over_7d_ratio": 0.69,
    "soil_moisture": 0.58,
    "soil_moisture_available": 1,
}

REPORT_BODY = {
    "longitude": 92.62,
    "latitude": 27.47,
    "report": "Landslide reported",
    "report_description": "Heavy rainfall caused soil movement",
}


def fake_critical_result(*_args, **_kwargs) -> PredictionResult:
    return PredictionResult(
        risk_score=0.91,
        risk_level=3,
        risk_tier="CRITICAL",
        alert_triggered=True,
        alert_message="IMMEDIATE WARNING: CRITICAL landslide risk! Evacuate and alert authorities.",
        model_version="test",
    )
