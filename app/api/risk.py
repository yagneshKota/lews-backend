from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.risk import PredictRequest, RiskPredictionResponse, RiskResult
from app.services.risk_service import RiskService


router = APIRouter(
    prefix="/api/risk",
    tags=["risk"],
)


@router.post(
    "/predict/{report_id}",
    response_model=RiskPredictionResponse,
    status_code=201,
)
async def predict_risk(
    report_id: UUID,
    payload: PredictRequest,
    db: Session = Depends(get_db),
):
    service = RiskService(db)

    try:
        return service.predict(
            report_id=report_id,
            features=payload.get_features(),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )


@router.get(
    "/{report_id}",
    response_model=RiskPredictionResponse,
)
def get_latest_risk(
    report_id: UUID,
    db: Session = Depends(get_db),
):
    service = RiskService(db)
    latest = service.latest(report_id)
    if latest is None:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found",
        )
    return latest


@router.post(
    "/predict-live",
    response_model=RiskResult,
)
async def predict_live_risk(
    payload: PredictRequest,
):
    """Direct live inference using the Phase 3 ML model."""
    from app.services.feature_service import FeatureService
    from app.ml.model_adapter import ModelAdapter

    model = ModelAdapter()
    try:
        prepared = FeatureService.prepare(payload.model_dump())
        result = model.predict(prepared)
        return RiskResult(
            risk_score=result.risk_score,
            risk_level=result.risk_level,
            risk_tier=result.risk_tier,
            alert_triggered=result.alert_triggered,
            alert_message=result.alert_message,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {exc}",
        )