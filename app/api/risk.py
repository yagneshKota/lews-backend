from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.risk import (
    LiveRiskRequest,
    LiveRiskResponse,
    PredictRequest,
    RiskPredictionResponse,
    RiskResult,
)
from app.services.live_feature_service import LiveFeatureService
from app.services.risk_service import RiskService

router = APIRouter(
    prefix="/api/risk",
    tags=["risk"],
)


@router.get(
    "/live",
    response_model=LiveRiskResponse,
    summary="Fetch live telemetry, engineer 12 ML features, and compute real-time landslide risk for any coordinate",
)
async def get_live_risk(
    lat: float = Query(..., ge=-90.0, le=90.0, description="WGS84 GPS Latitude"),
    lng: float = Query(..., ge=-180.0, le=180.0, description="WGS84 GPS Longitude"),
):
    """
    Centralized live endpoint:
    Accepts arbitrary latitude & longitude -> fetches real-time Open-Meteo & Copernicus DEM telemetry ->
    calculates exact 12 ML features -> runs Phase 3 ML model -> returns live risk assessment.
    """
    try:
        return await LiveFeatureService.get_live_risk_for_coordinate(latitude=lat, longitude=lng)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Live feature extraction error: {exc}")


@router.post(
    "/live",
    response_model=LiveRiskResponse,
    summary="POST coordinate to compute live landslide risk",
)
async def post_live_risk(
    payload: LiveRiskRequest,
):
    """POST JSON body with latitude & longitude."""
    try:
        return await LiveFeatureService.get_live_risk_for_coordinate(
            latitude=payload.latitude,
            longitude=payload.longitude,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Live feature extraction error: {exc}")


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
    """Direct inference using the Phase 3 ML model with explicit feature dictionary."""
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