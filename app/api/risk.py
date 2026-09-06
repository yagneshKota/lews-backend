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
import logging
import time
from fastapi.responses import JSONResponse

from app.services.live_feature_service import LiveFeatureService, LiveTelemetryUnavailableError
from app.services.risk_service import RiskService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/risk",
    tags=["risk"],
)


@router.get(
    "/live",
    response_model=LiveRiskResponse,
    responses={
        503: {
            "model": LiveRiskResponse,
            "description": "Live environmental telemetry is unavailable from external sources. No ML prediction produced.",
        }
    },
    summary="Fetch live telemetry, engineer 12 ML features, and compute real-time landslide risk for any coordinate",
)
async def get_live_risk(
    lat: float = Query(..., ge=-90.0, le=90.0, description="WGS84 GPS Latitude"),
    lng: float = Query(..., ge=-180.0, le=180.0, description="WGS84 GPS Longitude"),
):
    """
    Centralized live endpoint:
    Accepts arbitrary latitude & longitude -> fetches real-time NASA POWER & Open Topo Data telemetry ->
    calculates exact 10 ML features -> runs Phase 3 ML model -> returns live risk assessment.
    """
    try:
        return await LiveFeatureService.get_live_risk_for_coordinate(latitude=lat, longitude=lng)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except LiveTelemetryUnavailableError as exc:
        logger.warning("Live telemetry unavailable for (lat=%.4f, lng=%.4f): %s", lat, lng, exc)
        rate_limited = bool(getattr(exc, "rate_limited", False))
        message = (
            f"Live data provider '{exc.missing_source or 'External API'}' is temporarily rate-limited. Please try again later."
            if rate_limited
            else f"Live environmental telemetry is currently unavailable: {exc.message}"
        )
        return JSONResponse(
            status_code=503,
            content={
                "data_status": "UNAVAILABLE",
                "missing_source": exc.missing_source or "external-api",
                "rate_limited": rate_limited,
                "prediction": None,
                "risk_score": None,
                "risk_tier": "UNAVAILABLE",
                "message": message,
                "location": {"latitude": lat, "longitude": lng},
                "features": None,
                "environmental": None,
                "data_sources": {},
                "data_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "data_age_seconds": 0,
            },
        )
    except Exception as exc:
        logger.exception("Unexpected error in live risk endpoint for (lat=%.4f, lng=%.4f)", lat, lng)
        return JSONResponse(
            status_code=500,
            content={
                "data_status": "UNAVAILABLE",
                "prediction": None,
                "message": f"Internal live risk computation error: {exc}",
                "location": {"latitude": lat, "longitude": lng},
                "features": None,
                "environmental": None,
                "data_sources": {},
                "data_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "data_age_seconds": 0,
            },
        )


@router.post(
    "/live",
    response_model=LiveRiskResponse,
    responses={
        503: {
            "model": LiveRiskResponse,
            "description": "Live environmental telemetry is unavailable from external sources. No ML prediction produced.",
        }
    },
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
    except LiveTelemetryUnavailableError as exc:
        logger.warning(
            "Live telemetry unavailable for (lat=%.4f, lng=%.4f): %s",
            payload.latitude,
            payload.longitude,
            exc,
        )
        rate_limited = bool(getattr(exc, "rate_limited", False))
        message = (
            f"Live data provider '{exc.missing_source or 'External API'}' is temporarily rate-limited. Please try again later."
            if rate_limited
            else f"Live environmental telemetry is currently unavailable: {exc.message}"
        )
        return JSONResponse(
            status_code=503,
            content={
                "data_status": "UNAVAILABLE",
                "missing_source": exc.missing_source or "external-api",
                "rate_limited": rate_limited,
                "prediction": None,
                "risk_score": None,
                "risk_tier": "UNAVAILABLE",
                "message": message,
                "location": {"latitude": payload.latitude, "longitude": payload.longitude},
                "features": None,
                "environmental": None,
                "data_sources": {},
                "data_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "data_age_seconds": 0,
            },
        )
    except Exception as exc:
        logger.exception(
            "Unexpected error in live risk endpoint for (lat=%.4f, lng=%.4f)",
            payload.latitude,
            payload.longitude,
        )
        return JSONResponse(
            status_code=500,
            content={
                "data_status": "UNAVAILABLE",
                "prediction": None,
                "message": f"Internal live risk computation error: {exc}",
                "location": {"latitude": payload.latitude, "longitude": payload.longitude},
                "features": None,
                "environmental": None,
                "data_sources": {},
                "data_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "data_age_seconds": 0,
            },
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