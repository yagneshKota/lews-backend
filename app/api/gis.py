from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.report import Report
from app.models.risk_prediction import RiskPrediction
from app.services.weather_gis_service import WeatherGisService

router = APIRouter(prefix="/api/gis", tags=["gis"])


@router.get("/live-telemetry")
async def get_live_telemetry(
    lat: float = Query(..., ge=-90, le=90),
    lng: float = Query(..., ge=-180, le=180),
):
    """Fetches real-time Open-Meteo precipitation, soil moisture, and GIS elevation for any GPS coordinate."""
    return await WeatherGisService.get_realtime_telemetry(lat, lng)


@router.get("/reports")
def gis_reports(
    offset: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=5000),
    db: Session = Depends(get_db),
):
    ranked = (
        select(
            RiskPrediction.id.label("prediction_id"),
            RiskPrediction.report_id.label("prediction_report_id"),
            RiskPrediction.risk_score,
            RiskPrediction.risk_level,
            RiskPrediction.risk_tier,
            RiskPrediction.created_at.label("prediction_created_at"),
            func.row_number()
            .over(
                partition_by=RiskPrediction.report_id,
                order_by=RiskPrediction.created_at.desc(),
            )
            .label("rn"),
        )
        .subquery()
    )

    rows = db.execute(
        select(Report, ranked)
        .outerjoin(
            ranked,
            (ranked.c.prediction_report_id == Report.id)
            & (ranked.c.rn == 1),
        )
        .order_by(Report.created_at.desc())
        .offset(offset)
        .limit(limit)
    ).all()

    return [
    {
        "id": row[0].id,
        "latitude": row[0].latitude,
        "longitude": row[0].longitude,
        "report": row[0].report,
        "report_description": row[0].report_description,
        "risk_score": row.risk_score,
        "risk_level": row.risk_level,
        "risk_tier": row.risk_tier,
        "timestamp": row.prediction_created_at or row[0].created_at,
    }
    for row in rows
]


@router.get("/risk")
def gis_risk(
    offset: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=5000),
    db: Session = Depends(get_db),
):
    rows = db.execute(
        select(Report, RiskPrediction)
        .join(
            RiskPrediction,
            RiskPrediction.report_id == Report.id,
        )
        .order_by(RiskPrediction.created_at.desc())
        .offset(offset)
        .limit(limit)
    ).all()

    return [
    {
        "id": row[0].id,
        "latitude": row[0].latitude,
        "longitude": row[0].longitude,
        "risk_score": row[1].risk_score,
        "risk_level": row[1].risk_level,
        "risk_tier": row[1].risk_tier,
        "timestamp": row[1].created_at,
    }
    for row in rows
]