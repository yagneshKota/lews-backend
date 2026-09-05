from typing import Any
import httpx
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


@router.get("/search")
async def search_places(
    q: str = Query(..., min_length=1, description="Location search query (town, city, district)"),
    limit: int = Query(8, ge=1, le=20),
) -> list[dict[str, Any]]:
    """Geocodes place names across India and globally using Open-Meteo and OpenStreetMap."""
    query = q.strip()
    if not query:
        return []

    results: list[dict[str, Any]] = []
    seen_coords = set()

    try:
        async with httpx.AsyncClient(timeout=3.5, headers={"User-Agent": "BHU-GUARD-LEWS/1.0"}) as client:
            # 1. Try Open-Meteo Geocoding
            om_url = f"https://geocoding-api.open-meteo.com/v1/search?name={query}&count={limit}&language=en&format=json"
            res = await client.get(om_url)
            if res.status_code == 200:
                data = res.json()
                for r in data.get("results", []):
                    lat = round(float(r.get("latitude")), 4)
                    lon = round(float(r.get("longitude")), 4)
                    coord_key = (round(lat, 2), round(lon, 2))
                    if coord_key not in seen_coords:
                        seen_coords.add(coord_key)
                        results.append({
                            "id": f"om-{r.get('id', len(results))}",
                            "name": r.get("name"),
                            "district": r.get("admin2") or r.get("admin1") or "",
                            "state": r.get("admin1") or "",
                            "country": r.get("country") or "India",
                            "coordinates": [lat, lon],
                            "elevation_m": r.get("elevation") or 1000,
                        })

            # 2. If fewer than 2 results, try OpenStreetMap Nominatim with India prioritization
            if len(results) < 2:
                nom_url = f"https://nominatim.openstreetmap.org/search?q={query}&countrycodes=in&format=json&limit={limit}"
                nom_res = await client.get(nom_url)
                if nom_res.status_code == 200:
                    for item in nom_res.json():
                        lat = round(float(item["lat"]), 4)
                        lon = round(float(item["lon"]), 4)
                        coord_key = (round(lat, 2), round(lon, 2))
                        if coord_key not in seen_coords:
                            seen_coords.add(coord_key)
                            display_parts = [p.strip() for p in item.get("display_name", "").split(",")]
                            name = display_parts[0] if display_parts else query
                            state = display_parts[-2] if len(display_parts) >= 2 else "India"
                            district = display_parts[1] if len(display_parts) >= 3 else state
                            results.append({
                                "id": f"osm-{item.get('osm_id', len(results))}",
                                "name": name,
                                "district": district,
                                "state": state,
                                "country": "India",
                                "coordinates": [lat, lon],
                                "elevation_m": 1200,
                            })
    except Exception:
        pass

    return results[:limit]


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