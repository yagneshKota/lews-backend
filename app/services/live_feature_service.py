"""
Centralized Live Feature Engineering & ML Risk Service.

Fetches live real-time weather, historical precipitation time-series (past 30 days),
volumetric soil moisture, and DEM elevation grid from free, open-source APIs (Open-Meteo & Open GIS).
Derives the exact 12-feature geotechnical dataset and performs authoritative ML inference.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any
import httpx

from app.ml.features import ALL_FEATURES
from app.ml.model_adapter import ModelAdapter
from app.ml.risk_mapping import PredictionResult

logger = logging.getLogger(__name__)

# In-memory cache keyed by "round(lat, 3)_round(lng, 3)" with a 5-minute TTL
_FEATURE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
CACHE_TTL_SECONDS = 300.0  # 5 minutes


class LiveFeatureService:
    """Service to acquire live environmental data, calculate ML features, and run risk inference."""

    @staticmethod
    def calculate_slope_and_aspect(
        elevations: list[float],
        center_lat: float,
        step_deg: float = 0.001,
    ) -> tuple[float, float]:
        """
        Calculates terrain slope (degrees) and aspect (degrees clockwise from North)
        using central finite differences on a 5-point cross:
        [center, north, south, east, west].
        """
        if len(elevations) < 5 or any(e is None for e in elevations[:5]):
            return 25.0, 140.0

        zc, zn, zs, ze, zw = [float(e) for e in elevations[:5]]

        # Distance conversions (meters)
        dy = step_deg * 111320.0
        dx = step_deg * 111320.0 * max(0.01, math.cos(math.radians(center_lat)))

        dz_dy = (zn - zs) / (2.0 * dy)
        dz_dx = (ze - zw) / (2.0 * dx)

        slope_rad = math.atan(math.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = round(math.degrees(slope_rad), 1)

        # Aspect: Compass direction facing downhill (0 = North, 90 = East, 180 = South, 270 = West)
        # Downhill gradient vector is (-dz_dx, -dz_dy)
        aspect_rad = math.atan2(dz_dy, -dz_dx)
        aspect_deg = round((math.degrees(aspect_rad) + 360.0) % 360.0, 1)

        return slope_deg, aspect_deg

    @staticmethod
    def extract_rainfall_features(
        daily_precip: list[float | None],
    ) -> tuple[float, float, float, float, float, float, float]:
        """
        Calculates true antecedent rainfall aggregations from daily precipitation series.
        Given past 30 days series + today (indices 0..30):
        - rainfall_1d_before: previous 1 day accumulation (index 30 / today or index 29)
        - rainfall_3d_before: past 3 days accumulation
        - rainfall_7d_before: past 7 days accumulation
        - rainfall_14d_before: past 14 days accumulation
        - rainfall_30d_before: past 30 days accumulation
        - rainfall_7d_max1d: maximum single-day rainfall over past 7 days
        - rainfall_3d_over_7d_ratio: ratio of 3d / 7d accumulation
        """
        # Clean nulls to 0.0
        clean_series = [float(p or 0.0) for p in daily_precip]

        if not clean_series:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # If Open-Meteo returns past_days=30 + 7 forecast days = 38 items,
        # the past 30 days + today are clean_series[:31]
        past_window = clean_series[:31] if len(clean_series) >= 31 else clean_series

        # 1d before (most recent day)
        rainfall_1d = round(past_window[-1] if past_window else 0.0, 1)

        # 3d before (last 3 days)
        last_3 = past_window[-3:] if len(past_window) >= 3 else past_window
        rainfall_3d = round(sum(last_3), 1)

        # 7d before (last 7 days)
        last_7 = past_window[-7:] if len(past_window) >= 7 else past_window
        rainfall_7d = round(sum(last_7), 1)

        # 14d before (last 14 days)
        last_14 = past_window[-14:] if len(past_window) >= 14 else past_window
        rainfall_14d = round(sum(last_14), 1)

        # 30d before (last 30 days)
        last_30 = past_window[-30:] if len(past_window) >= 30 else past_window
        rainfall_30d = round(sum(last_30), 1)

        # 7d max single day
        rainfall_7d_max = round(max(last_7) if last_7 else rainfall_1d, 1)

        # 3d over 7d ratio
        if rainfall_7d > 0.0:
            ratio = round(min(1.0, max(0.0, rainfall_3d / rainfall_7d)), 3)
        else:
            ratio = 0.0

        return (
            rainfall_1d,
            rainfall_3d,
            rainfall_7d,
            rainfall_14d,
            rainfall_30d,
            rainfall_7d_max,
            ratio,
        )

    @classmethod
    async def get_live_risk_for_coordinate(
        cls,
        latitude: float,
        longitude: float,
    ) -> dict[str, Any]:
        """
        Main pipeline:
        Coordinate -> Live Weather + DEM Topography -> 12 ML Features -> Phase 3 ML Model -> Live Result.
        """
        # 1. Validate coordinates
        if not (-90.0 <= latitude <= 90.0) or not (-180.0 <= longitude <= 180.0):
            raise ValueError(f"Invalid GPS coordinates: latitude={latitude}, longitude={longitude}")

        cache_key = f"{round(latitude, 3)}_{round(longitude, 3)}"
        now = time.time()

        if cache_key in _FEATURE_CACHE:
            cached_time, cached_result = _FEATURE_CACHE[cache_key]
            if now - cached_time < CACHE_TTL_SECONDS:
                cached_copy = dict(cached_result)
                cached_copy["data_age_seconds"] = int(now - cached_time)
                return cached_copy

        # Default baselines for fallback if external network is disconnected
        data_status = "LIVE"
        weather_source = "Open-Meteo Free API"
        terrain_source = "Open-Meteo Copernicus DEM"
        soil_source = "Open-Meteo ECMWF IFS"

        # Defaults in case of network unavailability
        elevation_m = 1200.0
        slope_deg = 25.0
        aspect_deg = 135.0
        rainfall_1d = 20.0
        rainfall_3d = 55.0
        rainfall_7d = 110.0
        rainfall_14d = 180.0
        rainfall_30d = 310.0
        rainfall_7d_max = 35.0
        rainfall_ratio = 0.50
        soil_moisture = 0.525
        soil_moisture_available = 0

        temperature = 19.0
        humidity = 80.0
        wind_speed = 12.0

        step = 0.001

        try:
            async with httpx.AsyncClient(timeout=4.5) as client:
                # A. Fetch Weather, 30-Day Historical Precipitation & Soil Moisture
                weather_url = (
                    f"https://api.open-meteo.com/v1/forecast"
                    f"?latitude={latitude}&longitude={longitude}"
                    f"&past_days=30"
                    f"&daily=precipitation_sum,rain_sum"
                    f"&hourly=soil_moisture_0_to_1cm,soil_moisture_1_to_3cm,temperature_2m,relative_humidity_2m,wind_speed_10m"
                    f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
                    f"&timezone=auto"
                )
                weather_res = await client.get(weather_url)

                if weather_res.status_code == 200:
                    wdata = weather_res.json()
                    current = wdata.get("current", {})
                    daily = wdata.get("daily", {})
                    hourly = wdata.get("hourly", {})

                    if current:
                        temperature = float(current.get("temperature_2m", temperature))
                        humidity = float(current.get("relative_humidity_2m", humidity))
                        wind_speed = float(current.get("wind_speed_10m", wind_speed))

                    precip_daily = daily.get("precipitation_sum", [])
                    if precip_daily and len(precip_daily) >= 7:
                        (
                            rainfall_1d,
                            rainfall_3d,
                            rainfall_7d,
                            rainfall_14d,
                            rainfall_30d,
                            rainfall_7d_max,
                            rainfall_ratio,
                        ) = cls.extract_rainfall_features(precip_daily)

                    # Soil Moisture from Open-Meteo
                    sm1 = hourly.get("soil_moisture_0_to_1cm", [])
                    sm2 = hourly.get("soil_moisture_1_to_3cm", [])
                    # Pick the last valid non-null reading
                    valid_sm = None
                    for sm_list in (sm1, sm2):
                        for val in reversed(sm_list):
                            if val is not None:
                                valid_sm = float(val)
                                break
                        if valid_sm is not None:
                            break

                    if valid_sm is not None:
                        # Open-Meteo volumetric soil moisture is m³/m³ (0.1 to 0.7)
                        soil_moisture = round(valid_sm, 3)
                        soil_moisture_available = 1
                    else:
                        soil_moisture = 0.525
                        soil_moisture_available = 0
                else:
                    data_status = "PARTIAL"
                    logger.warning("Open-Meteo weather responded with status %s", weather_res.status_code)

                # B. Fetch DEM Elevation 5-point cross for Slope & Aspect
                lats = f"{latitude},{latitude+step},{latitude-step},{latitude},{latitude}"
                lons = f"{longitude},{longitude},{longitude},{longitude+step},{longitude-step}"
                elev_url = f"https://api.open-meteo.com/v1/elevation?latitude={lats}&longitude={lons}"

                elev_res = await client.get(elev_url)
                if elev_res.status_code == 200:
                    edata = elev_res.json()
                    elev_list = edata.get("elevation", [])
                    if elev_list and len(elev_list) >= 5 and elev_list[0] is not None:
                        elevation_m = round(float(elev_list[0]), 1)
                        calc_slope, calc_aspect = cls.calculate_slope_and_aspect(
                            elev_list, latitude, step
                        )
                        slope_deg = calc_slope
                        aspect_deg = calc_aspect
                else:
                    data_status = "PARTIAL" if data_status == "LIVE" else "UNAVAILABLE"

        except Exception as exc:
            logger.warning("Live telemetry fetch exception for (%s, %s): %s", latitude, longitude, exc)
            data_status = "STALE"

        # 2. Assemble the exact 12 ML features required by the LightGBM/XGBoost model
        features: dict[str, float | int] = {
            "elevation_m": elevation_m,
            "slope_degrees": slope_deg,
            "aspect_degrees": aspect_deg,
            "rainfall_1d_before": rainfall_1d,
            "rainfall_3d_before": rainfall_3d,
            "rainfall_7d_before": rainfall_7d,
            "rainfall_14d_before": rainfall_14d,
            "rainfall_30d_before": rainfall_30d,
            "rainfall_7d_max1d": rainfall_7d_max,
            "rainfall_3d_over_7d_ratio": rainfall_ratio,
            "soil_moisture": soil_moisture,
            "soil_moisture_available": soil_moisture_available,
        }

        # Validate that all 12 features are present
        for f in ALL_FEATURES:
            if f not in features:
                features[f] = 0.0

        # 3. Call authoritative trained ML model
        model_adapter = ModelAdapter()
        pred: PredictionResult = model_adapter.predict(features)

        result_payload = {
            "location": {
                "latitude": latitude,
                "longitude": longitude,
            },
            "features": features,
            "prediction": {
                "risk_score": pred.risk_score,
                "risk_level": pred.risk_level,
                "risk_tier": pred.risk_tier,
                "alert_triggered": pred.alert_triggered,
                "alert_message": pred.alert_message,
                "model_version": pred.model_version,
            },
            "environmental": {
                "temperature": temperature,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "rainfall_24h": rainfall_1d,
                "rainfall_3d": rainfall_3d,
                "rainfall_7d": rainfall_7d,
                "soil_moisture": soil_moisture,
                "elevation_m": elevation_m,
                "slope_degrees": slope_deg,
                "aspect_degrees": aspect_deg,
            },
            "data_sources": {
                "weather": weather_source,
                "terrain": terrain_source,
                "soil_moisture": soil_source,
            },
            "data_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "data_age_seconds": 0,
            "data_status": data_status,
        }

        # Cache result
        _FEATURE_CACHE[cache_key] = (now, result_payload)
        return result_payload
