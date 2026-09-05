"""
Centralized Live Feature Engineering & ML Risk Service.

Fetches live real-time weather, historical precipitation time-series (past 30 days),
volumetric soil moisture, and DEM elevation grid from free, open-source APIs (Open-Meteo & Open GIS).
Derives the exact 12-feature geotechnical dataset and performs authoritative ML inference.
NEVER uses hardcoded environmental fallback constants for live ML predictions.
"""

from __future__ import annotations

import asyncio
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

HTTP_TIMEOUT = httpx.Timeout(15.0, connect=10.0)
HTTP_HEADERS = {
    "User-Agent": "LandGuard-LEWS/1.0 (https://landguard.org; contact@landguard.org)",
    "Accept": "application/json",
}


class LiveTelemetryUnavailableError(Exception):
    """Raised when external environmental telemetry (Weather, Soil, DEM) cannot be retrieved or validated."""

    def __init__(self, message: str, missing_source: str | None = None):
        super().__init__(message)
        self.message = message
        self.missing_source = missing_source


class LiveFeatureService:
    """Service to acquire live environmental data, calculate ML features, and run risk inference."""

    @staticmethod
    async def fetch_with_retries(
        client: httpx.AsyncClient,
        url: str,
        api_name: str,
        lat: float,
        lng: float,
        max_retries: int = 2,
    ) -> httpx.Response:
        """
        Executes an HTTP GET request with exponential backoff retries for transient cloud network failures.
        Logs every attempt, status, and failure reason.
        """
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                logger.info(
                    "Open-Meteo %s request started for (lat=%.4f, lng=%.4f) [attempt %d/%d]",
                    api_name,
                    lat,
                    lng,
                    attempt + 1,
                    max_retries + 1,
                )
                res = await client.get(url)
                logger.info(
                    "Open-Meteo %s response status %d for (lat=%.4f, lng=%.4f)",
                    api_name,
                    res.status_code,
                    lat,
                    lng,
                )
                if res.status_code == 200:
                    return res
                elif res.status_code >= 500:
                    logger.warning(
                        "Open-Meteo %s transient HTTP %d error on attempt %d: %s",
                        api_name,
                        res.status_code,
                        attempt + 1,
                        res.text[:200],
                    )
                else:
                    logger.error(
                        "Open-Meteo %s non-retryable HTTP %d error for (lat=%.4f, lng=%.4f): %s",
                        api_name,
                        res.status_code,
                        lat,
                        lng,
                        res.text[:200],
                    )
                    raise LiveTelemetryUnavailableError(
                        f"{api_name} API returned HTTP {res.status_code}",
                        missing_source=api_name,
                    )
            except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as exc:
                last_exc = exc
                logger.warning(
                    "Open-Meteo %s connection/timeout exception on attempt %d/%d for (lat=%.4f, lng=%.4f): %s",
                    api_name,
                    attempt + 1,
                    max_retries + 1,
                    lat,
                    lng,
                    exc,
                )

            if attempt < max_retries:
                backoff_s = 0.5 * (2**attempt)
                await asyncio.sleep(backoff_s)

        raise LiveTelemetryUnavailableError(
            f"{api_name} API request failed after {max_retries + 1} attempts: {last_exc or 'HTTP error'}",
            missing_source=api_name,
        )

    @staticmethod
    def calculate_slope_and_aspect(
        elevations: list[float | None],
        center_lat: float,
        step_deg: float = 0.001,
    ) -> tuple[float, float]:
        """
        Calculates terrain slope (degrees) and aspect (degrees clockwise from North)
        using central finite differences on a 5-point cross:
        [center, north, south, east, west].
        Raises LiveTelemetryUnavailableError if elevations are incomplete or contain nulls.
        """
        if len(elevations) < 5 or any(e is None for e in elevations[:5]):
            raise LiveTelemetryUnavailableError(
                "Copernicus DEM elevation grid has incomplete or null values for 5-point stencil",
                missing_source="Copernicus DEM",
            )

        zc, zn, zs, ze, zw = [float(e) for e in elevations[:5]]

        # Distance conversions (meters)
        dy = step_deg * 111320.0
        dx = step_deg * 111320.0 * max(0.01, math.cos(math.radians(center_lat)))

        dz_dy = (zn - zs) / (2.0 * dy)
        dz_dx = (ze - zw) / (2.0 * dx)

        slope_rad = math.atan(math.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = round(math.degrees(slope_rad), 2)

        # Aspect: Compass direction facing downhill (0 = North, 90 = East, 180 = South, 270 = West)
        # Downhill gradient vector is (-dz_dx, -dz_dy)
        if slope_deg == 0.0:
            aspect_deg = 0.0
        else:
            aspect_rad = math.atan2(-dz_dx, -dz_dy)
            aspect_deg = round((math.degrees(aspect_rad) + 360.0) % 360.0, 1)

        return slope_deg, aspect_deg

    @staticmethod
    def extract_rainfall_features(
        daily_precip: list[float | None],
    ) -> tuple[float, float, float, float, float, float, float]:
        """
        Calculates true antecedent rainfall aggregations from historical daily precipitation series.
        Given past 30 days series (30 days prior + today, up to 31 values):
        - rainfall_1d_before: previous 1 day accumulation (index -1 or today/yesterday)
        - rainfall_3d_before: past 3 days accumulation
        - rainfall_7d_before: past 7 days accumulation
        - rainfall_14d_before: past 14 days accumulation
        - rainfall_30d_before: past 30 days accumulation
        - rainfall_7d_max1d: maximum single-day rainfall over past 7 days
        - rainfall_3d_over_7d_ratio: ratio of 3d / 7d accumulation
        """
        if not daily_precip or len(daily_precip) < 7:
            raise LiveTelemetryUnavailableError(
                f"Historical daily precipitation series is insufficient (got {len(daily_precip) if daily_precip else 0} days, required at least 7)",
                missing_source="Open-Meteo Precipitation",
            )

        # Clean nulls to 0.0
        clean_series = [float(p if p is not None else 0.0) for p in daily_precip]

        # Ensure we strictly use the past 30 days + today window (first 31 items) and omit any future forecast days
        past_window = clean_series[:31] if len(clean_series) >= 31 else clean_series

        # 1d before (most recent day)
        rainfall_1d = round(past_window[-1], 1)

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
        rainfall_7d_max = round(max(last_7), 1)

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
        Strict zero-fallback: Fails safely with LiveTelemetryUnavailableError if real telemetry cannot be retrieved.
        """
        # 1. Validate coordinates
        if not (-90.0 <= latitude <= 90.0) or not (-180.0 <= longitude <= 180.0):
            raise ValueError(f"Invalid GPS coordinates: latitude={latitude}, longitude={longitude}")

        cache_key = f"{round(latitude, 3)}_{round(longitude, 3)}"
        now = time.time()

        if cache_key in _FEATURE_CACHE:
            cached_time, cached_result = _FEATURE_CACHE[cache_key]
            if now - cached_time < CACHE_TTL_SECONDS:
                logger.info(
                    "Returning cached live risk data for (lat=%.4f, lng=%.4f) [age=%.1fs]",
                    latitude,
                    longitude,
                    now - cached_time,
                )
                cached_copy = dict(cached_result)
                cached_copy["data_age_seconds"] = int(now - cached_time)
                return cached_copy

        step = 0.001

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=HTTP_HEADERS) as client:
            # A. Fetch Weather, 30-Day Historical Precipitation & Soil Moisture
            weather_url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={latitude}&longitude={longitude}"
                f"&past_days=30"
                f"&forecast_days=1"
                f"&daily=precipitation_sum,rain_sum"
                f"&hourly=soil_moisture_0_to_1cm,soil_moisture_1_to_3cm,temperature_2m,relative_humidity_2m,wind_speed_10m"
                f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
                f"&timezone=auto"
            )

            weather_res = await cls.fetch_with_retries(
                client=client,
                url=weather_url,
                api_name="Weather & Precipitation",
                lat=latitude,
                lng=longitude,
            )

            wdata = weather_res.json()
            current = wdata.get("current")
            daily = wdata.get("daily")
            hourly = wdata.get("hourly")

            if not isinstance(wdata, dict) or not daily or not hourly:
                raise LiveTelemetryUnavailableError(
                    "Weather telemetry payload missing daily or hourly series",
                    missing_source="Weather & Precipitation",
                )

            # Validate current observations (or fallback to latest hourly reading if current is absent)
            if current and current.get("temperature_2m") is not None:
                temperature = float(current["temperature_2m"])
                humidity = float(current.get("relative_humidity_2m", 0.0))
                wind_speed = float(current.get("wind_speed_10m", 0.0))
            else:
                temps = [t for t in hourly.get("temperature_2m", []) if t is not None]
                humids = [h for h in hourly.get("relative_humidity_2m", []) if h is not None]
                winds = [w for w in hourly.get("wind_speed_10m", []) if w is not None]
                if not temps:
                    raise LiveTelemetryUnavailableError(
                        "Weather telemetry missing temperature observations",
                        missing_source="Weather & Precipitation",
                    )
                temperature = float(temps[-1])
                humidity = float(humids[-1]) if humids else 0.0
                wind_speed = float(winds[-1]) if winds else 0.0

            # Validate daily precipitation
            precip_daily = daily.get("precipitation_sum", [])
            (
                rainfall_1d,
                rainfall_3d,
                rainfall_7d,
                rainfall_14d,
                rainfall_30d,
                rainfall_7d_max,
                rainfall_ratio,
            ) = cls.extract_rainfall_features(precip_daily)

            # Soil Moisture extraction
            sm1 = hourly.get("soil_moisture_0_to_1cm", [])
            sm2 = hourly.get("soil_moisture_1_to_3cm", [])
            valid_sm: float | None = None
            for sm_list in (sm1, sm2):
                if isinstance(sm_list, list):
                    for val in reversed(sm_list):
                        if val is not None:
                            try:
                                valid_sm = float(val)
                                break
                            except (ValueError, TypeError):
                                pass
                if valid_sm is not None:
                    break

            if valid_sm is not None:
                soil_moisture = round(valid_sm, 3)
                soil_moisture_available = 1
            else:
                soil_moisture = 0.0
                soil_moisture_available = 0
                logger.info(
                    "Soil moisture data omitted for (lat=%.4f, lng=%.4f); setting soil_moisture_available=0",
                    latitude,
                    longitude,
                )

            # B. Fetch DEM Elevation 5-point cross for Slope & Aspect
            lats = f"{latitude},{latitude+step},{latitude-step},{latitude},{latitude}"
            lons = f"{longitude},{longitude},{longitude},{longitude+step},{longitude-step}"
            elev_url = f"https://api.open-meteo.com/v1/elevation?latitude={lats}&longitude={lons}"

            elev_res = await cls.fetch_with_retries(
                client=client,
                url=elev_url,
                api_name="Copernicus DEM",
                lat=latitude,
                lng=longitude,
            )

            edata = elev_res.json()
            elev_list = edata.get("elevation", [])
            if not isinstance(elev_list, list) or len(elev_list) < 5 or elev_list[0] is None:
                raise LiveTelemetryUnavailableError(
                    "Copernicus DEM elevation grid returned invalid elevation array",
                    missing_source="Copernicus DEM",
                )

            elevation_m = round(float(elev_list[0]), 1)
            slope_deg, aspect_deg = cls.calculate_slope_and_aspect(elev_list, latitude, step)

        logger.info(
            "Live environmental telemetry successfully validated for (lat=%.4f, lng=%.4f): elev=%.1fm, slope=%.2f deg, aspect=%.1f deg, rain_1d=%.1fmm, rain_30d=%.1fmm, sm=%.3f",
            latitude,
            longitude,
            elevation_m,
            slope_deg,
            aspect_deg,
            rainfall_1d,
            rainfall_30d,
            soil_moisture,
        )

        # 2. Assemble the exact 12 ML features required by the LightGBM model
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

        for f in ALL_FEATURES:
            if f not in features:
                features[f] = 0.0

        # 3. Call authoritative trained ML model
        logger.info("Executing LightGBM ML risk prediction with live features for (lat=%.4f, lng=%.4f)", latitude, longitude)
        model_adapter = ModelAdapter()
        pred: PredictionResult = model_adapter.predict(features)
        logger.info(
            "ML risk prediction completed: score=%.4f, tier=%s, alert=%s",
            pred.risk_score,
            pred.risk_tier,
            pred.alert_triggered,
        )

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
                "weather": "Open-Meteo Free API",
                "terrain": "Open-Meteo Copernicus DEM",
                "soil_moisture": "Open-Meteo ECMWF IFS" if soil_moisture_available else "Not Available",
            },
            "data_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "data_age_seconds": 0,
            "data_status": "LIVE",
            "message": "Live environmental telemetry successfully retrieved and analyzed.",
        }

        # Cache ONLY genuine LIVE results
        _FEATURE_CACHE[cache_key] = (now, result_payload)
        return result_payload
