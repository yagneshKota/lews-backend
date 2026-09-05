"""
Centralized Live Feature Engineering & ML Risk Service.

Fetches live real-time weather, historical precipitation time-series (past 30 days),
volumetric soil moisture, and DEM elevation grid from free, open-source APIs (Open-Meteo & Open GIS).
Derives the exact 12-feature geotechnical dataset and performs authoritative ML inference.
NEVER uses hardcoded environmental fallback constants for live ML predictions.

Rate-limit handling:
  - In-flight deduplication: concurrent requests for the same coordinate share one Open-Meteo call.
  - Provider cooldown: after HTTP 429 exhaustion, provider is skipped for PROVIDER_COOLDOWN_SECONDS.
  - Retry-After header: respected when present; otherwise uses 5s / 15s backoff (max 2 retries).
  - Reduced payload: only the minimum required Open-Meteo variables are requested.
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

# ---------------------------------------------------------------------------
# Success cache — keyed by "round(lat,3)_round(lng,3)", 5-minute TTL
# Only LIVE results are cached; UNAVAILABLE results are never cached.
# ---------------------------------------------------------------------------
_FEATURE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
CACHE_TTL_SECONDS = 300.0  # 5 minutes

# ---------------------------------------------------------------------------
# In-flight deduplication — if a request for the same coordinate is already
# running, subsequent requests await the existing asyncio.Future instead of
# launching a redundant Open-Meteo call.
# ---------------------------------------------------------------------------
_IN_FLIGHT: dict[str, asyncio.Future] = {}

# ---------------------------------------------------------------------------
# Provider cooldown — after HTTP 429 is exhausted, the provider is blocked
# for PROVIDER_COOLDOWN_SECONDS to prevent hammering a rate-limited API.
# key = api_name string, value = UNIX timestamp when cooldown expires.
# ---------------------------------------------------------------------------
_PROVIDER_COOLDOWN: dict[str, float] = {}
PROVIDER_COOLDOWN_SECONDS = 60.0  # block for 60 s after exhausting 429 retries

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
        Executes an HTTP GET with proper 429 handling, Retry-After header respect,
        exponential backoff for transient failures, and provider-level cooldown.

        429 strategy:
          - attempt 1: wait Retry-After or 5s
          - attempt 2: wait Retry-After or 15s
          - attempt 3: set 60s provider cooldown, raise UNAVAILABLE

        Logs: provider, status_code, retry_after, attempt, elapsed_time.
        """
        # Check provider cooldown — skip Open-Meteo entirely if rate-limited recently
        if api_name in _PROVIDER_COOLDOWN:
            cooldown_until = _PROVIDER_COOLDOWN[api_name]
            remaining = cooldown_until - time.time()
            if remaining > 0:
                logger.warning(
                    "[RATE-LIMIT] %s provider in cooldown for %.0fs more — skipping request "
                    "(lat=%.4f, lng=%.4f)",
                    api_name, remaining, lat, lng,
                )
                raise LiveTelemetryUnavailableError(
                    f"Live weather data temporarily rate-limited. Retry in ~{remaining:.0f}s.",
                    missing_source="open-meteo",
                )
            else:
                _PROVIDER_COOLDOWN.pop(api_name, None)
                logger.info("[RATE-LIMIT] %s provider cooldown expired — resuming requests.", api_name)

        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            start_time = time.time()
            try:
                logger.info(
                    "[REQUEST] %s attempt %d/%d started (lat=%.4f, lng=%.4f)",
                    api_name, attempt + 1, max_retries + 1, lat, lng,
                )
                res = await client.get(url)
                elapsed = time.time() - start_time

                logger.info(
                    "[RESPONSE] %s HTTP %d in %.2fs [attempt %d/%d] (lat=%.4f, lng=%.4f)",
                    api_name, res.status_code, elapsed, attempt + 1, max_retries + 1, lat, lng,
                )

                if res.status_code == 200:
                    return res

                elif res.status_code == 429:
                    # Respect Retry-After header when present
                    retry_after_header = res.headers.get("Retry-After", "")
                    try:
                        retry_after_s = float(retry_after_header)
                    except (ValueError, TypeError):
                        # Default backoff: 5s first, 15s second attempt
                        retry_after_s = 5.0 if attempt == 0 else 15.0

                    retry_after_s = min(retry_after_s, 30.0)  # cap per-attempt wait at 30s

                    logger.warning(
                        "[RATE-LIMIT] %s HTTP 429 [attempt %d/%d] (lat=%.4f, lng=%.4f). "
                        "Retry-After header: '%s' → waiting %.1fs",
                        api_name, attempt + 1, max_retries + 1, lat, lng,
                        retry_after_header or "N/A", retry_after_s,
                    )

                    if attempt < max_retries:
                        await asyncio.sleep(retry_after_s)
                        continue
                    else:
                        # Exhausted all 429 retries — apply server-side cooldown
                        _PROVIDER_COOLDOWN[api_name] = time.time() + PROVIDER_COOLDOWN_SECONDS
                        logger.error(
                            "[RATE-LIMIT] %s HTTP 429 exhausted after %d attempts. "
                            "Cooldown applied for %ds. (lat=%.4f, lng=%.4f)",
                            api_name, max_retries + 1, PROVIDER_COOLDOWN_SECONDS, lat, lng,
                        )
                        raise LiveTelemetryUnavailableError(
                            f"Live weather data temporarily rate-limited. Please retry in ~{PROVIDER_COOLDOWN_SECONDS:.0f}s.",
                            missing_source="open-meteo",
                        )

                elif res.status_code >= 500:
                    # Transient server error — retry with backoff
                    logger.warning(
                        "[ERROR] %s transient HTTP %d [attempt %d/%d] (lat=%.4f, lng=%.4f): %s",
                        api_name, res.status_code, attempt + 1, max_retries + 1,
                        lat, lng, res.text[:200],
                    )

                else:
                    # Non-retryable client error (4xx except 429)
                    logger.error(
                        "[ERROR] %s non-retryable HTTP %d (lat=%.4f, lng=%.4f): %s",
                        api_name, res.status_code, lat, lng, res.text[:200],
                    )
                    raise LiveTelemetryUnavailableError(
                        f"{api_name} API returned HTTP {res.status_code}",
                        missing_source=api_name,
                    )

            except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as exc:
                last_exc = exc
                elapsed = time.time() - start_time
                logger.warning(
                    "[ERROR] %s network/timeout error [attempt %d/%d] after %.2fs (lat=%.4f, lng=%.4f): %s",
                    api_name, attempt + 1, max_retries + 1, elapsed, lat, lng, exc,
                )

            if attempt < max_retries:
                backoff_s = 1.0 * (2 ** attempt)  # 1s, 2s for network failures
                await asyncio.sleep(backoff_s)

        raise LiveTelemetryUnavailableError(
            f"{api_name} request failed after {max_retries + 1} attempts: {last_exc or 'HTTP error'}",
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
        - rainfall_1d_before: previous 1 day accumulation
        - rainfall_3d_before: past 3 days accumulation
        - rainfall_7d_before: past 7 days accumulation
        - rainfall_14d_before: past 14 days accumulation
        - rainfall_30d_before: past 30 days accumulation
        - rainfall_7d_max1d: maximum single-day rainfall over past 7 days
        - rainfall_3d_over_7d_ratio: ratio of 3d / 7d accumulation
        """
        if not daily_precip or len(daily_precip) < 7:
            raise LiveTelemetryUnavailableError(
                f"Historical daily precipitation series insufficient (got {len(daily_precip) if daily_precip else 0} days, need ≥7)",
                missing_source="Open-Meteo Precipitation",
            )

        clean_series = [float(p if p is not None else 0.0) for p in daily_precip]
        past_window = clean_series[:31] if len(clean_series) >= 31 else clean_series

        rainfall_1d = round(past_window[-1], 1)

        last_3 = past_window[-3:] if len(past_window) >= 3 else past_window
        rainfall_3d = round(sum(last_3), 1)

        last_7 = past_window[-7:] if len(past_window) >= 7 else past_window
        rainfall_7d = round(sum(last_7), 1)

        last_14 = past_window[-14:] if len(past_window) >= 14 else past_window
        rainfall_14d = round(sum(last_14), 1)

        last_30 = past_window[-30:] if len(past_window) >= 30 else past_window
        rainfall_30d = round(sum(last_30), 1)

        rainfall_7d_max = round(max(last_7), 1)

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
    async def _fetch_telemetry_and_compute(
        cls,
        latitude: float,
        longitude: float,
        cache_key: str,
        request_time: float,
    ) -> dict[str, Any]:
        """
        Internal: performs the actual Open-Meteo + DEM fetch and ML computation.
        Called only once per coordinate; in-flight dedup ensures concurrency safety.
        """
        step = 0.001

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=HTTP_HEADERS) as client:
            # ----------------------------------------------------------------
            # A. Fetch Weather, 30-Day Historical Precipitation & Soil Moisture
            #    Minimal payload: only the variables actually required.
            #    Removed: rain_sum (duplicate), soil_moisture_1_to_3cm (secondary),
            #    hourly temp/humidity/wind (covered by current=).
            # ----------------------------------------------------------------
            weather_url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={latitude}&longitude={longitude}"
                f"&past_days=30"
                f"&forecast_days=1"
                f"&daily=precipitation_sum"
                f"&hourly=soil_moisture_0_to_1cm"
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

            # Current weather observations
            if current and current.get("temperature_2m") is not None:
                temperature = float(current["temperature_2m"])
                humidity = float(current.get("relative_humidity_2m", 0.0))
                wind_speed = float(current.get("wind_speed_10m", 0.0))
            else:
                raise LiveTelemetryUnavailableError(
                    "Weather telemetry missing current temperature observation",
                    missing_source="Weather & Precipitation",
                )

            # Daily precipitation (needed for all 7 rainfall features)
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

            # Soil Moisture — primary layer only
            sm1 = hourly.get("soil_moisture_0_to_1cm", [])
            valid_sm: float | None = None
            if isinstance(sm1, list):
                for val in reversed(sm1):
                    if val is not None:
                        try:
                            valid_sm = float(val)
                            break
                        except (ValueError, TypeError):
                            pass

            if valid_sm is not None:
                soil_moisture = round(valid_sm, 3)
                soil_moisture_available = 1
            else:
                soil_moisture = 0.0
                soil_moisture_available = 0
                logger.info(
                    "Soil moisture unavailable for (lat=%.4f, lng=%.4f); soil_moisture_available=0",
                    latitude, longitude,
                )

            # ----------------------------------------------------------------
            # B. Fetch DEM Elevation 5-point cross for Slope & Aspect
            # ----------------------------------------------------------------
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
            "Live telemetry validated (lat=%.4f, lng=%.4f): elev=%.1fm slope=%.2f° "
            "rain_1d=%.1fmm rain_30d=%.1fmm sm=%.3f",
            latitude, longitude, elevation_m, slope_deg,
            rainfall_1d, rainfall_30d, soil_moisture,
        )

        # 2. Assemble the exact 12 ML features
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

        # 3. ML inference
        logger.info(
            "Executing LightGBM ML risk prediction with live features (lat=%.4f, lng=%.4f)",
            latitude, longitude,
        )
        model_adapter = ModelAdapter()
        pred: PredictionResult = model_adapter.predict(features)
        logger.info(
            "ML prediction: score=%.4f tier=%s alert=%s",
            pred.risk_score, pred.risk_tier, pred.alert_triggered,
        )

        result_payload = {
            "location": {"latitude": latitude, "longitude": longitude},
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

        # Cache ONLY genuine LIVE results — never cache UNAVAILABLE
        _FEATURE_CACHE[cache_key] = (request_time, result_payload)
        return result_payload

    @classmethod
    async def get_live_risk_for_coordinate(
        cls,
        latitude: float,
        longitude: float,
    ) -> dict[str, Any]:
        """
        Main pipeline: Coordinate → Live Weather + DEM Topography → 12 ML Features → ML Model → Live Result.

        Implements:
          1. 5-minute success cache (never caches UNAVAILABLE)
          2. In-flight deduplication (concurrent requests share one Open-Meteo call)
          3. Provider cooldown (60s after 429 exhaustion)
          4. Strict zero-fallback: raises LiveTelemetryUnavailableError if real telemetry fails
        """
        # Validate coordinates
        if not (-90.0 <= latitude <= 90.0) or not (-180.0 <= longitude <= 180.0):
            raise ValueError(f"Invalid GPS coordinates: latitude={latitude}, longitude={longitude}")

        cache_key = f"{round(latitude, 3)}_{round(longitude, 3)}"
        now = time.time()

        # 1. Check 5-minute success cache
        if cache_key in _FEATURE_CACHE:
            cached_time, cached_result = _FEATURE_CACHE[cache_key]
            if now - cached_time < CACHE_TTL_SECONDS:
                age = int(now - cached_time)
                logger.info(
                    "[CACHE HIT] Returning cached LIVE result for (lat=%.4f, lng=%.4f) [age=%ds]",
                    latitude, longitude, age,
                )
                cached_copy = dict(cached_result)
                cached_copy["data_age_seconds"] = age
                return cached_copy
            else:
                _FEATURE_CACHE.pop(cache_key, None)

        # 2. In-flight deduplication — if another coroutine is already fetching
        #    this coordinate, await its Future instead of hitting Open-Meteo again.
        if cache_key in _IN_FLIGHT:
            existing_future = _IN_FLIGHT[cache_key]
            logger.info(
                "[DEDUP] Request already in-flight for (lat=%.4f, lng=%.4f) — awaiting existing result.",
                latitude, longitude,
            )
            # asyncio.shield prevents cancellation of the in-flight future if our caller cancels
            return await asyncio.shield(existing_future)

        # 3. Launch new computation — register Future for dedup
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        _IN_FLIGHT[cache_key] = future

        try:
            result = await cls._fetch_telemetry_and_compute(latitude, longitude, cache_key, now)
            if not future.done():
                future.set_result(result)
            return result
        except Exception as exc:
            if not future.done():
                future.set_exception(exc)
            raise
        finally:
            _IN_FLIGHT.pop(cache_key, None)
