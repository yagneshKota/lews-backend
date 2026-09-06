"""
Centralized live feature engineering and ML risk service.

Cache-first, low-request Open-Meteo usage:
  - Combined weather + soil + current observations in ONE forecast request
  - Batched elevation (center + 4 neighbors) in ONE elevation request
  - Coordinate-normalized TTL caches (weather ~10m, terrain ~24h, prediction ~10m)
  - In-flight deduplication for concurrent identical coordinates
  - HTTP 429: no retries; provider cooldown; stale cache or UNAVAILABLE
  - 5xx / network: at most one retry
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

COORD_DECIMALS = 4
WEATHER_TTL_SECONDS = 600.0  # 10 minutes
ELEVATION_TTL_SECONDS = 86400.0  # 24 hours
PREDICTION_TTL_SECONDS = 600.0  # 10 minutes
STALE_MAX_AGE_SECONDS = 86400.0  # stale prediction usable up to 24h
PROVIDER_COOLDOWN_SECONDS = 60.0
PROVIDER_NAME = "open-meteo"
ELEVATION_STEP_DEG = 0.001

HTTP_TIMEOUT = httpx.Timeout(15.0, connect=10.0)
HTTP_HEADERS = {
    "User-Agent": "LandGuard-LEWS/1.0 (https://landguard.org; contact@landguard.org)",
    "Accept": "application/json",
}

# Caches: key -> (stored_at, payload)
_WEATHER_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_ELEVATION_CACHE: dict[str, tuple[float, list[float | None]]] = {}
_PREDICTION_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_IN_FLIGHT: dict[str, asyncio.Future] = {}
_PROVIDER_COOLDOWN: dict[str, float] = {}


def reset_live_runtime_state() -> None:
    """Test helper: clear caches, in-flight map, and provider cooldown."""
    _WEATHER_CACHE.clear()
    _ELEVATION_CACHE.clear()
    _PREDICTION_CACHE.clear()
    _IN_FLIGHT.clear()
    _PROVIDER_COOLDOWN.clear()


def normalize_coordinate(latitude: float, longitude: float) -> tuple[float, float, str]:
    nlat = round(float(latitude), COORD_DECIMALS)
    nlng = round(float(longitude), COORD_DECIMALS)
    return nlat, nlng, f"{nlat:.{COORD_DECIMALS}f}_{nlng:.{COORD_DECIMALS}f}"


class LiveTelemetryUnavailableError(Exception):
    """Raised when external environmental telemetry cannot be retrieved or validated."""

    def __init__(self, message: str, missing_source: str | None = None, rate_limited: bool = False):
        super().__init__(message)
        self.message = message
        self.missing_source = missing_source
        self.rate_limited = rate_limited


def _cache_get(cache: dict[str, tuple[float, Any]], key: str, ttl: float, now: float) -> tuple[Any | None, float, bool]:
    """Return (value, age, is_fresh). Value is None if missing or older than STALE_MAX_AGE."""
    entry = cache.get(key)
    if not entry:
        return None, 0.0, False
    stored_at, value = entry
    age = now - stored_at
    if age > STALE_MAX_AGE_SECONDS and ttl < STALE_MAX_AGE_SECONDS:
        return None, age, False
    # Elevation uses a 24h fresh TTL; keep it even if older than STALE_MAX when ttl is large
    if ttl >= STALE_MAX_AGE_SECONDS and age > ttl:
        return value, age, False
    return value, age, age < ttl


def _provider_cooldown_remaining(now: float) -> float:
    until = _PROVIDER_COOLDOWN.get(PROVIDER_NAME, 0.0)
    remaining = until - now
    if remaining <= 0:
        _PROVIDER_COOLDOWN.pop(PROVIDER_NAME, None)
        return 0.0
    return remaining


def _set_provider_cooldown(seconds: float) -> None:
    cooldown = max(1.0, min(float(seconds), 300.0))
    _PROVIDER_COOLDOWN[PROVIDER_NAME] = time.time() + cooldown
    logger.warning("LIVE_RISK provider=OPEN_METEO status=429 cooldown=%.0fs", cooldown)


class LiveFeatureService:
    """Acquire live environmental data, calculate ML features, and run risk inference."""

    @staticmethod
    async def fetch_with_retries(
        client: httpx.AsyncClient,
        url: str,
        api_name: str,
        lat: float,
        lng: float,
        max_retries: int = 1,
        stats: dict[str, int] | None = None,
    ) -> httpx.Response:
        """
        GET with conservative retries.

        429: never retry; apply provider cooldown immediately.
        5xx / network / timeout: at most one retry with exponential backoff.
        """
        remaining = _provider_cooldown_remaining(time.time())
        if remaining > 0:
            logger.warning(
                "LIVE_RISK provider=OPEN_METEO status=COOLDOWN cooldown=%.0fs lat=%.4f lng=%.4f",
                remaining,
                lat,
                lng,
            )
            raise LiveTelemetryUnavailableError(
                f"Live data provider is temporarily rate-limited. Please try again later. (~{remaining:.0f}s)",
                missing_source="open-meteo",
                rate_limited=True,
            )

        last_exc: Exception | None = None
        attempts = max_retries + 1

        for attempt in range(attempts):
            start_time = time.time()
            try:
                logger.info(
                    "[REQUEST] %s attempt %d/%d started (lat=%.4f, lng=%.4f)",
                    api_name,
                    attempt + 1,
                    attempts,
                    lat,
                    lng,
                )
                if stats is not None:
                    stats["external_calls"] = stats.get("external_calls", 0) + 1
                res = await client.get(url)
                elapsed = time.time() - start_time
                logger.info(
                    "[RESPONSE] %s HTTP %d in %.2fs [attempt %d/%d] (lat=%.4f, lng=%.4f)",
                    api_name,
                    res.status_code,
                    elapsed,
                    attempt + 1,
                    attempts,
                    lat,
                    lng,
                )

                if res.status_code == 200:
                    return res

                if res.status_code == 429:
                    retry_after_header = res.headers.get("Retry-After", "")
                    try:
                        retry_after_s = float(retry_after_header)
                    except (ValueError, TypeError):
                        retry_after_s = PROVIDER_COOLDOWN_SECONDS
                    _set_provider_cooldown(retry_after_s if retry_after_s > 0 else PROVIDER_COOLDOWN_SECONDS)
                    raise LiveTelemetryUnavailableError(
                        "Live data provider is temporarily rate-limited. Please try again later.",
                        missing_source="open-meteo",
                        rate_limited=True,
                    )

                if res.status_code >= 500:
                    last_exc = LiveTelemetryUnavailableError(
                        f"{api_name} API returned HTTP {res.status_code}",
                        missing_source=api_name,
                    )
                    logger.warning(
                        "[ERROR] %s transient HTTP %d [attempt %d/%d] (lat=%.4f, lng=%.4f)",
                        api_name,
                        res.status_code,
                        attempt + 1,
                        attempts,
                        lat,
                        lng,
                    )
                else:
                    raise LiveTelemetryUnavailableError(
                        f"{api_name} API returned HTTP {res.status_code}",
                        missing_source=api_name,
                    )

            except LiveTelemetryUnavailableError:
                raise
            except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as exc:
                last_exc = exc
                logger.warning(
                    "[ERROR] %s network/timeout error [attempt %d/%d] after %.2fs (lat=%.4f, lng=%.4f): %s",
                    api_name,
                    attempt + 1,
                    attempts,
                    time.time() - start_time,
                    lat,
                    lng,
                    exc,
                )

            if attempt < max_retries:
                await asyncio.sleep(1.0 * (2**attempt))

        raise LiveTelemetryUnavailableError(
            f"{api_name} request failed after {attempts} attempts: {last_exc or 'HTTP error'}",
            missing_source=api_name,
        )

    @staticmethod
    def calculate_slope_and_aspect(
        elevations: list[float | None],
        center_lat: float,
        step_deg: float = ELEVATION_STEP_DEG,
    ) -> tuple[float, float]:
        if len(elevations) < 5 or any(e is None for e in elevations[:5]):
            raise LiveTelemetryUnavailableError(
                "Copernicus DEM elevation grid has incomplete or null values for 5-point stencil",
                missing_source="Copernicus DEM",
            )

        zc, zn, zs, ze, zw = [float(e) for e in elevations[:5]]
        dy = step_deg * 111320.0
        dx = step_deg * 111320.0 * max(0.01, math.cos(math.radians(center_lat)))
        dz_dy = (zn - zs) / (2.0 * dy)
        dz_dx = (ze - zw) / (2.0 * dx)
        slope_rad = math.atan(math.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = round(math.degrees(slope_rad), 2)

        if slope_deg == 0.0:
            aspect_deg = 0.0
        else:
            aspect_rad = math.atan2(-dz_dx, -dz_dy)
            aspect_deg = round((math.degrees(aspect_rad) + 360.0) % 360.0, 1)

        return slope_deg, aspect_deg

    @staticmethod
    def extract_rainfall_features(
        daily_precip: list[float | None],
        exclude_current_day: bool = True,
    ) -> tuple[float, float, float, float, float, float, float]:
        """
        Antecedent rainfall from completed historical days only.
        Does not invent missing days; does not include today's incomplete total.
        """
        if not daily_precip:
            raise LiveTelemetryUnavailableError(
                "Historical daily precipitation series missing",
                missing_source="Open-Meteo Precipitation",
            )

        completed = list(daily_precip[:-1] if exclude_current_day and len(daily_precip) > 1 else daily_precip)
        if any(p is None for p in completed):
            raise LiveTelemetryUnavailableError(
                "Historical daily precipitation contains missing days; values were not invented",
                missing_source="Open-Meteo Precipitation",
            )
        if len(completed) < 30:
            raise LiveTelemetryUnavailableError(
                f"Historical daily precipitation series insufficient (got {len(completed)} completed days, need ≥30)",
                missing_source="Open-Meteo Precipitation",
            )

        past_window = [float(p) for p in completed[-30:]]
        rainfall_1d = round(past_window[-1], 1)
        rainfall_3d = round(sum(past_window[-3:]), 1)
        rainfall_7d = round(sum(past_window[-7:]), 1)
        rainfall_14d = round(sum(past_window[-14:]), 1)
        rainfall_30d = round(sum(past_window[-30:]), 1)
        rainfall_7d_max = round(max(past_window[-7:]), 1)

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
    def _build_result_payload(
        cls,
        *,
        latitude: float,
        longitude: float,
        features: dict[str, float | int],
        pred: PredictionResult,
        temperature: float,
        humidity: float,
        wind_speed: float,
        soil_moisture_available: int,
        soil_moisture_display: float | None,
        data_status: str,
        message: str,
        data_age_seconds: int,
    ) -> dict[str, Any]:
        rainfall_1d = float(features["rainfall_1d_before"])
        return {
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
                "rainfall_3d": float(features["rainfall_3d_before"]),
                "rainfall_7d": float(features["rainfall_7d_before"]),
                "rainfall_14d": float(features["rainfall_14d_before"]),
                "rainfall_30d": float(features["rainfall_30d_before"]),
                "soil_moisture": soil_moisture_display,
                "soil_moisture_available": soil_moisture_available,
                "elevation_m": float(features["elevation_m"]),
                "slope_degrees": float(features["slope_degrees"]),
                "aspect_degrees": float(features["aspect_degrees"]),
            },
            "data_sources": {
                "weather": "Open-Meteo Free API",
                "terrain": "Open-Meteo Copernicus DEM",
                "soil_moisture": "Open-Meteo ECMWF IFS" if soil_moisture_available else "Not Available",
            },
            "data_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "data_age_seconds": data_age_seconds,
            "data_status": data_status,
            "message": message,
        }

    @classmethod
    def _features_and_prediction_from_telemetry(
        cls,
        latitude: float,
        longitude: float,
        weather: dict[str, Any],
        elev_list: list[float | None],
    ) -> dict[str, Any]:
        (
            rainfall_1d,
            rainfall_3d,
            rainfall_7d,
            rainfall_14d,
            rainfall_30d,
            rainfall_7d_max,
            rainfall_ratio,
        ) = cls.extract_rainfall_features(weather["precip_daily"])

        soil_moisture_available = int(weather["soil_moisture_available"])
        soil_moisture_model = float(weather["soil_moisture_model"])
        soil_moisture_display = weather["soil_moisture_display"]

        elevation_m = round(float(elev_list[0]), 1)  # type: ignore[arg-type]
        slope_deg, aspect_deg = cls.calculate_slope_and_aspect(elev_list, latitude, ELEVATION_STEP_DEG)

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
        }
        missing = [name for name in ALL_FEATURES if name not in features]
        if missing:
            raise LiveTelemetryUnavailableError(f"Incomplete feature set: {missing}", missing_source="features")

        logger.info(
            "Executing LightGBM ML risk prediction with live features (lat=%.4f, lng=%.4f)",
            latitude,
            longitude,
        )
        pred: PredictionResult = ModelAdapter().predict(features)
        logger.info(
            "ML prediction: score=%.4f tier=%s alert=%s",
            pred.risk_score,
            pred.risk_tier,
            pred.alert_triggered,
        )

        return cls._build_result_payload(
            latitude=latitude,
            longitude=longitude,
            features=features,
            pred=pred,
            temperature=float(weather["temperature"]),
            humidity=float(weather["humidity"]),
            wind_speed=float(weather["wind_speed"]),
            soil_moisture_available=soil_moisture_available,
            soil_moisture_display=soil_moisture_display,
            data_status="LIVE",
            message="Live environmental telemetry successfully retrieved and analyzed.",
            data_age_seconds=0,
        )

    @classmethod
    def _parse_weather_payload(cls, wdata: Any, latitude: float, longitude: float) -> dict[str, Any]:
        if not isinstance(wdata, dict):
            raise LiveTelemetryUnavailableError(
                "Weather telemetry payload invalid",
                missing_source="Weather & Precipitation",
            )
        current = wdata.get("current")
        daily = wdata.get("daily")
        hourly = wdata.get("hourly") or {}
        if not daily:
            raise LiveTelemetryUnavailableError(
                "Weather telemetry payload missing daily precipitation series",
                missing_source="Weather & Precipitation",
            )
        if not current or current.get("temperature_2m") is None:
            raise LiveTelemetryUnavailableError(
                "Weather telemetry missing current temperature observation",
                missing_source="Weather & Precipitation",
            )

        precip_daily = daily.get("precipitation_sum", [])
        sm1 = hourly.get("soil_moisture_0_to_1cm", []) if isinstance(hourly, dict) else []
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
            soil_moisture_model = round(valid_sm, 3)
            soil_moisture_available = 1
            soil_moisture_display: float | None = soil_moisture_model
        else:
            soil_moisture_model = 0.0
            soil_moisture_available = 0
            soil_moisture_display = None
            logger.info(
                "Soil moisture unavailable for (lat=%.4f, lng=%.4f); soil_moisture_available=0",
                latitude,
                longitude,
            )

        return {
            "temperature": float(current["temperature_2m"]),
            "humidity": float(current.get("relative_humidity_2m", 0.0)),
            "wind_speed": float(current.get("wind_speed_10m", 0.0)),
            "precip_daily": precip_daily,
            "soil_moisture_model": soil_moisture_model,
            "soil_moisture_available": soil_moisture_available,
            "soil_moisture_display": soil_moisture_display,
        }

    @classmethod
    async def _fetch_weather(
        cls,
        client: httpx.AsyncClient,
        latitude: float,
        longitude: float,
        cache_key: str,
        stats: dict[str, int],
        now: float,
    ) -> dict[str, Any]:
        cached, age, fresh = _cache_get(_WEATHER_CACHE, cache_key, WEATHER_TTL_SECONDS, now)
        if fresh and cached is not None:
            logger.info("LIVE_RISK weather cache=HIT age=%.0fs lat=%.4f lng=%.4f", age, latitude, longitude)
            return cached

        weather_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={latitude}&longitude={longitude}"
            f"&past_days=30"
            f"&forecast_days=1"
            f"&daily=precipitation_sum"
            f"&hourly=soil_moisture_0_to_1cm"
            f"&past_hours=3"
            f"&forecast_hours=1"
            f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
            f"&timezone=auto"
        )
        weather_res = await cls.fetch_with_retries(
            client=client,
            url=weather_url,
            api_name="Weather & Precipitation",
            lat=latitude,
            lng=longitude,
            stats=stats,
        )
        parsed = cls._parse_weather_payload(weather_res.json(), latitude, longitude)
        _WEATHER_CACHE[cache_key] = (now, parsed)
        return parsed

    @classmethod
    async def _fetch_elevation(
        cls,
        client: httpx.AsyncClient,
        latitude: float,
        longitude: float,
        cache_key: str,
        stats: dict[str, int],
        now: float,
    ) -> list[float | None]:
        cached, age, fresh = _cache_get(_ELEVATION_CACHE, cache_key, ELEVATION_TTL_SECONDS, now)
        if fresh and cached is not None:
            logger.info("LIVE_RISK elevation cache=HIT age=%.0fs lat=%.4f lng=%.4f", age, latitude, longitude)
            return cached

        step = ELEVATION_STEP_DEG
        lats = f"{latitude},{latitude + step},{latitude - step},{latitude},{latitude}"
        lons = f"{longitude},{longitude},{longitude},{longitude + step},{longitude - step}"
        elev_url = f"https://api.open-meteo.com/v1/elevation?latitude={lats}&longitude={lons}"
        elev_res = await cls.fetch_with_retries(
            client=client,
            url=elev_url,
            api_name="Copernicus DEM",
            lat=latitude,
            lng=longitude,
            stats=stats,
        )
        edata = elev_res.json()
        elev_list = edata.get("elevation", [])
        if not isinstance(elev_list, list) or len(elev_list) < 5 or elev_list[0] is None:
            raise LiveTelemetryUnavailableError(
                "Copernicus DEM elevation grid returned invalid elevation array",
                missing_source="Copernicus DEM",
            )
        _ELEVATION_CACHE[cache_key] = (now, elev_list)
        return elev_list

    @classmethod
    def _stale_prediction(cls, cache_key: str, now: float) -> dict[str, Any] | None:
        cached, age, _fresh = _cache_get(_PREDICTION_CACHE, cache_key, PREDICTION_TTL_SECONDS, now)
        if cached is None:
            return None
        if age > STALE_MAX_AGE_SECONDS:
            return None
        copy = dict(cached)
        copy["data_status"] = "STALE"
        copy["data_age_seconds"] = int(age)
        copy["message"] = "Showing cached data"
        return copy

    @classmethod
    async def _fetch_telemetry_and_compute(
        cls,
        latitude: float,
        longitude: float,
        cache_key: str,
        request_time: float,
        stats: dict[str, int],
    ) -> dict[str, Any]:
        cooldown = _provider_cooldown_remaining(request_time)
        if cooldown > 0:
            stale = cls._stale_prediction(cache_key, request_time)
            if stale:
                logger.info(
                    "LIVE_RISK lat=%.4f lng=%.4f cache=STALE external_calls=%d status=STALE cooldown=%.0fs",
                    latitude,
                    longitude,
                    stats.get("external_calls", 0),
                    cooldown,
                )
                return stale
            raise LiveTelemetryUnavailableError(
                "Live data provider is temporarily rate-limited. Please try again later.",
                missing_source="open-meteo",
                rate_limited=True,
            )

        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=HTTP_HEADERS) as client:
                weather = await cls._fetch_weather(client, latitude, longitude, cache_key, stats, request_time)
                elev_list = await cls._fetch_elevation(client, latitude, longitude, cache_key, stats, request_time)
        except LiveTelemetryUnavailableError:
            stale = cls._stale_prediction(cache_key, request_time)
            if stale:
                logger.info(
                    "LIVE_RISK lat=%.4f lng=%.4f cache=STALE external_calls=%d status=STALE (provider error)",
                    latitude,
                    longitude,
                    stats.get("external_calls", 0),
                )
                return stale
            raise

        payload = cls._features_and_prediction_from_telemetry(latitude, longitude, weather, elev_list)
        _PREDICTION_CACHE[cache_key] = (request_time, payload)
        logger.info(
            "LIVE_RISK lat=%.4f lng=%.4f cache=MISS external_calls=%d status=LIVE",
            latitude,
            longitude,
            stats.get("external_calls", 0),
        )
        return payload

    @classmethod
    async def get_live_risk_for_coordinate(
        cls,
        latitude: float,
        longitude: float,
    ) -> dict[str, Any]:
        if not (-90.0 <= latitude <= 90.0) or not (-180.0 <= longitude <= 180.0):
            raise ValueError(f"Invalid GPS coordinates: latitude={latitude}, longitude={longitude}")

        nlat, nlng, cache_key = normalize_coordinate(latitude, longitude)
        now = time.time()
        stats: dict[str, int] = {"external_calls": 0}

        cached_pred, age, fresh = _cache_get(_PREDICTION_CACHE, cache_key, PREDICTION_TTL_SECONDS, now)
        if fresh and cached_pred is not None:
            copy = dict(cached_pred)
            copy["data_age_seconds"] = int(age)
            copy["data_status"] = "LIVE"
            logger.info(
                "LIVE_RISK lat=%.4f lng=%.4f cache=HIT external_calls=0 status=LIVE",
                nlat,
                nlng,
            )
            return copy

        if cache_key in _IN_FLIGHT:
            existing_future = _IN_FLIGHT[cache_key]
            logger.info(
                "LIVE_RISK lat=%.4f lng=%.4f cache=DEDUP awaiting in-flight result",
                nlat,
                nlng,
            )
            return await asyncio.shield(existing_future)

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        _IN_FLIGHT[cache_key] = future

        try:
            result = await cls._fetch_telemetry_and_compute(nlat, nlng, cache_key, now, stats)
            if not future.done():
                future.set_result(result)
            return result
        except Exception as exc:
            if not future.done():
                future.set_exception(exc)
            raise
        finally:
            _IN_FLIGHT.pop(cache_key, None)
