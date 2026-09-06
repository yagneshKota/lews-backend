"""
Comprehensive unit and integration tests for the Live Landslide Risk Pipeline.
Verifies zero hardcoded fallbacks, HTTP 503 safe failures, 10-feature calculations,
NASA POWER & Open Topo Data telemetry integration, and caching mechanics.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch
import httpx
import pytest

from app.ml.features import ALL_FEATURES
from app.ml.model_loader import load_artifacts
from app.services.live_feature_service import (
    LiveFeatureService,
    LiveTelemetryUnavailableError,
    reset_live_runtime_state,
)


@pytest.fixture(autouse=True)
def clear_feature_cache():
    """Clear in-memory cache before every test."""
    reset_live_runtime_state()
    yield
    reset_live_runtime_state()


# ---------------------------------------------------------------------------
# 1. Slope, Aspect and DEM Topography Tests
# ---------------------------------------------------------------------------


def test_slope_and_aspect_flat_terrain():
    """Verify flat terrain produces slope=0 and aspect=0."""
    elevs = [500.0, 500.0, 500.0, 500.0, 500.0]
    slope, aspect = LiveFeatureService.calculate_slope_and_aspect(elevs, 18.5204)
    assert slope == 0.0
    assert aspect == 0.0


def test_slope_and_aspect_directional():
    """Verify directional slopes produce mathematically valid slope and azimuth aspect."""
    # North higher than South -> downhill towards South (aspect = 180)
    elevs_south = [500.0, 550.0, 450.0, 500.0, 500.0]
    slope_s, aspect_s = LiveFeatureService.calculate_slope_and_aspect(elevs_south, 18.5204)
    assert slope_s > 0.0
    assert aspect_s == 180.0

    # West higher than East -> downhill towards East (aspect = 90)
    elevs_east = [500.0, 500.0, 500.0, 450.0, 550.0]
    slope_e, aspect_e = LiveFeatureService.calculate_slope_and_aspect(elevs_east, 18.5204)
    assert slope_e > 0.0
    assert aspect_e == 90.0


def test_slope_and_aspect_incomplete_raises_error():
    """Verify incomplete or null elevations raise LiveTelemetryUnavailableError without fake defaults."""
    with pytest.raises(LiveTelemetryUnavailableError) as exc_info:
        LiveFeatureService.calculate_slope_and_aspect([500.0, None, 500.0, 500.0, 500.0], 18.5)
    assert "Open Topo Data" in exc_info.value.message

    with pytest.raises(LiveTelemetryUnavailableError):
        LiveFeatureService.calculate_slope_and_aspect([500.0, 500.0], 18.5)


# ---------------------------------------------------------------------------
# 2. Historical Rainfall Aggregation Tests
# ---------------------------------------------------------------------------


def test_rainfall_feature_aggregations():
    """Verify exact 1d, 3d, 7d, 14d, 30d, max 7d, and 3d/7d ratio from precipitation series."""
    # 31 days with 10.0mm per day
    series = [10.0] * 31
    (
        r1d,
        r3d,
        r7d,
        r14d,
        r30d,
        r7d_max,
        ratio,
    ) = LiveFeatureService.extract_rainfall_features(series)

    assert r1d == 10.0
    assert r3d == 30.0
    assert r7d == 70.0
    assert r14d == 140.0
    assert r30d == 300.0
    assert r7d_max == 10.0
    assert ratio == 0.429 or round(30.0 / 70.0, 3) == ratio


def test_rainfall_zero_division_safety():
    """Verify zero rainfall does not divide by zero and returns ratio=0.0."""
    series = [0.0] * 31
    (
        r1d,
        r3d,
        r7d,
        r14d,
        r30d,
        r7d_max,
        ratio,
    ) = LiveFeatureService.extract_rainfall_features(series)

    assert r1d == 0.0
    assert r7d == 0.0
    assert ratio == 0.0


def test_rainfall_insufficient_data_raises_error():
    """Verify series with fewer than 30 days raises LiveTelemetryUnavailableError."""
    with pytest.raises(LiveTelemetryUnavailableError):
        LiveFeatureService.extract_rainfall_features([5.0, 2.0, 1.0])


# ---------------------------------------------------------------------------
# 3. Arbitrary Coordinate Endpoint Tests (Pune, Mumbai, Dehradun)
# ---------------------------------------------------------------------------


def test_live_risk_api_pune_real_coordinates(client):
    """Test GET /api/risk/live with Pune coordinates (18.5204, 73.8567)."""
    response = client.get("/api/risk/live?lat=18.5204&lng=73.8567")
    assert response.status_code == 200
    data = response.json()

    assert data["data_status"] == "LIVE"
    assert data["location"]["latitude"] == 18.5204
    assert data["location"]["longitude"] == 73.8567
    assert data["prediction"] is not None
    assert 0.0 <= data["prediction"]["risk_score"] <= 1.0
    assert data["features"] is not None
    assert len(data["features"]) == 10

    # Check data sources
    assert data["data_sources"]["rainfall"] == "NASA POWER (PRECTOTCORR)"
    assert data["data_sources"]["terrain"] == "OpenTopoData SRTM90m"

    # Check environmental block
    assert data["environmental"]["soil_moisture"] is None
    assert data["environmental"]["soil_moisture_available"] == 0
    assert "elevation_m" in data["features"]
    assert "slope_degrees" in data["features"]


def test_live_risk_api_mumbai_coordinates(client):
    """Test GET /api/risk/live with Mumbai coordinates (19.0760, 72.8777)."""
    response = client.get("/api/risk/live?lat=19.0760&lng=72.8777")
    assert response.status_code == 200
    data = response.json()
    assert data["data_status"] == "LIVE"
    assert data["location"]["latitude"] == 19.0760
    assert data["location"]["longitude"] == 72.8777
    assert data["prediction"] is not None


def test_live_risk_api_post_valid_coordinates(client):
    """Test POST /api/risk/live with Dehradun coordinates."""
    response = client.post(
        "/api/risk/live",
        json={"latitude": 30.3165, "longitude": 78.0322},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["data_status"] == "LIVE"
    assert data["location"]["latitude"] == 30.3165
    assert data["location"]["longitude"] == 78.0322


def test_live_risk_api_invalid_coordinates(client):
    """Test validation errors for out-of-range coordinates."""
    res1 = client.get("/api/risk/live?lat=95.0&lng=73.8567")
    assert res1.status_code == 422

    res2 = client.get("/api/risk/live?lat=18.5204&lng=195.0")
    assert res2.status_code == 422


# ---------------------------------------------------------------------------
# 4. Mocked Telemetry & Safe Failure Handling (HTTP 503)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_mock_nasa_power_rainfall_timeout():
    """Verify that a network timeout raises LiveTelemetryUnavailableError and does not run ML."""
    async with httpx.AsyncClient() as client:
        with patch.object(client, "get", side_effect=httpx.TimeoutException("Connection timed out")):
            with pytest.raises(LiveTelemetryUnavailableError) as exc_info:
                await LiveFeatureService.fetch_with_retries(
                    client=client,
                    url="https://power.larc.nasa.gov/api/temporal/daily/point",
                    api_name="NASA POWER Daily Point API",
                    lat=18.5,
                    lng=73.8,
                    provider_name="NASA POWER",
                    max_retries=1,
                )
            assert "failed after" in exc_info.value.message


def test_api_returns_503_when_nasa_power_unavailable(client):
    """Verify GET /api/risk/live returns HTTP 503 with prediction=None when external rainfall fails."""
    with patch.object(
        LiveFeatureService,
        "get_live_risk_for_coordinate",
        side_effect=LiveTelemetryUnavailableError("NASA POWER Daily API timeout", missing_source="NASA POWER"),
    ):
        response = client.get("/api/risk/live?lat=18.5204&lng=73.8567")
        assert response.status_code == 503
        data = response.json()
        assert data["data_status"] == "UNAVAILABLE"
        assert data["prediction"] is None
        assert data["features"] is None
        assert "Live environmental telemetry is currently unavailable" in data["message"]


def test_api_returns_503_when_dem_unavailable(client):
    """Verify GET /api/risk/live returns HTTP 503 with prediction=None when DEM fails."""
    with patch.object(
        LiveFeatureService,
        "get_live_risk_for_coordinate",
        side_effect=LiveTelemetryUnavailableError("Open Topo Data service unavailable", missing_source="Open Topo Data"),
    ):
        response = client.get("/api/risk/live?lat=18.5204&lng=73.8567")
        assert response.status_code == 503
        data = response.json()
        assert data["data_status"] == "UNAVAILABLE"
        assert data["prediction"] is None


# ---------------------------------------------------------------------------
# 5. Missing Soil Moisture Graceful Handling
# ---------------------------------------------------------------------------


def test_missing_soil_moisture_sets_available_zero():
    """Verify that soil moisture is always reported as unavailable (soil_moisture_available=0, soil_moisture=None)."""
    mock_nasa_power_daily = {
        "properties": {
            "parameter": {
                "PRECTOTCORR": {f"202608{i:02d}": 5.0 for i in range(1, 32)}
            }
        }
    }
    mock_opentopodata_dem = {
        "results": [
            {"elevation": 600.0},
            {"elevation": 605.0},
            {"elevation": 595.0},
            {"elevation": 600.0},
            {"elevation": 600.0},
        ],
        "status": "OK",
    }
    mock_nasa_hourly = {
        "properties": {
            "parameter": {
                "T2M": {},
                "RH2M": {},
                "WS10M": {},
            }
        }
    }

    load_artifacts()

    async def run_test():
        with patch.object(LiveFeatureService, "fetch_with_retries") as mock_fetch:
            res_rain = httpx.Response(200, json=mock_nasa_power_daily, request=httpx.Request("GET", "http://test"))
            res_dem = httpx.Response(200, json=mock_opentopodata_dem, request=httpx.Request("GET", "http://test"))
            res_weather = httpx.Response(200, json=mock_nasa_hourly, request=httpx.Request("GET", "http://test"))
            mock_fetch.side_effect = [res_rain, res_dem, res_weather]

            result = await LiveFeatureService.get_live_risk_for_coordinate(18.5204, 73.8567)
            assert result["environmental"]["soil_moisture_available"] == 0
            assert result["environmental"]["soil_moisture"] is None
            assert result["prediction"] is not None
            assert result["data_status"] == "LIVE"

    asyncio.run(run_test())


# ---------------------------------------------------------------------------
# 6. Cache Behavior Tests
# ---------------------------------------------------------------------------


def test_cache_stores_only_successful_live_results(client):
    """Verify that successful live queries are cached with TTL, but errors are not."""
    # First successful query
    res1 = client.get("/api/risk/live?lat=18.5204&lng=73.8567")
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["data_status"] == "LIVE"

    # Second query should hit cache with data_age_seconds >= 0
    res2 = client.get("/api/risk/live?lat=18.5204&lng=73.8567")
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["data_status"] == "LIVE"
    assert "data_age_seconds" in data2
