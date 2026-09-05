import asyncio
from app.services.live_feature_service import LiveFeatureService
from app.ml.features import ALL_FEATURES
from app.ml.model_loader import load_artifacts


def test_slope_and_aspect_calculation():
    """Verify finite-difference slope and aspect math on a 5-point elevation cross."""
    # Flat terrain: all elevations equal 1000m
    flat_elevs = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
    slope, aspect = LiveFeatureService.calculate_slope_and_aspect(flat_elevs, 27.5)
    assert slope == 0.0

    # North-sloping terrain (North higher than South)
    elevs = [2000.0, 2100.0, 1900.0, 2000.0, 2000.0]
    slope, aspect = LiveFeatureService.calculate_slope_and_aspect(elevs, 27.5)
    assert slope > 0.0
    assert 0.0 <= aspect <= 360.0


def test_rainfall_feature_aggregations():
    """Verify exact multi-day aggregations, max 1-day, and 3d/7d ratio."""
    # Construct 35 days of synthetic precipitation: 10mm per day
    series = [10.0] * 35
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
    """Verify zero rainfall does not crash division and produces ratio = 0.0."""
    series = [0.0] * 35
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


def test_live_feature_service_returns_all_12_features():
    """Verify all 12 model features are computed and present."""
    load_artifacts()
    result = asyncio.run(LiveFeatureService.get_live_risk_for_coordinate(27.5861, 91.8660))
    features = result["features"]
    for f in ALL_FEATURES:
        assert f in features, f"Missing feature {f}"
        assert isinstance(features[f], (int, float))

    assert "prediction" in result
    assert "risk_score" in result["prediction"]
    assert "risk_tier" in result["prediction"]
    assert "data_status" in result


def test_live_risk_api_get_valid(client):
    """Test GET /api/risk/live with valid coordinates (e.g. Tawang)."""
    response = client.get("/api/risk/live?lat=27.5861&lng=91.8660")
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert "prediction" in data
    assert "environmental" in data
    assert len(data["features"]) == 12


def test_live_risk_api_get_pune(client):
    """Test GET /api/risk/live with arbitrary non-Northeast coordinate (Pune)."""
    response = client.get("/api/risk/live?lat=18.5204&lng=73.8567")
    assert response.status_code == 200
    data = response.json()
    assert data["location"]["latitude"] == 18.5204
    assert data["location"]["longitude"] == 73.8567
    assert data["prediction"]["risk_score"] >= 0.0


def test_live_risk_api_post_valid(client):
    """Test POST /api/risk/live with JSON coordinate payload."""
    response = client.post(
        "/api/risk/live",
        json={"latitude": 27.3389, "longitude": 88.6065},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["location"]["latitude"] == 27.3389
    assert data["location"]["longitude"] == 88.6065


def test_live_risk_api_invalid_latitude(client):
    """Test validation failure for latitude > 90."""
    response = client.get("/api/risk/live?lat=95.0&lng=91.8660")
    assert response.status_code == 422


def test_live_risk_api_invalid_longitude(client):
    """Test validation failure for longitude > 180."""
    response = client.get("/api/risk/live?lat=27.5861&lng=195.0")
    assert response.status_code == 422


def test_gis_search_endpoint(client):
    """Test GET /api/gis/search returns geocoded results."""
    response = client.get("/api/gis/search?q=Shillong")
    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)
    if len(results) > 0:
        assert "coordinates" in results[0]
        assert len(results[0]["coordinates"]) == 2
