from unittest.mock import patch

from tests.conftest import CRITICAL_FEATURES, REPORT_BODY, fake_critical_result


def test_predict_without_features_422(client):
    created = client.post("/api/reports", json=REPORT_BODY).json()
    response = client.post(f"/api/risk/predict/{created['id']}", json={})
    assert response.status_code == 422


def test_predict_and_fetch(client):
    created = client.post("/api/reports", json=REPORT_BODY).json()
    with patch("app.services.risk_service.ModelAdapter.predict", side_effect=fake_critical_result):
        predicted = client.post(
            f"/api/risk/predict/{created['id']}",
            json={"features": CRITICAL_FEATURES},
        )
    assert predicted.status_code == 201
    body = predicted.json()
    assert body["risk_tier"] == "CRITICAL"
    assert body["risk_score"] == 0.91
    assert body["risk_label"] == "CRITICAL"

    latest = client.get(f"/api/risk/{created['id']}")
    assert latest.status_code == 200
    assert latest.json()["id"] == body["id"]

    gis = client.get("/api/gis/risk")
    assert len(gis.json()) == 1
    assert gis.json()[0]["risk_tier"] == "CRITICAL"


def test_create_report_with_features_runs_prediction(client):
    payload = {**REPORT_BODY, "features": CRITICAL_FEATURES}
    with patch("app.services.risk_service.ModelAdapter.predict", side_effect=fake_critical_result):
        created = client.post("/api/reports", json=payload)
    assert created.status_code == 201
    assert created.json()["risk_tier"] == "CRITICAL"
