"""
Open-Source Real-Time Weather & Open GIS Service (Open-Meteo & Open GIS)
Delegates to LiveFeatureService. Never invents environmental fallbacks.
"""

from typing import Any

from app.services.live_feature_service import LiveFeatureService, LiveTelemetryUnavailableError


class WeatherGisService:
    @staticmethod
    async def get_realtime_telemetry(
        latitude: float,
        longitude: float,
    ) -> dict[str, Any]:
        """Fetch live real-time open-source weather and GIS telemetry."""
        result = await LiveFeatureService.get_live_risk_for_coordinate(latitude, longitude)
        env = result.get("environmental") or {}
        feat = result.get("features") or {}
        status = result.get("data_status", "UNAVAILABLE")
        if status == "UNAVAILABLE" or not env or not feat:
            raise LiveTelemetryUnavailableError(
                result.get("message") or "Live environmental telemetry unavailable",
                missing_source="open-meteo",
            )

        return {
            "source": result.get("data_sources", {}).get("weather", "Open-Meteo Free API & Open GIS"),
            "latitude": latitude,
            "longitude": longitude,
            "rainfall_24h": env.get("rainfall_24h"),
            "rainfall_3d": env.get("rainfall_3d"),
            "rainfall_7d": env.get("rainfall_7d"),
            "soil_moisture": env.get("soil_moisture"),
            "soil_moisture_available": env.get("soil_moisture_available", 0),
            "temperature": env.get("temperature"),
            "humidity": env.get("humidity"),
            "wind_speed": env.get("wind_speed"),
            "elevation_m": feat.get("elevation_m"),
            "slope_degrees": feat.get("slope_degrees"),
            "aspect_degrees": feat.get("aspect_degrees"),
            "timestamp": result.get("data_timestamp"),
            "data_status": status,
            "data_age_seconds": result.get("data_age_seconds", 0),
            "message": result.get("message"),
        }
