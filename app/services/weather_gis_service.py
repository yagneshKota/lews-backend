"""
Open-Source Real-Time Weather & Open GIS Service (Open-Meteo & Open GIS)
Delegates to LiveFeatureService to provide unified, dynamic real-time telemetry.
"""

from typing import Any
from app.services.live_feature_service import LiveFeatureService


class WeatherGisService:
    @staticmethod
    async def get_realtime_telemetry(
        latitude: float,
        longitude: float,
        fallback_elevation: float = 2000.0,
        fallback_slope: float = 30.0,
    ) -> dict[str, Any]:
        """Fetch live real-time open-source weather and GIS telemetry."""
        result = await LiveFeatureService.get_live_risk_for_coordinate(latitude, longitude)
        env = result.get("environmental", {})
        feat = result.get("features", {})

        return {
            "source": result.get("data_sources", {}).get("weather", "Open-Meteo Free API & Open GIS"),
            "latitude": latitude,
            "longitude": longitude,
            "rainfall_24h": env.get("rainfall_24h", 0.0),
            "rainfall_3d": env.get("rainfall_3d", 0.0),
            "rainfall_7d": env.get("rainfall_7d", 0.0),
            "soil_moisture": env.get("soil_moisture", 0.52),
            "temperature": env.get("temperature", 18.0),
            "humidity": env.get("humidity", 82.0),
            "wind_speed": env.get("wind_speed", 14.0),
            "elevation_m": feat.get("elevation_m", fallback_elevation),
            "slope_degrees": feat.get("slope_degrees", fallback_slope),
            "aspect_degrees": feat.get("aspect_degrees", 135.0),
            "timestamp": result.get("data_timestamp"),
            "data_status": result.get("data_status", "LIVE"),
        }
