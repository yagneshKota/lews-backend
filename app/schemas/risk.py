from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RiskFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    elevation_m: float = Field(ge=0)
    slope_degrees: float = Field(ge=0, le=90)
    aspect_degrees: float = Field(ge=0, le=360)

    rainfall_1d_before: float = Field(ge=0)
    rainfall_3d_before: float = Field(ge=0)
    rainfall_7d_before: float = Field(ge=0)
    rainfall_14d_before: float = Field(ge=0)
    rainfall_30d_before: float = Field(ge=0)

    rainfall_7d_max1d: float = Field(ge=0)
    rainfall_3d_over_7d_ratio: float = Field(ge=0)

    soil_moisture: float = Field(ge=0, le=1)
    soil_moisture_available: int = Field(ge=0, le=1)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    features: RiskFeatures | None = None

    @model_validator(mode="after")
    def validate_features_present(self):
        dump = self.model_dump()
        feat = self.features.model_dump() if self.features else dump
        # If no core features are present, raise ValueError so FastAPI returns 422
        if not feat or "elevation_m" not in feat:
            raise ValueError("Features are required for risk prediction")
        return self

    def get_features(self) -> dict[str, Any]:
        if self.features is not None:
            return self.features.model_dump()
        dump = self.model_dump()
        dump.pop("features", None)
        return dump


class RiskPredictionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    report_id: UUID
    risk_score: float
    risk_level: int
    risk_tier: str
    risk_label: str | None = None
    alert_triggered: bool
    alert_message: str
    created_at: datetime

    @model_validator(mode="after")
    def populate_risk_label(self):
        if not self.risk_label:
            self.risk_label = self.risk_tier
        return self


class RiskResult(BaseModel):
    risk_score: float
    risk_level: int
    risk_tier: str
    alert_triggered: bool
    alert_message: str