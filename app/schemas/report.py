from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ReportCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    longitude: float = Field(ge=-180, le=180)
    latitude: float = Field(ge=-90, le=90)
    report: str = Field(min_length=1, max_length=255)
    report_description: str = Field(min_length=1, max_length=5000)
    features: dict[str, Any] | None = None


class ReportResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    longitude: float
    latitude: float
    report: str
    report_description: str
    created_at: datetime
    updated_at: datetime
    image_url: str | None = None
    risk_score: float | None = None
    risk_tier: str | None = None
    risk_label: str | None = None


class PaginatedReportsResponse(BaseModel):
    items: list[ReportResponse]
    total: int
    offset: int
    limit: int