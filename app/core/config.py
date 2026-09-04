from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

RiskTier = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = Field(
        default="postgresql+psycopg://landguard:landguard@localhost:5432/landguard"
    )
    auto_create_tables: bool = Field(default=True)
    ml_model_path: str = "./ml_artifacts/landslide_xgboost_model.pkl"
    ml_preprocessor_path: str = "./ml_artifacts/preprocessor.pkl"
    ml_model_version: str = "lgbm-phase3-v1"
    environment: str = "development"
    cors_origins: str = (
        "http://localhost:3000,http://localhost:5173,https://landguard-eight.vercel.app"
    )
    alert_min_tier: RiskTier = "CRITICAL"
    log_level: str = "INFO"

    @field_validator("alert_min_tier", mode="before")
    @classmethod
    def normalize_tier(cls, value: str) -> str:
        return str(value).upper()

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
