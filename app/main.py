from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import alerts, gis, health, locations, reports, risk, ws
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import setup_logging
from app.ml.model_loader import load_artifacts


from pathlib import Path
from app.core.database import Base, engine
from app.models import alert, report, risk_prediction  # noqa: F401


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_logging()
    load_artifacts()

    # Ensure uploads directory exists
    Path("uploads/reports").mkdir(parents=True, exist_ok=True)

    if settings.auto_create_tables:
        try:
            Base.metadata.create_all(bind=engine)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Table auto-creation failed: %s", e)

    yield


app = FastAPI(
    title="LandGuard Backend",
    description=(
        "Central hub for LandGuard: field reports, PostgreSQL persistence, "
        "Phase 3 LightGBM risk inference, GIS GeoJSON-friendly points, and WebSocket alerts."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.mount(
    "/uploads",
    StaticFiles(directory="uploads"),
    name="uploads",
)

origins = [origin for origin in settings.cors_origin_list if origin != "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)
app.include_router(health.router)
app.include_router(locations.router)
app.include_router(reports.router)
app.include_router(risk.router)
app.include_router(gis.router)
app.include_router(alerts.router)
app.include_router(ws.router)

