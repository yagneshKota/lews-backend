from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings


class Base(DeclarativeBase):
    pass


import logging

logger = logging.getLogger(__name__)

connect_args: dict = {}
engine_kwargs: dict = {"pool_pre_ping": True}
if settings.is_sqlite:
    connect_args = {"check_same_thread": False}
    if settings.database_url.endswith(":memory:"):
        engine_kwargs["poolclass"] = StaticPool
else:
    connect_args = {"connect_timeout": 10}

try:
    engine = create_engine(
        settings.database_url,
        connect_args=connect_args,
        **engine_kwargs,
    )
    # Test connection
    with engine.connect() as conn:
        logger.info("Successfully connected to database at %s", settings.database_url.split('@')[-1] if '@' in settings.database_url else settings.database_url)
except Exception as exc:
    logger.warning(
        "Could not connect to PostgreSQL at %s (%s). Falling back to local SQLite database: sqlite:///./landguard.db",
        settings.database_url,
        exc,
    )
    engine = create_engine(
        "sqlite:///./landguard.db",
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
