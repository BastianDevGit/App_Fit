"""Persistencia (SQLAlchemy 2). SQLite en desarrollo; PostgreSQL en staging/prod vía DATABASE_URL.

Diseño multi-tenant desde el inicio (A7): toda tabla de datos de usuario lleva tenant_id.
En PostgreSQL se activa Row Level Security con sql/rls.sql. Las series ejecutadas son
append-only (A1): las correcciones son eventos nuevos, nunca UPDATE.
"""

from __future__ import annotations

import os
import uuid
from datetime import date, datetime, timezone

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, ForeignKey, Integer, String, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dev.db")
DEFAULT_TENANT = "b2c"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class TenantMixin:
    tenant_id: Mapped[str] = mapped_column(String(40), default=DEFAULT_TENANT, index=True)


class User(TenantMixin, Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    role: Mapped[str] = mapped_column(String(20), default="usuario")  # usuario | coach | admin_gimnasio
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    consent_version: Mapped[str] = mapped_column(String(20))
    consent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    consent_health: Mapped[bool] = mapped_column(Boolean)
    consent_camera: Mapped[bool] = mapped_column(Boolean, default=False)
    consent_analytics: Mapped[bool] = mapped_column(Boolean, default=False)
    consent_food_photo: Mapped[bool] = mapped_column(Boolean, default=False)
    parq: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    profile: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    equipment: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    avoid_zones: Mapped[list] = mapped_column(JSON, default=list)
    calibration_skipped: Mapped[bool] = mapped_column(Boolean, default=False)


class E1RMRecord(TenantMixin, Base):
    __tablename__ = "e1rm_history"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    exercise_id: Mapped[str] = mapped_column(String(20), index=True)
    value: Mapped[float] = mapped_column(Float)
    raw_session_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(20))  # calibration | session
    session_key: Mapped[str | None] = mapped_column(String(80), nullable=True)
    engine_version: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Mesocycle(TenantMixin, Base):
    __tablename__ = "mesocycles"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    draft: Mapped[dict] = mapped_column(JSON)
    meta: Mapped[dict] = mapped_column(JSON)
    prompt_version: Mapped[str] = mapped_column(String(40))
    engine_version: Mapped[str] = mapped_column(String(20))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class PerformedSetEvent(TenantMixin, Base):
    __tablename__ = "performed_set_events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    session_key: Mapped[str] = mapped_column(String(80), index=True)
    exercise_id: Mapped[str] = mapped_column(String(20), index=True)
    set_index: Mapped[int] = mapped_column(Integer)
    load: Mapped[float] = mapped_column(Float)
    reps: Mapped[int] = mapped_column(Integer)
    rir: Mapped[int | None] = mapped_column(Integer, nullable=True)
    target_rir: Mapped[int | None] = mapped_column(Integer, nullable=True)
    velocity_drop: Mapped[float | None] = mapped_column(Float, nullable=True)
    alerts: Mapped[list] = mapped_column(JSON, default=list)
    source: Mapped[str] = mapped_column(String(10), default="manual")  # manual | vision
    kind: Mapped[str] = mapped_column(String(12), default="original")  # original | correction
    corrects_event_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    engine_version: Mapped[str] = mapped_column(String(20))
    prompt_version: Mapped[str | None] = mapped_column(String(40), nullable=True)
    cv_config_version: Mapped[str | None] = mapped_column(String(40), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class SessionLog(TenantMixin, Base):
    __tablename__ = "session_logs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    session_key: Mapped[str] = mapped_column(String(80), index=True)
    short: Mapped[bool] = mapped_column(Boolean, default=False)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class DailyReadiness(TenantMixin, Base):
    __tablename__ = "daily_readiness"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    day: Mapped[date] = mapped_column(Date, index=True)
    fatigue: Mapped[int] = mapped_column(Integer)
    sleep: Mapped[int] = mapped_column(Integer)
    soreness: Mapped[int] = mapped_column(Integer)
    stress: Mapped[int] = mapped_column(Integer)
    skipped: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class FoodLog(TenantMixin, Base):
    __tablename__ = "food_logs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    day: Mapped[date] = mapped_column(Date, index=True)
    meal: Mapped[str] = mapped_column(String(40))
    items: Mapped[list] = mapped_column(JSON)
    totals: Mapped[dict] = mapped_column(JSON)
    source: Mapped[str] = mapped_column(String(10), default="texto")  # texto | foto
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


@event.listens_for(PerformedSetEvent, "before_update")
def _forbid_update(mapper, connection, target):  # noqa: ARG001
    raise RuntimeError("performed_set_events es append-only: registra una corrección en vez de editar (A1)")


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_session():
    with SessionLocal() as s:
        yield s


__all__ = ["Session", "init_db", "get_session"]
