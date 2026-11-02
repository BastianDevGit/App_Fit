"""API del MVP (FastAPI). Ejecutar: uvicorn app.main:app --reload  → demo en http://localhost:8000"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app import services as sv
from app.catalog.service import UserEquipment, get_catalog
from app.db import (
    DailyReadiness,
    E1RMRecord,
    FoodLog,
    Mesocycle,
    PerformedSetEvent,
    SessionLog,
    User,
    get_session,
    init_db,
)
from app.engine import ENGINE_VERSION
from app.engine.loads import estimate_e1rm, is_valid_for_e1rm
from app.engine.progression import next_set
from app.llm.blocks import BlockRequest, generate_block
from app.llm.provider import TRACER, get_provider
from app.nutrition.pipeline import (
    ParsedItem,
    TargetsInput,
    choose_portion,
    daily_targets,
    parse_meal,
    summarize,
)
from app.safety.classifier import check_text

CONSENT_VERSION = "1.0"
MIN_AGE = 16  # hipótesis: la edad mínima real la fijan los términos (paso 0.7)
STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="Personal Trainer IA — API del MVP", version="0.1.0", lifespan=lifespan)


@app.exception_handler(sv.DomainError)
async def domain_error(_: Request, exc: sv.DomainError):
    return JSONResponse(status_code=exc.status, content={"detail": str(exc)})


DB = Depends(get_session)


# ---------------------------------------------------------------------------
# Esquemas de entrada
# ---------------------------------------------------------------------------

class ConsentIn(BaseModel):
    health: bool
    camera: bool = False
    analytics: bool = False


class ParqIn(BaseModel):
    answers: list[bool] = Field(min_length=7, max_length=7)
    authorized: bool = False


class ProfileIn(BaseModel):
    age: int = Field(ge=10, le=99)
    weight_kg: float = Field(gt=30, lt=300)
    level: Literal["principiante", "intermedio", "avanzado"]
    goal: Literal["fuerza", "masa", "bajar_grasa", "salud"]
    days_per_week: int = Field(ge=1, le=6)
    minutes_per_session: int = Field(ge=15, le=120)
    reminder_time: str | None = Field(default=None, pattern=r"^\d{2}:\d{2}$")
    sex: Literal["f", "m"] | None = None
    height_cm: float | None = Field(default=None, gt=120, lt=230)


class CalibrationSet(BaseModel):
    exercise_id: str
    load: float = Field(gt=0)
    reps: int = Field(ge=1, le=30)
    rir: int = Field(ge=0, le=6)


class CalibrationIn(BaseModel):
    skipped: bool = False
    sets: list[CalibrationSet] = Field(default_factory=list)


class ReadinessIn(BaseModel):
    fatigue: int = Field(default=3, ge=1, le=5)
    sleep: int = Field(default=3, ge=1, le=5)
    soreness: int = Field(default=3, ge=1, le=5)
    stress: int = Field(default=3, ge=1, le=5)
    skipped: bool = False


class SetIn(BaseModel):
    session_key: str
    exercise_id: str
    set_index: int = Field(ge=1, le=20)
    load: float = Field(ge=0)
    reps: int = Field(ge=0, le=100)
    rir: int | None = Field(default=None, ge=0, le=10)
    target_rir: int | None = Field(default=None, ge=0, le=6)
    velocity_drop: float | None = Field(default=None, ge=0, le=1)
    alerts: list[str] = Field(default_factory=list)
    source: Literal["manual", "vision"] = "manual"


class CorrectionIn(BaseModel):
    load: float | None = Field(default=None, ge=0)
    reps: int | None = Field(default=None, ge=0, le=100)
    rir: int | None = Field(default=None, ge=0, le=10)


class SwapIn(BaseModel):
    exercise_id: str
    reason: Literal["maquina_ocupada", "me_molesta"]
    zone: str | None = None
    text: str = ""
    acute: bool = False
    reps: int | None = None
    target_rir: int = 2


class TextIn(BaseModel):
    text: str = Field(max_length=2000)


class PortionIn(BaseModel):
    item: ParsedItem
    unit: str
    quantity: float = Field(default=1, gt=0, le=20)


class MealIn(BaseModel):
    meal: str = "comida"
    items: list[ParsedItem]
    source: Literal["texto", "foto"] = "texto"


# ---------------------------------------------------------------------------
# Onboarding (EP-01)
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"ok": True, "engine_version": ENGINE_VERSION}


@app.post("/api/users", status_code=201)
def create_user(body: ConsentIn, s: Session = DB):
    if not body.health:
        raise HTTPException(422, "Necesitamos el permiso de datos de salud para crear tu plan. Puedes salir o eliminar tu cuenta.")
    user = User(consent_version=CONSENT_VERSION, consent_health=True, consent_camera=body.camera,
                consent_analytics=body.analytics)
    s.add(user)
    s.commit()
    return {"id": user.id, "consent_version": user.consent_version}


@app.post("/api/users/{user_id}/parq")
def submit_parq(user_id: str, body: ParqIn, s: Session = DB):
    user = sv.get_user(s, user_id)
    any_yes = any(body.answers)
    cleared = (not any_yes) or body.authorized
    user.parq = {"answers": body.answers, "authorized": body.authorized, "cleared": cleared, "at": date.today().isoformat()}
    s.commit()
    msg = "" if not any_yes else "Te recomendamos consultar a un profesional de salud antes de empezar."
    return {"cleared": cleared, "any_yes": any_yes, "message": msg}


@app.put("/api/users/{user_id}/profile")
def put_profile(user_id: str, body: ProfileIn, s: Session = DB):
    if body.age < MIN_AGE:
        raise HTTPException(422, f"Debes tener al menos {MIN_AGE} años para usar la app.")
    user = sv.get_user(s, user_id)
    user.profile = body.model_dump()
    s.commit()
    return user.profile


@app.put("/api/users/{user_id}/equipment")
def put_equipment(user_id: str, body: UserEquipment, s: Session = DB):
    user = sv.get_user(s, user_id)
    user.equipment = body.model_dump(mode="json")
    s.commit()
    return {"ok": True, "compatible_exercises": len(get_catalog().compatible(body, (user.profile or {}).get("level", "principiante"),
                                                                                  user.avoid_zones or []))}


@app.post("/api/users/{user_id}/calibration")
def calibration(user_id: str, body: CalibrationIn, s: Session = DB):
    user = sv.get_user(s, user_id)
    if body.skipped:
        user.calibration_skipped = True
        s.commit()
        return {"skipped": True, "message": "Partiremos con cargas conservadoras para tu nivel."}
    out = []
    for c in body.sets:
        if get_catalog().get(c.exercise_id) is None:
            raise HTTPException(422, f"{c.exercise_id} no existe en el catálogo")
        if not is_valid_for_e1rm(c.load, c.reps, c.rir):
            out.append({"exercise_id": c.exercise_id, "e1rm": None, "message": "Serie fuera de rango: no se usa"})
            continue
        value = estimate_e1rm(c.load, c.reps, c.rir)
        s.add(E1RMRecord(user_id=user.id, tenant_id=user.tenant_id, exercise_id=c.exercise_id, value=value,
                         source="calibration", engine_version=ENGINE_VERSION))
        out.append({"exercise_id": c.exercise_id, "e1rm": round(value, 1)})
    s.commit()
    return {"skipped": False, "estimates": out}


# ---------------------------------------------------------------------------
# Bloques (EP-03) y sesión (EP-04)
# ---------------------------------------------------------------------------

@app.post("/api/users/{user_id}/blocks", status_code=201)
def create_block(user_id: str, s: Session = DB):
    user = sv.get_user(s, user_id)
    if not (user.parq or {}).get("cleared"):
        raise HTTPException(409, "Completa el PAR-Q (o confirma la autorización profesional) antes de generar el plan.")
    if not user.profile:
        raise HTTPException(409, "Completa tu perfil antes de generar el plan.")
    eq = sv.equipment_of(user)
    p = user.profile
    req = BlockRequest(days_per_week=p["days_per_week"], minutes_per_session=p["minutes_per_session"],
                       level=p["level"], goal=p["goal"], avoid_zones=user.avoid_zones or [])
    draft, meta = generate_block(req, eq, get_catalog(), get_provider(), user_id=user.id)
    for old in s.scalars(select(Mesocycle).where(Mesocycle.user_id == user.id, Mesocycle.active.is_(True))):
        old.active = False
    meso = Mesocycle(user_id=user.id, tenant_id=user.tenant_id, draft=draft.model_dump(), meta=meta,
                     prompt_version=meta["prompt_version"], engine_version=ENGINE_VERSION)
    s.add(meso)
    s.commit()
    cat = get_catalog()
    return {"id": meso.id, "source": meta["source"], "attempts": meta["attempts"], "weeks": draft.weeks,
            "rationale": draft.rationale,
            "days": [{"name": d.name, "focus": d.focus,
                      "exercises": [{**e.model_dump(), "name": cat.get(e.exercise_id).name} for e in d.exercises]}
                     for d in draft.days]}


@app.post("/api/users/{user_id}/readiness")
def post_readiness(user_id: str, body: ReadinessIn, s: Session = DB):
    user = sv.get_user(s, user_id)
    s.add(DailyReadiness(user_id=user.id, tenant_id=user.tenant_id, day=date.today(), **body.model_dump()))
    s.commit()
    return {"score": sv.readiness_for(s, user.id, date.today())}


@app.get("/api/users/{user_id}/sessions/today")
def today(user_id: str, week: int = 1, day: int = 0, hard_day: bool = False, s: Session = DB):
    return sv.plan_session(s, sv.get_user(s, user_id), week, day, hard_day)


@app.post("/api/users/{user_id}/sets", status_code=201)
def log_set(user_id: str, body: SetIn, s: Session = DB):
    user = sv.get_user(s, user_id)
    cat = get_catalog()
    ex = cat.get(body.exercise_id)
    if ex is None:
        raise HTTPException(422, "Ejercicio fuera del catálogo")
    meso = sv.active_mesocycle(s, user.id)
    ev = PerformedSetEvent(user_id=user.id, tenant_id=user.tenant_id, engine_version=ENGINE_VERSION,
                           prompt_version=meso.prompt_version if meso else None, cv_config_version=sv.CV_CONFIG_VERSION,
                           **body.model_dump(exclude={"target_rir"}), target_rir=body.target_rir)
    s.add(ev)
    s.commit()
    resp: dict = {"event_id": ev.id}
    if body.rir is not None and body.target_rir is not None:
        d = next_set(body.load, body.target_rir, body.rir, cat.options(ex, sv.equipment_of(user)), body.velocity_drop)
        resp["next_set"] = {"load": d.load, "reason": d.reason.value, "explanation": d.explanation, "flags": list(d.flags),
                            "engine_version": d.engine_version}
    return resp


@app.post("/api/users/{user_id}/sets/{event_id}/corrections", status_code=201)
def correct_set(user_id: str, event_id: str, body: CorrectionIn, s: Session = DB):
    user = sv.get_user(s, user_id)
    original = s.get(PerformedSetEvent, event_id)
    if original is None or original.user_id != user.id or original.kind != "original":
        raise HTTPException(404, "Serie no encontrada")
    fix = PerformedSetEvent(
        user_id=user.id, tenant_id=user.tenant_id, session_key=original.session_key, exercise_id=original.exercise_id,
        set_index=original.set_index, load=body.load if body.load is not None else original.load,
        reps=body.reps if body.reps is not None else original.reps, rir=body.rir if body.rir is not None else original.rir,
        target_rir=original.target_rir, velocity_drop=original.velocity_drop, alerts=original.alerts, source=original.source,
        kind="correction", corrects_event_id=original.id, engine_version=ENGINE_VERSION)
    s.add(fix)
    s.commit()
    return {"event_id": fix.id, "corrects": original.id}


@app.post("/api/users/{user_id}/sessions/{key}/complete")
def complete(user_id: str, key: str, short: bool = False, s: Session = DB):
    return sv.complete_session(s, sv.get_user(s, user_id), key, short)


@app.post("/api/users/{user_id}/swap")
def swap(user_id: str, body: SwapIn, s: Session = DB):
    """Máquina ocupada (HU-035) y me molesta (HU-036, con capa de seguridad HU-025)."""
    user = sv.get_user(s, user_id)
    cat = get_catalog()
    eq = sv.equipment_of(user)
    level = (user.profile or {}).get("level", "principiante")
    zones = list(user.avoid_zones or [])
    if body.reason == "me_molesta":
        safety = check_text(body.text, acute_selected=body.acute)
        if safety.action == "derivar":
            return {"action": "derivar", "message": safety.message, "matched": safety.matched}
        if body.zone and body.zone not in zones:
            zones.append(body.zone)
            user.avoid_zones = zones
            s.commit()
    alts = cat.alternatives(body.exercise_id, eq, level, zones)
    out = []
    for ex in alts[:3]:
        options = cat.options(ex, eq)
        reps = body.reps or min(ex.rep_range[1], ex.rep_range[0] + 2)
        if options.is_bodyweight or ex.method != "carga_rir":
            load = 0.0 if options.is_bodyweight else options.minimum
        else:
            load = sv.suggested_load(s, user, ex, options, reps, body.target_rir)[0]
        out.append({"exercise_id": ex.id, "name": ex.name, "load": load,
                    "load_label": sv._load_label(load, cat.load_implement(ex, eq), single=ex.units == 1),
                    "reps": reps})
    return {"action": "cambiar", "alternatives": out, "avoid_zones": zones}


@app.get("/api/users/{user_id}/progress")
def get_progress(user_id: str, s: Session = DB):
    return sv.progress(s, sv.get_user(s, user_id))


# ---------------------------------------------------------------------------
# Dieta (EP-06) y seguridad (A5)
# ---------------------------------------------------------------------------

@app.post("/api/nutrition/parse")
def nutrition_parse(body: TextIn):
    result = parse_meal(body.text, get_provider())
    if not result.items:
        return {**result.model_dump(), "message": "No reconocimos alimentos. Prueba describiéndolo de otra forma."}
    return result


@app.post("/api/nutrition/portion")
def nutrition_portion(body: PortionIn):
    try:
        return choose_portion(body.item, body.unit, body.quantity)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/users/{user_id}/meals", status_code=201)
def save_meal(user_id: str, body: MealIn, s: Session = DB):
    user = sv.get_user(s, user_id)
    summary = summarize(body.items)
    if not summary.complete:
        raise HTTPException(422, "Confirma la porción de todos los alimentos antes de guardar.")
    s.add(FoodLog(user_id=user.id, tenant_id=user.tenant_id, day=date.today(), meal=body.meal,
                  items=[i.model_dump() for i in body.items], totals=summary.totals.model_dump(), source=body.source))
    s.commit()
    return sv.food_day(s, user.id, date.today())


@app.get("/api/users/{user_id}/nutrition/today")
def nutrition_today(user_id: str, s: Session = DB):
    user = sv.get_user(s, user_id)
    p = user.profile or {}
    targets = None
    if p:
        targets = daily_targets(TargetsInput(weight_kg=p["weight_kg"], goal=p["goal"], sex=p.get("sex"),
                                             height_cm=p.get("height_cm"), age=p.get("age"),
                                             training_days=p.get("days_per_week", 3)))
    return {**sv.food_day(s, user.id, date.today()), "targets": targets}


@app.post("/api/safety/check")
def safety_check(body: TextIn):
    return check_text(body.text)


# ---------------------------------------------------------------------------
# Derechos del titular (HU-093, HU-094) y operación
# ---------------------------------------------------------------------------

USER_TABLES = [E1RMRecord, Mesocycle, PerformedSetEvent, SessionLog, DailyReadiness, FoodLog]


@app.get("/api/users/{user_id}/export")
def export(user_id: str, s: Session = DB):
    user = sv.get_user(s, user_id)
    data = {"user": {c.name: getattr(user, c.name) for c in User.__table__.columns}}
    for model in USER_TABLES:
        rows = s.scalars(select(model).where(model.user_id == user.id))
        data[model.__tablename__] = [{c.name: getattr(r, c.name) for c in model.__table__.columns} for r in rows]
    return JSONResponse(content=_jsonable(data))


@app.delete("/api/users/{user_id}", status_code=204)
def delete_user(user_id: str, s: Session = DB):
    user = sv.get_user(s, user_id)
    for model in USER_TABLES:
        s.execute(delete(model).where(model.user_id == user.id))
    s.delete(user)
    s.commit()


@app.get("/api/ops/llm-traces")
def llm_traces(limit: int = 20):
    return [t.__dict__ for t in TRACER.traces[-limit:]]


@app.get("/api/catalog/exercises")
def catalog():
    cat = get_catalog()
    return {"meta": cat.meta, "exercises": [e.model_dump() for e in cat.all]}


@app.get("/")
def demo():
    return FileResponse(STATIC / "index.html")


def _jsonable(x):
    from datetime import date as _d
    from datetime import datetime as _dt
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    if isinstance(x, (_dt, _d)):
        return x.isoformat()
    return x

