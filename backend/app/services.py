"""Casos de uso: une catálogo, motor, LLM y persistencia. Sin lógica de progresión propia."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.catalog.service import Exercise, UserEquipment, get_catalog
from app.db import DailyReadiness, E1RMRecord, FoodLog, Mesocycle, PerformedSetEvent, SessionLog, User
from app.engine import ENGINE_VERSION
from app.engine.loads import LoadOptions, load_for_target, plates_per_side
from app.engine.progression import (
    PerformedSet,
    apply_deload,
    apply_readiness,
    decision_table,
    deload_reason,
    double_progression,
    next_session_load,
    readiness_score,
    reentry_factor,
    session_e1rm,
    short_session,
    update_e1rm,
    warmup_sets,
)
from app.engine.reasons import Reason, explain
from app.llm.blocks import MesocycleDraft

LEVEL_FACTOR = {"principiante": 0.6, "intermedio": 0.8, "avanzado": 0.9}
CV_CONFIG_VERSION = None  # la visión llega en la Fase 1/6


class DomainError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# Lecturas
# ---------------------------------------------------------------------------

def get_user(s: Session, user_id: str) -> User:
    user = s.get(User, user_id)
    if user is None:
        raise DomainError("Usuario no encontrado", 404)
    return user


def equipment_of(user: User) -> UserEquipment:
    if not user.equipment:
        raise DomainError("Faltan los implementos del usuario (HU-005)")
    return UserEquipment(**user.equipment)


def active_mesocycle(s: Session, user_id: str) -> Mesocycle | None:
    return s.scalars(select(Mesocycle).where(Mesocycle.user_id == user_id, Mesocycle.active.is_(True))
                     .order_by(Mesocycle.created_at.desc())).first()


def effective_events(s: Session, user_id: str, exercise_id: str | None = None) -> list[PerformedSetEvent]:
    """Series vigentes: cada original reemplazada por su última corrección (A1)."""
    q = select(PerformedSetEvent).where(PerformedSetEvent.user_id == user_id)
    if exercise_id:
        q = q.where(PerformedSetEvent.exercise_id == exercise_id)
    events = list(s.scalars(q.order_by(PerformedSetEvent.created_at)))
    latest_fix: dict[str, PerformedSetEvent] = {}
    for e in events:
        if e.kind == "correction" and e.corrects_event_id:
            latest_fix[e.corrects_event_id] = e
    return [latest_fix.get(e.id, e) for e in events if e.kind == "original"]


def last_session_sets(s: Session, user_id: str, exercise_id: str) -> list[PerformedSetEvent]:
    evs = effective_events(s, user_id, exercise_id)
    if not evs:
        return []
    last_key = max(evs, key=lambda e: e.created_at).session_key
    return sorted([e for e in evs if e.session_key == last_key], key=lambda e: e.set_index)


def current_e1rm(s: Session, user_id: str, exercise_id: str) -> E1RMRecord | None:
    return s.scalars(select(E1RMRecord).where(E1RMRecord.user_id == user_id, E1RMRecord.exercise_id == exercise_id)
                     .order_by(E1RMRecord.created_at.desc())).first()


def e1rm_series(s: Session, user_id: str) -> dict[str, list[E1RMRecord]]:
    out: dict[str, list[E1RMRecord]] = defaultdict(list)
    for r in s.scalars(select(E1RMRecord).where(E1RMRecord.user_id == user_id).order_by(E1RMRecord.created_at)):
        out[r.exercise_id].append(r)
    return out


def readiness_for(s: Session, user_id: str, day: date) -> float | None:
    r = s.scalars(select(DailyReadiness).where(DailyReadiness.user_id == user_id, DailyReadiness.day == day)
                  .order_by(DailyReadiness.created_at.desc())).first()
    return None if r is None or r.skipped else readiness_score(r.fatigue, r.sleep, r.soreness, r.stress)


def recent_readiness(s: Session, user_id: str, days: int = 7) -> list[float]:
    since = date.today() - timedelta(days=days)
    rows = s.scalars(select(DailyReadiness).where(DailyReadiness.user_id == user_id, DailyReadiness.day >= since,
                                                  DailyReadiness.skipped.is_(False)).order_by(DailyReadiness.day))
    return [readiness_score(r.fatigue, r.sleep, r.soreness, r.stress) for r in rows]


# ---------------------------------------------------------------------------
# Cargas iniciales
# ---------------------------------------------------------------------------

def conservative_e1rm(ex: Exercise, user: User) -> float | None:
    if not ex.conservative_ratio or not user.profile:
        return None
    return user.profile["weight_kg"] * ex.conservative_ratio * LEVEL_FACTOR.get(user.profile["level"], 0.6)


def suggested_load(s: Session, user: User, ex: Exercise, options: LoadOptions, reps: int, rir: int) -> tuple[float, Reason, str]:
    """Carga para la próxima sesión de un ejercicio con carga (HU-006, HU-011, HU-014)."""
    rec = current_e1rm(s, user.id, ex.id)
    last = last_session_sets(s, user.id, ex.id)
    prev_load = max((e.load for e in last), default=None)
    if rec is None:
        e1rm = conservative_e1rm(ex, user)
        if e1rm is None:
            return options.minimum, Reason.INITIAL_CONSERVATIVE, explain(Reason.INITIAL_CONSERVATIVE)
        return load_for_target(e1rm, reps, rir, options), Reason.INITIAL_CONSERVATIVE, explain(Reason.INITIAL_CONSERVATIVE)
    if prev_load is None:
        reason = Reason.INITIAL_FROM_CALIBRATION if rec.source == "calibration" else Reason.ON_TARGET
        return load_for_target(rec.value, reps, rir, options), reason, explain(reason)
    d = next_session_load(rec.value, reps, rir, options, prev_load)
    return d.load, d.reason, d.explanation


# ---------------------------------------------------------------------------
# Planificación de la sesión del día
# ---------------------------------------------------------------------------

def session_key(meso_id: str, week: int, day_index: int) -> str:
    return f"{meso_id[:8]}-w{week}-d{day_index}"


def plan_session(s: Session, user: User, week: int, day_index: int, hard_day: bool = False) -> dict:
    meso = active_mesocycle(s, user.id)
    if meso is None:
        raise DomainError("No hay un bloque activo. Genera uno primero.", 409)
    cat = get_catalog()
    eq = equipment_of(user)
    draft = MesocycleDraft(**meso.draft)
    day = draft.days[day_index % len(draft.days)]
    today_score = readiness_for(s, user.id, date.today())
    series = {k: [r.value for r in v if r.source == "session"] for k, v in e1rm_series(s, user.id).items()}
    deload = deload_reason(week, draft.weeks, series, recent_readiness(s, user.id))
    last_log = s.scalars(select(SessionLog).where(SessionLog.user_id == user.id).order_by(SessionLog.completed_at.desc())).first()
    weeks_off = ((datetime.now(timezone.utc) - _aware(last_log.completed_at)).days / 7) if last_log else 0
    reentry = reentry_factor(weeks_off)

    exercises = []
    for item in sorted(day.exercises, key=lambda e: e.order):
        ex = cat.get(item.exercise_id)
        if ex is None:
            continue
        options = cat.options(ex, eq)
        implement = cat.load_implement(ex, eq)
        sets, reps, rir = item.sets, item.reps, item.target_rir
        notes: list[dict] = []
        if ex.method in ("doble_progresion", "variante", "tiempo") or options.is_bodyweight:
            last = last_session_sets(s, user.id, ex.id)
            load = max((e.load for e in last), default=None)
            if load is None:
                load = 0.0 if options.is_bodyweight else suggested_load(s, user, ex, options, reps, rir)[0]
            if last and ex.method == "doble_progresion":
                d = double_progression(load, [e.reps for e in last], [e.rir or 0 for e in last], ex.rep_range, rir, options)
                load, reps = d.load, d.reps or reps
                notes.append({"reason": d.reason.value, "text": d.explanation})
        else:
            load, reason, text = suggested_load(s, user, ex, options, reps, rir)
            notes.append({"reason": reason.value, "text": text})
        if reentry < 1 and not options.is_bodyweight:
            load = options.floor(load * reentry)
            notes.append({"reason": Reason.REENTRY_AFTER_BREAK.value, "text": explain(Reason.REENTRY_AFTER_BREAK)})
        if deload:
            sets, load = apply_deload(sets, load, options)
            notes.append({"reason": deload.value, "text": explain(deload)})
        r = apply_readiness(sets, load, today_score, options)
        if r.reason is not Reason.ON_TARGET:
            sets, load = r.sets or sets, r.load
            notes.append({"reason": r.reason.value, "text": r.explanation})
        bar = eq.barra.bar_kg if implement == "barra" and eq.barra else None
        plates = None
        if bar and eq.barra:
            res = plates_per_side(load, bar, eq.barra.plates)
            plates = {"bar_kg": bar, "per_side": res[1]} if res else None
        table = decision_table(load, rir, options)
        exercises.append({
            "exercise_id": ex.id, "name": ex.name, "type": ex.type, "method": ex.method, "order": item.order,
            "implement": implement, "sets": sets, "reps": reps, "target_rir": rir,
            "unit": "s" if ex.method == "tiempo" else "reps", "load": load,
            "load_label": _load_label(load, implement, single=ex.units == 1), "vision_cfg": ex.vision_cfg,
            "warmup": [{"load": w, "reps": n} for w, n in warmup_sets(load, options, bar)] if ex.type == "compuesto" else [],
            "plates": plates, "notes": notes,
            "decision_table": {str(k): {"load": v.load, "reason": v.reason.value, "text": v.explanation} for k, v in table.items()},
            "counts_for_progression": True,
        })
    if hard_day:
        exercises = short_session(exercises)
    minutes = 5 + sum(e["sets"] * (2.5 if e["type"] == "compuesto" else 1.5) for e in exercises)
    return {
        "session_key": session_key(meso.id, week, day_index), "mesocycle_id": meso.id, "week": week,
        "day": day.name, "focus": day.focus, "short": hard_day, "deload": deload.value if deload else None,
        "readiness_score": today_score, "estimated_minutes": round(minutes),
        "engine_version": ENGINE_VERSION, "prompt_version": meso.prompt_version, "cv_config_version": CV_CONFIG_VERSION,
        "exercises": exercises,
    }


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _load_label(load: float, implement: str | None, single: bool = False) -> str:
    if not implement or load <= 0:
        return "peso corporal"
    n = f"{load:g}".replace(".", ",")
    return f"2 × {n} kg" if implement in {"mancuernas", "kettlebell"} and not single else f"{n} kg"


# ---------------------------------------------------------------------------
# Cierre de sesión, progreso y adherencia
# ---------------------------------------------------------------------------

def complete_session(s: Session, user: User, key: str, short: bool) -> dict:
    events = [e for e in effective_events(s, user.id) if e.session_key == key]
    if not events:
        raise DomainError("La sesión no tiene series registradas")
    by_ex: dict[str, list[PerformedSetEvent]] = defaultdict(list)
    for e in events:
        by_ex[e.exercise_id].append(e)
    records, prs = [], []
    if not short:  # la versión corta no penaliza ni empuja la progresión (HU-034)
        series = e1rm_series(s, user.id)
        for ex_id, evs in by_ex.items():
            value = session_e1rm([PerformedSet(e.load, e.reps, e.rir) for e in evs])
            if value is None:
                continue
            prev = current_e1rm(s, user.id, ex_id)
            new = update_e1rm(prev.value if prev else None, value)
            best_before = max((r.value for r in series.get(ex_id, [])), default=0)
            s.add(E1RMRecord(user_id=user.id, tenant_id=user.tenant_id, exercise_id=ex_id, value=new,
                             raw_session_value=value, source="session", session_key=key, engine_version=ENGINE_VERSION))
            records.append({"exercise_id": ex_id, "e1rm": round(new, 1)})
            if series.get(ex_id) and new > best_before + 1e-6:
                prs.append({"exercise_id": ex_id, "name": get_catalog().get(ex_id).name, "e1rm": round(new, 1),
                            "delta": round(new - best_before, 1)})
    s.add(SessionLog(user_id=user.id, tenant_id=user.tenant_id, session_key=key, short=short))
    s.commit()
    return {"session_key": key, "e1rm_updates": records, "personal_records": prs}


def streak(s: Session, user: User) -> dict:
    """Racha semanal con días de gracia (HU-071): cuenta semanas que cumplen la meta, no días."""
    goal = int((user.profile or {}).get("days_per_week", 3))
    weeks: dict[tuple[int, int], int] = defaultdict(int)
    for log in s.scalars(select(SessionLog).where(SessionLog.user_id == user.id)):
        iso = _aware(log.completed_at).isocalendar()
        weeks[(iso[0], iso[1])] += 1
    today = date.today().isocalendar()
    current = (today[0], today[1])
    count, cursor = 0, date.today() - timedelta(days=7)
    while True:
        iso = cursor.isocalendar()
        if weeks.get((iso[0], iso[1]), 0) >= goal:
            count += 1
            cursor -= timedelta(days=7)
        else:
            break
    if weeks.get(current, 0) >= goal:
        count += 1
    return {"weeks": count, "goal_per_week": goal, "this_week": weeks.get(current, 0)}


def progress(s: Session, user: User) -> dict:
    cat = get_catalog()
    series = e1rm_series(s, user.id)
    out = []
    for ex_id, recs in series.items():
        ex = cat.get(ex_id)
        out.append({"exercise_id": ex_id, "name": ex.name if ex else ex_id,
                    "points": [{"at": _aware(r.created_at).isoformat(), "e1rm": round(r.value, 1), "source": r.source} for r in recs],
                    "best": round(max(r.value for r in recs), 1)})
    return {"streak": streak(s, user), "e1rm": out, "note": "Solo te comparamos contigo mismo."}


def food_day(s: Session, user_id: str, day: date) -> dict:
    logs = list(s.scalars(select(FoodLog).where(FoodLog.user_id == user_id, FoodLog.day == day)))
    total = {"kcal": 0.0, "p": 0.0, "c": 0.0, "f": 0.0}
    for log in logs:
        for k in total:
            total[k] += float(log.totals.get(k, 0))
    return {"day": day.isoformat(), "meals": [{"id": g.id, "meal": g.meal, "totals": g.totals, "source": g.source} for g in logs],
            "totals": {k: round(v, 1) for k, v in total.items()}}
