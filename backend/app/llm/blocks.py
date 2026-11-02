"""Generación de bloques con LLM restringido al catálogo cerrado (HU-021, HU-022, HU-023).

Flujo: contexto filtrado → LLM → validación con Pydantic + reglas → reintento con errores →
plantilla base si falla todo. El LLM nunca entrega cargas; las completa el motor.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.catalog.service import Catalog, UserEquipment

from .provider import LLMProvider, _estimate_minutes, call_traced

PROMPT_VERSION = "blocks-0.1"
MAX_RETRIES = 2

PROMPT = """Eres un entrenador que arma bloques de entrenamiento.
Usa SOLO ejercicios de `allowed_exercises` (por ID). No inventes IDs. No indiques cargas en kg.
Respeta days_per_week, minutes_per_session y evita las zonas en avoid_zones.
Responde solo JSON con: weeks (3-6), days[{name, focus, exercises[{exercise_id, sets, reps, target_rir, order}]}], rationale.
"""


class PlannedExercise(BaseModel):
    exercise_id: str
    sets: int = Field(ge=1, le=6)
    reps: int = Field(ge=1, le=60)
    target_rir: int = Field(ge=0, le=4)
    order: int = Field(ge=1)


class DayPlan(BaseModel):
    name: str
    focus: str = ""
    exercises: list[PlannedExercise] = Field(min_length=2, max_length=8)

    @field_validator("exercises")
    @classmethod
    def no_duplicates(cls, v: list[PlannedExercise]) -> list[PlannedExercise]:
        ids = [e.exercise_id for e in v]
        if len(ids) != len(set(ids)):
            raise ValueError("ejercicio repetido en el mismo día")
        return v


class MesocycleDraft(BaseModel):
    weeks: int = Field(ge=3, le=6)
    days: list[DayPlan] = Field(min_length=1, max_length=6)
    rationale: str = ""


class BlockRequest(BaseModel):
    days_per_week: int = Field(ge=1, le=6)
    minutes_per_session: int = Field(ge=15, le=120)
    level: str
    goal: str
    avoid_zones: list[str] = Field(default_factory=list)


def build_context(req: BlockRequest, eq: UserEquipment, catalog: Catalog) -> dict[str, Any]:
    allowed = catalog.compatible(eq, req.level, req.avoid_zones)
    return {
        "days_per_week": req.days_per_week,
        "minutes_per_session": req.minutes_per_session,
        "level": req.level,
        "goal": req.goal,
        "avoid_zones": req.avoid_zones,
        "allowed_exercises": [
            {"id": e.id, "name": e.name, "pattern": e.pattern, "type": e.type, "rep_range": list(e.rep_range),
             "method": e.method}
            for e in allowed
        ],
    }


def validate_draft(raw: str, req: BlockRequest, eq: UserEquipment, catalog: Catalog) -> tuple[MesocycleDraft | None, list[str]]:
    """Valida forma (schema) y fondo (catálogo, compatibilidad, tiempo). Devuelve (draft, errores)."""
    try:
        draft = MesocycleDraft.model_validate_json(raw)
    except ValidationError as exc:
        return None, [f"schema: {e['loc']}: {e['msg']}" for e in exc.errors()]
    errors: list[str] = []
    if len(draft.days) != req.days_per_week:
        errors.append(f"se pidieron {req.days_per_week} días y llegaron {len(draft.days)}")
    by_id = {e.id: e.model_dump() for e in catalog.all}
    for day in draft.days:
        for item in day.exercises:
            ex = catalog.get(item.exercise_id)
            if ex is None:
                errors.append(f"{item.exercise_id} no existe en el catálogo")
                continue
            why = catalog.incompatibility(ex, eq, req.level, req.avoid_zones)
            if why:
                errors.append(f"{item.exercise_id} incompatible: {why}")
        minutes = _estimate_minutes([e.model_dump() for e in day.exercises], by_id)
        if minutes > req.minutes_per_session:
            errors.append(f"día {day.name} dura ~{minutes:.0f} min y el máximo es {req.minutes_per_session}")
    return (draft if not errors else None), errors


def fallback_template(req: BlockRequest, eq: UserEquipment, catalog: Catalog) -> MesocycleDraft:
    """Plantilla base determinista: cuerpo completo con un ejercicio por patrón principal."""
    allowed = catalog.compatible(eq, req.level, req.avoid_zones)
    picks = []
    for pattern in ["rodilla", "empuje_h", "traccion_h", "cadera", "empuje_v", "traccion_v", "core"]:
        cands = sorted([e for e in allowed if e.pattern == pattern], key=lambda e: (e.type != "compuesto", e.id))
        if cands:
            picks.append(cands[0])
    rir = 3 if req.level == "principiante" else 2
    by_id = {e.id: e.model_dump() for e in catalog.all}
    exercises = [PlannedExercise(exercise_id=e.id, sets=3, reps=e.rep_range[0], target_rir=rir, order=i + 1)
                 for i, e in enumerate(picks)]
    while len(exercises) > 2 and _estimate_minutes([x.model_dump() for x in exercises], by_id) > req.minutes_per_session:
        exercises.pop()
    if len(exercises) < 2:
        raise ValueError("No hay suficientes ejercicios compatibles para armar un bloque")
    days = [DayPlan(name=chr(65 + d), focus="cuerpo completo", exercises=exercises) for d in range(req.days_per_week)]
    return MesocycleDraft(weeks=4, days=days, rationale="Plantilla base (el LLM no entregó un bloque válido).")


def generate_block(req: BlockRequest, eq: UserEquipment, catalog: Catalog, provider: LLMProvider,
                   user_id: str | None = None) -> tuple[MesocycleDraft, dict[str, Any]]:
    ctx = build_context(req, eq, catalog)
    prompt = PROMPT
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, MAX_RETRIES + 2):
        raw = call_traced(provider, "mesocycle", prompt, ctx, PROMPT_VERSION, attempt, user_id=user_id, critical=True)
        draft, errors = validate_draft(raw, req, eq, catalog)
        attempts.append({"attempt": attempt, "errors": errors})
        if draft:
            return draft, {"source": "llm", "attempts": attempts, "prompt_version": PROMPT_VERSION,
                           "provider": provider.name, "model": provider.model}
        prompt = PROMPT + "\nTu respuesta anterior tenía estos errores, corrígelos:\n- " + "\n- ".join(errors)
    draft = fallback_template(req, eq, catalog)
    return draft, {"source": "fallback", "attempts": attempts, "prompt_version": PROMPT_VERSION,
                   "provider": provider.name, "model": provider.model}


def draft_to_json(draft: MesocycleDraft) -> str:
    return json.dumps(draft.model_dump(), ensure_ascii=False)
