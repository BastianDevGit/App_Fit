"""Catálogo cerrado de ejercicios e implementos del usuario (HU-005, HU-021, HU-035, HU-036)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.engine.loads import LoadOptions

DATA = Path(__file__).parent / "data" / "exercises.json"

Level = Literal["principiante", "intermedio", "avanzado"]
Environment = Literal["casa", "gimnasio", "ambos"]
LEVEL_ORDER = {"principiante": 0, "intermedio": 1, "avanzado": 2}


class Exercise(BaseModel):
    id: str
    name: str
    pattern: str
    type: Literal["compuesto", "aislado"]
    requires: list[str] = Field(default_factory=list)
    any_of: list[str] = Field(default_factory=list)
    load_implement: str | None = None
    environment: Environment
    levels: list[Level]
    rep_range: tuple[int, int]
    method: Literal["carga_rir", "doble_progresion", "variante", "tiempo"]
    caution_zones: list[str] = Field(default_factory=list)
    regression: str | None = None
    progression: str | None = None
    vision_cfg: str | None = None
    camera_angle: str | None = None
    conservative_ratio: float | None = None
    units: int | None = None  # cuántas mancuernas/kettlebells se usan; None = 2 si aplica


# --- Implementos del usuario ---------------------------------------------------

class Barbell(BaseModel):
    bar_kg: float = 20.0
    plates: dict[float, int] = Field(default_factory=lambda: {20: 1, 15: 1, 10: 1, 5: 2, 2.5: 2, 1.25: 1})


class Stepped(BaseModel):
    min: float
    max: float
    step: float


class Explicit(BaseModel):
    weights: list[float]


class UserEquipment(BaseModel):
    """Lo que el usuario declara en el onboarding (HU-005). Los implementos sin carga van en `simple`."""

    environment: Literal["casa", "gimnasio"] = "gimnasio"
    barra: Barbell | None = None
    mancuernas: Stepped | None = None
    kettlebell: Explicit | None = None
    polea: Stepped | None = None
    maquinas: Stepped | None = None
    simple: list[str] = Field(default_factory=list, description="banco, rack, barra_dominadas, bandas…")

    def has(self, item: str) -> bool:
        if item in {"barra", "mancuernas", "kettlebell", "polea", "maquinas"}:
            return getattr(self, item) is not None
        return item in self.simple

    def options_for(self, implement: str | None) -> LoadOptions:
        if implement is None:
            return LoadOptions()
        if implement == "barra" and self.barra:
            return LoadOptions.for_barbell(self.barra.bar_kg, self.barra.plates)
        if implement in {"mancuernas", "polea", "maquinas"}:
            s: Stepped | None = getattr(self, implement)
            if s:
                return LoadOptions.stepped(s.min, s.max, s.step)
        if implement == "kettlebell" and self.kettlebell:
            return LoadOptions.explicit(self.kettlebell.weights)
        return LoadOptions()


# --- Catálogo ------------------------------------------------------------------

class Catalog:
    def __init__(self, exercises: list[Exercise], meta: dict):
        self.meta = meta
        self._by_id = {e.id: e for e in exercises}

    @property
    def all(self) -> list[Exercise]:
        return list(self._by_id.values())

    def get(self, exercise_id: str) -> Exercise | None:
        return self._by_id.get(exercise_id)

    def __contains__(self, exercise_id: str) -> bool:
        return exercise_id in self._by_id

    def load_implement(self, ex: Exercise, eq: UserEquipment) -> str | None:
        if ex.load_implement == "any":
            return next((i for i in ex.any_of if eq.has(i)), None)
        return ex.load_implement

    def options(self, ex: Exercise, eq: UserEquipment) -> LoadOptions:
        return eq.options_for(self.load_implement(ex, eq))

    def incompatibility(self, ex: Exercise, eq: UserEquipment, level: str, avoid_zones: list[str]) -> str | None:
        """Motivo por el que el ejercicio no sirve para este usuario, o None si es compatible."""
        if ex.environment != "ambos" and eq.environment != ex.environment:
            return f"requiere entorno {ex.environment}"
        missing = [i for i in ex.requires if not eq.has(i)]
        if missing:
            return "faltan implementos: " + ", ".join(missing)
        if ex.any_of and not any(eq.has(i) for i in ex.any_of):
            return "requiere uno de: " + ", ".join(ex.any_of)
        if level not in ex.levels:
            return f"no apto para nivel {level}"
        blocked = sorted(set(ex.caution_zones) & set(avoid_zones))
        if blocked:
            return "carga una zona marcada: " + ", ".join(blocked)
        return None

    def compatible(self, eq: UserEquipment, level: str, avoid_zones: list[str] | None = None) -> list[Exercise]:
        z = avoid_zones or []
        return [e for e in self.all if self.incompatibility(e, eq, level, z) is None]

    def alternatives(self, exercise_id: str, eq: UserEquipment, level: str, avoid_zones: list[str] | None = None) -> list[Exercise]:
        """Mismo patrón, compatibles, sin el original (HU-035 máquina ocupada, HU-036 me molesta).

        Primero la regresión indicada por el catálogo; luego el resto del patrón.
        """
        base = self.get(exercise_id)
        if base is None:
            return []
        pool = [e for e in self.compatible(eq, level, avoid_zones) if e.pattern == base.pattern and e.id != base.id]
        pool.sort(key=lambda e: (0 if e.id == base.regression else 1, abs(LEVEL_ORDER[e.levels[0]] - LEVEL_ORDER[level])))
        return pool


@lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    raw = json.loads(DATA.read_text(encoding="utf-8"))
    return Catalog([Exercise(**e) for e in raw["exercises"]], raw.get("meta", {}))
