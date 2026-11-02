"""Registro de comidas en lenguaje natural (HU-060, HU-061, HU-062) y metas seguras (HU-064).

El LLM solo extrae alimentos y porciones. La búsqueda y el cálculo de macros son deterministas.
"""

from __future__ import annotations

import difflib
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.llm.provider import LLMProvider, _norm, call_traced

PROMPT_VERSION = "food-extraction-0.1"
PROMPT = """Extrae los alimentos y porciones del texto. No calcules calorías ni macros.
Responde solo JSON: {"items": [{"food_id_hint", "mention", "quantity", "unit"}]}. Usa null si no se sabe."""
DATA = Path(__file__).parent / "data" / "foods.json"


class Macros(BaseModel):
    kcal: float = 0
    p: float = 0
    c: float = 0
    f: float = 0

    def __add__(self, other: "Macros") -> "Macros":
        return Macros(kcal=self.kcal + other.kcal, p=self.p + other.p, c=self.c + other.c, f=self.f + other.f)

    def rounded(self) -> "Macros":
        return Macros(kcal=round(self.kcal), p=round(self.p, 1), c=round(self.c, 1), f=round(self.f, 1))


class Food(BaseModel):
    id: str
    name: str
    aliases: list[str]
    per100: Macros
    portions: dict[str, float]
    default_portion: str | None


class ParsedItem(BaseModel):
    food_id: str | None
    food_name: str | None
    mention: str
    quantity: float | None = None
    unit: str | None = None
    grams: float | None = None
    macros: Macros | None = None
    confidence: Literal["alta", "media", "baja"] = "media"
    needs_portion: bool = False
    portion_options: list[dict] = Field(default_factory=list)


class ParseResult(BaseModel):
    items: list[ParsedItem]
    totals: Macros
    complete: bool
    note: str = "Valores calculados desde la base nutricional, no inventados por la IA."


@lru_cache(maxsize=1)
def food_db() -> dict[str, Food]:
    raw = json.loads(DATA.read_text(encoding="utf-8"))
    return {f["id"]: Food(**f) for f in raw["foods"]}


def search_food(query: str) -> tuple[Food | None, bool]:
    """Busca por alias exacto y luego aproximado (equivalente local a pg_trgm). Devuelve (alimento, exacto)."""
    q = _norm(query)
    alias_map = {_norm(a): f for f in food_db().values() for a in [f.name, *f.aliases]}
    if q in alias_map:
        return alias_map[q], True
    close = difflib.get_close_matches(q, list(alias_map), n=1, cutoff=0.75)
    return (alias_map[close[0]], False) if close else (None, False)


def macros_for(food: Food, grams: float) -> Macros:
    k = grams / 100.0
    m = food.per100
    return Macros(kcal=m.kcal * k, p=m.p * k, c=m.c * k, f=m.f * k).rounded()


def resolve_item(food: Food, mention: str, quantity: float | None, unit: str | None, exact: bool) -> ParsedItem:
    qty = quantity if quantity else 1.0
    base = ParsedItem(food_id=food.id, food_name=food.name, mention=mention, quantity=quantity, unit=unit)
    unit_n = _norm(unit) if unit else None
    portions = {_norm(k): v for k, v in food.portions.items()}
    grams: float | None = None
    confidence = "alta"
    if unit_n in {"g", "gr", "gramos"}:
        grams = qty
    elif unit_n and unit_n in portions:
        grams = qty * portions[unit_n]
    elif food.default_portion:
        grams = qty * food.portions[food.default_portion]
        confidence = "media"
        base.unit = food.default_portion
    if grams is None:
        base.needs_portion = True
        base.confidence = "media"
        base.portion_options = [{"unit": k, "grams": v} for k, v in food.portions.items()]
        return base
    if not exact:
        confidence = "baja"
    base.grams = round(grams, 1)
    base.macros = macros_for(food, grams)
    base.confidence = confidence  # type: ignore[assignment]
    return base


def parse_meal(text: str, provider: LLMProvider, user_id: str | None = None) -> ParseResult:
    db = food_db()
    units = sorted({u for f in db.values() for u in f.portions} | {"g", "gramos"})
    ctx = {"text": text, "food_aliases": {f.id: [f.name, *f.aliases] for f in db.values()}, "unit_words": units}
    raw = call_traced(provider, "food_extraction", PROMPT, ctx, PROMPT_VERSION, 1, user_id=user_id)
    extracted = json.loads(raw).get("items", [])
    items: list[ParsedItem] = []
    for it in extracted:
        food = db.get(it.get("food_id_hint") or "")
        exact = food is not None
        if food is None:
            food, exact = search_food(it.get("mention", ""))
        if food is None:
            items.append(ParsedItem(food_id=None, food_name=None, mention=it.get("mention", ""), confidence="baja"))
            continue
        items.append(resolve_item(food, it.get("mention", food.name), it.get("quantity"), it.get("unit"), exact))
    return summarize(items)


def summarize(items: list[ParsedItem]) -> ParseResult:
    totals = Macros()
    for i in items:
        if i.macros:
            totals = totals + i.macros
    complete = bool(items) and all(i.macros is not None for i in items)
    return ParseResult(items=items, totals=totals.rounded(), complete=complete)


def choose_portion(item: ParsedItem, unit: str, quantity: float = 1.0) -> ParsedItem:
    """Respuesta del usuario a la pregunta de porción ambigua (HU-061)."""
    food = food_db()[item.food_id] if item.food_id else None
    if food is None or unit not in food.portions:
        raise ValueError("Porción no válida para este alimento")
    return resolve_item(food, item.mention, quantity, unit, True)


# ---------------------------------------------------------------------------
# Metas con mínimos seguros (HU-064)
# ---------------------------------------------------------------------------

class TargetsInput(BaseModel):
    weight_kg: float = Field(gt=30, lt=300)
    goal: Literal["fuerza", "masa", "bajar_grasa", "salud"]
    sex: Literal["f", "m"] | None = None
    height_cm: float | None = Field(default=None, gt=120, lt=230)
    age: int | None = Field(default=None, ge=16, le=99)
    training_days: int = Field(default=3, ge=0, le=7)


class Range(BaseModel):
    min: float
    max: float


class Targets(BaseModel):
    kcal: Range
    protein_g: Range
    carbs_g: Range
    fat_g: Range
    floor_kcal: float
    message: str


SAFE_FLOOR = {"f": 1200, "m": 1500, None: 1400}  # hipótesis: a validar con nutricionista
MAX_DEFICIT = 0.20


def daily_targets(t: TargetsInput) -> Targets:
    if t.sex and t.height_cm and t.age:
        bmr = 10 * t.weight_kg + 6.25 * t.height_cm - 5 * t.age + (5 if t.sex == "m" else -161)
    else:
        bmr = 22 * t.weight_kg  # aproximación cuando faltan datos
    activity = 1.375 if t.training_days <= 3 else 1.55
    tdee = bmr * activity
    factor = {"bajar_grasa": 0.85, "masa": 1.10, "fuerza": 1.0, "salud": 1.0}[t.goal]
    center = tdee * max(factor, 1 - MAX_DEFICIT)
    floor = max(SAFE_FLOOR[t.sex], bmr)
    lo, hi = max(center * 0.95, floor), max(center * 1.05, floor + 100)
    protein = Range(min=round(1.6 * t.weight_kg), max=round(2.0 * t.weight_kg))
    fat = Range(min=round(0.7 * t.weight_kg), max=round(1.0 * t.weight_kg))
    carbs = Range(min=max(0, round((lo - protein.max * 4 - fat.max * 9) / 4)),
                  max=max(0, round((hi - protein.min * 4 - fat.min * 9) / 4)))
    return Targets(kcal=Range(min=round(lo, -1), max=round(hi, -1)), protein_g=protein, carbs_g=carbs, fat_g=fat,
                   floor_kcal=round(floor), message="Son rangos de referencia, no metas exactas. Un día fuera del rango no pasa nada.")
