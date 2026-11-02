"""Capa de abstracción de proveedor LLM (HU-024) con proveedor simulado y control de costo (HU-026).

El código de negocio solo conoce `LLMProvider`. Cambiar de proveedor es cambiar la variable
de entorno LLM_PROVIDER; los proveedores reales se conectan implementando `complete_json`.
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMProvider(Protocol):
    name: str
    model: str

    def complete_json(self, task: str, prompt: str, context: dict[str, Any]) -> str:
        """Devuelve texto que debería ser JSON. La validación ocurre fuera del proveedor."""
        ...


# ---------------------------------------------------------------------------
# Trazas (equivalente local a Langfuse) y presupuesto de tokens
# ---------------------------------------------------------------------------

@dataclass
class Trace:
    task: str
    provider: str
    model: str
    prompt_version: str
    attempt: int
    ok: bool
    latency_ms: float
    est_tokens: int
    error: str | None = None


@dataclass
class Tracer:
    traces: list[Trace] = field(default_factory=list)

    def record(self, trace: Trace) -> None:
        self.traces.append(trace)


TRACER = Tracer()


class BudgetExceeded(Exception):
    pass


@dataclass
class TokenBudget:
    """Presupuesto mensual de tokens por usuario (A12). En producción vive en la base o Redis."""

    monthly_limit: int = 200_000
    used: dict[str, int] = field(default_factory=dict)

    def charge(self, user_id: str, tokens: int, critical: bool = False) -> None:
        current = self.used.get(user_id, 0)
        if not critical and current + tokens > self.monthly_limit:
            raise BudgetExceeded("Presupuesto mensual de IA agotado para funciones no críticas")
        self.used[user_id] = current + tokens


BUDGET = TokenBudget()


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def call_traced(provider: LLMProvider, task: str, prompt: str, context: dict, prompt_version: str, attempt: int,
                user_id: str | None = None, critical: bool = False) -> str:
    tokens = estimate_tokens(prompt + json.dumps(context, ensure_ascii=False))
    if user_id:
        BUDGET.charge(user_id, tokens, critical=critical)
    start = time.perf_counter()
    try:
        out = provider.complete_json(task, prompt, context)
        TRACER.record(Trace(task, provider.name, provider.model, prompt_version, attempt, True,
                            (time.perf_counter() - start) * 1000, tokens))
        return out
    except Exception as exc:  # noqa: BLE001 — se registra y se relanza
        TRACER.record(Trace(task, provider.name, provider.model, prompt_version, attempt, False,
                            (time.perf_counter() - start) * 1000, tokens, str(exc)))
        raise


# ---------------------------------------------------------------------------
# Proveedor simulado: determinista, sin red, útil para desarrollo, demo y tests
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower()).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip()


NUMBER_WORDS = {"un": 1, "una": 1, "uno": 1, "medio": 0.5, "media": 0.5, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5}


class FakeProvider:
    """Imita al LLM con reglas. `fail_first` permite simular salidas inválidas para probar reintentos."""

    name = "fake"
    model = "fake-rules-0.1"

    def __init__(self, fail_first: int = 0):
        self.fail_first = fail_first
        self.calls = 0

    def complete_json(self, task: str, prompt: str, context: dict[str, Any]) -> str:
        self.calls += 1
        if self.calls <= self.fail_first:
            bad = [{"exercise_id": "EJ-999", "sets": 3, "reps": 8, "target_rir": 2, "order": 1},
                   {"exercise_id": "EJ-998", "sets": 3, "reps": 8, "target_rir": 2, "order": 2}]
            n = int(context.get("days_per_week", 1))
            return json.dumps({"weeks": 4, "days": [{"name": chr(65 + i), "focus": "x", "exercises": bad} for i in range(n)],
                               "rationale": "inventado"})
        if task == "mesocycle":
            return json.dumps(self._mesocycle(context), ensure_ascii=False)
        if task == "food_extraction":
            return json.dumps(self._foods(context), ensure_ascii=False)
        raise ValueError(f"Tarea desconocida: {task}")

    # --- Bloques -------------------------------------------------------------
    TEMPLATES = {
        "full": [["rodilla", "empuje_h", "traccion_h", "cadera", "empuje_v", "core"],
                 ["cadera", "empuje_v", "traccion_v", "rodilla", "empuje_h", "pantorrilla"],
                 ["rodilla", "traccion_h", "empuje_h", "cadera", "brazos", "carga"]],
        "split": [["rodilla", "cadera", "rodilla", "pantorrilla", "core"],
                  ["empuje_h", "traccion_h", "empuje_v", "traccion_v", "brazos"]],
    }

    def _mesocycle(self, ctx: dict[str, Any]) -> dict[str, Any]:
        allowed: list[dict] = ctx["allowed_exercises"]
        days_per_week: int = ctx["days_per_week"]
        minutes: int = ctx["minutes_per_session"]
        level: str = ctx["level"]
        by_pattern: dict[str, list[dict]] = {}
        for e in allowed:
            by_pattern.setdefault(e["pattern"], []).append(e)
        for lst in by_pattern.values():
            lst.sort(key=lambda e: (e["type"] != "compuesto", e["id"]))
        template = self.TEMPLATES["split"] if days_per_week >= 4 else self.TEMPLATES["full"]
        used_count: dict[str, int] = {}
        days = []
        rir = 3 if level == "principiante" else 2
        for d in range(days_per_week):
            patterns = template[d % len(template)]
            exercises, seen = [], set()
            for p in patterns:
                options = [e for e in by_pattern.get(p, []) if e["id"] not in seen]
                if not options:
                    continue
                options.sort(key=lambda e: used_count.get(e["id"], 0))
                ex = options[0]
                seen.add(ex["id"])
                used_count[ex["id"]] = used_count.get(ex["id"], 0) + 1
                lo, hi = ex["rep_range"]
                reps = lo if ex["method"] == "doble_progresion" else min(hi, lo + 2)
                exercises.append({"exercise_id": ex["id"], "sets": 3, "reps": reps, "target_rir": rir,
                                  "order": len(exercises) + 1})
            # Ajuste al tiempo disponible: se quitan ejercicios del final.
            while exercises and _estimate_minutes(exercises, {e["id"]: e for e in allowed}) > minutes and len(exercises) > 2:
                exercises.pop()
            days.append({"name": chr(65 + d), "focus": " · ".join(patterns[:3]), "exercises": exercises})
        return {"weeks": 4, "days": days,
                "rationale": f"Bloque de 4 semanas, {days_per_week} días, priorizando compuestos y respetando {minutes} min."}

    # --- Dieta ---------------------------------------------------------------
    def _foods(self, ctx: dict[str, Any]) -> dict[str, Any]:
        text = _norm(ctx["text"])
        aliases: dict[str, str] = {_norm(a): fid for fid, names in ctx["food_aliases"].items() for a in names}
        units = sorted({_norm(u) for u in ctx["unit_words"]} | {"g", "gr", "gramos"}, key=len, reverse=True)
        qty_rx = r"\d+(?:[.,]\d+)?|" + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
        unit_rx = "|".join(re.escape(u) for u in units)
        items = []
        consumed = text
        for alias in sorted(aliases, key=len, reverse=True):
            m = re.search(rf"(?:\b(?P<q>{qty_rx})\s*)?(?:\b(?P<u>(?:{unit_rx})(?:es|s)?)\s+)?(?:de\s+)?\b{re.escape(alias)}\b",
                          consumed)
            if not m:
                continue
            qty, unit = None, m.group("u")
            q = m.group("q")
            if q:
                qty = NUMBER_WORDS.get(q) or float(q.replace(",", "."))
            if unit and unit not in units:
                unit = unit[:-2] if unit.endswith("es") and unit[:-2] in units else unit.rstrip("s")
            if unit in {"gr", "gramos"}:
                unit = "g"
            items.append({"food_id_hint": aliases[alias], "mention": alias, "quantity": qty, "unit": unit})
            consumed = consumed[: m.start()] + " " * (m.end() - m.start()) + consumed[m.end():]
        return {"items": items}


def _estimate_minutes(exercises: list[dict], by_id: dict[str, dict]) -> float:
    total = 5.0  # calentamiento general
    for item in exercises:
        ex = by_id.get(item["exercise_id"], {})
        per_set = 2.5 if ex.get("type") == "compuesto" else 1.5
        total += item["sets"] * per_set
    return total


class NotConfiguredProvider:
    def __init__(self, name: str):
        self.name, self.model = name, os.getenv("LLM_MODEL", "sin-configurar")

    def complete_json(self, task: str, prompt: str, context: dict[str, Any]) -> str:
        raise RuntimeError(
            f"El proveedor '{self.name}' aún no está conectado. Implementa complete_json en app/llm/provider.py "
            "con el SDK oficial y define la API key en el entorno."
        )


def get_provider() -> LLMProvider:
    name = os.getenv("LLM_PROVIDER", "fake").lower()
    if name == "fake":
        return FakeProvider()
    return NotConfiguredProvider(name)
