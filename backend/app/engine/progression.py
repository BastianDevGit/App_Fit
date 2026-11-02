"""Reglas de progresión: ajustes intra e intersesión, fatiga, deload, límites y doble progresión.

Todas las funciones son puras y deterministas (HU-011, escenario "Reproducibilidad").
Los parámetros numéricos son hipótesis iniciales: viven en `ProgressionConfig` para poder
calibrarlos sin tocar la lógica.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from . import ENGINE_VERSION
from .loads import LoadOptions, estimate_e1rm, is_valid_for_e1rm, load_for_target
from .reasons import Reason, explain


@dataclass(frozen=True)
class ProgressionConfig:
    up_margin_rir: int = 1               # RIR sobre el objetivo que justifica subir
    max_session_increase_pct: float = 0.05  # tope de subida entre sesiones (HU-014)
    max_session_decrease_pct: float = 0.15
    e1rm_smoothing: float = 0.5          # peso del dato nuevo en la media móvil del e1RM
    fatigue_high: float = 4.0            # promedio 1–5 (5 = peor)
    fatigue_moderate: float = 3.5
    fatigue_moderate_load_factor: float = 0.975
    fatigue_high_load_factor: float = 0.95
    deload_load_factor: float = 0.9
    deload_e1rm_drop_pct: float = 0.05
    deload_min_exercises_dropping: int = 2
    deload_fatigue_days: int = 3
    velocity_mismatch_drop: float = 0.20  # caída de velocidad que contradice un RIR alto (HU-031)
    reentry_weeks: int = 2
    reentry_load_factor: float = 0.85


DEFAULT_CONFIG = ProgressionConfig()


@dataclass(frozen=True)
class Decision:
    load: float
    reason: Reason
    explanation: str
    reps: int | None = None
    sets: int | None = None
    engine_version: str = ENGINE_VERSION
    flags: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Intra-sesión: carga de la próxima serie según el RIR reportado (HU-012, HU-015, HU-031)
# ---------------------------------------------------------------------------

def effective_rir(reported_rir: int, velocity_drop: float | None, cfg: ProgressionConfig = DEFAULT_CONFIG) -> tuple[int, bool]:
    """Corrige un RIR alto que contradice una caída de velocidad grande. Devuelve (rir, hubo_discrepancia)."""
    if velocity_drop is not None and velocity_drop >= cfg.velocity_mismatch_drop and reported_rir >= 3:
        return reported_rir - 1, True
    return reported_rir, False


def next_set(
    planned_load: float,
    target_rir: int,
    reported_rir: int,
    options: LoadOptions,
    velocity_drop: float | None = None,
    cfg: ProgressionConfig = DEFAULT_CONFIG,
) -> Decision:
    rir, mismatch = effective_rir(reported_rir, velocity_drop, cfg)
    diff = rir - target_rir
    flags = ("velocity_mismatch",) if mismatch else ()
    if options.is_bodyweight:
        return Decision(0.0, Reason.ON_TARGET, explain(Reason.ON_TARGET, reported=reported_rir), flags=flags)
    if diff <= -2:
        load, reason = options.step_down(planned_load, 2), Reason.MUCH_HARDER_THAN_PLANNED
    elif diff == -1:
        load, reason = options.step_down(planned_load, 1), Reason.HARDER_THAN_PLANNED
    elif diff >= cfg.up_margin_rir:
        load, reason = options.step_up(planned_load, 1), Reason.EASIER_THAN_PLANNED
    else:
        load, reason = options.floor(planned_load), Reason.ON_TARGET
    if mismatch and reason is Reason.ON_TARGET:
        reason = Reason.VELOCITY_MISMATCH
    return Decision(load, reason, explain(reason, reported=reported_rir, target=target_rir), flags=flags)


def decision_table(
    planned_load: float, target_rir: int, options: LoadOptions, max_rir: int = 5, cfg: ProgressionConfig = DEFAULT_CONFIG
) -> dict[int, Decision]:
    """Tabla precalculada RIR reportado → carga de la siguiente serie, para uso sin conexión (HU-015).

    La app solo busca en esta tabla; no contiene lógica de progresión propia.
    """
    return {r: next_set(planned_load, target_rir, r, options, cfg=cfg) for r in range(0, max_rir + 1)}


# ---------------------------------------------------------------------------
# Intersesión: e1RM, carga de la próxima sesión y límites de seguridad (HU-010, HU-011, HU-014)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PerformedSet:
    load: float
    reps: int
    rir: int | None


def session_e1rm(sets: Sequence[PerformedSet]) -> float | None:
    """Mejor e1RM de la sesión usando solo series aptas."""
    values = [estimate_e1rm(s.load, s.reps, s.rir) for s in sets if is_valid_for_e1rm(s.load, s.reps, s.rir)]
    return max(values) if values else None


def update_e1rm(previous: float | None, new_value: float | None, cfg: ProgressionConfig = DEFAULT_CONFIG) -> float | None:
    if new_value is None:
        return previous
    if previous is None:
        return new_value
    a = cfg.e1rm_smoothing
    return a * new_value + (1 - a) * previous


def clamp_change(previous_load: float | None, proposed: float, options: LoadOptions, cfg: ProgressionConfig = DEFAULT_CONFIG) -> tuple[float, bool]:
    """Aplica el tope de subida y bajada entre sesiones. Devuelve (carga, se_limitó_la_subida)."""
    if options.is_bodyweight or previous_load is None or previous_load <= 0:
        return proposed, False
    max_up = max(previous_load * (1 + cfg.max_session_increase_pct), options.step_up(previous_load, 1))
    min_down = previous_load * (1 - cfg.max_session_decrease_pct)
    if proposed > max_up + 1e-9:
        return options.floor(max_up), True
    if proposed < min_down - 1e-9:
        return options.nearest(min_down), False
    return proposed, False


def next_session_load(
    e1rm: float,
    reps: int,
    target_rir: int,
    options: LoadOptions,
    previous_load: float | None,
    cfg: ProgressionConfig = DEFAULT_CONFIG,
) -> Decision:
    proposed = load_for_target(e1rm, reps, target_rir, options)
    load, capped = clamp_change(previous_load, proposed, options, cfg)
    if capped:
        reason = Reason.CAPPED_INCREASE
    elif previous_load is None or abs(load - previous_load) < 1e-9:
        reason = Reason.ON_TARGET
    elif load > previous_load:
        reason = Reason.E1RM_UP
    else:
        reason = Reason.E1RM_DOWN
    text = "Mantenemos la carga de la sesión anterior." if reason is Reason.ON_TARGET else explain(reason)
    return Decision(load, reason, text, reps=reps)


def double_progression(
    load: float,
    reps_done: Sequence[int],
    rirs: Sequence[int],
    rep_range: tuple[int, int],
    target_rir: int,
    options: LoadOptions,
) -> Decision:
    """Doble progresión (método del catálogo): primero reps, luego carga (HU-011, escenario 3)."""
    low, high = rep_range
    top_reached = all(r >= high for r in reps_done) and all(x >= target_rir for x in rirs)
    if top_reached and not options.is_bodyweight:
        new_load = options.step_up(load, 1)
        if new_load > load:
            return Decision(new_load, Reason.LOAD_UP_TOP_OF_RANGE, explain(Reason.LOAD_UP_TOP_OF_RANGE), reps=low)
    next_reps = min(high, (min(reps_done) if reps_done else low) + 1)
    return Decision(load, Reason.REPS_UP, explain(Reason.REPS_UP), reps=max(low, next_reps))


# ---------------------------------------------------------------------------
# Fatiga del día y versión corta (HU-012 escenario 3, HU-034)
# ---------------------------------------------------------------------------

def readiness_score(fatigue: int, sleep: int, soreness: int, stress: int) -> float:
    """Promedio 1–5, donde 5 es peor. El check-in omitido usa 3 (neutro)."""
    return (fatigue + sleep + soreness + stress) / 4.0


def apply_readiness(sets: int, load: float, score: float | None, options: LoadOptions, cfg: ProgressionConfig = DEFAULT_CONFIG) -> Decision:
    if score is None or score < cfg.fatigue_moderate:
        return Decision(load, Reason.ON_TARGET, "", sets=sets)
    if score >= cfg.fatigue_high:
        return Decision(options.floor(load * cfg.fatigue_high_load_factor), Reason.FATIGUE_HIGH,
                        explain(Reason.FATIGUE_HIGH), sets=max(1, sets - 1))
    return Decision(options.floor(load * cfg.fatigue_moderate_load_factor), Reason.FATIGUE_MODERATE,
                    explain(Reason.FATIGUE_MODERATE), sets=sets)


def short_session(exercises: Sequence[dict], max_exercises: int = 3) -> list[dict]:
    """Versión de 15–20 min: compuestos primero, una serie menos (mín. 2). Sin tocar la progresión."""
    ordered = sorted(exercises, key=lambda e: (0 if e.get("type") == "compuesto" else 1, e.get("order", 0)))
    out = []
    for e in ordered[:max_exercises]:
        item = dict(e)
        item["sets"] = max(2, int(e.get("sets", 3)) - 1)
        item["counts_for_progression"] = False
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# Deload y reentrada (HU-013, HU-070)
# ---------------------------------------------------------------------------

def deload_reason(
    week_in_block: int,
    block_weeks: int,
    e1rm_series: dict[str, Sequence[float]],
    recent_readiness: Sequence[float],
    cfg: ProgressionConfig = DEFAULT_CONFIG,
) -> Reason | None:
    if week_in_block >= block_weeks:
        return Reason.DELOAD_END_OF_BLOCK
    dropping = 0
    for series in e1rm_series.values():
        if len(series) >= 3:
            peak = max(series[:-1])
            if peak > 0 and (peak - series[-1]) / peak >= cfg.deload_e1rm_drop_pct and series[-1] < series[-2]:
                dropping += 1
    if dropping >= cfg.deload_min_exercises_dropping:
        return Reason.DELOAD_PERFORMANCE_DROP
    tail = list(recent_readiness)[-cfg.deload_fatigue_days:]
    if len(tail) == cfg.deload_fatigue_days and all(s >= cfg.fatigue_high for s in tail):
        return Reason.DELOAD_SUSTAINED_FATIGUE
    return None


def apply_deload(sets: int, load: float, options: LoadOptions, cfg: ProgressionConfig = DEFAULT_CONFIG) -> tuple[int, float]:
    return max(1, math.ceil(sets / 2)), options.floor(load * cfg.deload_load_factor)


def reentry_factor(weeks_since_last_session: float, cfg: ProgressionConfig = DEFAULT_CONFIG) -> float:
    return cfg.reentry_load_factor if weeks_since_last_session >= cfg.reentry_weeks else 1.0


# ---------------------------------------------------------------------------
# Calentamiento (HU-019)
# ---------------------------------------------------------------------------

def warmup_sets(work_load: float, options: LoadOptions, bar_kg: float | None = None) -> list[tuple[float, int]]:
    """Series de aproximación (carga, reps) redondeadas a los implementos. Vacío en peso corporal."""
    if options.is_bodyweight or work_load <= 0:
        return []
    scheme = [(0.5, 5), (0.7, 3), (0.85, 1)] if work_load >= 40 else [(0.5, 8), (0.75, 4)]
    out: list[tuple[float, int]] = []
    if bar_kg:
        out.append((bar_kg, 10))
    for pct, reps in scheme:
        load = options.floor(work_load * pct)
        if load >= work_load or load < options.minimum:
            continue
        if out and load <= out[-1][0]:
            continue
        out.append((load, reps))
    return out
