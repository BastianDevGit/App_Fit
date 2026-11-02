"""Cálculos de carga: e1RM, carga para un objetivo de reps/RIR y redondeo al implemento.

Convenciones:
- Las cargas están en kg. En mancuernas y kettlebells la carga es por unidad (por mano).
- RIR = repeticiones en reserva. "Reps hasta el fallo" = reps + RIR.
- Fórmula de Epley ajustada por RIR: e1RM = carga × (1 + (reps + RIR) / 30).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from typing import Iterable

# Rango en que la estimación de e1RM es razonable (HU-010). Fuera de él la serie no actualiza el e1RM.
MIN_REPS_TO_FAILURE = 1
MAX_REPS_TO_FAILURE = 12
EPS = 1e-9


def reps_to_failure(reps: int, rir: float) -> float:
    return reps + max(rir, 0.0)


def is_valid_for_e1rm(load: float, reps: int, rir: float | None) -> bool:
    if rir is None or load <= 0 or reps <= 0:
        return False
    rtf = reps_to_failure(reps, rir)
    return MIN_REPS_TO_FAILURE <= rtf <= MAX_REPS_TO_FAILURE


def estimate_e1rm(load: float, reps: int, rir: float) -> float:
    """e1RM de una serie. Lanza ValueError si la serie no es apta (HU-010, escenario 2)."""
    if not is_valid_for_e1rm(load, reps, rir):
        raise ValueError("Serie fuera del rango válido para estimar e1RM")
    return load * (1 + reps_to_failure(reps, rir) / 30.0)


def raw_load_for_target(e1rm: float, reps: int, rir: float) -> float:
    """Carga teórica (sin redondear) para hacer `reps` dejando `rir` en reserva."""
    if e1rm <= 0:
        return 0.0
    return e1rm / (1 + reps_to_failure(reps, rir) / 30.0)


@dataclass(frozen=True)
class LoadOptions:
    """Cargas realizables con los implementos del usuario (HU-005).

    `values` es la lista ordenada de cargas posibles. Para barra se genera con `for_barbell`.
    Una lista vacía significa peso corporal: no hay carga que redondear.
    """

    values: tuple[float, ...] = field(default_factory=tuple)

    @staticmethod
    def stepped(minimum: float, maximum: float, step: float) -> "LoadOptions":
        if step <= 0 or maximum < minimum:
            raise ValueError("Rango de cargas inválido")
        n = int(round((maximum - minimum) / step))
        return LoadOptions(tuple(round(minimum + i * step, 3) for i in range(n + 1)))

    @staticmethod
    def explicit(values: Iterable[float]) -> "LoadOptions":
        return LoadOptions(tuple(sorted({round(v, 3) for v in values if v > 0})))

    @staticmethod
    def for_barbell(bar_kg: float, plate_pairs: dict[float, int]) -> "LoadOptions":
        """Cargas posibles con una barra y `plate_pairs` = {peso_disco: pares disponibles}."""
        per_side: set[float] = {0.0}
        plates: list[float] = []
        for w, pairs in plate_pairs.items():
            plates.extend([w] * max(int(pairs), 0))
        # Sumas de subconjuntos por lado (los discos van de a pares).
        for p in plates:
            per_side |= {round(s + p, 3) for s in per_side}
        return LoadOptions(tuple(sorted(round(bar_kg + 2 * s, 3) for s in per_side)))

    @property
    def is_bodyweight(self) -> bool:
        return not self.values

    @property
    def minimum(self) -> float:
        return self.values[0] if self.values else 0.0

    @property
    def maximum(self) -> float:
        return self.values[-1] if self.values else 0.0

    def floor(self, target: float) -> float:
        """Mayor carga realizable ≤ objetivo (redondeo conservador). Si nada cabe, la mínima."""
        if self.is_bodyweight:
            return 0.0
        best = self.values[0]
        for v in self.values:
            if v <= target + EPS:
                best = v
            else:
                break
        return best

    def nearest(self, target: float) -> float:
        """Carga realizable más cercana; empata hacia abajo."""
        if self.is_bodyweight:
            return 0.0
        return min(self.values, key=lambda v: (abs(v - target), v))

    def step_up(self, load: float, steps: int = 1) -> float:
        if self.is_bodyweight:
            return 0.0
        idx = self._index_at_or_below(load)
        return self.values[min(idx + steps, len(self.values) - 1)]

    def step_down(self, load: float, steps: int = 1) -> float:
        if self.is_bodyweight:
            return 0.0
        idx = self._index_at_or_below(load)
        return self.values[max(idx - steps, 0)]

    def _index_at_or_below(self, load: float) -> int:
        idx = 0
        for i, v in enumerate(self.values):
            if v <= load + EPS:
                idx = i
            else:
                break
        return idx


def load_for_target(e1rm: float, reps: int, rir: float, options: LoadOptions) -> float:
    """Carga sugerida redondeada hacia abajo a lo que el usuario puede armar (HU-011)."""
    return options.floor(raw_load_for_target(e1rm, reps, rir))


def plates_per_side(load: float, bar_kg: float, plate_pairs: dict[float, int]) -> tuple[float, list[float]] | None:
    """Discos por lado para armar `load` (HU-020).

    Devuelve (carga_armable, discos_por_lado). Si `load` no se puede armar exactamente,
    usa la mayor carga armable por debajo. None si ni la barra sola cabe.
    """
    options = LoadOptions.for_barbell(bar_kg, plate_pairs)
    if load + EPS < bar_kg:
        return None
    achievable = options.floor(load)
    target_side = round((achievable - bar_kg) / 2, 3)
    if target_side <= EPS:
        return achievable, []
    # Búsqueda con menos discos posible: primero combinaciones cortas, discos grandes primero.
    available = sorted(plate_pairs.items(), key=lambda kv: -kv[0])
    weights = [w for w, _ in available]
    for n in range(1, sum(p for _, p in available) + 1):
        for combo in combinations_with_replacement(weights, n):
            if abs(sum(combo) - target_side) > 1e-6:
                continue
            if all(combo.count(w) <= plate_pairs[w] for w in set(combo)):
                return achievable, sorted(combo, reverse=True)
    return None  # inalcanzable por construcción, pero no debería ocurrir
