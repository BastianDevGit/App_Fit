"""Tests del motor de progresión (EP-02). Los nombres citan la historia y el escenario que cubren."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.engine.loads import (
    LoadOptions,
    estimate_e1rm,
    is_valid_for_e1rm,
    load_for_target,
    plates_per_side,
    raw_load_for_target,
)
from app.engine.progression import (
    DEFAULT_CONFIG,
    PerformedSet,
    apply_deload,
    apply_readiness,
    clamp_change,
    decision_table,
    deload_reason,
    double_progression,
    next_session_load,
    next_set,
    session_e1rm,
    short_session,
    update_e1rm,
    warmup_sets,
)
from app.engine.reasons import Reason

DUMBBELLS = LoadOptions.stepped(2.5, 30, 2.5)
BARBELL = LoadOptions.for_barbell(20, {20: 1, 15: 1, 10: 1, 5: 2, 2.5: 2, 1.25: 1})


# --- HU-010: e1RM -----------------------------------------------------------

def test_hu010_e1rm_serie_valida():
    assert estimate_e1rm(60, 8, 2) == pytest.approx(60 * (1 + 10 / 30))


def test_hu010_serie_fuera_de_rango_no_es_apta():
    assert not is_valid_for_e1rm(40, 15, 2)
    assert not is_valid_for_e1rm(40, 8, None)
    with pytest.raises(ValueError):
        estimate_e1rm(40, 15, 2)


def test_hu010_session_e1rm_ignora_series_no_aptas():
    sets = [PerformedSet(60, 8, 2), PerformedSet(40, 20, 0), PerformedSet(60, 8, None)]
    assert session_e1rm(sets) == pytest.approx(80.0)


# --- HU-005 / HU-011: redondeo y carga sugerida -------------------------------

def test_hu005_redondeo_mancuernas_13_1_sugiere_12_5():
    assert DUMBBELLS.floor(13.1) == 12.5


def test_hu011_carga_redondeada_hacia_abajo():
    load = load_for_target(80, 8, 2, BARBELL)
    assert load <= raw_load_for_target(80, 8, 2) + 1e-9
    assert load in BARBELL.values


@settings(max_examples=300)
@given(
    e1rm=st.floats(min_value=10, max_value=300),
    reps=st.integers(min_value=1, max_value=12),
    rir=st.integers(min_value=0, max_value=4),
)
def test_hu011_reproducible_y_nunca_sobre_lo_teorico(e1rm, reps, rir):
    a = load_for_target(e1rm, reps, rir, BARBELL)
    b = load_for_target(e1rm, reps, rir, BARBELL)
    assert a == b
    assert a in BARBELL.values
    raw = raw_load_for_target(e1rm, reps, rir)
    assert a <= raw + 1e-9 or a == BARBELL.minimum


def test_peso_corporal_no_tiene_carga():
    bw = LoadOptions()
    assert bw.is_bodyweight
    assert load_for_target(100, 10, 2, bw) == 0.0


# --- HU-012 / S5: ajuste de la próxima serie ---------------------------------

@pytest.mark.parametrize(
    "reported, expected_load, reason",
    [
        (0, 55.0, Reason.MUCH_HARDER_THAN_PLANNED),
        (1, 57.5, Reason.HARDER_THAN_PLANNED),
        (2, 60.0, Reason.ON_TARGET),
        (3, 62.5, Reason.EASIER_THAN_PLANNED),
        (5, 62.5, Reason.EASIER_THAN_PLANNED),
    ],
)
def test_hu012_next_set_segun_rir(reported, expected_load, reason):
    d = next_set(60, 2, reported, BARBELL)
    assert d.load == expected_load
    assert d.reason is reason
    assert d.explanation


def test_hu031_velocidad_contradice_rir_alto():
    d = next_set(60, 2, 3, BARBELL, velocity_drop=0.25)
    assert d.reason is Reason.VELOCITY_MISMATCH
    assert d.load == 60.0
    assert "velocity_mismatch" in d.flags


# --- HU-015: tabla de decisión offline ---------------------------------------

def test_hu015_tabla_cubre_rir_0_a_5_y_coincide_con_el_motor():
    table = decision_table(60, 2, BARBELL)
    assert set(table) == {0, 1, 2, 3, 4, 5}
    for rir, decision in table.items():
        assert decision == next_set(60, 2, rir, BARBELL)


# --- HU-014: límites de seguridad --------------------------------------------

def test_hu014_tope_de_subida_entre_sesiones():
    d = next_session_load(e1rm=120, reps=8, target_rir=2, options=BARBELL, previous_load=60)
    assert d.load <= max(60 * 1.05, BARBELL.step_up(60))
    assert d.reason is Reason.CAPPED_INCREASE


@settings(max_examples=200)
@given(
    prev=st.sampled_from(BARBELL.values[2:]),
    e1rm=st.floats(min_value=20, max_value=400),
    reps=st.integers(min_value=3, max_value=12),
)
def test_hu014_nunca_sube_mas_del_tope(prev, e1rm, reps):
    d = next_session_load(e1rm, reps, 2, BARBELL, previous_load=prev)
    limit = max(prev * (1 + DEFAULT_CONFIG.max_session_increase_pct), BARBELL.step_up(prev))
    assert d.load <= limit + 1e-9
    assert BARBELL.minimum <= d.load <= BARBELL.maximum


def test_hu014_simulacion_12_semanas_sin_cargas_imposibles():
    """Perfil sintético que siempre reporta RIR alto: la carga sube, pero dentro de topes y del rango."""
    e1rm, load = 70.0, None
    loads = []
    for _ in range(12 * 3):
        d = next_session_load(e1rm, 8, 2, BARBELL, previous_load=load)
        load = d.load
        loads.append(load)
        performed = [PerformedSet(load, 8, 4)] * 3
        e1rm = update_e1rm(e1rm, session_e1rm(performed))
    assert all(BARBELL.minimum <= x <= BARBELL.maximum for x in loads)
    for a, b in zip(loads, loads[1:]):
        assert b <= max(a * 1.05, BARBELL.step_up(a)) + 1e-9


def test_clamp_bajada_maxima():
    load, capped = clamp_change(100, 50, BARBELL)
    assert load >= 85 - 2.5 and not capped


# --- Doble progresión ---------------------------------------------------------

def test_doble_progresion_suma_reps():
    d = double_progression(12.5, [10, 10, 9], [2, 2, 1], (8, 12), 2, DUMBBELLS)
    assert d.reason is Reason.REPS_UP and d.load == 12.5 and d.reps == 10


def test_doble_progresion_sube_carga_en_el_tope():
    d = double_progression(12.5, [12, 12, 12], [2, 2, 3], (8, 12), 2, DUMBBELLS)
    assert d.reason is Reason.LOAD_UP_TOP_OF_RANGE and d.load == 15.0 and d.reps == 8


# --- Fatiga, versión corta, deload -------------------------------------------

def test_hu012_fatiga_alta_reduce_series_y_carga():
    d = apply_readiness(3, 60, 4.25, BARBELL)
    assert d.reason is Reason.FATIGUE_HIGH and d.sets == 2 and d.load < 60


def test_fatiga_normal_no_cambia_nada():
    d = apply_readiness(3, 60, 2.5, BARBELL)
    assert d.sets == 3 and d.load == 60


def test_hu034_sesion_corta_prioriza_compuestos_y_no_cuenta_para_progresion():
    exs = [
        {"exercise_id": "a", "type": "aislado", "sets": 3, "order": 0},
        {"exercise_id": "b", "type": "compuesto", "sets": 3, "order": 1},
        {"exercise_id": "c", "type": "compuesto", "sets": 4, "order": 2},
        {"exercise_id": "d", "type": "aislado", "sets": 3, "order": 3},
    ]
    short = short_session(exs)
    assert [e["exercise_id"] for e in short] == ["b", "c", "a"]
    assert all(not e["counts_for_progression"] for e in short)
    assert short[1]["sets"] == 3


def test_hu013_deload_fin_de_bloque():
    assert deload_reason(4, 4, {}, []) is Reason.DELOAD_END_OF_BLOCK


def test_hu013_deload_por_caida_de_rendimiento():
    series = {"a": [100, 102, 96], "b": [80, 81, 76], "c": [50, 51, 52]}
    assert deload_reason(2, 4, series, []) is Reason.DELOAD_PERFORMANCE_DROP


def test_hu013_deload_por_fatiga_sostenida():
    assert deload_reason(2, 4, {}, [4.0, 4.5, 4.25]) is Reason.DELOAD_SUSTAINED_FATIGUE
    assert deload_reason(2, 4, {}, [4.0, 3.0, 4.25]) is None


def test_hu013_aplicar_deload():
    sets, load = apply_deload(4, 60, BARBELL)
    assert sets == 2 and load == 52.5


# --- HU-019 / HU-020: calentamiento y discos ----------------------------------

def test_hu019_calentamiento_barra_ascendente_y_bajo_la_carga():
    ws = warmup_sets(60, BARBELL, bar_kg=20)
    loads = [w for w, _ in ws]
    assert loads[0] == 20 and loads == sorted(loads) and loads[-1] < 60


def test_hu019_peso_corporal_sin_calentamiento_de_carga():
    assert warmup_sets(0, LoadOptions()) == []


def test_hu020_discos_por_lado():
    load, plates = plates_per_side(60, 20, {20: 1, 15: 1, 10: 1, 5: 2, 2.5: 2, 1.25: 1})
    assert load == 60 and sum(plates) == 20 and plates == [20]


def test_hu020_carga_no_armable_baja_a_la_mas_cercana():
    load, plates = plates_per_side(61, 20, {20: 1, 5: 1})
    assert load == 60 and plates == [20]


def test_hu020_bajo_la_barra_es_none():
    assert plates_per_side(10, 20, {5: 1}) is None
