"""Tests de catálogo, bloques con LLM (EP-03), dieta (EP-06) y capa de seguridad (A5)."""

import json

import pytest

from app.catalog.service import Barbell, Stepped, UserEquipment, get_catalog
from app.llm.blocks import BlockRequest, generate_block, validate_draft
from app.llm.provider import TRACER, BudgetExceeded, FakeProvider, TokenBudget
from app.nutrition.pipeline import TargetsInput, choose_portion, daily_targets, parse_meal
from app.safety.classifier import check_text

CAT = get_catalog()
GYM = UserEquipment(environment="gimnasio", barra=Barbell(), mancuernas=Stepped(min=2.5, max=20, step=2.5),
                    polea=Stepped(min=5, max=80, step=5), simple=["banco", "rack"])
HOME = UserEquipment(environment="casa", mancuernas=Stepped(min=2, max=12, step=2), simple=["bandas"])
REQ = BlockRequest(days_per_week=3, minutes_per_session=45, level="intermedio", goal="fuerza")


# --- Catálogo -------------------------------------------------------------------

def test_hu005_cambio_a_casa_solo_ejercicios_compatibles():
    ids = {e.id for e in CAT.compatible(HOME, "principiante")}
    assert "EJ-001" not in ids  # barra + rack + gimnasio
    assert "EJ-002" in ids      # goblet con mancuernas


def test_hu036_zona_marcada_excluye_ejercicios():
    ids = {e.id for e in CAT.compatible(GYM, "intermedio", ["rodilla"])}
    assert not any(CAT.get(i).pattern == "rodilla" and "rodilla" in CAT.get(i).caution_zones for i in ids)


def test_hu035_alternativas_mismo_patron_primero_la_regresion():
    alts = CAT.alternatives("EJ-001", GYM, "intermedio")
    assert alts and alts[0].id == "EJ-002"
    assert all(a.pattern == "rodilla" and a.id != "EJ-001" for a in alts)


def test_redondeo_segun_implemento_del_ejercicio():
    assert CAT.options(CAT.get("EJ-002"), HOME).values[:3] == (2.0, 4.0, 6.0)
    assert CAT.options(CAT.get("EJ-023"), HOME).is_bodyweight


# --- Bloques con LLM --------------------------------------------------------------

def test_hu021_bloque_valido_solo_con_ids_del_catalogo():
    draft, meta = generate_block(REQ, GYM, CAT, FakeProvider())
    assert meta["source"] == "llm" and len(draft.days) == 3
    for day in draft.days:
        for item in day.exercises:
            assert item.exercise_id in CAT
            assert CAT.incompatibility(CAT.get(item.exercise_id), GYM, "intermedio", []) is None


def test_hu021_id_inventado_se_rechaza_y_se_reintenta():
    draft, meta = generate_block(REQ, GYM, CAT, FakeProvider(fail_first=1))
    assert meta["source"] == "llm"
    assert any("EJ-999" in e for e in meta["attempts"][0]["errors"])


def test_hu021_tras_reintentos_usa_plantilla_base():
    draft, meta = generate_block(REQ, GYM, CAT, FakeProvider(fail_first=99))
    assert meta["source"] == "fallback" and len(meta["attempts"]) == 3
    assert all(i.exercise_id in CAT for d in draft.days for i in d.exercises)


def test_hu022_schema_invalido_se_rechaza():
    _, errors = validate_draft(json.dumps({"weeks": 9, "days": []}), REQ, GYM, CAT)
    assert errors and all(e.startswith("schema") for e in errors)


def test_hu004_sesiones_respetan_el_tiempo():
    req = BlockRequest(days_per_week=3, minutes_per_session=30, level="intermedio", goal="fuerza")
    draft, meta = generate_block(req, GYM, CAT, FakeProvider())
    _, errors = validate_draft(json.dumps(draft.model_dump()), req, GYM, CAT)
    assert errors == []


@pytest.mark.parametrize("days", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("eq", [GYM, HOME])
@pytest.mark.parametrize("level", ["principiante", "intermedio", "avanzado"])
def test_hu022_eval_bloques_validos_en_todas_las_combinaciones(days, eq, level):
    """Mini-eval: el criterio de Fase 4 es ≥ 98% de bloques válidos y 100% de restricciones de seguridad."""
    req = BlockRequest(days_per_week=days, minutes_per_session=45, level=level, goal="salud", avoid_zones=["hombro"])
    draft, meta = generate_block(req, eq, CAT, FakeProvider())
    for d in draft.days:
        for i in d.exercises:
            assert "hombro" not in CAT.get(i.exercise_id).caution_zones


def test_hu024_cada_llamada_queda_trazada_con_prompt_version():
    before = len(TRACER.traces)
    generate_block(REQ, GYM, CAT, FakeProvider())
    assert len(TRACER.traces) > before and TRACER.traces[-1].prompt_version == "blocks-0.1"


def test_hu026_presupuesto_bloquea_solo_funciones_no_criticas():
    b = TokenBudget(monthly_limit=100)
    b.charge("u", 90)
    with pytest.raises(BudgetExceeded):
        b.charge("u", 20)
    b.charge("u", 20, critical=True)  # sesiones y progresión nunca se degradan


# --- Dieta ----------------------------------------------------------------------------

def test_hu060_extrae_alimentos_y_calcula_macros_deterministas():
    r = parse_meal("2 huevos y una taza de leche", FakeProvider())
    assert r.complete and r.totals.kcal == 143 + 146
    assert {i.food_id for i in r.items} == {"huevo", "leche_entera"}


def test_hu061_porcion_ambigua_pide_eleccion():
    r = parse_meal("una marraqueta con palta", FakeProvider())
    palta = next(i for i in r.items if i.food_id == "palta")
    assert palta.needs_portion and palta.macros is None and not r.complete
    resolved = choose_portion(palta, "cucharada", 2)
    assert resolved.grams == 50 and resolved.macros.kcal == 80


def test_hu063_porciones_chilenas():
    r = parse_meal("un plato grande de cazuela", FakeProvider())
    assert r.items[0].grams == 550


def test_hu064_metas_respetan_minimo_seguro():
    t = daily_targets(TargetsInput(weight_kg=45, goal="bajar_grasa", sex="f", height_cm=150, age=30))
    assert t.kcal.min >= 1200 and t.kcal.min >= t.floor_kcal
    assert "no pasa nada" in t.message


# --- Seguridad --------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["siento dolor en el pecho", "me mareé en la serie", "sentí un tirón en la espalda",
                                  "tengo hormigueo en el brazo", "creo que me duele raro la rodilla"])
def test_hu025_menciones_de_riesgo_derivan(text):
    assert check_text(text).action == "derivar"


@pytest.mark.parametrize("text", ["me molesta un poco el hombro", "hoy entrené bien", ""])
def test_hu025_texto_normal_continua(text):
    assert check_text(text).action == "continuar"


def test_hu036_dolor_agudo_seleccionado_deriva_siempre():
    assert check_text("", acute_selected=True).action == "derivar"
