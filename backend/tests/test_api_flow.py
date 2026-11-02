"""Flujo completo por API: onboarding → bloque → sesión → series → cierre → progreso → dieta → derechos."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

GYM = {"environment": "gimnasio", "barra": {"bar_kg": 20, "plates": {"20": 1, "15": 1, "10": 1, "5": 2, "2.5": 2, "1.25": 1}},
       "mancuernas": {"min": 2.5, "max": 20, "step": 2.5}, "polea": {"min": 5, "max": 80, "step": 5}, "simple": ["banco", "rack"]}
PROFILE = {"age": 29, "weight_kg": 68, "level": "intermedio", "goal": "fuerza", "days_per_week": 3, "minutes_per_session": 45}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def onboard(c: TestClient, calibrate: bool = True) -> str:
    uid = c.post("/api/users", json={"health": True, "camera": True}).json()["id"]
    assert c.post(f"/api/users/{uid}/parq", json={"answers": [False] * 7}).json()["cleared"]
    assert c.put(f"/api/users/{uid}/profile", json=PROFILE).status_code == 200
    assert c.put(f"/api/users/{uid}/equipment", json=GYM).status_code == 200
    if calibrate:
        r = c.post(f"/api/users/{uid}/calibration", json={"sets": [{"exercise_id": "EJ-001", "load": 50, "reps": 8, "rir": 3}]})
        assert r.json()["estimates"][0]["e1rm"] == pytest.approx(50 * (1 + 11 / 30), abs=0.1)
    else:
        c.post(f"/api/users/{uid}/calibration", json={"skipped": True})
    return uid


def test_hu002_sin_consentimiento_de_salud_no_hay_cuenta(client):
    r = client.post("/api/users", json={"health": False})
    assert r.status_code == 422


def test_hu003_parq_con_si_bloquea_el_plan_hasta_autorizacion(client):
    uid = client.post("/api/users", json={"health": True}).json()["id"]
    r = client.post(f"/api/users/{uid}/parq", json={"answers": [True] + [False] * 6})
    assert not r.json()["cleared"] and r.json()["message"]
    client.put(f"/api/users/{uid}/profile", json=PROFILE)
    client.put(f"/api/users/{uid}/equipment", json=GYM)
    assert client.post(f"/api/users/{uid}/blocks").status_code == 409
    client.post(f"/api/users/{uid}/parq", json={"answers": [True] + [False] * 6, "authorized": True})
    assert client.post(f"/api/users/{uid}/blocks").status_code == 201


def test_hu004_edad_bajo_el_minimo(client):
    uid = client.post("/api/users", json={"health": True}).json()["id"]
    assert client.put(f"/api/users/{uid}/profile", json={**PROFILE, "age": 14}).status_code == 422


def test_flujo_completo_de_entrenamiento(client):
    uid = onboard(client)
    block = client.post(f"/api/users/{uid}/blocks").json()
    assert block["source"] == "llm" and len(block["days"]) == 3

    client.post(f"/api/users/{uid}/readiness", json={"fatigue": 2, "sleep": 2, "soreness": 2, "stress": 2})
    plan = client.get(f"/api/users/{uid}/sessions/today", params={"week": 1, "day": 0}).json()
    assert plan["engine_version"] and plan["prompt_version"] == "blocks-0.1"
    squat = next(e for e in plan["exercises"] if e["exercise_id"] == "EJ-001")
    assert squat["notes"][0]["reason"] == "INITIAL_FROM_CALIBRATION"
    assert squat["plates"]["bar_kg"] == 20 and squat["warmup"][0]["load"] == 20
    assert set(squat["decision_table"]) == {"0", "1", "2", "3", "4", "5"}
    assert plan["estimated_minutes"] <= PROFILE["minutes_per_session"]

    key, load, rir = plan["session_key"], squat["load"], squat["target_rir"]
    r = client.post(f"/api/users/{uid}/sets", json={"session_key": key, "exercise_id": "EJ-001", "set_index": 1,
                                                    "load": load, "reps": squat["reps"], "rir": rir + 2, "target_rir": rir})
    nxt = r.json()["next_set"]
    assert nxt["load"] > load and nxt["reason"] == "EASIER_THAN_PLANNED"
    assert nxt["load"] == squat["decision_table"][str(rir + 2)]["load"]  # tabla offline = motor

    for i in (2, 3):
        client.post(f"/api/users/{uid}/sets", json={"session_key": key, "exercise_id": "EJ-001", "set_index": i,
                                                    "load": nxt["load"], "reps": squat["reps"], "rir": rir, "target_rir": rir})
    done = client.post(f"/api/users/{uid}/sessions/{key}/complete").json()
    assert done["e1rm_updates"] and done["personal_records"]

    prog = client.get(f"/api/users/{uid}/progress").json()
    sq = next(p for p in prog["e1rm"] if p["exercise_id"] == "EJ-001")
    assert len(sq["points"]) == 2 and prog["streak"]["this_week"] == 1

    plan2 = client.get(f"/api/users/{uid}/sessions/today", params={"week": 1, "day": 0}).json()
    squat2 = next(e for e in plan2["exercises"] if e["exercise_id"] == "EJ-001")
    assert squat2["load"] <= max(nxt["load"] * 1.05, nxt["load"] + 2.5)  # tope de subida (HU-014)


def test_hu016_correccion_es_un_evento_nuevo(client):
    uid = onboard(client)
    client.post(f"/api/users/{uid}/blocks")
    ev = client.post(f"/api/users/{uid}/sets", json={"session_key": "k", "exercise_id": "EJ-001", "set_index": 1,
                                                     "load": 50, "reps": 8, "rir": 2}).json()["event_id"]
    fix = client.post(f"/api/users/{uid}/sets/{ev}/corrections", json={"reps": 6})
    assert fix.status_code == 201 and fix.json()["corrects"] == ev
    exported = client.get(f"/api/users/{uid}/export").json()["performed_set_events"]
    assert [e["kind"] for e in exported] == ["original", "correction"]
    assert exported[0]["reps"] == 8  # el original se conserva


def test_hu034_dia_dificil_version_corta(client):
    uid = onboard(client, calibrate=False)
    client.post(f"/api/users/{uid}/blocks")
    full = client.get(f"/api/users/{uid}/sessions/today").json()
    short = client.get(f"/api/users/{uid}/sessions/today", params={"hard_day": True}).json()
    assert len(short["exercises"]) <= 3 and short["estimated_minutes"] <= 20 < full["estimated_minutes"]
    assert all(not e["counts_for_progression"] for e in short["exercises"])
    assert full["exercises"][0]["notes"][0]["reason"] == "INITIAL_CONSERVATIVE"


def test_hu012_fatiga_alta_ajusta_la_sesion(client):
    uid = onboard(client)
    client.post(f"/api/users/{uid}/blocks")
    base = client.get(f"/api/users/{uid}/sessions/today").json()
    client.post(f"/api/users/{uid}/readiness", json={"fatigue": 5, "sleep": 4, "soreness": 4, "stress": 5})
    tired = client.get(f"/api/users/{uid}/sessions/today").json()
    assert tired["exercises"][0]["sets"] == base["exercises"][0]["sets"] - 1


def test_hu013_ultima_semana_es_deload(client):
    uid = onboard(client)
    client.post(f"/api/users/{uid}/blocks")
    plan = client.get(f"/api/users/{uid}/sessions/today", params={"week": 4}).json()
    assert plan["deload"] == "DELOAD_END_OF_BLOCK"


def test_hu035_y_hu036_cambio_de_ejercicio_y_seguridad(client):
    uid = onboard(client)
    busy = client.post(f"/api/users/{uid}/swap", json={"exercise_id": "EJ-001", "reason": "maquina_ocupada"}).json()
    assert busy["alternatives"][0]["exercise_id"] == "EJ-002"
    mild = client.post(f"/api/users/{uid}/swap", json={"exercise_id": "EJ-001", "reason": "me_molesta", "zone": "rodilla",
                                                       "text": "me molesta un poco"}).json()
    assert mild["action"] == "cambiar" and "rodilla" in mild["avoid_zones"]
    acute = client.post(f"/api/users/{uid}/swap", json={"exercise_id": "EJ-001", "reason": "me_molesta",
                                                        "text": "sentí un tirón y me duele mucho"}).json()
    assert acute["action"] == "derivar" and "131" in acute["message"]


def test_dieta_parse_porcion_guardar_y_metas(client):
    uid = onboard(client)
    parsed = client.post("/api/nutrition/parse", json={"text": "una marraqueta con palta"}).json()
    items = parsed["items"]
    palta = next(i for i in items if i["food_id"] == "palta")
    assert palta["needs_portion"]
    fixed = client.post("/api/nutrition/portion", json={"item": palta, "unit": "cucharada", "quantity": 2}).json()
    items = [fixed if i["food_id"] == "palta" else i for i in items]
    day = client.post(f"/api/users/{uid}/meals", json={"meal": "desayuno", "items": items}).json()
    assert day["totals"]["kcal"] == 270 + 80
    today = client.get(f"/api/users/{uid}/nutrition/today").json()
    assert today["targets"]["kcal"]["min"] >= today["targets"]["floor_kcal"]


def test_dieta_no_guarda_si_falta_porcion(client):
    uid = onboard(client)
    items = client.post("/api/nutrition/parse", json={"text": "palta"}).json()["items"]
    assert client.post(f"/api/users/{uid}/meals", json={"items": items}).status_code == 422


def test_hu093_y_hu094_exportar_y_eliminar(client):
    uid = onboard(client)
    assert client.get(f"/api/users/{uid}/export").json()["user"]["id"] == uid
    assert client.delete(f"/api/users/{uid}").status_code == 204
    assert client.get(f"/api/users/{uid}/progress").status_code == 404
