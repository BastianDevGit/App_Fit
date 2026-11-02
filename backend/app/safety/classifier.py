"""Capa de seguridad ante menciones médicas en texto libre (A5, HU-025).

Versión inicial basada en reglas: alta sensibilidad a propósito. Ante la duda, deriva.
Se reemplazará o complementará con un clasificador evaluado, manteniendo esta interfaz.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

from app.llm.provider import _norm

RED_FLAGS = {
    "dolor en el pecho": r"dolor (en el |de )?pecho|opresion (en el )?pecho|me aprieta el pecho",
    "falta de aire": r"falta (de )?aire|no puedo respirar|me ahogo|dificultad para respirar",
    "mareo o desmayo": r"\bmare(o|e|ad)|desmay|perdi el conocimiento|se me nublo la vista",
    "palpitaciones": r"palpitacion|el corazon (muy )?(rapido|acelerado)|taquicardia",
    "hormigueo o adormecimiento": r"hormigueo|adormec|no siento (la|el|los|las)",
    "dolor agudo": r"dolor (agudo|punzante|fuerte|intenso)|me duele (mucho|harto|demasiado)|crujio y (me )?duele|senti un tiron",
    "hinchazón": r"hinchad|inflamad|se me hincho",
    "lesión": r"(me )?lesione|esguince|desgarr|fractur|luxac",
}
UNCERTAIN = r"\b(dolor|duele|molest\w*|raro|extrano)\b"


class SafetyResult(BaseModel):
    action: Literal["continuar", "derivar"]
    matched: list[str]
    uncertain: bool
    message: str


DERIVE_MSG = ("Por lo que nos contaste, lo más seguro es no seguir entrenando hoy y consultar a un profesional de salud. "
              "Si el dolor es muy intenso, aparece en el pecho o te cuesta respirar, busca atención de urgencia (SAMU 131).")


def check_text(text: str, acute_selected: bool = False) -> SafetyResult:
    t = _norm(text or "")
    matched = [label for label, rx in RED_FLAGS.items() if re.search(rx, t)]
    if matched or acute_selected:
        return SafetyResult(action="derivar", matched=matched or ["dolor agudo (seleccionado)"], uncertain=False,
                            message=DERIVE_MSG)
    uncertain = bool(re.search(UNCERTAIN, t)) and bool(re.search(r"\b(no se|no estoy segur|creo|parece)\b", t))
    if uncertain:
        # Prima la derivación ante la duda (HU-025, escenario 2).
        return SafetyResult(action="derivar", matched=[], uncertain=True, message=DERIVE_MSG)
    return SafetyResult(action="continuar", matched=[], uncertain=False, message="")
