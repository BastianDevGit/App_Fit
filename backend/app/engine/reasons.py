"""Códigos de razón y plantillas de explicación (HU-018, HU-103).

Cada decisión del motor guarda un código. La app muestra la explicación desde estas plantillas,
lo que funciona también sin conexión. El backoffice podrá editarlas más adelante.
"""

from enum import Enum


class Reason(str, Enum):
    ON_TARGET = "ON_TARGET"
    HARDER_THAN_PLANNED = "HARDER_THAN_PLANNED"
    MUCH_HARDER_THAN_PLANNED = "MUCH_HARDER_THAN_PLANNED"
    EASIER_THAN_PLANNED = "EASIER_THAN_PLANNED"
    CAPPED_INCREASE = "CAPPED_INCREASE"
    FATIGUE_HIGH = "FATIGUE_HIGH"
    FATIGUE_MODERATE = "FATIGUE_MODERATE"
    DELOAD_END_OF_BLOCK = "DELOAD_END_OF_BLOCK"
    DELOAD_PERFORMANCE_DROP = "DELOAD_PERFORMANCE_DROP"
    DELOAD_SUSTAINED_FATIGUE = "DELOAD_SUSTAINED_FATIGUE"
    INITIAL_FROM_CALIBRATION = "INITIAL_FROM_CALIBRATION"
    INITIAL_CONSERVATIVE = "INITIAL_CONSERVATIVE"
    REPS_UP = "REPS_UP"
    LOAD_UP_TOP_OF_RANGE = "LOAD_UP_TOP_OF_RANGE"
    SHORT_SESSION = "SHORT_SESSION"
    REENTRY_AFTER_BREAK = "REENTRY_AFTER_BREAK"
    VELOCITY_MISMATCH = "VELOCITY_MISMATCH"
    E1RM_UP = "E1RM_UP"
    E1RM_DOWN = "E1RM_DOWN"


TEMPLATES: dict[Reason, str] = {
    Reason.ON_TARGET: "Te quedaron {reported} reps, justo lo planeado. Mantenemos la carga.",
    Reason.HARDER_THAN_PLANNED: (
        "Te costó un poco más de lo previsto (quedaban {reported} y el objetivo era {target}). "
        "Bajamos la carga para que la siguiente quede en el rango."
    ),
    Reason.MUCH_HARDER_THAN_PLANNED: (
        "Te costó bastante más de lo previsto (quedaban {reported} y el objetivo era {target}). "
        "Bajamos la carga dos saltos para cuidar la técnica."
    ),
    Reason.EASIER_THAN_PLANNED: (
        "Te sobró margen (quedaban {reported} y el objetivo era {target}). Subimos el salto mínimo de tus implementos."
    ),
    Reason.CAPPED_INCREASE: "Vas fuerte, pero limitamos la subida entre sesiones para progresar con seguridad.",
    Reason.FATIGUE_HIGH: "Llegas con fatiga alta. Hoy hacemos una serie menos y bajamos un poco la carga.",
    Reason.FATIGUE_MODERATE: "Llegas algo cansado. Bajamos levemente la carga de hoy.",
    Reason.DELOAD_END_OF_BLOCK: "Terminaste el bloque: esta semana es de descarga para recuperarte y seguir progresando.",
    Reason.DELOAD_PERFORMANCE_DROP: "Tu fuerza estimada bajó en varias sesiones. Una semana de descarga te ayudará a recuperarte.",
    Reason.DELOAD_SUSTAINED_FATIGUE: "Llevas varios días con fatiga alta. Programamos una semana de descarga.",
    Reason.INITIAL_FROM_CALIBRATION: "Carga calculada a partir de tu prueba inicial.",
    Reason.INITIAL_CONSERVATIVE: "Como no hiciste la prueba inicial, partimos con una carga conservadora para tu nivel.",
    Reason.REPS_UP: "Mantienes la carga y sumas repeticiones hasta llegar al tope del rango.",
    Reason.LOAD_UP_TOP_OF_RANGE: "Llegaste al tope del rango de reps con margen: subimos la carga y vuelves al inicio del rango.",
    Reason.SHORT_SESSION: "Versión corta de tu sesión: lo esencial, sin afectar tu progresión.",
    Reason.REENTRY_AFTER_BREAK: "Volviste después de una pausa: retomamos con cargas más suaves.",
    Reason.VELOCITY_MISMATCH: (
        "Tu velocidad bajó bastante al final, así que quizás te quedaban menos reps. Lo tomamos con cautela."
    ),
    Reason.E1RM_UP: "Tu fuerza estimada subió desde la última sesión, así que la carga sube.",
    Reason.E1RM_DOWN: "Tu fuerza estimada bajó un poco desde la última sesión; ajustamos la carga a la baja.",
}


def explain(reason: Reason, **kwargs: object) -> str:
    text = TEMPLATES[reason]
    try:
        return text.format(**kwargs)
    except KeyError:
        return text
