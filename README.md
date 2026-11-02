# Personal Trainer IA — MVP

App fitness con un Personal Trainer impulsado por IA: el LLM arma el bloque eligiendo ejercicios de un
catálogo cerrado, un **motor determinista en Python** calcula todas las cargas, y la visión por
computador (Fase 1/6) da feedback de técnica.

Este repositorio contiene el **primer corte vertical del backend**: se puede ejecutar, probar y evaluar
sin teléfono ni API key.

## Qué incluye

| Módulo | Qué hace | Historias |
|---|---|---|
| `app/engine` | e1RM, carga por RIR, redondeo a implementos, ajuste de la siguiente serie, tablas de decisión offline, topes de seguridad, fatiga, deload, doble progresión, calentamiento y discos por lado | HU-010 a HU-020 |
| `app/catalog` | Catálogo cerrado, compatibilidad por implementos/nivel/zonas, alternativas del mismo patrón | HU-005, HU-021, HU-035, HU-036 |
| `app/llm` | Abstracción de proveedor, proveedor simulado, validación con Pydantic + reglas, reintentos con errores, plantilla base, trazas y presupuesto de tokens | HU-021 a HU-024, HU-026 |
| `app/nutrition` | Extracción de alimentos (LLM) → búsqueda en base → macros deterministas; porción ambigua; metas con mínimos seguros | HU-060 a HU-064 |
| `app/safety` | Clasificador de menciones médicas (reglas, alta sensibilidad) | HU-025 |
| `app/db.py` | SQLite en dev / PostgreSQL en prod, `tenant_id` en todo (A7), series append-only con correcciones como eventos (A1), versionado de decisiones (A2) | HU-016, HU-017, HU-091, HU-092 |
| `app/main.py` | API REST + exportar y eliminar datos | HU-001 a HU-006, HU-030 a HU-036, HU-070 a HU-072, HU-093, HU-094 |
| `app/static` | Demo web móvil para recorrer el flujo completo | — |

## Ejecutar

```bash
cd backend
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

- Demo: <http://localhost:8000>
- Documentación interactiva de la API: <http://localhost:8000/docs>
- Tests: `pytest` (98 tests, incluye pruebas de propiedades con Hypothesis y una simulación de 12 semanas)

Con PostgreSQL: `docker compose up` y luego aplicar `backend/sql/rls.sql` para Row Level Security.

## Recorrido de la demo

1. **Perfil**: consentimientos, PAR-Q, perfil, implementos y prueba inicial → se genera el bloque.
2. **Hoy**: check-in de fatiga, día difícil opcional → sesión con cargas, calentamiento y discos.
   Registra cada serie con "¿cuántas reps más podías hacer?" y verás la carga de la siguiente y el "¿por qué?".
   Prueba "Máquina ocupada" y "Me molesta" (con "dolor agudo" la sesión se pausa y deriva).
3. **Terminar sesión**: actualiza la fuerza estimada y detecta récords.
4. **Dieta**: escribe lo que comiste; si la porción es ambigua te la pregunta.
5. **Progreso**: racha semanal y fuerza estimada por ejercicio.

## Lo que todavía es provisional

| Pieza | Estado | Para cerrarlo |
|---|---|---|
| Catálogo | 25 ejercicios de ejemplo con IDs del rango EJ-0xx | Importar `catalogo_ejercicios_mvp.xlsx` (EJ-001 a EJ-055) |
| LLM | Proveedor simulado con reglas (`LLM_PROVIDER=fake`) | Implementar `complete_json` con el SDK elegido y su API key |
| Base nutricional | ~25 alimentos con valores aproximados | Cargar USDA FoodData Central + tablas chilenas con su fuente |
| Parámetros del motor | Hipótesis iniciales en `ProgressionConfig` | Calibrar con datos de la beta y un experto |
| Clasificador de seguridad | Reglas por palabras clave | Evaluar con un set de frases etiquetadas |
| Visión, app Flutter, pagos, foto de comida | Fuera de este corte | Fases 1, 5, 6 y 8 del plan |

## Principios que el código hace cumplir

- El LLM nunca calcula cargas ni macros; si entrega un ID fuera del catálogo, se rechaza y se reintenta.
- La app móvil no tendrá lógica de progresión: usa las `decision_table` precalculadas por el backend.
- Las series ejecutadas no se editan: las correcciones son eventos nuevos.
- Cada carga sugerida guarda `engine_version`, `prompt_version` y `cv_config_version`.
