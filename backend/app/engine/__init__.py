"""Motor de progresión determinista (EP-02).

Única fuente de verdad para e1RM, cargas, ajustes, deloads, calentamiento y discos.
El LLM nunca calcula cargas: este paquete sí, y es puro (sin E/S) para poder testearlo.
"""

ENGINE_VERSION = "0.1.0"
