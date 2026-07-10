"""
Paquete de OBSERVABILIDAD del agente (EP3) — capa aditiva, cero cambios de
logica en agente.py. Ver instrumentacion.py para el diseño completo.
"""

from observabilidad.instrumentacion import (  # noqa: F401 — API publica del paquete
    RUTA_TRAZAS,
    TRAZADOR,
    activar_observabilidad,
    ejecutar_agente_observado,
)
