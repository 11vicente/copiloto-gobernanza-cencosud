"""
OBSERVABILIDAD/AUTO.PY — activacion del trazado con un solo import.

POR QUE EXISTE: app_agente.py no debe reescribirse (restriccion del proyecto).
Con este modulo, instrumentar la API web cuesta UNA linea aditiva:

    import observabilidad.auto  # activa trazado JSONL (EP3)

Al importarse, re-ata el grafo del agente con el trazador (zero-touch) y fija
el origen por defecto de las trazas en "api", para que en el dashboard se
distinga el trafico web real de las corridas de evaluacion/carga.
"""

from observabilidad.instrumentacion import TRAZADOR, activar_observabilidad

TRAZADOR.origen_por_defecto = "api"
activar_observabilidad()
