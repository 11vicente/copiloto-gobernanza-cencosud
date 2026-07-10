"""
=============================================================================
EVALUACION/EVALUAR.PY — EP3 (IE1): precision, consistencia y errores
=============================================================================
Corre el agente REAL sobre el set de casos de casos.json, VARIAS veces por
caso y con variantes de fraseo distintas (escenarios con variabilidad de
datos), y calcula las tres metricas que exige el indicador IE1:

    1. PRECISION: % de corridas cuyo resultado cumple el criterio de acierto
       del caso (contenido esperado, cifra exacta, rehusa correcta y/o
       herramientas correctas invocadas).
    2. CONSISTENCIA, en tres dimensiones complementarias:
       - de veredicto:  ¿las repeticiones aciertan/fallan de forma estable?
         (proporcion de corridas que coinciden con el veredicto mayoritario)
       - textual:       similitud media entre las respuestas de las
         repeticiones (difflib.SequenceMatcher, stdlib). Es una similitud
         LEXICA declarada como aproximacion: dos respuestas pueden decir lo
         mismo con otras palabras y puntuar <1.0.
       - operacional:   ¿el agente uso la MISMA secuencia de herramientas en
         cada repeticion? (n° de rutas distintas observadas)
    3. FRECUENCIA DE ERRORES, tipificada para poder analizarla (IE3/IE4):
       error_ejecucion (excepcion/limite del grafo), alucinacion (respondio
       en vez de rehusar), rehusa_incorrecta (rehuso habiendo informacion),
       herramienta_no_usada (p. ej. calculo mental en vez de calculadora),
       contenido_faltante y cifra_incorrecta.

POR QUE CRITERIOS DETERMINISTAS Y NO UN "JUEZ LLM":
    - Reproducibles: el evaluador de la asignatura puede verificar cada
      veredicto a mano contra la traza (transparencia).
    - Sin costo ni cuota extra: juzgar con otro LLM duplicaria las llamadas
      (el free tier de GitHub Models tiene limites estrictos por dia).
    - Sin ruido: un juez LLM introduce SU propia inconsistencia justo en la
      metrica que queremos medir. Limitacion declarada: los criterios
      lexicales son mas gruesos que una evaluacion semantica humana.

POR QUE historial=None EN CADA CORRIDA: cada repeticion debe ser
independiente; si compartieran memoria de corto plazo, la repeticion n°2
"veria" la respuesta de la n°1 y la consistencia quedaria contaminada.
(La memoria de LARGO plazo del agente si sigue activa —es su comportamiento
real de produccion— y se declara en el informe.)

Uso (desde la raiz del proyecto, SIEMPRE con el python del venv):
    venv\\Scripts\\python.exe evaluacion\\evaluar.py
    venv\\Scripts\\python.exe evaluacion\\evaluar.py --repeticiones 3 --pausa 8
    venv\\Scripts\\python.exe evaluacion\\evaluar.py --solo C1
    venv\\Scripts\\python.exe evaluacion\\evaluar.py --desde 4   (reanudar bateria)

Salidas:
    - logs/agente_trazas.jsonl        (una traza por corrida, origen="evaluacion")
    - evaluacion/resultados_evaluacion.json (metricas por caso y globales)
=============================================================================
"""

import argparse
import json
import re
import sys
import time
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path

# El script vive en evaluacion/ pero importa paquetes de la raiz del proyecto;
# agregamos la raiz al path para poder ejecutarlo desde cualquier directorio.
RUTA_RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUTA_RAIZ))

from observabilidad.instrumentacion import ejecutar_agente_observado  # noqa: E402

RUTA_CASOS = Path(__file__).parent / "casos.json"
RUTA_RESULTADOS = Path(__file__).parent / "resultados_evaluacion.json"

# Frase (normalizada) con la que el prompt del agente le exige rehusar cuando
# no hay evidencia en los documentos. Es el ancla de la metrica anti-alucinacion.
FRASE_REHUSA = "no tengo informacion suficiente"


# =============================================================================
# 1. UTILIDADES DE COMPARACION
# =============================================================================
def normalizar(texto: str) -> str:
    """Minusculas y sin tildes: 'Información' == 'informacion'. Evita que un
    acento decida un veredicto (el LLM alterna tildes segun la corrida)."""
    descompuesto = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in descompuesto if unicodedata.category(c) != "Mn")


def evaluar_corrida(caso: dict, registro: dict) -> tuple[bool, str | None]:
    """Aplica el criterio del caso a UNA corrida y tipifica el fallo.

    El ORDEN de las verificaciones importa y esta pensado para diagnosticar:
    primero errores de ejecucion (no hubo respuesta que juzgar), luego el
    contrato anti-alucinacion, luego el comportamiento (herramientas) y al
    final el contenido. Asi cada fallo recibe SU causa mas especifica.

    Returns:
        (acierto, tipo_fallo): tipo_fallo es None cuando acierto=True.
    """
    criterio = caso["criterio"]

    if not registro.get("exito"):
        return False, "error_ejecucion"

    respuesta = registro.get("respuesta") or ""
    resp_norm = normalizar(respuesta)
    usadas = {h["nombre"] for h in registro.get("herramientas", [])}
    rehusa = FRASE_REHUSA in resp_norm

    # Casos trampa: la UNICA respuesta correcta es rehusar.
    if criterio.get("debe_rehusar"):
        return (True, None) if rehusa else (False, "alucinacion")

    # Caso normal que rehuso: fallo de recuperacion/uso de contexto.
    if rehusa:
        return False, "rehusa_incorrecta"

    # Comportamiento: ¿invoco las herramientas que el caso exige?
    faltantes = [h for h in criterio.get("herramientas_esperadas", [])
                 if h not in usadas]
    if faltantes:
        return False, "herramienta_no_usada:" + ",".join(faltantes)

    # Contenido: al menos una de las palabras clave esperadas.
    esperados = criterio.get("debe_contener_alguno")
    if esperados and not any(normalizar(e) in resp_norm for e in esperados):
        return False, "contenido_faltante"

    # Cifra exacta (casos de calculo verificable).
    patron = criterio.get("regex_obligatorio")
    if patron and not re.search(patron, respuesta):
        return False, "cifra_incorrecta"

    return True, None


def similitud_textual_media(respuestas: list[str]) -> float | None:
    """Similitud lexica media entre TODOS los pares de respuestas de un caso.
    1.0 = respuestas identicas caracter a caracter. Se declara como
    aproximacion lexica (no semantica) en el informe."""
    validas = [r for r in respuestas if r]
    pares = list(combinations(validas, 2))
    if not pares:
        return None
    total = sum(SequenceMatcher(None, a, b).ratio() for a, b in pares)
    return round(total / len(pares), 3)


# =============================================================================
# 2. EJECUCION DE UN CASO (N repeticiones con variantes rotadas)
# =============================================================================
def correr_caso(caso: dict, repeticiones: int, pausa: float) -> dict:
    """Corre un caso N veces rotando sus variantes de fraseo (round-robin).

    POR QUE ROTAR VARIANTES: la rubrica exige medir "en escenarios con
    variabilidad de datos". Rotar el fraseo en cada repeticion mide la
    estabilidad del agente ante entradas equivalentes pero distintas, que es
    exactamente esa variabilidad (ademas de la aleatoriedad propia del LLM).

    Manejo de rate limit (free tier de GitHub Models): si la corrida termina
    en error 429/RateLimit se espera 70 s y se reintenta UNA vez. La traza
    fallida queda igual en el JSONL: el limite de cuota es un dato real de
    operacion, no algo que esconder.
    """
    corridas = []
    for i in range(repeticiones):
        variante_idx = i % len(caso["variantes"])
        pregunta = caso["variantes"][variante_idx]
        print(f"  corrida {i + 1}/{repeticiones} (variante {variante_idx + 1}): "
              f"{pregunta[:60]}...")

        registro = ejecutar_agente_observado(
            pregunta,
            historial=None,  # independencia entre corridas (ver docstring)
            origen="evaluacion",
            caso_id=caso["id"],
            variante=str(variante_idx + 1),
        )

        # Reintento unico ante rate limit: pausa larga y misma variante.
        error = registro.get("error") or {}
        if not registro.get("exito") and "ratelimit" in str(error.get("tipo", "")).lower():
            print("    [!] Rate limit del proveedor: espero 70 s y reintento una vez...")
            time.sleep(70)
            registro = ejecutar_agente_observado(
                pregunta, historial=None, origen="evaluacion",
                caso_id=caso["id"], variante=str(variante_idx + 1),
            )

        acierto, tipo_fallo = evaluar_corrida(caso, registro)
        print(f"    -> {'ACIERTO' if acierto else 'FALLO(' + str(tipo_fallo) + ')'} | "
              f"{registro.get('latencia_total_s')}s | "
              f"herramientas={[h['nombre'] for h in registro.get('herramientas', [])]}")

        corridas.append({
            "repeticion": i + 1,
            "variante": variante_idx + 1,
            "pregunta": pregunta,
            "acierto": acierto,
            "tipo_fallo": tipo_fallo,
            "latencia_total_s": registro.get("latencia_total_s"),
            "tokens_total": registro.get("tokens_total"),
            "num_pasos_react": registro.get("num_pasos_react"),
            "ruta_herramientas": [h["nombre"] for h in registro.get("herramientas", [])],
            "respuesta": registro.get("respuesta"),
            "trace_id": registro.get("trace_id"),
        })
        time.sleep(pausa)  # respeto del limite de requests/minuto del proveedor

    # --- Metricas del caso ---
    aciertos = sum(1 for c in corridas if c["acierto"])
    veredictos = [c["acierto"] for c in corridas]
    mayoritario = max(set(veredictos), key=veredictos.count)
    rutas_distintas = len({tuple(c["ruta_herramientas"]) for c in corridas})

    return {
        "id": caso["id"],
        "categoria": caso["categoria"],
        "descripcion": caso["descripcion"],
        "corridas": corridas,
        "precision": round(aciertos / len(corridas), 3),
        "consistencia_veredicto": round(
            veredictos.count(mayoritario) / len(veredictos), 3),
        "consistencia_textual": similitud_textual_media(
            [c["respuesta"] for c in corridas]),
        "rutas_herramientas_distintas": rutas_distintas,
        "errores": {c["tipo_fallo"]: sum(1 for x in corridas
                                         if x["tipo_fallo"] == c["tipo_fallo"])
                    for c in corridas if c["tipo_fallo"]},
    }


# =============================================================================
# 3. METRICAS GLOBALES Y PERSISTENCIA
# =============================================================================
def calcular_globales(resultados_casos: list[dict]) -> dict:
    """Agrega las metricas de todos los casos evaluados (vista de conjunto)."""
    todas = [c for r in resultados_casos for c in r["corridas"]]
    if not todas:
        return {}

    errores_por_tipo: dict[str, int] = {}
    for c in todas:
        if c["tipo_fallo"]:
            errores_por_tipo[c["tipo_fallo"]] = errores_por_tipo.get(c["tipo_fallo"], 0) + 1

    # Tasa de alucinacion medida SOLO sobre los casos trampa, que es donde
    # una alucinacion puede manifestarse con este diseño de criterios.
    trampa = [c for r in resultados_casos if r["categoria"] == "anti_alucinacion"
              for c in r["corridas"]]
    alucinaciones = sum(1 for c in trampa if c["tipo_fallo"] == "alucinacion")

    precision_por_categoria: dict[str, list[float]] = {}
    for r in resultados_casos:
        precision_por_categoria.setdefault(r["categoria"], []).append(r["precision"])

    consistencias = [r["consistencia_textual"] for r in resultados_casos
                     if r["consistencia_textual"] is not None]

    return {
        "total_corridas": len(todas),
        "precision_global": round(sum(1 for c in todas if c["acierto"]) / len(todas), 3),
        "precision_por_categoria": {
            k: round(sum(v) / len(v), 3) for k, v in precision_por_categoria.items()},
        "tasa_error_ejecucion": round(
            errores_por_tipo.get("error_ejecucion", 0) / len(todas), 3),
        "tasa_alucinacion_casos_trampa": (
            round(alucinaciones / len(trampa), 3) if trampa else None),
        "consistencia_textual_media": (
            round(sum(consistencias) / len(consistencias), 3) if consistencias else None),
        "errores_por_tipo": errores_por_tipo,
        "latencia_media_s": round(
            sum(c["latencia_total_s"] or 0 for c in todas) / len(todas), 2),
        "tokens_promedio": round(
            sum(c["tokens_total"] or 0 for c in todas) / len(todas)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluacion de precision/consistencia/errores del agente (EP3, IE1).")
    parser.add_argument("--repeticiones", type=int, default=3,
                        help="corridas por caso (default 3; cada una rota la variante)")
    parser.add_argument("--pausa", type=float, default=8.0,
                        help="segundos entre corridas para respetar el rate limit (default 8)")
    parser.add_argument("--solo", type=str, default=None,
                        help="ejecutar SOLO el caso con este id (ej: C1)")
    parser.add_argument("--desde", type=int, default=0,
                        help="indice del caso inicial, para reanudar una bateria cortada por cuota")
    args = parser.parse_args()

    with open(RUTA_CASOS, "r", encoding="utf-8") as f:
        casos = json.load(f)["casos"]

    if args.solo:
        casos = [c for c in casos if c["id"] == args.solo]
        if not casos:
            sys.exit(f"[ERROR] No existe el caso '{args.solo}' en casos.json")
    casos = casos[args.desde:]

    print("=" * 74)
    print(f"EVALUACION DEL AGENTE — {len(casos)} caso(s) x {args.repeticiones} "
          f"repeticiones (pausa {args.pausa}s)")
    print("=" * 74)

    resultados_nuevos = []
    for n, caso in enumerate(casos, 1):
        print(f"\n[{n}/{len(casos)}] Caso {caso['id']} ({caso['categoria']}): "
              f"{caso['descripcion'][:70]}...")
        resultados_nuevos.append(correr_caso(caso, args.repeticiones, args.pausa))

    # --- Fusion con resultados previos (workflow de cuota partida) ---
    # Si se corrio con --solo/--desde, los casos NO corridos ahora conservan su
    # resultado anterior; las metricas globales se recalculan sobre TODO.
    previos = {}
    if RUTA_RESULTADOS.exists():
        try:
            with open(RUTA_RESULTADOS, "r", encoding="utf-8") as f:
                previos = {r["id"]: r for r in json.load(f).get("casos", [])}
        except (json.JSONDecodeError, OSError):
            previos = {}
    for r in resultados_nuevos:
        previos[r["id"]] = r
    resultados_todos = list(previos.values())

    salida = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": {"repeticiones": args.repeticiones, "pausa_s": args.pausa},
        "casos": resultados_todos,
        "global": calcular_globales(resultados_todos),
    }
    with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)

    # --- Resumen en consola (evidencia inmediata para el informe) ---
    g = salida["global"]
    print("\n" + "=" * 74)
    print("RESUMEN GLOBAL")
    print("=" * 74)
    print(f"  Corridas totales:        {g['total_corridas']}")
    print(f"  Precision global:        {g['precision_global']:.1%}")
    for cat, p in g["precision_por_categoria"].items():
        print(f"    - {cat:<18} {p:.1%}")
    print(f"  Consistencia textual:    {g['consistencia_textual_media']}")
    print(f"  Tasa error de ejecucion: {g['tasa_error_ejecucion']:.1%}")
    print(f"  Alucinacion (trampa):    {g['tasa_alucinacion_casos_trampa']}")
    print(f"  Errores por tipo:        {g['errores_por_tipo'] or 'ninguno'}")
    print(f"  Latencia media:          {g['latencia_media_s']} s")
    print(f"  Tokens promedio/corrida: {g['tokens_promedio']}")
    print(f"\nResultados detallados en: {RUTA_RESULTADOS}")
    print(f"Trazas de cada corrida en: logs/agente_trazas.jsonl (origen=evaluacion)")


if __name__ == "__main__":
    main()
