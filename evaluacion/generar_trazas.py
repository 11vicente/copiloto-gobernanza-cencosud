"""
=============================================================================
EVALUACION/GENERAR_TRAZAS.PY — EP3: bateria de carga reproducible
=============================================================================
Ejecuta una bateria FIJA de consultas variadas contra el agente real para
poblar logs/agente_trazas.jsonl con datos de operacion realistas. Es el
complemento de evaluar.py: aquel mide precision/consistencia con criterios;
este aporta VOLUMEN y VARIEDAD para el analisis (IE3/IE4) y el dashboard (IE5).

POR QUE UNA LISTA FIJA EN EL CODIGO (y no preguntas aleatorias): la rubrica
premia la reproducibilidad. Cualquier evaluador que ejecute este script
genera la MISMA bateria; las diferencias que observe (latencias, rutas,
tokens) son del sistema, no del experimento.

DISEÑO DE LA BATERIA (variabilidad de datos deliberada):
    - Factuales sobre TRES años distintos (2023/2024/2025) y DOS documentos
      (Memorias y Codigo de Etica): variabilidad de fuente y periodo.
    - Comparativas multi-año: fuerzan VARIAS busquedas encadenadas (estresan
      el ciclo ReAct y alargan la ruta del grafo).
    - Un reporte ejecutivo: ejercita la herramienta mas costosa en tokens.
    - Calculos: delegacion a la calculadora segura.
    - Casos borde: consulta de UNA palabra (ambigua), consulta muy larga
      multi-requisito, y una pregunta fuera de dominio (debe rehusar).

MANEJO DE CUOTA (free tier de GitHub Models): pausa configurable entre
consultas y reintento unico tras 70 s si aparece un error de rate limit.
Si la cuota DIARIA se agota, el script se detiene con instrucciones para
reanudar con --desde N (el JSONL es acumulativo, no se pierde nada).

Uso (desde la raiz del proyecto, SIEMPRE con el python del venv):
    venv\\Scripts\\python.exe evaluacion\\generar_trazas.py
    venv\\Scripts\\python.exe evaluacion\\generar_trazas.py --pausa 10
    venv\\Scripts\\python.exe evaluacion\\generar_trazas.py --desde 7
    venv\\Scripts\\python.exe evaluacion\\generar_trazas.py --limite 5
=============================================================================
"""

import argparse
import sys
import time
from pathlib import Path

# Raiz del proyecto al path (mismo motivo que en evaluar.py).
RUTA_RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUTA_RAIZ))

from observabilidad.instrumentacion import RUTA_TRAZAS, ejecutar_agente_observado  # noqa: E402

# =============================================================================
# BATERIA FIJA (16 consultas). El numero al imprimirse es el indice para --desde.
# =============================================================================
BATERIA: list[str] = [
    # --- factuales por año/documento (variabilidad de periodo y fuente) ---
    "¿Cuáles son los principales riesgos identificados en la Memoria Anual 2023?",
    "¿Qué información entrega la Memoria Anual 2025 sobre los resultados financieros de Cencosud?",
    "¿Qué dice el Código de Ética de Cencosud sobre el uso de información confidencial?",
    "¿Qué canales de denuncia establece el Código de Ética de Cencosud?",
    "¿Cómo gestiona Cencosud la ciberseguridad según sus memorias anuales?",
    "¿Qué países concentran las operaciones de Cencosud según la memoria más reciente?",
    "¿Quiénes componen el directorio de Cencosud según la memoria más reciente?",
    "¿Qué avances en transformación digital o e-commerce reporta la Memoria Anual 2024?",
    "¿Cuál es la política de dividendos de Cencosud?",
    # --- comparativa multi-año: fuerza varias busquedas encadenadas ---
    "Compara los riesgos reportados en la Memoria 2023 con los de la Memoria 2024: ¿cuáles se repiten y cuáles son nuevos?",
    # --- reporte ejecutivo: la herramienta de escritura, costosa en tokens ---
    "Genera un reporte ejecutivo sobre la estrategia de sostenibilidad de Cencosud según las memorias anuales.",
    # --- calculos: delegacion a la calculadora segura ---
    "Si el EBITDA fue 1200 millones en 2023 y 1380 millones en 2024, calcula la variación porcentual exacta.",
    "Busca las emisiones o indicadores ambientales de 2023 y 2024 en las memorias y calcula su variación porcentual.",
    # --- casos borde de forma (estresan el sistema de maneras distintas) ---
    "riesgos",
    ("Necesito que revises las memorias anuales de Cencosud de 2023 y 2024, identifiques los riesgos "
     "financieros y operacionales de cada año, me digas cuáles se repiten en ambos años y me expliques "
     "qué medidas de mitigación declara la compañía para los riesgos que se repiten."),
    # --- fuera de dominio: debe rehusar (grounding bajo carga) ---
    "¿Cuál es la receta de la torta de mil hojas?",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bateria de carga reproducible del agente (EP3).")
    parser.add_argument("--pausa", type=float, default=8.0,
                        help="segundos entre consultas para respetar el rate limit (default 8)")
    parser.add_argument("--desde", type=int, default=0,
                        help="indice de la consulta inicial (reanudar bateria cortada)")
    parser.add_argument("--limite", type=int, default=None,
                        help="ejecutar solo las primeras N consultas (tras --desde)")
    args = parser.parse_args()

    consultas = BATERIA[args.desde:]
    if args.limite:
        consultas = consultas[:args.limite]

    print("=" * 74)
    print(f"BATERIA DE CARGA — {len(consultas)} consultas (pausa {args.pausa}s)")
    print(f"Trazas -> {RUTA_TRAZAS}")
    print("=" * 74)

    fallidas = 0
    for i, pregunta in enumerate(consultas, start=args.desde):
        print(f"\n[{i}] {pregunta[:80]}{'...' if len(pregunta) > 80 else ''}")
        registro = ejecutar_agente_observado(pregunta, origen="carga")

        # Reintento unico ante rate limit por minuto; si es la cuota DIARIA,
        # el reintento tambien fallara y se aborta con instrucciones claras.
        error = registro.get("error") or {}
        if not registro.get("exito") and "ratelimit" in str(error.get("tipo", "")).lower():
            print("    [!] Rate limit: espero 70 s y reintento una vez...")
            time.sleep(70)
            registro = ejecutar_agente_observado(pregunta, origen="carga")

        if registro.get("exito"):
            print(f"    OK  {registro['latencia_total_s']}s | ruta: {registro['ruta']} | "
                  f"tokens: {registro['tokens_total']} | "
                  f"herramientas: {[h['nombre'] for h in registro['herramientas']]}")
        else:
            fallidas += 1
            err = registro.get("error") or {}
            print(f"    FALLO ({err.get('tipo')}): {str(err.get('mensaje'))[:120]}")
            if "ratelimit" in str(err.get("tipo", "")).lower():
                print(f"\n[ABORTADO] Cuota del proveedor agotada. Reanudar mañana con:\n"
                      f"    venv\\Scripts\\python.exe evaluacion\\generar_trazas.py --desde {i}")
                break

        time.sleep(args.pausa)

    print(f"\nBateria terminada. Consultas fallidas: {fallidas}. "
          f"Analiza con: venv\\Scripts\\python.exe analisis\\analizar_logs.py")


if __name__ == "__main__":
    main()
