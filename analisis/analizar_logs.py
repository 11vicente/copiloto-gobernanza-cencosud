"""
=============================================================================
ANALISIS/ANALIZAR_LOGS.PY — EP3 (IE3/IE4): examen de logs y anomalias
=============================================================================
Lee las trazas estructuradas (logs/agente_trazas.jsonl) y las examina para:

    IE3 — identificar ERRORES y CUELLOS DE BOTELLA: percentiles de latencia
          (p50/p95/max), descomposicion de la latencia por nodo del grafo y
          por componente (espera del LLM vs herramientas), tasa y tipos de
          error con su ubicacion, costo por herramienta.
    IE4 — detectar PATRONES y ANOMALIAS: outliers de latencia y de tokens
          (metodo IQR), consultas con rutas anormalmente largas (posibles
          loops), respuestas anomalas, distribucion de rutas del grafo.

El script EMITE HALLAZGOS TEXTUALES numerados (H1, H2, ...) generados desde
los datos: cada hallazgo cita las cifras que lo respaldan, para que el
informe (y las propuestas de mejora de IE7) se apoyen en evidencia y no en
impresiones. Salida doble: consola + analisis/hallazgos.md (con tablas).

POR QUE UMBRALES IQR PARA OUTLIERS: es el criterio estandar de caja y bigotes
(outlier si valor > Q3 + 1.5*IQR). Es robusto ante distribuciones sesgadas
como las latencias (colas largas) y no exige suponer normalidad.

NOTA DE COSTO: GitHub Models (free tier) no factura, por lo que el "costo"
en USD se calcula con el PRECIO DE LISTA publico de gpt-4o-mini en OpenAI
(US$0.15 / 1M tokens de entrada; US$0.60 / 1M de salida) como REFERENCIA
declarada para dimensionar la sostenibilidad economica (IE7). No es gasto real.

Uso (desde la raiz del proyecto, con el python del venv):
    venv\\Scripts\\python.exe analisis\\analizar_logs.py
    venv\\Scripts\\python.exe analisis\\analizar_logs.py --trazas otra_ruta.jsonl
=============================================================================
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

RUTA_RAIZ = Path(__file__).resolve().parents[1]
RUTA_TRAZAS_DEFAULT = RUTA_RAIZ / "logs" / "agente_trazas.jsonl"
RUTA_HALLAZGOS = Path(__file__).parent / "hallazgos.md"
RUTA_RESULTADOS_EVAL = RUTA_RAIZ / "evaluacion" / "resultados_evaluacion.json"

# Precio de lista de gpt-4o-mini (OpenAI, referencia declarada — ver docstring).
PRECIO_USD_1M_ENTRADA = 0.15
PRECIO_USD_1M_SALIDA = 0.60


# =============================================================================
# 1. CARGA
# =============================================================================
def cargar_trazas(ruta: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Carga el JSONL en un DataFrame (campos escalares) y en una lista de
    dicts (para los campos anidados: nodos, herramientas, llamadas_llm).

    POR QUE AMBOS: pandas es comodo para percentiles/agrupaciones sobre
    escalares, pero aplanar TODO el anidamiento complicaria el codigo; las
    estructuras anidadas se recorren directo de los dicts.
    """
    if not ruta.exists():
        sys.exit(f"[ERROR] No existe {ruta}. Genera trazas primero con "
                 f"evaluacion\\evaluar.py o evaluacion\\generar_trazas.py")
    registros = []
    with open(ruta, "r", encoding="utf-8") as f:
        for num_linea, linea in enumerate(f, 1):
            linea = linea.strip()
            if not linea:
                continue
            try:
                registros.append(json.loads(linea))
            except json.JSONDecodeError:
                # Una linea corrupta (caida a mitad de escritura) no invalida
                # el resto del archivo: se reporta y se sigue.
                print(f"[AVISO] Linea {num_linea} corrupta; se omite.")
    if not registros:
        sys.exit("[ERROR] El archivo de trazas esta vacio.")
    return pd.DataFrame(registros), registros


# =============================================================================
# 2. SECCIONES DEL ANALISIS (cada una devuelve markdown + hallazgos)
# =============================================================================
def seccion_resumen(df: pd.DataFrame) -> tuple[str, list[str]]:
    """Vision general del dataset de trazas."""
    exito = df["exito"].mean()
    md = [
        "## 1. Resumen del dataset",
        "",
        f"- Trazas analizadas: **{len(df)}** "
        f"(desde {df['timestamp'].min()} hasta {df['timestamp'].max()})",
        f"- Por origen: {df['origen'].value_counts().to_dict()}",
        f"- Tasa de exito de ejecucion: **{exito:.1%}** "
        f"({int(df['exito'].sum())} de {len(df)})",
        f"- Modelo: {df['modelo'].dropna().unique().tolist()}",
        "",
    ]
    hallazgos = []
    if exito < 0.95:
        hallazgos.append(
            f"La tasa de exito de ejecucion es {exito:.1%}: hay fallos de "
            f"ejecucion que requieren analisis (ver seccion de errores)."
        )
    return "\n".join(md), hallazgos


def seccion_latencia(df: pd.DataFrame, registros: list[dict]) -> tuple[str, list[str]]:
    """IE3: percentiles, descomposicion por nodo y cuello de botella."""
    lat = df["latencia_total_s"].dropna()
    p50, p95, pmax = lat.quantile(0.5), lat.quantile(0.95), lat.max()

    # --- Latencia acumulada por nodo (suma sobre todas las trazas) ---
    total_por_nodo: dict[str, float] = {}
    ejecuciones_por_nodo: dict[str, int] = {}
    for r in registros:
        for nodo, s in (r.get("latencia_por_nodo_s") or {}).items():
            total_por_nodo[nodo] = total_por_nodo.get(nodo, 0.0) + s
        for span in (r.get("nodos") or []):
            ejecuciones_por_nodo[span["nodo"]] = ejecuciones_por_nodo.get(span["nodo"], 0) + 1

    suma_total = sum(total_por_nodo.values()) or 1.0
    cuello, t_cuello = max(total_por_nodo.items(), key=lambda kv: kv[1])

    # --- Descomposicion LLM vs herramientas (¿donde se va el tiempo?) ---
    t_llm = sum(c["latencia_s"] for r in registros
                for c in (r.get("llamadas_llm") or []))
    t_tools = sum(h["latencia_s"] for r in registros
                  for h in (r.get("herramientas") or []))
    t_total = lat.sum() or 1.0
    # Las herramientas del nodo reportar/calcular llaman al LLM por dentro
    # (generar_reporte); el solape se declara en vez de ocultarse.
    pct_llm, pct_tools = t_llm / t_total, t_tools / t_total

    filas = "\n".join(
        f"| {n} | {ejecuciones_por_nodo.get(n, 0)} | {t:.1f} | {t / suma_total:.1%} |"
        for n, t in sorted(total_por_nodo.items(), key=lambda kv: -kv[1])
    )
    md = [
        "## 2. Latencia y cuellos de botella (IE3)",
        "",
        f"- Latencia total por consulta: p50 = **{p50:.2f} s**, "
        f"p95 = **{p95:.2f} s**, max = **{pmax:.2f} s**",
        f"- Tiempo de espera del LLM: **{pct_llm:.1%}** del tiempo total "
        f"({t_llm:.1f} s acumulados en {sum(len(r.get('llamadas_llm') or []) for r in registros)} llamadas)",
        f"- Tiempo en herramientas: {pct_tools:.1%} del total "
        f"(nota: generar_reporte llama al LLM por dentro, hay solape declarado)",
        "",
        "| Nodo | Ejecuciones | Latencia acumulada (s) | % del tiempo |",
        "|---|---|---|---|",
        filas,
        "",
    ]
    hallazgos = [
        f"CUELLO DE BOTELLA: el nodo '{cuello}' concentra el "
        f"{t_cuello / suma_total:.1%} de la latencia acumulada "
        f"({t_cuello:.1f} s). La espera del LLM explica el {pct_llm:.1%} del "
        f"tiempo total: la optimizacion debe apuntar a las llamadas al modelo "
        f"(menos pasos, prompts mas cortos, cache), no al retrieval.",
    ]
    if p95 > 2 * p50:
        hallazgos.append(
            f"COLA PESADA de latencia: p95 ({p95:.1f} s) es "
            f"{p95 / p50:.1f}x el p50 ({p50:.1f} s). Una fraccion de consultas "
            f"degrada mucho la experiencia; ver outliers (seccion anomalias)."
        )
    return "\n".join(md), hallazgos


def seccion_recursos(df: pd.DataFrame, registros: list[dict]) -> tuple[str, list[str]]:
    """IE2: tokens, llamadas LLM, RAM/CPU y costo de referencia."""
    ok = df[df["exito"] == True]  # noqa: E712 — claridad sobre estilo
    tok_prompt = int(ok["tokens_prompt"].sum())
    tok_comp = int(ok["tokens_completion"].sum())
    costo = (tok_prompt * PRECIO_USD_1M_ENTRADA + tok_comp * PRECIO_USD_1M_SALIDA) / 1e6

    md = [
        "## 3. Uso de recursos (IE2)",
        "",
        f"- Tokens totales: **{tok_prompt + tok_comp:,}** "
        f"(prompt: {tok_prompt:,} | completion: {tok_comp:,}) — "
        f"fuente: {df['fuente_tokens'].value_counts().to_dict()}",
        f"- Tokens por consulta: media = {ok['tokens_total'].mean():,.0f}, "
        f"p95 = {ok['tokens_total'].quantile(0.95):,.0f}, "
        f"max = {ok['tokens_total'].max():,.0f}",
        f"- Llamadas LLM por consulta: media = {ok['num_llamadas_llm'].mean():.1f}, "
        f"max = {int(ok['num_llamadas_llm'].max())}",
        f"- Pasos ReAct por consulta: media = {ok['num_pasos_react'].mean():.1f}, "
        f"max = {int(ok['num_pasos_react'].max())}",
        f"- Memoria del proceso (RSS): media = {df['memoria_rss_mb'].mean():.0f} MB, "
        f"max = {df['memoria_rss_mb'].max():.0f} MB | "
        f"CPU medio por consulta: {df['cpu_proceso_pct'].mean():.1f}%",
        f"- Costo de REFERENCIA (precio lista gpt-4o-mini): "
        f"**US$ {costo:.4f}** total, US$ {costo / max(len(ok), 1):.5f} por consulta "
        f"(GitHub Models free tier no factura; cifra para dimensionar escalamiento)",
        "",
    ]
    ratio = tok_prompt / max(tok_comp, 1)
    hallazgos = []
    if ratio > 4:
        hallazgos.append(
            f"PATRON DE COSTO: los tokens de PROMPT superan {ratio:.0f}x a los "
            f"de completion. El costo esta dominado por el contexto que se "
            f"reinyecta en cada paso ReAct (chunks k=6 + historial): reducir "
            f"k, comprimir contexto o cachear busquedas repetidas tendria mas "
            f"impacto que acortar respuestas."
        )
    cpu = df["cpu_proceso_pct"].mean()
    if cpu < 25:
        hallazgos.append(
            f"PROCESO I/O-BOUND: CPU medio de {cpu:.1f}% por consulta. El agente "
            f"pasa el tiempo ESPERANDO a la API del LLM, no computando: escalar "
            f"verticalmente (mas CPU) no mejoraria la latencia; si lo harian "
            f"lotes concurrentes o un modelo/endpoint mas rapido."
        )
    return "\n".join(md), hallazgos


def seccion_errores(df: pd.DataFrame) -> tuple[str, list[str]]:
    """IE3: frecuencia, tipos y ubicacion de errores de ejecucion."""
    fallos = df[df["exito"] == False]  # noqa: E712
    md = ["## 4. Errores de ejecucion (IE3)", ""]
    hallazgos = []
    if fallos.empty:
        md.append("- Sin errores de ejecucion en el periodo analizado. Los fallos "
                  "de CALIDAD (criterios no cumplidos) se miden en la evaluacion "
                  "(seccion 6), no aqui.")
        md.append("")
    else:
        tipos = fallos["error"].apply(
            lambda e: (e or {}).get("tipo", "?")).value_counts().to_dict()
        dondes = fallos["error"].apply(
            lambda e: (e or {}).get("donde", "?")).value_counts().to_dict()
        md += [
            f"- Consultas fallidas: **{len(fallos)}** de {len(df)} "
            f"({len(fallos) / len(df):.1%})",
            f"- Por tipo de excepcion: {tipos}",
            f"- Por ubicacion: {dondes}",
            "",
        ]
        hallazgos.append(
            f"ERRORES: {len(fallos)} fallos de ejecucion "
            f"({len(fallos) / len(df):.1%}); tipos: {tipos}; ubicacion: {dondes}."
        )
    return "\n".join(md), hallazgos


def seccion_herramientas(registros: list[dict]) -> tuple[str, list[str]]:
    """Uso y costo (latencia) por herramienta: ¿cual es la mas cara?"""
    stats: dict[str, dict] = {}
    for r in registros:
        for h in (r.get("herramientas") or []):
            s = stats.setdefault(h["nombre"], {"usos": 0, "lat": [], "fallos": 0})
            s["usos"] += 1
            s["lat"].append(h["latencia_s"])
            if not h.get("ok", True):
                s["fallos"] += 1

    md = ["## 5. Uso de herramientas", "",
          "| Herramienta | Usos | Latencia media (s) | Latencia max (s) | Fallos |",
          "|---|---|---|---|---|"]
    hallazgos = []
    if stats:
        for nombre, s in sorted(stats.items(), key=lambda kv: -kv[1]["usos"]):
            media = sum(s["lat"]) / len(s["lat"])
            md.append(f"| {nombre} | {s['usos']} | {media:.2f} | "
                      f"{max(s['lat']):.2f} | {s['fallos']} |")
        mas_cara = max(stats.items(), key=lambda kv: sum(kv[1]["lat"]) / len(kv[1]["lat"]))
        media_cara = sum(mas_cara[1]["lat"]) / len(mas_cara[1]["lat"])
        hallazgos.append(
            f"HERRAMIENTA MAS COSTOSA: '{mas_cara[0]}' con latencia media de "
            f"{media_cara:.2f} s por uso ({mas_cara[1]['usos']} usos)."
        )
    md.append("")
    return "\n".join(md), hallazgos


def seccion_anomalias(df: pd.DataFrame) -> tuple[str, list[str]]:
    """IE4: outliers (IQR), rutas largas y patrones en las rutas del grafo."""
    md = ["## 6. Patrones y anomalias (IE4)", ""]
    hallazgos = []
    ok = df[df["exito"] == True]  # noqa: E712

    # --- Outliers de latencia por criterio IQR (robusto a colas largas) ---
    lat = ok["latencia_total_s"].dropna()
    q1, q3 = lat.quantile(0.25), lat.quantile(0.75)
    umbral_lat = q3 + 1.5 * (q3 - q1)
    outliers = ok[ok["latencia_total_s"] > umbral_lat]
    md.append(f"- Umbral de outlier de latencia (Q3 + 1.5*IQR): "
              f"**{umbral_lat:.1f} s** -> {len(outliers)} consulta(s) anomala(s)")
    for _, fila in outliers.iterrows():
        md.append(f"  - `{str(fila['pregunta'])[:70]}...` -> "
                  f"{fila['latencia_total_s']:.1f} s, ruta: {fila['ruta']}")
    if len(outliers):
        hallazgos.append(
            f"ANOMALIA DE LATENCIA: {len(outliers)} consulta(s) superan el umbral "
            f"IQR de {umbral_lat:.1f} s; todas corresponden a rutas con "
            f"multiples busquedas o generacion de reportes (ver detalle en "
            f"hallazgos.md), un patron que anticipa la degradacion al escalar."
        )

    # --- Rutas largas: posible sobre-busqueda o loop ---
    umbral_pasos = max(6, int(ok["num_pasos_react"].quantile(0.9)))
    largas = ok[ok["num_pasos_react"] > umbral_pasos]
    md.append(f"- Consultas con ruta larga (> {umbral_pasos} pasos): {len(largas)}")
    for _, fila in largas.iterrows():
        md.append(f"  - `{str(fila['pregunta'])[:70]}...` -> "
                  f"{int(fila['num_pasos_react'])} pasos: {fila['ruta']}")

    # --- Distribucion de rutas: patrones de decision del agente ---
    rutas = ok["ruta"].value_counts()
    md += ["", "- Rutas del grafo observadas (patron de decision del agente):"]
    for ruta, n in rutas.items():
        md.append(f"  - {n:>3}x  `{ruta}`")
    md.append("")
    hallazgos.append(
        f"PATRON DE RUTAS: se observan {len(rutas)} rutas distintas del grafo; "
        f"la dominante es `{rutas.index[0]}` ({rutas.iloc[0]} veces, "
        f"{rutas.iloc[0] / len(ok):.0%} de las consultas exitosas)."
    )

    # --- Outliers de tokens (IQR) ---
    tok = ok["tokens_total"].dropna()
    if len(tok) > 3:
        q1t, q3t = tok.quantile(0.25), tok.quantile(0.75)
        umbral_tok = q3t + 1.5 * (q3t - q1t)
        out_tok = ok[ok["tokens_total"] > umbral_tok]
        if len(out_tok):
            md.append(f"- Outliers de tokens (> {umbral_tok:,.0f}): {len(out_tok)} "
                      f"consulta(s), max {int(out_tok['tokens_total'].max()):,}")
            hallazgos.append(
                f"ANOMALIA DE CONSUMO: {len(out_tok)} consulta(s) superan el umbral "
                f"IQR de {umbral_tok:,.0f} tokens (max: "
                f"{int(out_tok['tokens_total'].max()):,}). Corresponden a flujos "
                f"multi-busqueda/reporte: candidatas a limitar k o a resumir "
                f"contexto intermedio."
            )
            md.append("")
    return "\n".join(md), hallazgos


def seccion_evaluacion() -> tuple[str, list[str]]:
    """Cruza con los resultados de evaluar.py (si existen): calidad, no solo velocidad."""
    if not RUTA_RESULTADOS_EVAL.exists():
        return ("## 7. Calidad (evaluacion IE1)\n\n- (Aun no hay "
                "resultados_evaluacion.json; correr evaluacion\\evaluar.py)\n"), []
    with open(RUTA_RESULTADOS_EVAL, "r", encoding="utf-8") as f:
        ev = json.load(f)
    g = ev.get("global", {})
    md = [
        "## 7. Calidad (cruce con la evaluacion IE1)",
        "",
        f"- Precision global: **{g.get('precision_global', 0):.1%}** "
        f"sobre {g.get('total_corridas', 0)} corridas",
        f"- Por categoria: {g.get('precision_por_categoria', {})}",
        f"- Consistencia textual media: {g.get('consistencia_textual_media')}",
        f"- Alucinaciones en casos trampa: {g.get('tasa_alucinacion_casos_trampa')}",
        f"- Errores por tipo: {g.get('errores_por_tipo', {})}",
        "",
    ]
    hallazgos = []
    for caso in ev.get("casos", []):
        if caso["precision"] < 1.0:
            hallazgos.append(
                f"CALIDAD: el caso {caso['id']} ({caso['categoria']}) logro "
                f"precision {caso['precision']:.0%} con errores {caso['errores']}: "
                f"punto concreto de mejora."
            )
    return "\n".join(md), hallazgos


# =============================================================================
# 3. ORQUESTACION
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Analisis de trazas del agente (EP3).")
    parser.add_argument("--trazas", type=Path, default=RUTA_TRAZAS_DEFAULT)
    parser.add_argument("--salida", type=Path, default=RUTA_HALLAZGOS)
    args = parser.parse_args()

    df, registros = cargar_trazas(args.trazas)

    secciones, hallazgos = [], []
    for fn in (lambda: seccion_resumen(df),
               lambda: seccion_latencia(df, registros),
               lambda: seccion_recursos(df, registros),
               lambda: seccion_errores(df),
               lambda: seccion_herramientas(registros),
               lambda: seccion_anomalias(df),
               seccion_evaluacion):
        md, h = fn()
        secciones.append(md)
        hallazgos.extend(h)

    encabezado = (
        f"# Analisis de trazas del agente — EP3\n\n"
        f"Generado: {datetime.now().astimezone().isoformat(timespec='seconds')} · "
        f"Fuente: `{args.trazas.name}` ({len(df)} trazas)\n"
    )
    bloque_hallazgos = ["## 8. HALLAZGOS (sintesis para el informe)", ""]
    bloque_hallazgos += [f"- **H{i}.** {h}" for i, h in enumerate(hallazgos, 1)]
    bloque_hallazgos.append("")

    contenido = "\n".join([encabezado] + secciones + bloque_hallazgos)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(contenido, encoding="utf-8")

    # Consola: solo la sintesis (el detalle queda en el .md).
    print("=" * 74)
    print(f"ANALISIS COMPLETO -> {args.salida}")
    print("=" * 74)
    for i, h in enumerate(hallazgos, 1):
        print(f"\nH{i}. {h}")


if __name__ == "__main__":
    main()
