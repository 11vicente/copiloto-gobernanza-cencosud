"""
=============================================================================
DASHBOARD/DASHBOARD.PY — EP3 (IE5): dashboard de observabilidad del agente
=============================================================================
Dashboard Streamlit + Plotly que visualiza TODAS las metricas clave del
agente a partir de las trazas locales (logs/agente_trazas.jsonl) y de los
resultados de la evaluacion (evaluacion/resultados_evaluacion.json):

    - KPIs generales (consultas, tasa de exito, p50/p95, tokens, costo ref.)
    - Latencia: linea de tiempo interactiva + descomposicion por nodo
      (cuello de botella) + distribucion de pasos ReAct           (IE2/IE3)
    - Recursos: tokens prompt/completion por consulta, llamadas LLM (IE2)
    - Herramientas y rutas del grafo (patrones de decision)        (IE4)
    - Calidad: precision/consistencia por caso de evaluacion       (IE1)
    - Errores: tabla de fallos con tipo y ubicacion                (IE3)
    - Datos: tabla completa filtrable (vista accesible/auditable)

POR QUE STREAMLIT (y no Grafana/Kibana/PowerBI): es 100% Python (el mismo
stack del proyecto), corre local sin infraestructura extra (Grafana/Kibana
exigen un servidor y una base de series de tiempo), es interactivo (filtros
en vivo) y figura explicitamente como opcion valida en el encargo EP3.

POR QUE PLOTLY: graficos con tooltip/zoom/pan nativos (la rubrica IE5 pide
"interactivo"), integrados en Streamlit sin dependencias adicionales.

DISEÑO VISUAL: paleta categorica validada para daltonismo (ΔE adyacente
minimo 24.2 en superficie clara) con ORDEN FIJO; cada entidad (nodo,
herramienta, origen) conserva SU color en todos los graficos ("el color
sigue a la entidad, no al orden de aparicion"). Un solo eje Y por grafico.
La tabla de datos completa al final es la vista accesible del dataset.

Uso (desde la raiz del proyecto, con el python del venv):
    venv\\Scripts\\python.exe -m streamlit run dashboard\\dashboard.py
=============================================================================
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# 1. RUTAS Y PALETA (validada; orden fijo, color por entidad)
# =============================================================================
RUTA_RAIZ = Path(__file__).resolve().parents[1]
RUTA_TRAZAS = RUTA_RAIZ / "logs" / "agente_trazas.jsonl"
RUTA_EVAL = RUTA_RAIZ / "evaluacion" / "resultados_evaluacion.json"

# Tinta y cromo del grafico (superficie clara).
INK = "#0b0b0b"          # texto principal
INK_2 = "#52514e"        # texto secundario
MUTED = "#898781"        # ejes / etiquetas terciarias
GRID = "#e1e0d9"         # gridlines (recesivas)
BASELINE = "#c3c2b7"     # linea de eje

# Slots categoricos en ORDEN FIJO (paleta validada para vision de color).
AZUL, AQUA, AMARILLO, VERDE, VIOLETA = "#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7"

# Estados (reservados para exito/fallo, nunca para series tematicas).
OK_VERDE, CRITICO = "#0ca30c", "#d03b3b"

# El color SIGUE a la entidad: un nodo/herramienta luce igual en TODOS los
# graficos, y las herramientas heredan el color del nodo que las ejecuta.
COLOR_NODO = {"razonar": AZUL, "buscar": AQUA, "calcular": AMARILLO,
              "reportar": VIOLETA, "responder": VERDE}
COLOR_HERRAMIENTA = {"buscar_documentos": AQUA, "calcular_metricas": AMARILLO,
                     "generar_reporte": VIOLETA}
COLOR_ORIGEN = {"evaluacion": AZUL, "carga": AQUA, "verificacion": AMARILLO,
                "api": VIOLETA, "debug": MUTED, "script": MUTED,
                "desconocido": MUTED}
COLOR_CATEGORIA = {"factual": AZUL, "calculo": AQUA, "anti_alucinacion": AMARILLO,
                   "reporte": VIOLETA, "robustez": VERDE}


def estilizar(fig: go.Figure, altura: int = 340) -> go.Figure:
    """Cromo comun: tipografia de sistema, grid recesivo, tooltip legible,
    esquinas 4px en barras. Centralizado para que TODO el dashboard sea
    un solo sistema visual."""
    fig.update_layout(
        template="plotly_white",
        height=altura,
        margin=dict(l=10, r=10, t=48, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family='system-ui, "Segoe UI", sans-serif', color=INK_2, size=13),
        title=dict(font=dict(size=15, color=INK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                    font=dict(color=INK_2)),
        hoverlabel=dict(bgcolor="#ffffff", font=dict(color=INK),
                        bordercolor=GRID),
        barcornerradius=4,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=BASELINE, zerolinecolor=GRID,
                     tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor=BASELINE, zerolinecolor=GRID,
                     tickfont=dict(color=MUTED))
    return fig


# =============================================================================
# 2. CARGA DE DATOS (cacheada; se invalida sola si el JSONL cambia)
# =============================================================================
@st.cache_data(show_spinner=False)
def cargar_trazas(mtime: float) -> pd.DataFrame:
    """Lee el JSONL a DataFrame. El parametro mtime existe SOLO para que la
    cache de Streamlit se invalide cuando el archivo cambia (nuevas trazas
    aparecen al refrescar, sin reiniciar el server)."""
    registros = []
    with open(RUTA_TRAZAS, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                try:
                    registros.append(json.loads(linea))
                except json.JSONDecodeError:
                    continue  # linea corrupta: se omite sin romper el dashboard
    df = pd.DataFrame(registros)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def cargar_eval(mtime: float) -> dict | None:
    with open(RUTA_EVAL, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# 3. APP
# =============================================================================
st.set_page_config(page_title="Observabilidad — Co-piloto Cencosud",
                   page_icon=":bar_chart:", layout="wide")

st.title("Observabilidad del Agente — Co-piloto de Gobernanza (EP3)")
st.caption("Fuente: `logs/agente_trazas.jsonl` (trazas locales estructuradas; "
           "independientes de LangSmith). Trace por consulta con spans de "
           "nodo, herramienta y llamada LLM.")

if not RUTA_TRAZAS.exists():
    st.error("No existe `logs/agente_trazas.jsonl`. Genera trazas primero:\n\n"
             "`venv\\Scripts\\python.exe evaluacion\\evaluar.py` y/o "
             "`venv\\Scripts\\python.exe evaluacion\\generar_trazas.py`")
    st.stop()

df = cargar_trazas(RUTA_TRAZAS.stat().st_mtime)
eval_data = cargar_eval(RUTA_EVAL.stat().st_mtime) if RUTA_EVAL.exists() else None

# ----------------------------------------------------------------- filtros ---
# Sidebar = panel de filtros interactivos (IE5). Afectan a TODO el dashboard.
with st.sidebar:
    st.header("Filtros")
    origenes = sorted(df["origen"].dropna().unique().tolist())
    filtro_origen = st.multiselect("Origen de la traza", origenes, default=origenes)

    fecha_min = df["timestamp"].min().date()
    fecha_max = df["timestamp"].max().date()
    rango = st.date_input("Rango de fechas", (fecha_min, fecha_max),
                          min_value=fecha_min, max_value=fecha_max)

    solo_errores = st.checkbox("Solo consultas con error", value=False)
    buscar_texto = st.text_input("Buscar en la pregunta", "")

    st.divider()
    st.caption(f"{len(df)} trazas totales en el archivo. "
               f"Ultima: {df['timestamp'].max():%d-%m-%Y %H:%M}")

mask = df["origen"].isin(filtro_origen)
if isinstance(rango, tuple) and len(rango) == 2:
    mask &= (df["timestamp"].dt.date >= rango[0]) & (df["timestamp"].dt.date <= rango[1])
if solo_errores:
    mask &= ~df["exito"].fillna(False)
if buscar_texto:
    mask &= df["pregunta"].str.contains(buscar_texto, case=False, na=False)
dfv = df[mask].copy()

if dfv.empty:
    st.warning("Ningun registro pasa los filtros actuales.")
    st.stop()

registros_v = dfv.to_dict("records")  # para recorrer campos anidados
ok = dfv[dfv["exito"] == True]  # noqa: E712

# -------------------------------------------------------------------- KPIs ---
lat = ok["latencia_total_s"].dropna()
tok_prompt = int(ok["tokens_prompt"].fillna(0).sum())
tok_comp = int(ok["tokens_completion"].fillna(0).sum())
costo_ref = (tok_prompt * 0.15 + tok_comp * 0.60) / 1e6  # precio lista gpt-4o-mini

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Consultas", len(dfv))
k2.metric("Tasa de exito", f"{dfv['exito'].mean():.0%}")
k3.metric("Latencia p50", f"{lat.quantile(0.5):.1f} s" if len(lat) else "—")
k4.metric("Latencia p95", f"{lat.quantile(0.95):.1f} s" if len(lat) else "—")
k5.metric("Tokens totales", f"{tok_prompt + tok_comp:,}")
k6.metric("Costo referencia", f"US$ {costo_ref:.4f}",
          help="Precio de lista de gpt-4o-mini (OpenAI). GitHub Models free "
               "tier no factura: es una referencia para dimensionar costos.")

# -------------------------------------------------------------------- tabs ---
tab_lat, tab_rec, tab_herr, tab_cal, tab_err, tab_datos = st.tabs(
    ["Latencia", "Recursos", "Herramientas y rutas", "Calidad (IE1)",
     "Errores", "Datos"])

# ============================================================= TAB LATENCIA ==
with tab_lat:
    # --- Linea de tiempo: cada punto una consulta (tooltip con la pregunta) ---
    linea = dfv.dropna(subset=["latencia_total_s"]).sort_values("timestamp")
    fig = px.scatter(
        linea, x="timestamp", y="latencia_total_s", color="origen",
        color_discrete_map=COLOR_ORIGEN,
        hover_data={"pregunta": True, "ruta": True, "origen": False,
                    "timestamp": False, "latencia_total_s": ":.2f"},
        labels={"timestamp": "", "latencia_total_s": "latencia (s)"},
        title="Linea de tiempo de consultas (latencia total)",
    )
    fig.update_traces(marker=dict(size=10, opacity=0.85,
                                  line=dict(width=1, color="#ffffff")))
    st.plotly_chart(estilizar(fig), use_container_width=True,
                    config={"displayModeBar": False})

    c1, c2 = st.columns(2)
    # --- Cuello de botella: latencia ACUMULADA por nodo ---
    acum: dict[str, float] = {}
    ejec: dict[str, int] = {}
    for r in registros_v:
        for nodo, s in (r.get("latencia_por_nodo_s") or {}).items():
            acum[nodo] = acum.get(nodo, 0.0) + s
        for span in (r.get("nodos") or []):
            ejec[span["nodo"]] = ejec.get(span["nodo"], 0) + 1
    if acum:
        nodos_df = pd.DataFrame({
            "nodo": list(acum.keys()),
            "latencia_acumulada_s": [round(v, 1) for v in acum.values()],
            "ejecuciones": [ejec.get(n, 0) for n in acum.keys()],
        }).sort_values("latencia_acumulada_s", ascending=False)
        fig = px.bar(nodos_df, x="nodo", y="latencia_acumulada_s", color="nodo",
                     color_discrete_map=COLOR_NODO, text="latencia_acumulada_s",
                     hover_data={"ejecuciones": True},
                     labels={"latencia_acumulada_s": "latencia acumulada (s)",
                             "nodo": ""},
                     title="Latencia acumulada por nodo (cuello de botella)")
        fig.update_traces(textposition="outside", showlegend=False,
                          marker_line=dict(width=0))
        c1.plotly_chart(estilizar(fig), use_container_width=True,
                        config={"displayModeBar": False})

        media_df = nodos_df.assign(
            media_s=(nodos_df["latencia_acumulada_s"] / nodos_df["ejecuciones"]).round(2))
        fig = px.bar(media_df, x="nodo", y="media_s", color="nodo",
                     color_discrete_map=COLOR_NODO, text="media_s",
                     labels={"media_s": "latencia media por ejecucion (s)", "nodo": ""},
                     title="Latencia media por ejecucion de nodo")
        fig.update_traces(textposition="outside", showlegend=False,
                          marker_line=dict(width=0))
        c2.plotly_chart(estilizar(fig), use_container_width=True,
                        config={"displayModeBar": False})

    # --- Distribucion de pasos ReAct (complejidad de las rutas) ---
    pasos = ok["num_pasos_react"].dropna().astype(int).value_counts().sort_index()
    fig = go.Figure(go.Bar(x=pasos.index.astype(str), y=pasos.values,
                           marker_color=AZUL, text=pasos.values,
                           textposition="outside"))
    fig.update_layout(title="Distribucion de pasos ReAct por consulta",
                      xaxis_title="pasos (transiciones de nodo)",
                      yaxis_title="consultas")
    st.plotly_chart(estilizar(fig, altura=300), use_container_width=True,
                    config={"displayModeBar": False})

# ============================================================= TAB RECURSOS ==
with tab_rec:
    # --- Tokens por consulta, apilado prompt/completion (mismo eje) ---
    tok = ok.dropna(subset=["tokens_total"]).sort_values("timestamp").reset_index()
    tok["consulta"] = tok.index + 1
    fig = go.Figure()
    fig.add_bar(x=tok["consulta"], y=tok["tokens_prompt"], name="prompt",
                marker_color=AZUL, marker_line=dict(width=1, color="#ffffff"),
                customdata=tok["pregunta"].str.slice(0, 90),
                hovertemplate="consulta %{x}<br>prompt: %{y:,}<br>%{customdata}<extra></extra>")
    fig.add_bar(x=tok["consulta"], y=tok["tokens_completion"], name="completion",
                marker_color=AQUA, marker_line=dict(width=1, color="#ffffff"),
                hovertemplate="completion: %{y:,}<extra></extra>")
    fig.update_layout(barmode="stack",
                      title="Tokens por consulta (prompt vs completion)",
                      xaxis_title="consulta (orden cronologico)",
                      yaxis_title="tokens")
    st.plotly_chart(estilizar(fig), use_container_width=True,
                    config={"displayModeBar": False})

    c1, c2 = st.columns(2)
    # --- Llamadas LLM por consulta ---
    llm_counts = ok["num_llamadas_llm"].dropna().astype(int).value_counts().sort_index()
    fig = go.Figure(go.Bar(x=llm_counts.index.astype(str), y=llm_counts.values,
                           marker_color=AZUL, text=llm_counts.values,
                           textposition="outside"))
    fig.update_layout(title="Llamadas al LLM por consulta",
                      xaxis_title="n° de llamadas", yaxis_title="consultas")
    c1.plotly_chart(estilizar(fig, altura=300), use_container_width=True,
                    config={"displayModeBar": False})

    # --- RAM del proceso a lo largo del tiempo ---
    ram = dfv.dropna(subset=["memoria_rss_mb"]).sort_values("timestamp")
    fig = px.line(ram, x="timestamp", y="memoria_rss_mb", markers=True,
                  labels={"timestamp": "", "memoria_rss_mb": "RSS (MB)"},
                  title="Memoria del proceso (RSS) por consulta")
    fig.update_traces(line=dict(color=AZUL, width=2),
                      marker=dict(size=8, color=AZUL))
    c2.plotly_chart(estilizar(fig, altura=300), use_container_width=True,
                    config={"displayModeBar": False})

    st.caption("Los tokens provienen del proveedor (usage real de la API; "
               "`fuente_tokens='proveedor'`). Los tokens de embeddings de la "
               "busqueda vectorial no los expone la API y no se contabilizan "
               "(limitacion declarada).")

# ================================================== TAB HERRAMIENTAS/RUTAS ===
with tab_herr:
    c1, c2 = st.columns(2)
    filas_h = [
        {"herramienta": h["nombre"], "latencia_s": h["latencia_s"], "ok": h.get("ok", True)}
        for r in registros_v for h in (r.get("herramientas") or [])
    ]
    if filas_h:
        hdf = pd.DataFrame(filas_h)
        usos = hdf.groupby("herramienta").agg(
            usos=("latencia_s", "size"),
            latencia_media_s=("latencia_s", "mean")).round(2).reset_index()

        fig = px.bar(usos, x="herramienta", y="usos", color="herramienta",
                     color_discrete_map=COLOR_HERRAMIENTA, text="usos",
                     labels={"herramienta": "", "usos": "invocaciones"},
                     title="Uso de cada herramienta")
        fig.update_traces(textposition="outside", showlegend=False,
                          marker_line=dict(width=0))
        c1.plotly_chart(estilizar(fig), use_container_width=True,
                        config={"displayModeBar": False})

        fig = px.bar(usos, x="herramienta", y="latencia_media_s",
                     color="herramienta", color_discrete_map=COLOR_HERRAMIENTA,
                     text="latencia_media_s",
                     labels={"herramienta": "", "latencia_media_s": "latencia media (s)"},
                     title="Costo (latencia media) por herramienta")
        fig.update_traces(textposition="outside", showlegend=False,
                          marker_line=dict(width=0))
        c2.plotly_chart(estilizar(fig), use_container_width=True,
                        config={"displayModeBar": False})
    else:
        st.info("Sin invocaciones de herramientas en los registros filtrados.")

    # --- Rutas del grafo (patron de decision del agente, IE4) ---
    rutas = ok["ruta"].value_counts().head(10).iloc[::-1]
    fig = go.Figure(go.Bar(x=rutas.values, y=rutas.index, orientation="h",
                           marker_color=AZUL, text=rutas.values,
                           textposition="outside"))
    fig.update_layout(title="Rutas del grafo mas frecuentes (top 10)",
                      xaxis_title="consultas", yaxis_title="")
    st.plotly_chart(estilizar(fig, altura=380), use_container_width=True,
                    config={"displayModeBar": False})

# ============================================================== TAB CALIDAD ==
with tab_cal:
    if not eval_data:
        st.info("Aun no hay `evaluacion/resultados_evaluacion.json`. "
                "Correr: `venv\\Scripts\\python.exe evaluacion\\evaluar.py`")
    else:
        g = eval_data.get("global", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision global", f"{g.get('precision_global', 0):.0%}")
        c2.metric("Consistencia textual", f"{g.get('consistencia_textual_media', 0)}",
                  help="Similitud lexica media entre repeticiones del mismo caso "
                       "(difflib; 1.0 = respuestas identicas). Aproximacion "
                       "declarada: no mide equivalencia semantica.")
        tasa_aluc = g.get("tasa_alucinacion_casos_trampa")
        c3.metric("Alucinaciones (casos trampa)",
                  f"{tasa_aluc:.0%}" if tasa_aluc is not None else "—")
        c4.metric("Errores de ejecucion", f"{g.get('tasa_error_ejecucion', 0):.0%}")

        casos = pd.DataFrame([
            {"caso": c["id"], "categoria": c["categoria"],
             "precision": c["precision"],
             "consistencia_veredicto": c["consistencia_veredicto"],
             "consistencia_textual": c["consistencia_textual"],
             "rutas_distintas": c["rutas_herramientas_distintas"]}
            for c in eval_data.get("casos", [])
        ])
        c1, c2 = st.columns(2)
        fig = px.bar(casos, x="caso", y="precision", color="categoria",
                     color_discrete_map=COLOR_CATEGORIA, text="precision",
                     labels={"caso": "", "precision": "precision"},
                     title="Precision por caso (variantes de fraseo rotadas)")
        fig.update_traces(texttemplate="%{text:.0%}", textposition="outside",
                          marker_line=dict(width=0))
        fig.update_yaxes(range=[0, 1.15], tickformat=".0%")
        c1.plotly_chart(estilizar(fig), use_container_width=True,
                        config={"displayModeBar": False})

        fig = px.bar(casos, x="caso", y="consistencia_textual", color="categoria",
                     color_discrete_map=COLOR_CATEGORIA, text="consistencia_textual",
                     labels={"caso": "", "consistencia_textual": "similitud media"},
                     title="Consistencia textual entre repeticiones")
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside",
                          marker_line=dict(width=0))
        fig.update_yaxes(range=[0, 1.15])
        c2.plotly_chart(estilizar(fig), use_container_width=True,
                        config={"displayModeBar": False})

        st.dataframe(casos, use_container_width=True, hide_index=True)

# ============================================================== TAB ERRORES ==
with tab_err:
    fallos = dfv[dfv["exito"] == False]  # noqa: E712
    if fallos.empty:
        st.success("Sin errores de ejecucion en los registros filtrados.")
    else:
        tipos = fallos["error"].apply(lambda e: (e or {}).get("tipo", "?")).value_counts()
        fig = go.Figure(go.Bar(x=tipos.index, y=tipos.values,
                               marker_color=CRITICO, text=tipos.values,
                               textposition="outside"))
        fig.update_layout(title="Errores por tipo de excepcion",
                          xaxis_title="", yaxis_title="ocurrencias")
        st.plotly_chart(estilizar(fig, altura=300), use_container_width=True,
                        config={"displayModeBar": False})

        detalle = fallos.assign(
            tipo=fallos["error"].apply(lambda e: (e or {}).get("tipo")),
            donde=fallos["error"].apply(lambda e: (e or {}).get("donde")),
            mensaje=fallos["error"].apply(lambda e: str((e or {}).get("mensaje"))[:160]),
        )[["timestamp", "origen", "pregunta", "tipo", "donde", "mensaje"]]
        st.dataframe(detalle, use_container_width=True, hide_index=True)

# ================================================================ TAB DATOS ==
with tab_datos:
    st.caption("Vista tabular completa (accesible y auditable): cada fila es "
               "una traza del JSONL con sus metricas planas.")
    columnas = ["timestamp", "origen", "caso_id", "variante", "pregunta", "exito",
                "latencia_total_s", "num_pasos_react", "num_llamadas_llm",
                "tokens_prompt", "tokens_completion", "tokens_total",
                "num_recuperaciones", "docs_recuperados", "memoria_rss_mb",
                "cpu_proceso_pct", "ruta", "longitud_respuesta", "trace_id"]
    presentes = [c for c in columnas if c in dfv.columns]
    st.dataframe(dfv[presentes].sort_values("timestamp", ascending=False),
                 use_container_width=True, hide_index=True, height=480)
