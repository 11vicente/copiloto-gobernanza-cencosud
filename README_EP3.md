# EP3 — Observabilidad y Trazabilidad del Agente

> Capa de **observabilidad** sobre el agente ReAct de la EP2 (`agente.py`):
> trazas estructuradas por consulta, evaluación de precisión/consistencia,
> análisis de cuellos de botella y anomalías, y dashboard interactivo.
> **Cero cambios de lógica** en el agente: toda la instrumentación es aditiva.

**Asignatura:** ISY0101 — Ingeniería de Soluciones con IA · **Evaluación Parcial N°3**
**Autor:** Vicente Varela Rios

Documentos relacionados: [README.md](README.md) (instalación base del proyecto) ·
[INFORME_EP3.md](INFORME_EP3.md) (informe técnico) ·
[SEGURIDAD_Y_ETICA.md](SEGURIDAD_Y_ETICA.md) (protocolos IE6).

---

## 1. Arquitectura de la capa de observabilidad

```
                       (sin cambios de logica)
  evaluacion/evaluar.py ----------+
  evaluacion/generar_trazas.py ---+---> agente.py (grafo LangGraph)
  app_agente.py (:8001) ----------+          |
                                             | eventos LangChain/LangGraph
                                             v
                            observabilidad/instrumentacion.py
                            (TrazadorJSONL: callback handler)
                                             |
                                             v  1 linea JSON por consulta
                              logs/agente_trazas.jsonl  <== FUENTE DE VERDAD
                                   |                  |
                                   v                  v
                     analisis/analizar_logs.py   dashboard/dashboard.py
                     (hallazgos IE3/IE4)         (Streamlit + Plotly, IE5)
```

- El trazador se **suscribe a los eventos** del grafo (`with_config(callbacks=…)`,
  mecanismo oficial de LangChain) y registra por consulta: latencia total y por
  nodo, ruta del grafo, pasos ReAct, herramientas con su latencia, llamadas LLM
  con **tokens reales del proveedor**, recuperaciones vectoriales, RAM/CPU del
  proceso y errores tipificados con su ubicación.
- **No depende de LangSmith** (cuya credencial responde 403 en este proyecto):
  las trazas locales son la fuente de verdad, con `trace_id` y spans al estilo
  del estándar de trazabilidad distribuida.

## 2. Archivos nuevos de la EP3

| Ruta | Rol |
|---|---|
| `observabilidad/instrumentacion.py` | Callback handler + `ejecutar_agente_observado()` |
| `observabilidad/auto.py` | Activa el trazado con un import (lo usa la API web) |
| `evaluacion/casos.json` | 9 casos × 3 variantes de fraseo con criterio de acierto |
| `evaluacion/evaluar.py` | Precisión, consistencia y errores (IE1) |
| `evaluacion/generar_trazas.py` | Batería fija de 16 consultas variadas (carga) |
| `analisis/analizar_logs.py` | Agregados, cuellos de botella y anomalías (IE3/IE4) |
| `dashboard/dashboard.py` | Dashboard Streamlit + Plotly (IE5) |
| `logs/agente_trazas.jsonl` | Evidencia versionada: las trazas reales generadas |
| `.streamlit/config.toml` | Tema fijo del dashboard (capturas reproducibles) |
| `SEGURIDAD_Y_ETICA.md` | Protocolos de seguridad y uso responsable (IE6) |

## 3. Requisitos

Los de la EP1/EP2 (ver [README.md](README.md) §4–6: venv con **Python 3.11**,
`.env` con credenciales) más las dependencias nuevas, ya declaradas en
`requirements.txt` (pandas, streamlit, plotly, psutil):

```powershell
venv\Scripts\python.exe -m pip install -r requirements.txt
```

> **Todos los comandos** se ejecutan desde la **raíz del proyecto** y con el
> **python del venv** (`venv\Scripts\python.exe`), nunca el python global.

## 4. Flujo completo de validación (paso a paso)

### Paso 0 — Verificar credenciales (opcional)

```powershell
venv\Scripts\python.exe test_conexion.py
```

### Paso 1 — Generar trazas reales

**Opción A — Evaluación con criterios (IE1):** corre los 9 casos × 3
repeticiones (27 consultas reales, ~15 min con pausas de cortesía):

```powershell
venv\Scripts\python.exe evaluacion\evaluar.py --repeticiones 3 --pausa 8
```

Produce `evaluacion/resultados_evaluacion.json` (métricas por caso y globales)
y agrega 27 trazas a `logs/agente_trazas.jsonl` (origen=`evaluacion`).

**Opción B — Batería de carga (volumen y variedad):** 16 consultas fijas
(años distintos, comparativas, reporte, cálculos, casos borde):

```powershell
venv\Scripts\python.exe evaluacion\generar_trazas.py --pausa 8
```

> **Límites del free tier de GitHub Models:** si aparece `RateLimitError`, los
> scripts esperan 70 s y reintentan una vez; si la cuota **diaria** se agota,
> se detienen indicando el comando de reanudación (`--desde N`). Las trazas ya
> escritas nunca se pierden (el JSONL es acumulativo).
>
> **Nota:** el repositorio ya incluye `logs/agente_trazas.jsonl` con la corrida
> oficial documentada en el informe, de modo que los pasos 2 y 3 pueden
> validarse **sin gastar cuota**.

### Paso 2 — Analizar los logs (IE3/IE4)

```powershell
venv\Scripts\python.exe analisis\analizar_logs.py
```

Imprime los HALLAZGOS numerados (cuello de botella, outliers IQR, patrones de
rutas, costos) y escribe el detalle con tablas en `analisis/hallazgos.md`.

### Paso 3 — Levantar el dashboard (IE5)

```powershell
venv\Scripts\python.exe -m streamlit run dashboard\dashboard.py
```

Abre `http://localhost:8501`. Pestañas: **Latencia** (línea de tiempo, cuello
de botella por nodo, distribución de pasos ReAct), **Recursos** (tokens
prompt/completion, llamadas LLM, RAM), **Herramientas y rutas**, **Calidad
(IE1)**, **Errores** y **Datos** (tabla completa). Filtros en la barra lateral:
origen, rango de fechas, solo-errores y búsqueda de texto.

### Paso 4 (opcional) — Trazar el uso web real

Levantar el sistema completo (ver README.md §7.4); las consultas hechas desde
el navegador quedan trazadas con `origen="api"` gracias al import aditivo de
`observabilidad.auto` en `app_agente.py`:

```powershell
# Terminal 1        # Terminal 2                 # Terminal 3
uvicorn app:app --port 8000
                    uvicorn app_agente:app --port 8001
                                                 cd frontend; python -m http.server 3000
```

## 5. Esquema de cada traza (JSON Lines)

Campos principales de cada línea de `logs/agente_trazas.jsonl`:

| Campo | Contenido |
|---|---|
| `trace_id`, `timestamp`, `origen`, `caso_id`, `variante` | Identidad y procedencia de la consulta |
| `pregunta`, `respuesta`, `respuesta_sha256`, `longitud_respuesta` | Contenido (con hash de integridad) |
| `exito`, `error {tipo, mensaje, donde}` | Resultado; el error indica el nodo/herramienta/LLM donde ocurrió |
| `latencia_total_s`, `latencia_por_nodo_s`, `nodos[]`, `ruta` | Descomposición temporal por nodo del grafo (spans) |
| `num_pasos_react`, `num_ciclos_razonar` | Complejidad del ciclo ReAct |
| `herramientas[] {nombre, nodo, latencia_s, ok}` | Cada invocación de herramienta |
| `num_llamadas_llm`, `llamadas_llm[]`, `tokens_prompt/completion/total`, `fuente_tokens` | Uso del LLM; tokens **reales del proveedor** (`fuente_tokens="proveedor"`) |
| `num_recuperaciones`, `docs_recuperados` | Actividad de búsqueda vectorial |
| `memoria_rss_mb`, `delta_rss_mb`, `cpu_proceso_pct` | Recursos del proceso (psutil) |

## 6. Set de evaluación (cómo se mide IE1)

`evaluacion/casos.json` define 9 casos en 5 categorías (factual, cálculo,
anti-alucinación, reporte, robustez), cada uno con **3 variantes de fraseo**
(variabilidad de datos) y un **criterio determinista** de acierto: palabras
clave, regex de cifra exacta, frase literal de rehúsa y/o herramientas que
deben haberse invocado. `evaluar.py` rota las variantes entre repeticiones y
calcula precisión, consistencia (de veredicto, textual y operacional) y
frecuencia de errores tipificada. Para agregar un caso basta añadir un objeto
al JSON — no hay que tocar código.

## 7. Notas de honestidad métrica

- **Tokens:** exactos, informados por GitHub Models en cada respuesta
  (`usage`); si faltaran, el trazador estima con tiktoken y lo declara en
  `fuente_tokens`. Los tokens de *embeddings* no los expone la API y **no se
  contabilizan** (se registra el n° de recuperaciones y documentos).
- **CPU:** % del proceso medido entre inicio y fin de cada consulta (Windows
  puede reportar >100% con varios núcleos).
- **Consistencia textual:** similitud léxica (difflib), no semántica —
  aproximación declarada.
- **LangSmith:** el cableado por variables de entorno existe, pero la
  credencial responde `403 Forbidden`; por diseño, **nada de la EP3 depende de
  LangSmith** (ver `[observabilidad] Trazado JSONL activo` en el arranque).
