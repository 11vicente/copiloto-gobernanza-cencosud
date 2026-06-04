# Co-piloto de Gobernanza Corporativa para Cencosud S.A.

> Sistema RAG (Retrieval-Augmented Generation) que permite a directores de
> Cencosud S.A. consultar la Memoria Anual y el Código de Ética en lenguaje
> natural, con respuestas trazables, libres de alucinaciones y entregadas
> en streaming token a token.
>
> **Fase 4 (nuevo):** además del chat RAG, el proyecto incorpora un **agente
> funcional construido con LangGraph** (ciclo ReAct) capaz de razonar paso a
> paso, decidir qué herramienta usar, encadenar varios pasos, calcular cifras
> exactas y generar reportes ejecutivos, con memoria de corto y largo plazo.

**Asignatura:** ISY0101 — Ingeniería de Soluciones con IA
**Autor:** Vicente Varela Rios
**Docente:** Francisco Miqueles Varela

---

> **Nota para el evaluador:** el archivo `.env` con las credenciales
> reales (MongoDB Atlas, GitHub Models, LangSmith) **no está incluido en
> este repositorio** por motivos de seguridad. El autor lo envió por canal
> privado al docente. Una vez que pongas el `.env` en la raíz del proyecto,
> los pasos de este README permiten ejecutar y validar todo el sistema.

---

## Tabla de contenidos

1. [Descripción del sistema](#1-descripción-del-sistema)
2. [Arquitectura](#2-arquitectura)
3. [Estructura del repositorio](#3-estructura-del-repositorio)
4. [Requisitos previos](#4-requisitos-previos)
5. [Instalación paso a paso](#5-instalación-paso-a-paso)
6. [Configuración (`.env`)](#6-configuración-env)
7. [Ejecución del sistema](#7-ejecución-del-sistema)
8. [Verificación y pruebas](#8-verificación-y-pruebas)
9. [Solución de problemas frecuentes](#9-solución-de-problemas-frecuentes)
10. [Stack tecnológico](#10-stack-tecnológico)

---

## 1. Descripción del sistema

El proyecto resuelve el problema de **infoxicación** en directorios
corporativos: las Memorias Anuales de Cencosud superan las 200 páginas y
los directores deben revisarlas en plazos acotados. Este sistema permite
hacerles preguntas en lenguaje natural y obtener respuestas ejecutivas con
cita explícita a la fuente y página exacta.

### Características principales

- **RAG funcional** sobre 3.433 chunks vectorizados de 4 documentos
  corporativos (1.213 páginas en total).
- **Respuestas trazables** con cita formato `[Fuente N] archivo (pág. X)`.
- **Anti-alucinación** mediante restricción explícita al contexto
  recuperado y temperatura baja del LLM.
- **Memoria conversacional** con compresión progresiva
  (`ConversationSummaryBufferMemory`) para conversaciones largas sin
  saturar la ventana de tokens.
- **Reformulador history-aware**: reformula preguntas anafóricas
  ("¿y en 2024?") antes de buscar en el vector store.
- **Streaming token a token** vía Server-Sent Events (SSE).
- **Observabilidad completa** con LangSmith: cada consulta queda trazada.
- **Few-Shot + Chain of Thought** para calibrar tono ejecutivo y forzar
  razonamiento explícito ("Análisis Previo" antes de "Respuesta Final").

### Agente funcional con LangGraph (Fase 4)

Sobre el RAG anterior se construyó un **agente ReAct** (`agente.py`) que
**decide** qué hacer en lugar de seguir un flujo fijo:

- **Grafo de decisión explícito (LangGraph)** con 4 nodos —
  `razonar → buscar / calcular / reportar → responder` — y aristas
  condicionales. Cada paso vuelve a `razonar`, lo que permite
  encadenar múltiples pasos.
- **Tres herramientas (`@tool`):**
  1. `buscar_documentos` — reutiliza el `retriever` (k=6) y reformateo de
     citas de `motor_rag.py`.
  2. `generar_reporte` — sintetiza un reporte ejecutivo en markdown
     (resumen, hallazgos, recomendaciones, fuentes).
  3. `calcular_metricas` — **calculadora aritmética exacta y segura**
     (variaciones %, ratios, crecimientos) con evaluador `ast` de lista
     blanca (sin `eval`, a prueba de inyección de código).
- **Memoria de corto plazo:** buffer de los últimos 4 mensajes (alta fidelidad).
- **Memoria de largo plazo:** persistencia en `memoria_largo_plazo.json`
  (resumen de la sesión anterior, temas consultados, reportes generados);
  se inyecta en el prompt al iniciar.
- **Planificación explícita:** en cada paso el agente escribe un `plan`
  (qué busca, por qué esa herramienta) visible en los logs.
- **Observabilidad:** logging de cada transición de nodo + integración
  LangSmith con las mismas variables de entorno que el RAG.

### Interfaz web unificada (RAG + Agente)

`frontend/index.html` es ahora una **UI única con un selector** que permite
alternar entre los dos motores en la misma ventana:

- **Chat RAG** → backend `app.py` (`:8000`), streaming SSE.
- **Agente** → backend `app_agente.py` (`:8001`), request/response.

---

## 2. Arquitectura

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NAVEGADOR DEL DIRECTOR                           │
│         frontend/index.html  (HTML + JS vanilla + marked.js)        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ HTTP / SSE
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                BACKEND FastAPI  (app.py)                            │
│   GET  /health   POST /chat (streaming)   POST /reset               │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ import
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                MOTOR RAG  (motor_rag.py)                            │
│   Reformulador → Retriever → Prompt(CoT+Few-Shot) → LLM → Memoria   │
│                       Cadena LCEL de LangChain                      │
└──────┬──────────────────┬──────────────────┬────────────────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌──────────────┐  ┌────────────────┐  ┌────────────────┐
│  MongoDB     │  │  GitHub Models │  │  LangSmith     │
│  Atlas       │  │  (Azure)       │  │  (trazas)      │
│  Vector      │  │  embeddings +  │  │                │
│  Search      │  │  gpt-4o-mini   │  │                │
└──────────────┘  └────────────────┘  └────────────────┘

      ▲ ingesta única (offline)
┌─────┴───────────────────────────────────────────────────────────────┐
│            SCRIPT DE INGESTA  (ingesta.py)                          │
│   PDFs → PyPDFLoader → Splitter → Embeddings → MongoDB Atlas        │
└─────────────────────────────────────────────────────────────────────┘
```

### Grafo del agente (Fase 4 — `agente.py`)

```
                          +-------------+
             (entrada)--->|   razonar   |<--------------------------+
                          +------+------+                           |
                                 | router() inspecciona             |
                                 | el ultimo mensaje                |
       buscar    calcular        |  reportar      (sin tool_call)   |
         +----------+------------+------------+----------+          |
         |          |                         |          |          |
         v          v                         v          v          |
   +----------+ +-----------+           +-----------+ +-----------+  |
   | buscar   | | calcular  |           | reportar  | | responder |  |
   +----+-----+ +-----+-----+           +-----+-----+ +-----+-----+  |
        |             |                       |             |        |
        +----- vuelve a razonar (multi-paso) -+             v        |
                                                          +----+     |
                                                          | END|     |
                                                          +----+     |
                                                                     |
   buscar / calcular / reportar SIEMPRE vuelven a razonar -----------+
```

---

## 3. Estructura del repositorio

```
Proyecto_RAG/
│
├── data/                       # PDFs corporativos (Memorias 2023-2025, Código de Ética)
│   ├── codigo_etica.pdf
│   ├── memoria_2023.pdf
│   ├── memoria_2024.pdf
│   └── memoria_2025.pdf
│
├── frontend/
│   ├── index.html              # UI unificada (selector Chat RAG / Agente)
│   ├── index_rag_backup.html   # Respaldo de la UI solo-RAG original
│   └── agente.html             # UI solo-agente (respaldo)
│
├── ingesta.py                  # Fase 1: ETL de PDFs a MongoDB Atlas
├── motor_rag.py                # Fase 2: Cadena RAG conversacional (LCEL)
├── app.py                      # Fase 3: API REST del RAG (FastAPI + SSE)  :8000
├── agente.py                   # Fase 4: Agente ReAct con LangGraph (3 tools)
├── app_agente.py               # Fase 4: API REST del agente (FastAPI)      :8001
├── test_conexion.py            # Utilitario: diagnóstico de credenciales Mongo
│
├── requirements.txt            # Dependencias Python con versiones fijas
├── .env.example                # Plantilla de credenciales (sin valores reales)
├── .env                        # Credenciales reales — NO versionado (.gitignore)
│
├── memoria_largo_plazo.json    # Estado del agente — NO versionado (lo genera en runtime)
└── README.md                   # Este archivo
```

---

## 4. Requisitos previos

| Requisito | Versión | Notas |
|---|---|---|
| **Python** | **3.11.x** | NO usar 3.13 ni 3.14: faltan wheels pre-compilados de varias dependencias en Windows |
| **pip** | 24+ | Incluido con Python; actualizar con `python -m pip install --upgrade pip` |
| **Sistema operativo** | Windows / Linux / macOS | El proyecto fue desarrollado y probado en Windows 10 |
| **Acceso a internet** | — | Necesario para llamadas a GitHub Models, MongoDB Atlas y LangSmith |
| **Archivo `.env`** | — | Provisto por el autor en privado |

---

## 5. Instalación paso a paso

### 5.1. Obtener el código

```powershell
git clone <URL_DEL_REPO>
cd Proyecto_RAG
```

### 5.2. Crear y activar entorno virtual con Python 3.11

**Windows (PowerShell):**
```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

Confirma la versión:
```powershell
python --version    # debe responder: Python 3.11.x
```

### 5.3. Instalar dependencias

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

La instalación tarda 2–5 minutos. Todas las dependencias se descargan
como *wheels* pre-compilados; no requiere compilador C/C++.

### 5.4. Verificar instalación

```powershell
python -c "import langchain, langchain_openai, langchain_mongodb, pypdf, pymongo, fastapi; print('OK')"
```

Debe imprimir `OK` sin errores.

---

## 6. Configuración (`.env`)

El archivo `.env` con las credenciales reales **se entrega por canal
privado** y debe colocarse en la raíz del proyecto (al mismo nivel que
`app.py`, `motor_rag.py` e `ingesta.py`).

La estructura del archivo está documentada en `.env.example`. Las claves
que contiene son:

| Clave | Propósito |
|---|---|
| `GITHUB_TOKEN` | PAT de GitHub para acceder a GitHub Models (LLM y embeddings) |
| `GITHUB_BASE_URL` | Endpoint de GitHub Models (Azure) |
| `MONGO_URI` | Cadena de conexión al cluster de MongoDB Atlas |
| `DB_NAME` | Base de datos donde residen los chunks vectorizados |
| `COLLECTION_NAME` | Colección de chunks (`documentos`) |
| `INDEX_NAME` | Nombre del índice vectorial en Atlas Search (`vector_index`) |
| `LANGCHAIN_TRACING_V2` | Activa trazabilidad de LangSmith |
| `LANGCHAIN_API_KEY` | Credencial de LangSmith |
| `LANGCHAIN_PROJECT` | Nombre del proyecto en el dashboard de LangSmith |

> El cluster de MongoDB ya tiene los **3.433 chunks vectorizados** de los
> 4 documentos cargados, y el índice vectorial ya está creado y activo.
> No es necesario re-ejecutar la ingesta; el evaluador puede ir
> directamente a la sección 7.2 o 7.3 para probar el sistema.

### Validación rápida de credenciales

Antes de ejecutar el sistema, conviene verificar que el `.env` está bien
configurado:

```powershell
python test_conexion.py
```

Salida esperada si todo está bien:
```
[OK] Ping exitoso
[OK] Bases visibles: [...]
[OK] Escritura exitosa
[OK] Limpieza exitosa
[EXITO] Las credenciales y permisos son correctos.
```

---

## 7. Ejecución del sistema

El sistema tiene **tres formas de uso**, en orden de complejidad:

### 7.1. Ingesta de documentos (no requerido para el evaluador)

> Este paso ya fue ejecutado por el autor. La colección de Mongo ya
> contiene los 3.433 chunks vectorizados. Esta sección queda documentada
> solo para fines de reproducibilidad si se quisiera re-cargar otros PDFs.

```powershell
python ingesta.py
```

Tarda 3–7 minutos: lee los PDFs en `data/`, los chunkea y vectoriza, y
los inserta en MongoDB Atlas.

### 7.2. Modo CLI (consola interactiva)

Ideal para pruebas rápidas, sin frontend ni servidor:

```powershell
python motor_rag.py
```

```
======================================================================
CO-PILOTO DE GOBERNANZA CORPORATIVA - Cencosud S.A.
Modo CLI. Escribe 'salir' o Ctrl+C para terminar.
======================================================================

[Director] ¿Cuáles son los principales riesgos identificados en la Memoria 2024?

[Co-piloto] **Análisis Previo:**
- Reviso la sección de "Gestión Integral de Riesgos"...
...
```

### 7.3. Modo API + Frontend (uso completo)

Requiere **dos terminales abiertas en paralelo**.

**Terminal 1 — Backend API:**
```powershell
.\venv\Scripts\Activate.ps1
uvicorn app:app --port 8000
```

Espera 5–10 segundos hasta ver:
```
INFO: copiloto.api: Co-piloto de Gobernanza Corporativa - Backend listo
INFO: Application startup complete.
INFO: Uvicorn running on http://127.0.0.1:8000
```

**Terminal 2 — Frontend (servidor estático):**
```powershell
cd frontend
python -m http.server 3000
```

Luego abre en el navegador:
- **http://localhost:3000** — interfaz web del Co-piloto
- **http://localhost:8000/docs** — Swagger UI auto-generado del API

> **¿Por qué `python -m http.server` y no abrir `index.html` directamente?**
> Algunos navegadores tratan `file://` como origen `null` y bloquean
> `fetch()` por CORS. Servir el HTML por HTTP local lo evita.

### 7.4. Agente con LangGraph (Fase 4)

El agente se puede usar por **CLI** o por **web**.

**Opción A — CLI interactivo:**
```powershell
.\venv\Scripts\Activate.ps1
python agente.py
```
En los logs verás el ciclo ReAct paso a paso (`[NODO] razonar`,
`[ROUTER] ...`, `[PLAN] ...`) y, entre sesiones, el agente recuerda los
temas de la sesión anterior (memoria de largo plazo en
`memoria_largo_plazo.json`).

**Opción B — Web (UI unificada con selector RAG / Agente):**

Requiere **tres terminales**: ambos backends + el servidor estático.

```powershell
# Terminal 1 — Backend RAG (:8000)
.\venv\Scripts\Activate.ps1
uvicorn app:app --port 8000

# Terminal 2 — Backend Agente (:8001)
.\venv\Scripts\Activate.ps1
uvicorn app_agente:app --port 8001

# Terminal 3 — Frontend estático
cd frontend
python -m http.server 3000
```

Abre **http://localhost:3000** y usa el selector **[ Chat RAG | Agente ]**
de la cabecera para alternar entre los dos motores en la misma página.

> El backend del agente expone: `GET /health`, `POST /preguntar`
> (request/response, no streaming) y `POST /reset`.

---

## 8. Verificación y pruebas

Esta sección permite validar end-to-end que el sistema funciona.

### 8.1. Verificar el backend (sin frontend)

Con el servidor corriendo en `localhost:8000`:

```powershell
# Health check
curl http://localhost:8000/health
# Esperado: {"status":"ok","servicio":"co-piloto-gobernanza","version":"1.0.0"}

# Reiniciar memoria
curl -X POST http://localhost:8000/reset
# Esperado: {"status":"ok","mensaje":"Memoria conversacional reiniciada."}

# Streaming SSE (el flag -N evita que curl bufferee)
curl -N -X POST http://localhost:8000/chat `
     -H "Content-Type: application/json" `
     -d '{\"pregunta\":\"En una frase, ¿que es Cencosud?\"}'
```

La última llamada debe devolver eventos SSE en streaming:
```
event: token
data: {"token": "C"}

event: token
data: {"token": "enc"}

...

event: done
data: [FIN]
```

### 8.2. Verificar el frontend

Con backend (puerto 8000) y http.server (puerto 3000) corriendo:

1. Abre http://localhost:3000.
2. El indicador inferior debe mostrar **"Backend conectado en http://localhost:8000"** con punto verde.
3. Click en cualquiera de las 4 sugerencias de inicio.
4. Verifica que:
   - Tu pregunta aparece en burbuja azul a la derecha.
   - La respuesta del agente aparece **token a token** (efecto typewriter)
     en burbuja blanca a la izquierda.
   - El cursor parpadeante desaparece al terminar el stream.
   - El markdown se renderiza correctamente (negritas, bullets, citas).
5. Haz una pregunta de seguimiento como **"¿Y en 2023?"**: el reformulador
   history-aware debe interpretar correctamente el "y en 2023?" según el
   contexto previo.

### 8.3. Verificar trazabilidad en LangSmith

1. Ve a https://smith.langchain.com → proyecto `copiloto-gobernanza-cencosud`.
2. Después de cada pregunta, refresca la página.
3. Click en el run más reciente. Debes ver el árbol de ejecución:
   ```
   RunnableSequence
   ├── preparar_inputs
   │   ├── ChatOpenAI (reformulador)        ← solo si hay historial
   │   └── MongoDBAtlasVectorSearch         ← retrieval k=6
   ├── ChatPromptTemplate
   ├── ChatOpenAI (respuesta principal)     ← genera CoT + respuesta
   └── StrOutputParser
   ```
4. Click en cualquier nodo para ver inputs, outputs, tokens consumidos
   y latencia.

### 8.4. Casos de prueba sugeridos

| # | Pregunta | Capacidad que valida |
|---|---|---|
| 1 | ¿Cuáles son los principales riesgos identificados en la Memoria 2024? | Retrieval simple, CoT, formato ejecutivo |
| 2 | Compara los desafíos entre la Memoria 2023 y 2024. ¿Qué riesgos nuevos aparecen? | Retrieval cross-document, síntesis comparativa |
| 3 | Y de esos riesgos nuevos, ¿cuál es el más urgente? | Reformulador history-aware (anafórica) |
| 4 | ¿Qué dice el Código de Ética sobre conflictos de interés? | Retrieval sobre documento distinto |
| 5 | ¿Cuál es el color favorito del CEO? | Anti-alucinación: debe responder *"No tengo información suficiente"* |

El caso 5 es crítico: si el sistema **inventa** una respuesta, las
restricciones anti-alucinación no están funcionando.

### 8.5. Casos de prueba del agente (Fase 4)

Con el agente corriendo (CLI o `:8001`):

| # | Pregunta | Flujo esperado / capacidad |
|---|---|---|
| 1 | ¿Cuáles son los riesgos de la Memoria 2024? | `razonar → buscar → razonar → responder` |
| 2 | Genera un reporte ejecutivo sobre los riesgos de 2023 y 2024 | `razonar → buscar → … → reportar → responder` |
| 3 | ¿Cuánto variaron las emisiones de GEI entre 2023 y 2024? Calcula el % exacto. | Encadena `buscar → calcular`; el cálculo lo hace `calcular_metricas`, no el LLM |
| 4 | ¿Cuál es el color favorito del CEO? | Anti-alucinación: *"No tengo información suficiente…"* |
| 5 | (2ª sesión, tras reiniciar) cualquier pregunta | El banner inicial menciona los temas de la sesión anterior (memoria de largo plazo) |

---

## 9. Solución de problemas frecuentes

### "ModuleNotFoundError" al ejecutar cualquier script
- Asegúrate de que el venv está activado: el prompt debe iniciar con `(venv)`.
- Si recién creaste el venv, instala dependencias: `pip install -r requirements.txt`.

### Error de compilación al instalar numpy / pydantic
- Causa: estás usando Python 3.13 o 3.14 (sin wheels pre-compilados).
- Solución: recrear el venv con Python 3.11 (`py -3.11 -m venv venv`).

### `bad auth: authentication failed` al conectar a Mongo
- Verifica que el archivo `.env` esté en la raíz del proyecto.
- Confirma que la IP de la máquina del evaluador esté autorizada en
  Atlas → Network Access (durante el período de evaluación está abierta a `0.0.0.0/0`).

### `403 Forbidden` al consumir LangSmith
- API key inválida o expirada en `.env`.
- Como workaround temporal, deshabilitar el tracing en `.env`:
  `LANGCHAIN_TRACING_V2=false` (no afecta la funcionalidad del RAG).

### El frontend muestra mensajes vacíos pero el backend logea respuestas correctas
- Causa común: el navegador está cacheando una versión antigua del HTML.
- Solución: refrescar con `Ctrl+F5` (hard reload).

### Uvicorn dice "Started reloader process" y aparenta colgarse
- No está colgado: está cargando el motor RAG (5–10 segundos en silencio).
- Espera hasta ver `INFO: Application startup complete.`
- Si te molesta el delay, ejecuta sin reload: `uvicorn app:app --port 8000`.

---

## 10. Stack tecnológico

| Capa | Tecnología | Versión |
|---|---|---|
| Lenguaje | Python | 3.11.9 |
| Web framework | FastAPI | 0.115.6 |
| Servidor ASGI | Uvicorn | 0.32.1 |
| Streaming | sse-starlette | 2.1.3 |
| Orquestador IA | LangChain | 0.3.13 |
| Agente / grafo de estados | LangGraph | 0.2.76 |
| Embeddings y LLM | GitHub Models (vía OpenAI-compatible API) | text-embedding-3-small + gpt-4o-mini |
| Base vectorial | MongoDB Atlas Vector Search | — |
| Driver MongoDB | pymongo | 4.10.1 |
| Carga de PDFs | pypdf (vía langchain-community) | 5.1.0 |
| Memoria | langchain.memory.ConversationSummaryBufferMemory | 0.3.13 |
| Observabilidad | LangSmith | 0.2.3 |
| Frontend | HTML5 + JavaScript vanilla + marked.js | — |
| Gestión de secretos | python-dotenv | 1.0.1 |

---

## Declaración de uso de IA

Este proyecto utilizó IA generativa (Claude / Gemini) como apoyo en:
generación de scaffolding de código, redacción descriptiva del informe
y diagramas. Las decisiones técnicas, justificaciones, conclusiones y
reflexiones son del autor.
