"""
=============================================================================
APP.PY - Fase 3: API REST con streaming SSE
=============================================================================
Co-piloto de Gobernanza Corporativa - Backend FastAPI

Expone el motor RAG (Fase 2) como una API HTTP. Tres endpoints:

    GET  /health       Health check para monitoreo basico.
    POST /chat         Recibe una pregunta y devuelve la respuesta del
                       agente RAG via Server-Sent Events (token a token).
    POST /reset        Limpia el historial conversacional.

DECISIONES DE DISENO:

- **SSE (Server-Sent Events) en lugar de WebSockets.** SSE es unidireccional
  (servidor -> cliente), justo lo que se necesita para streaming de tokens.
  WebSockets seria sobre-ingenieria. SSE ya es soportado nativamente por
  fetch + ReadableStream en cualquier frontend moderno, sin librerias.

- **POST en /chat (no GET).** Aunque el EventSource API nativo solo soporta
  GET, las preguntas ejecutivas pueden superar facilmente el limite de URL.
  Usamos POST + body JSON, y el frontend consume el stream via fetch +
  response.body.getReader(). Ver snippet del cliente al final del archivo.

- **Memoria compartida (single-session).** El motor_rag.py mantiene una
  memoria global. Es una decision deliberada por simplicidad academica.
  Para produccion multi-usuario habria que migrar a un dict de memorias
  por session_id o un store externo (Redis), pero queda fuera del alcance
  de la rubrica. /reset permite reiniciar la conversacion cuando haga falta.

- **CORS abierto en desarrollo.** En produccion habria que listar origenes
  permitidos (ej: el dominio del frontend desplegado).

- **Importacion eager del motor RAG.** Al cargar este modulo, motor_rag.py
  ya inicializa embeddings, retriever, LLM y prompts. Sucede UNA SOLA VEZ
  al arrancar uvicorn, no por cada request: critico para latencia.

Uso:
    uvicorn app:app --reload --port 8000

Documentacion interactiva auto-generada por FastAPI:
    http://localhost:8000/docs        (Swagger UI)
    http://localhost:8000/redoc       (ReDoc)
=============================================================================
"""

import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

# Importamos del motor RAG la funcion publica (streaming) y la memoria.
# Al hacer este import, motor_rag.py ejecuta toda su inicializacion modular:
# conexion a MongoDB, configuracion de embeddings, instanciacion del LLM,
# construccion de la cadena LCEL. Por eso el "primer arranque" del servidor
# tarda unos segundos: es cuando todo se conecta y queda listo en memoria.
from motor_rag import generar_respuesta_streaming, memoria


# =============================================================================
# 1. LOGGING
# =============================================================================
# Logging estructurado para debug en consola. En produccion se reemplazaria
# por un handler hacia ELK/Datadog/Cloud Logging.
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("copiloto.api")


# =============================================================================
# 2. CICLO DE VIDA DE LA APLICACION
# =============================================================================
# El "lifespan" de FastAPI permite ejecutar codigo al iniciar y al detener
# el servidor. Lo usamos para emitir un log claro de estado, util para que
# en la demo se vea cuando el motor RAG ya esta listo para recibir trafico.
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("================================================================")
    log.info("Co-piloto de Gobernanza Corporativa - Backend listo")
    log.info("Motor RAG cargado, vector store conectado, memoria inicializada")
    log.info("Documentacion: http://localhost:8000/docs")
    log.info("================================================================")
    yield
    log.info("Co-piloto detenido. Conexiones cerradas.")


# =============================================================================
# 3. INSTANCIA DE FastAPI
# =============================================================================
app = FastAPI(
    title="Co-piloto de Gobernanza Corporativa",
    description=(
        "API REST con streaming SSE sobre arquitectura RAG. "
        "Permite a directores de Cencosud S.A. consultar Memorias Anuales "
        "y Codigo de Etica en lenguaje natural, con respuestas trazables."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# 4. CORS
# =============================================================================
# Habilitamos CORS para que el frontend (Angular en dev: localhost:4200,
# o cualquier otro) pueda llamar al API desde un origen distinto al backend.
# allow_origins=["*"] es practico en desarrollo; en produccion se restringe.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,    # debe ser False cuando allow_origins=["*"]
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# =============================================================================
# 5. MODELOS Pydantic (validacion y documentacion automatica)
# =============================================================================
class ChatRequest(BaseModel):
    """Cuerpo del POST /chat. Pydantic valida tipo, limites y serializa.
    Si el cliente manda algo invalido, FastAPI responde 422 automaticamente."""
    pregunta: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Pregunta del director en lenguaje natural.",
        examples=[
            "¿Cuáles son los principales riesgos identificados en la Memoria 2024?"
        ],
    )


class HealthResponse(BaseModel):
    status: str
    servicio: str
    version: str


class ResetResponse(BaseModel):
    status: str
    mensaje: str


# =============================================================================
# 6. ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["util"])
def health_check():
    """Health check sin dependencias. Devuelve 200 si el proceso esta vivo.
    Util para load balancers, monitoring y para que el frontend verifique
    rapido si el backend esta disponible antes de hacer la primera consulta."""
    return HealthResponse(
        status="ok",
        servicio="co-piloto-gobernanza",
        version=app.version,
    )


@app.post("/chat", tags=["rag"])
async def chat_streaming(req: ChatRequest):
    """Endpoint principal del Co-piloto.

    Recibe una pregunta y devuelve la respuesta del agente RAG en streaming
    via Server-Sent Events. Cada token aparece en el cliente apenas el LLM
    lo genera, sin esperar a la respuesta completa.

    Formato de los eventos SSE enviados al cliente:

        event: token
        data: {"token": "fragmento de texto"}

        event: done
        data: [FIN]

        event: error
        data: {"error": "mensaje"}

    El cliente debe parsear cada chunk con JSON.parse(event.data) para
    obtener el token. El evento "done" marca el fin del stream.
    """
    log.info(f"Pregunta recibida: {req.pregunta[:100]}...")
    inicio = time.time()

    async def event_generator():
        """Generador async que SSE-Starlette consumira para emitir cada token.

        Internamente, generar_respuesta_streaming es un generador SINCRONO
        (motor_rag.py lo construye con `yield` sobre cadena_rag.stream()).
        sse_starlette detecta que es sync y lo ejecuta en un thread executor
        para no bloquear el event loop de uvicorn. No requiere refactor.
        """
        tokens_emitidos = 0
        try:
            for chunk in generar_respuesta_streaming(req.pregunta):
                tokens_emitidos += 1
                # Serializamos cada token como JSON: asi pasamos saltos de
                # linea, comillas, emojis y cualquier caracter especial sin
                # romper el formato SSE (que es text-based con delimitadores).
                yield {
                    "event": "token",
                    "data": json.dumps({"token": chunk}, ensure_ascii=False),
                }

            # Senal explicita de fin de stream para que el cliente cierre
            # la conexion limpiamente y libere el reader.
            yield {"event": "done", "data": "[FIN]"}

            duracion = time.time() - inicio
            log.info(
                f"Respuesta completada: {tokens_emitidos} chunks "
                f"en {duracion:.2f}s"
            )

        except Exception as e:
            # Capturamos cualquier error del pipeline (LLM, MongoDB, etc.)
            # y lo enviamos como evento SSE en lugar de cortar la conexion
            # abruptamente. El frontend puede mostrar un mensaje claro.
            log.exception("Error en el pipeline RAG")
            yield {
                "event": "error",
                "data": json.dumps(
                    {"error": f"{type(e).__name__}: {str(e)}"},
                    ensure_ascii=False,
                ),
            }

    return EventSourceResponse(event_generator())


@app.post("/reset", response_model=ResetResponse, tags=["util"])
def reset_memoria():
    """Limpia el historial conversacional.

    Util en demos para empezar una nueva consulta sin contaminacion del
    contexto previo, o cuando el director cambia de tema y no quiere que
    el reformulador history-aware mezcle preguntas anteriores."""
    memoria.clear()
    log.info("Memoria conversacional reiniciada por request del cliente.")
    return ResetResponse(
        status="ok",
        mensaje="Memoria conversacional reiniciada.",
    )


# =============================================================================
# CLIENTE DE REFERENCIA (snippet documental para el frontend)
# =============================================================================
# El siguiente fragmento NO se ejecuta en Python; queda aqui como referencia
# del consumo del API desde el frontend. La idea es que el cliente abra una
# conexion fetch al endpoint /chat y lea el ReadableStream a medida que llega:
#
#     async function preguntar(pregunta, onToken, onFin, onError) {
#         const resp = await fetch('http://localhost:8000/chat', {
#             method: 'POST',
#             headers: {'Content-Type': 'application/json'},
#             body: JSON.stringify({pregunta})
#         });
#         const reader = resp.body.getReader();
#         const decoder = new TextDecoder();
#         let buffer = '';
#         while (true) {
#             const {done, value} = await reader.read();
#             if (done) break;
#             buffer += decoder.decode(value, {stream: true});
#             const eventos = buffer.split('\n\n');
#             buffer = eventos.pop();
#             for (const ev of eventos) {
#                 if (ev.includes('event: done')) { onFin(); return; }
#                 if (ev.includes('event: error')) {
#                     const m = ev.match(/data: (.+)/);
#                     if (m) onError(JSON.parse(m[1]).error);
#                     return;
#                 }
#                 const m = ev.match(/data: (.+)/);
#                 if (m) {
#                     try {
#                         const {token} = JSON.parse(m[1]);
#                         onToken(token);
#                     } catch {}
#                 }
#             }
#         }
#     }
#
# Asi, el frontend solo necesita acumular los tokens en un string y mostrarlo
# en la UI con un efecto "typewriter", sin librerias externas.
