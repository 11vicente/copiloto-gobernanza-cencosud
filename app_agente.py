"""
=============================================================================
APP_AGENTE.PY - API REST para el Agente LangGraph (Fase 4)
=============================================================================
Co-piloto de Gobernanza Corporativa - Backend del AGENTE

Este backend expone el agente de agente.py (grafo ReAct con LangGraph) por
HTTP, para poder probarlo desde una interfaz web (frontend/agente.html), no
solo por su CLI.

POR QUE UN ARCHIVO APARTE Y NO TOCAR app.py:
    - app.py sirve el RAG conversacional (motor_rag.py) con streaming SSE y NO
      debe modificarse (restriccion del proyecto). El agente es un sistema
      distinto, asi que vive en su propio proceso/puerto (8001) sin interferir.
    - Asi pueden coexistir: app.py en :8000 (RAG) y app_agente.py en :8001 (agente).

DIFERENCIA CLAVE CON app.py (streaming):
    El agente NO entrega tokens uno a uno: ejecutar_agente() corre todo el
    ciclo ReAct (razonar -> buscar/reportar -> ... -> responder) y devuelve la
    respuesta final completa de una vez. Por eso aqui usamos un endpoint JSON
    normal (request/response), no SSE. La UI muestra un indicador "pensando..."
    mientras el grafo trabaja.

MEMORIA DE CORTO PLAZO ENTRE PETICIONES:
    Igual que app.py mantiene una memoria global single-session, aqui guardamos
    un historial_sesion en memoria de proceso para que el agente recuerde los
    turnos anteriores dentro de la misma sesion del navegador. /reset lo limpia.
    (Decision academica simple; produccion multiusuario requeriria sesiones.)

Endpoints:
    GET  /health         Health check.
    POST /preguntar      Recibe una pregunta, corre el agente, devuelve respuesta.
    POST /reset          Limpia la memoria de corto plazo de la sesion.

Uso:
    uvicorn app_agente:app --port 8001
=============================================================================
"""

import logging
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage

# Importamos la API publica del agente y utilidades de memoria de largo plazo.
# Al importar agente.py se ejecuta su inicializacion (que a su vez importa
# motor_rag.py: MongoDB, embeddings, retriever). Sucede UNA vez al arrancar.
from agente import ejecutar_agente, cargar_memoria_largo_plazo

# [EP3 — OBSERVABILIDAD, cambio ADITIVO] Este unico import activa el trazado
# JSONL del agente (logs/agente_trazas.jsonl) sin tocar la logica de este
# archivo ni de agente.py: cada consulta web queda registrada con origen="api"
# (latencia por nodo, tokens, pasos ReAct, errores). Quitarlo desactiva la
# observabilidad y todo vuelve a funcionar exactamente igual.
import observabilidad.auto  # noqa: F401,E402


# =============================================================================
# 1. LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("copiloto.api_agente")


# =============================================================================
# 2. INSTANCIA FastAPI + CORS
# =============================================================================
app = FastAPI(
    title="Co-piloto de Gobernanza - Agente (LangGraph)",
    description=(
        "API REST que expone el agente ReAct (agente.py) construido con "
        "LangGraph. Permite probar desde la web el razonamiento multi-paso, "
        "la busqueda RAG y la generacion de reportes ejecutivos."
    ),
    version="1.0.0",
)

# CORS abierto en desarrollo (igual que app.py) para que el frontend servido
# desde otro origen/puerto pueda llamar a este backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# =============================================================================
# 3. MEMORIA DE CORTO PLAZO DE LA SESION (en memoria de proceso)
# =============================================================================
# Buffer de mensajes de la conversacion en curso. ejecutar_agente lo recibe
# para que el agente "recuerde" turnos previos; al terminar conservamos los
# ultimos 4 mensajes (memoria de corto plazo de alta fidelidad).
historial_sesion: list = []


# =============================================================================
# 4. MODELOS Pydantic
# =============================================================================
class PreguntaRequest(BaseModel):
    """Cuerpo del POST /preguntar."""
    pregunta: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Pregunta del director en lenguaje natural.",
        examples=[
            "Genera un reporte ejecutivo sobre los riesgos identificados en 2023 y 2024"
        ],
    )


class RespuestaResponse(BaseModel):
    respuesta: str
    duracion_seg: float


class HealthResponse(BaseModel):
    status: str
    servicio: str
    version: str
    temas_memoria_largo_plazo: int


class ResetResponse(BaseModel):
    status: str
    mensaje: str


# =============================================================================
# 5. ENDPOINTS
# =============================================================================
@app.get("/health", response_model=HealthResponse, tags=["util"])
def health_check():
    """Health check. Reporta tambien cuantos temas hay en la memoria de largo
    plazo, util para verificar de un vistazo que la persistencia funciona."""
    memoria_lp = cargar_memoria_largo_plazo()
    return HealthResponse(
        status="ok",
        servicio="co-piloto-agente",
        version=app.version,
        temas_memoria_largo_plazo=len(memoria_lp.get("temas_consultados", [])),
    )


@app.post("/preguntar", response_model=RespuestaResponse, tags=["agente"])
def preguntar(req: PreguntaRequest):
    """Corre el agente para una pregunta y devuelve la respuesta final.

    A diferencia de /chat en app.py (streaming), aqui esperamos a que el grafo
    complete TODO su ciclo ReAct y devolvemos la respuesta de una vez. El
    razonamiento paso a paso (los 'plan') queda en los logs del servidor.
    """
    global historial_sesion
    log.info("Pregunta al agente: %s", req.pregunta[:100])
    inicio = time.time()

    # Pasamos el historial de la sesion para memoria de corto plazo.
    respuesta = ejecutar_agente(req.pregunta, historial=historial_sesion)

    # Actualizamos el buffer de corto plazo y lo recortamos a 4 mensajes.
    historial_sesion.append(HumanMessage(content=req.pregunta))
    historial_sesion.append(AIMessage(content=respuesta))
    historial_sesion = historial_sesion[-4:]

    duracion = time.time() - inicio
    log.info("Respuesta del agente lista en %.2fs", duracion)
    return RespuestaResponse(respuesta=respuesta, duracion_seg=round(duracion, 2))


@app.post("/reset", response_model=ResetResponse, tags=["util"])
def reset():
    """Limpia la memoria de corto plazo de la sesion (no toca la de largo plazo,
    que persiste entre sesiones en el JSON)."""
    global historial_sesion
    historial_sesion = []
    log.info("Memoria de corto plazo del agente reiniciada.")
    return ResetResponse(
        status="ok",
        mensaje="Memoria de corto plazo reiniciada (la de largo plazo persiste).",
    )
