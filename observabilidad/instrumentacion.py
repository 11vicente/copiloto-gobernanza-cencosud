"""
=============================================================================
OBSERVABILIDAD/INSTRUMENTACION.PY — EP3: trazas estructuradas del agente
=============================================================================
Co-piloto de Gobernanza Corporativa — Capa de OBSERVABILIDAD (aditiva)

Este modulo instrumenta el agente de agente.py SIN MODIFICARLO. Genera, por
cada consulta, UNA linea JSON (formato JSON Lines) en logs/agente_trazas.jsonl
con las metricas que exige la rubrica EP3:

    - IE1: exito/fallo, tipo de error, respuesta (para medir precision y
      consistencia desde evaluacion/evaluar.py).
    - IE2: latencia total y por nodo, latencia por herramienta y por llamada
      LLM, tokens prompt/completion, n° de llamadas LLM, n° de pasos ReAct,
      RAM (RSS) y CPU del proceso.
    - IE3/IE4: secuencia exacta de nodos (ruta), errores con su ubicacion,
      recuperaciones vectoriales y documentos recuperados.

POR QUE UN CALLBACK HANDLER DE LANGCHAIN (y no prints ni ediciones al agente):
    - LangChain/LangGraph emiten eventos estandar (inicio/fin de cadena, de
      llamada al LLM, de herramienta, de retriever) hacia cualquier handler
      registrado. Suscribirse a esos eventos permite medir DESDE FUERA, sin
      tocar una sola linea de agente.py (restriccion dura del proyecto).
    - Es el mismo mecanismo que usa LangSmith. Como la credencial de LangSmith
      del proyecto responde 403, esta capa local es la FUENTE DE VERDAD de
      trazabilidad: no dependemos de ningun servicio externo (diseño
      "vendor-agnostic", alineado con el PDF 3.1.1 que menciona OpenTelemetry
      como estandar agnostico al proveedor).

POR QUE "ZERO-TOUCH" VIA with_config (envoltura, no monkey-patch invasivo):
    - agente.ejecutar_agente() resuelve `app_grafo` desde las variables del
      modulo EN CADA LLAMADA. Por lo tanto, re-atar agente.app_grafo a
      `app_grafo.with_config(callbacks=[...])` (un wrapper OFICIAL de
      LangChain que solo agrega configuracion) hace que TODAS las ejecuciones
      del grafo emitan eventos a nuestro handler, con 0 lineas cambiadas en
      agente.py. Si se quita este modulo, el agente queda exactamente igual.

CORRELACION DE EVENTOS (terminologia del PDF 3.2.1 "Trazabilidad"):
    - trace_id  : identificador unico de TODA la ejecucion de una consulta
                  (= run_id del grafo raiz, un UUID).
    - "spans"   : cada nodo del grafo, cada llamada LLM y cada herramienta se
                  registran como sub-operaciones con su latencia, correlados
                  por run_id/parent_run_id que LangChain propaga solo.
    - La ejecucion raiz se detecta por parent_run_id=None (robusto: no
      depende del nombre interno "LangGraph"), y un nodo del grafo se
      reconoce por (nombre en NODOS_GRAFO) Y (hijo directo de la raiz), lo
      que evita contar dos veces los runnables internos de LangGraph.

LIMITES DECLARADOS (honestidad metrica, exigida por el encargo):
    - Tokens: se toman EXACTOS del proveedor (usage_metadata / token_usage de
      GitHub Models, disponibles porque los LLM del agente no usan streaming).
      Si el proveedor no los entregara, se ESTIMAN con tiktoken y el registro
      queda marcado con fuente_tokens="estimado(tiktoken)".
    - Los tokens de EMBEDDINGS (busqueda vectorial) NO los expone la API de
      embeddings via callbacks; se registra el n° de recuperaciones y de
      documentos, no sus tokens.
    - CPU: psutil.cpu_percent mide el % del proceso entre inicio y fin de la
      consulta (en Windows puede superar 100% si usa varios nucleos).
    - Concurrencia: el handler asume ejecucion SECUENCIAL (una consulta a la
      vez), igual que la memoria de sesion global de app_agente.py. Para
      multiusuario real habria que llevar el estado por trace_id.

Uso programatico (scripts de evaluacion / carga):
    from observabilidad.instrumentacion import ejecutar_agente_observado
    registro = ejecutar_agente_observado("¿Riesgos 2024?", origen="evaluacion")

Uso desde la API web: basta `import observabilidad.auto` (ver auto.py).
=============================================================================
"""

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import psutil
from langchain_core.callbacks import BaseCallbackHandler

# Importar agente.py dispara su inicializacion completa (y la de motor_rag.py:
# MongoDB, embeddings, retriever). Es el MISMO costo que paga app_agente.py.
import agente

# =============================================================================
# 1. CONFIGURACION DEL MODULO
# =============================================================================
RUTA_PROYECTO = Path(__file__).resolve().parents[1]

# Ruta del archivo de trazas. Se puede redirigir con la variable de entorno
# OBS_RUTA_TRAZAS (util para pruebas sin contaminar la evidencia oficial).
RUTA_TRAZAS = Path(os.getenv("OBS_RUTA_TRAZAS",
                             str(RUTA_PROYECTO / "logs" / "agente_trazas.jsonl")))

# Nombres de los nodos declarados en agente.py. Se usan para FILTRAR los
# eventos de cadena: LangGraph emite tambien runnables internos (escrituras de
# canal, el router, etc.) que no son nodos y no deben contarse como pasos.
NODOS_GRAFO = {"razonar", "buscar", "calcular", "reportar", "responder"}

# OBS_DEBUG=1 imprime cada evento de cadena recibido (nombre y jerarquia).
# Sirvio para VERIFICAR empiricamente como nombra LangGraph 0.2.x sus runs,
# y queda disponible para que el evaluador audite el mecanismo.
_DEBUG = os.getenv("OBS_DEBUG", "0") == "1"

VERSION_ESQUEMA = 1          # por si el esquema del JSONL evoluciona
_proceso = psutil.Process()  # proceso actual (agente + esta capa)


def _ahora_iso() -> str:
    """Timestamp local en ISO 8601 con zona horaria (trazabilidad temporal)."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256(texto: str) -> str:
    """Hash de la respuesta: permite detectar respuestas identicas (consistencia)
    y da integridad a la evidencia sin depender del texto completo."""
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


# --- Estimacion de tokens (SOLO respaldo, ver docstring del modulo) ---
_encoder_tiktoken = None


def _estimar_tokens(texto: str) -> Optional[int]:
    """Estima tokens con tiktoken (encoding o200k_base, el de gpt-4o-mini).
    Devuelve None si tiktoken no esta disponible: preferimos declarar el dato
    como ausente antes que inventarlo (regla del encargo: no inventar metricas).
    """
    global _encoder_tiktoken
    try:
        if _encoder_tiktoken is None:
            import tiktoken
            _encoder_tiktoken = tiktoken.get_encoding("o200k_base")
        return len(_encoder_tiktoken.encode(texto))
    except Exception:
        return None


def _extraer_tokens_llm(response) -> tuple[Optional[int], Optional[int], str]:
    """Extrae (tokens_prompt, tokens_completion, fuente) de un LLMResult.

    Camino 1 (exacto): response.llm_output["token_usage"], que langchain-openai
    rellena con el usage REAL que devuelve GitHub Models en llamadas sin
    streaming (los dos LLM del agente son no-streaming, asi que aplica).
    Camino 2 (exacto): usage_metadata del AIMessage generado.
    Si ninguno existe, se devuelve (None, None, "no_disponible") y el llamador
    decide si estima. NUNCA se inventa el dato.
    """
    llm_output = getattr(response, "llm_output", None) or {}
    uso = llm_output.get("token_usage") or {}
    if uso.get("prompt_tokens") is not None:
        return uso.get("prompt_tokens"), uso.get("completion_tokens"), "proveedor"

    try:
        mensaje = response.generations[0][0].message
        um = getattr(mensaje, "usage_metadata", None)
        if um:
            return um.get("input_tokens"), um.get("output_tokens"), "proveedor"
    except (IndexError, AttributeError):
        pass
    return None, None, "no_disponible"


# =============================================================================
# 2. EL CALLBACK HANDLER (corazon de la instrumentacion)
# =============================================================================
class TrazadorJSONL(BaseCallbackHandler):
    """Suscriptor de eventos LangChain/LangGraph que arma un registro por
    consulta y lo persiste como una linea JSON.

    Ciclo de vida de una traza:
        on_chain_start(parent=None)  -> abre la traza (trace_id, t0, RAM/CPU)
        on_chain_start(nodo)         -> abre span de nodo
        on_chat_model_start/_end     -> span de llamada LLM (+tokens)
        on_tool_start/_end/_error    -> span de herramienta
        on_retriever_start/_end      -> conteo de recuperaciones vectoriales
        on_chain_end(nodo)           -> cierra span de nodo
        on_chain_end(parent raiz)    -> cierra la traza y ESCRIBE la linea
        on_chain_error(parent raiz)  -> idem pero con exito=False

    raise_error=False (default de BaseCallbackHandler): un bug del trazador
    JAMAS debe tumbar una respuesta del agente; la observabilidad es un
    ciudadano de segunda clase frente a la funcionalidad.
    """

    def __init__(self) -> None:
        super().__init__()
        # Metadatos de la PROXIMA consulta (los fija preparar(); si nadie los
        # fija —p. ej. trafico web via observabilidad/auto.py— se usa el default).
        self.origen_por_defecto = "desconocido"
        self._origen: Optional[str] = None
        self._caso_id: Optional[str] = None
        self._variante: Optional[str] = None

        # Ultimo registro finalizado (lo lee ejecutar_agente_observado).
        self.ultimo_registro: Optional[dict] = None

        self._reiniciar_estado()

    # ------------------------------------------------------------------ estado
    def _reiniciar_estado(self) -> None:
        """Limpia el estado interno entre consultas (una traza a la vez)."""
        self._root_run_id: Optional[UUID] = None
        self._t0_traza: float = 0.0
        self._pregunta: str = ""
        self._nodo_actual: Optional[str] = None
        self._t_por_run: dict[UUID, float] = {}       # run_id -> t_inicio
        self._nodo_por_run: dict[UUID, str] = {}      # run_id de nodo -> nombre
        self._tool_por_run: dict[UUID, dict] = {}     # run_id de tool -> parcial
        self._llm_por_run: dict[UUID, dict] = {}      # run_id de llm -> parcial
        self._nodos: list[dict] = []                  # spans de nodo, en orden
        self._llamadas_llm: list[dict] = []
        self._herramientas: list[dict] = []
        self._num_recuperaciones: int = 0
        self._docs_recuperados: int = 0
        self._ultimo_error: Optional[dict] = None

    def preparar(self, origen: str, caso_id: Optional[str] = None,
                 variante: Optional[str] = None) -> None:
        """Fija los metadatos de la consulta que viene (la llama el wrapper).
        Tambien limpia `ultimo_registro` para poder distinguir 'la traza se
        escribio' de 'la ejecucion reviento antes de entrar al grafo'."""
        self._origen = origen
        self._caso_id = caso_id
        self._variante = variante
        self.ultimo_registro = None
        self._reiniciar_estado()

    # ------------------------------------------------------- eventos de cadena
    def on_chain_start(self, serialized: Optional[dict], inputs: Any, *,
                       run_id: UUID, parent_run_id: Optional[UUID] = None,
                       **kwargs: Any) -> None:
        nombre = kwargs.get("name") or (serialized or {}).get("name") or "?"
        if _DEBUG:
            print(f"[OBS-DEBUG] chain_start nombre={nombre!r} run={str(run_id)[:8]} "
                  f"parent={str(parent_run_id)[:8] if parent_run_id else None}")

        # --- Raiz del grafo: parent_run_id=None es la señal robusta ---
        if parent_run_id is None:
            self._reiniciar_estado()
            self._root_run_id = run_id
            self._t0_traza = time.perf_counter()
            self._timestamp = _ahora_iso()
            # El estado inicial del grafo trae la pregunta original.
            if isinstance(inputs, dict):
                self._pregunta = str(inputs.get("pregunta", ""))
            # Linea base de recursos: cpu_percent(None) RESETEA el contador,
            # de modo que la lectura al final mida SOLO esta consulta.
            _proceso.cpu_percent(None)
            self._rss0_mb = _proceso.memory_info().rss / (1024 * 1024)
            return

        # --- Span de nodo: nombre conocido Y colgando directo de la raiz ---
        # (evita contar runnables internos homonimos o anidados de LangGraph)
        if nombre in NODOS_GRAFO and parent_run_id == self._root_run_id:
            self._nodo_actual = nombre
            self._nodo_por_run[run_id] = nombre
            self._t_por_run[run_id] = time.perf_counter()

    def on_chain_end(self, outputs: Any, *, run_id: UUID,
                     parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        # --- Cierre de un span de nodo ---
        if run_id in self._nodo_por_run:
            nombre = self._nodo_por_run.pop(run_id)
            t0 = self._t_por_run.pop(run_id, None)
            self._nodos.append({
                "nodo": nombre,
                "orden": len(self._nodos) + 1,
                "latencia_s": round(time.perf_counter() - t0, 3) if t0 else None,
            })
            self._nodo_actual = None
            return

        # --- Cierre de la traza completa ---
        if run_id == self._root_run_id:
            respuesta = ""
            if isinstance(outputs, dict):
                respuesta = str(outputs.get("respuesta_final", ""))
            self._finalizar(exito=True, respuesta=respuesta)

    def on_chain_error(self, error: BaseException, *, run_id: UUID,
                       parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        # Error dentro de un nodo: registramos donde ocurrio y dejamos que el
        # error burbujee hasta la raiz (alli se cierra la traza).
        if run_id in self._nodo_por_run:
            nombre = self._nodo_por_run.pop(run_id)
            self._t_por_run.pop(run_id, None)
            self._registrar_error(error, donde=f"nodo:{nombre}")
            self._nodo_actual = None
            return

        if run_id == self._root_run_id:
            # GraphRecursionError (limite de 25 pasos) y cualquier excepcion
            # no controlada terminan aqui: la traza se escribe IGUAL, porque
            # los fallos son precisamente lo que IE1/IE3 piden medir.
            self._registrar_error(error, donde=self._ultimo_error["donde"]
                                  if self._ultimo_error else "grafo")
            self._finalizar(exito=False, respuesta=None)

    # ---------------------------------------------------------- eventos de LLM
    def on_chat_model_start(self, serialized: Optional[dict], messages: Any, *,
                            run_id: UUID, parent_run_id: Optional[UUID] = None,
                            **kwargs: Any) -> None:
        # Guardamos el texto del prompt SOLO para poder estimar tokens si el
        # proveedor no informara el uso real (respaldo declarado).
        texto_prompt = ""
        try:
            texto_prompt = "\n".join(
                str(getattr(m, "content", "")) for lote in messages for m in lote
            )
        except TypeError:
            pass
        self._llm_por_run[run_id] = {
            "t0": time.perf_counter(),
            "nodo": self._nodo_actual,
            "texto_prompt": texto_prompt,
        }

    def on_llm_end(self, response: Any, *, run_id: UUID,
                   parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        parcial = self._llm_por_run.pop(run_id, None)
        if parcial is None:
            return
        tokens_in, tokens_out, fuente = _extraer_tokens_llm(response)

        # Respaldo declarado: estimacion con tiktoken solo si falta el dato real.
        if fuente == "no_disponible":
            est_in = _estimar_tokens(parcial["texto_prompt"])
            texto_salida = ""
            try:
                texto_salida = response.generations[0][0].text or ""
            except (IndexError, AttributeError):
                pass
            est_out = _estimar_tokens(texto_salida)
            if est_in is not None or est_out is not None:
                tokens_in, tokens_out, fuente = est_in, est_out, "estimado(tiktoken)"

        self._llamadas_llm.append({
            "orden": len(self._llamadas_llm) + 1,
            "nodo": parcial["nodo"],
            "latencia_s": round(time.perf_counter() - parcial["t0"], 3),
            "tokens_prompt": tokens_in,
            "tokens_completion": tokens_out,
            "fuente_tokens": fuente,
        })

    def on_llm_error(self, error: BaseException, *, run_id: UUID,
                     parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        parcial = self._llm_por_run.pop(run_id, None)
        nodo = parcial["nodo"] if parcial else self._nodo_actual
        self._registrar_error(error, donde=f"llm@{nodo or '?'}")

    # -------------------------------------------------- eventos de herramienta
    def on_tool_start(self, serialized: Optional[dict], input_str: str, *,
                      run_id: UUID, parent_run_id: Optional[UUID] = None,
                      **kwargs: Any) -> None:
        nombre = (serialized or {}).get("name") or kwargs.get("name") or "?"
        self._tool_por_run[run_id] = {
            "nombre": nombre,
            "nodo": self._nodo_actual,
            "t0": time.perf_counter(),
        }

    def on_tool_end(self, output: Any, *, run_id: UUID,
                    parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        parcial = self._tool_por_run.pop(run_id, None)
        if parcial is None:
            return
        self._herramientas.append({
            "nombre": parcial["nombre"],
            "nodo": parcial["nodo"],
            "orden": len(self._herramientas) + 1,
            "latencia_s": round(time.perf_counter() - parcial["t0"], 3),
            "ok": True,
        })

    def on_tool_error(self, error: BaseException, *, run_id: UUID,
                      parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        parcial = self._tool_por_run.pop(run_id, None)
        nombre = parcial["nombre"] if parcial else "?"
        if parcial:
            self._herramientas.append({
                "nombre": nombre,
                "nodo": parcial["nodo"],
                "orden": len(self._herramientas) + 1,
                "latencia_s": round(time.perf_counter() - parcial["t0"], 3),
                "ok": False,
            })
        self._registrar_error(error, donde=f"tool:{nombre}")

    # ---------------------------------------------------- eventos de retriever
    def on_retriever_start(self, serialized: Optional[dict], query: str, *,
                           run_id: UUID, parent_run_id: Optional[UUID] = None,
                           **kwargs: Any) -> None:
        self._num_recuperaciones += 1

    def on_retriever_end(self, documents: Any, *, run_id: UUID,
                         parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        try:
            self._docs_recuperados += len(documents)
        except TypeError:
            pass

    # ------------------------------------------------------------- cierre/util
    def _registrar_error(self, error: BaseException, donde: str) -> None:
        """Conserva el PRIMER error observado (la causa raiz); los siguientes
        suelen ser el mismo error burbujeando por la jerarquia de runs."""
        if self._ultimo_error is None:
            self._ultimo_error = {
                "tipo": type(error).__name__,
                "mensaje": str(error)[:500],
                "donde": donde,
            }

    def _finalizar(self, exito: bool, respuesta: Optional[str]) -> None:
        """Arma el registro completo de la consulta y lo persiste en el JSONL."""
        latencia_total = round(time.perf_counter() - self._t0_traza, 3)

        # Agregado de latencia por nodo (un nodo puede ejecutarse varias veces
        # en el ciclo ReAct: aqui se SUMAN sus ejecuciones).
        latencia_por_nodo: dict[str, float] = {}
        for span in self._nodos:
            if span["latencia_s"] is not None:
                latencia_por_nodo[span["nodo"]] = round(
                    latencia_por_nodo.get(span["nodo"], 0.0) + span["latencia_s"], 3
                )

        # Totales de tokens: si CUALQUIER llamada quedo sin dato real, el total
        # se declara con la fuente mas debil (transparencia metrica).
        tokens_prompt = sum(c["tokens_prompt"] or 0 for c in self._llamadas_llm)
        tokens_completion = sum(c["tokens_completion"] or 0 for c in self._llamadas_llm)
        fuentes = {c["fuente_tokens"] for c in self._llamadas_llm}
        if not fuentes:
            fuente_tokens = "sin_llamadas"
        elif fuentes == {"proveedor"}:
            fuente_tokens = "proveedor"
        elif "proveedor" in fuentes:
            fuente_tokens = "mixto"
        else:
            fuente_tokens = next(iter(fuentes))

        registro = {
            "version_esquema": VERSION_ESQUEMA,
            "trace_id": str(self._root_run_id),
            "timestamp": getattr(self, "_timestamp", _ahora_iso()),
            "origen": self._origen or self.origen_por_defecto,
            "caso_id": self._caso_id,
            "variante": self._variante,
            "modelo": agente.LLM_MODEL,
            "pregunta": self._pregunta,
            "exito": exito,
            "error": self._ultimo_error,
            "latencia_total_s": latencia_total,
            "nodos": self._nodos,
            "latencia_por_nodo_s": latencia_por_nodo,
            "ruta": " > ".join(s["nodo"] for s in self._nodos),
            "num_pasos_react": len(self._nodos),
            "num_ciclos_razonar": sum(1 for s in self._nodos if s["nodo"] == "razonar"),
            "herramientas": self._herramientas,
            "num_llamadas_llm": len(self._llamadas_llm),
            "llamadas_llm": self._llamadas_llm,
            "tokens_prompt": tokens_prompt,
            "tokens_completion": tokens_completion,
            "tokens_total": tokens_prompt + tokens_completion,
            "fuente_tokens": fuente_tokens,
            "num_recuperaciones": self._num_recuperaciones,
            "docs_recuperados": self._docs_recuperados,
            "memoria_rss_mb": round(_proceso.memory_info().rss / (1024 * 1024), 1),
            "delta_rss_mb": round(
                _proceso.memory_info().rss / (1024 * 1024)
                - getattr(self, "_rss0_mb", 0.0), 1),
            "cpu_proceso_pct": _proceso.cpu_percent(None),
            "respuesta": respuesta,
            "respuesta_sha256": _sha256(respuesta) if respuesta else None,
            "longitud_respuesta": len(respuesta) if respuesta else 0,
        }

        self._escribir(registro)
        self.ultimo_registro = registro
        # Los metadatos de consulta NO se arrastran a la siguiente traza.
        self._origen = self._caso_id = self._variante = None
        self._reiniciar_estado()

    def _escribir(self, registro: dict) -> None:
        """Persiste el registro como UNA linea JSON (JSON Lines).

        POR QUE JSONL y no un unico JSON: el archivo se escribe en modo append
        linea a linea, de modo que (1) una caida a mitad de bateria no corrompe
        lo ya registrado y (2) pandas/Streamlit lo leen directo con read_json(
        lines=True). ensure_ascii=False mantiene el español legible a ojo.
        """
        try:
            RUTA_TRAZAS.parent.mkdir(parents=True, exist_ok=True)
            with open(RUTA_TRAZAS, "a", encoding="utf-8") as f:
                f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        except OSError as e:
            # Nunca tumbar al agente por un problema de disco/permisos.
            print(f"[observabilidad] AVISO: no se pudo escribir la traza: {e}")


# =============================================================================
# 3. ACTIVACION ZERO-TOUCH Y API PUBLICA
# =============================================================================
# Instancia unica del trazador: todos los caminos de entrada (scripts, API web)
# comparten el mismo archivo de trazas y el mismo esquema.
TRAZADOR = TrazadorJSONL()

_activada = False


def activar_observabilidad() -> None:
    """Re-ata agente.app_grafo a una version con el trazador suscrito.

    with_config(callbacks=[...]) es la via OFICIAL de LangChain para adjuntar
    configuracion a un Runnable: devuelve un RunnableBinding que, al invocarse,
    FUSIONA esta config con la de cada llamada (el recursion_limit=25 que pasa
    ejecutar_agente() se conserva intacto). Como ejecutar_agente() busca
    `app_grafo` en el modulo agente en cada llamada, este re-atado es
    suficiente y no requiere tocar agente.py. Idempotente: llamadas repetidas
    no apilan trazadores duplicados.
    """
    global _activada
    if _activada:
        return
    agente.app_grafo = agente.app_grafo.with_config({"callbacks": [TRAZADOR]})
    _activada = True
    print(f"[observabilidad] Trazado JSONL activo -> {RUTA_TRAZAS}")


def ejecutar_agente_observado(pregunta: str, historial: list | None = None,
                              origen: str = "script",
                              caso_id: Optional[str] = None,
                              variante: Optional[str] = None) -> dict:
    """Ejecuta el agente REAL y devuelve el registro de observabilidad completo.

    Envuelve agente.ejecutar_agente() sin alterar su comportamiento: la
    respuesta del agente viene dentro del registro (clave 'respuesta').

    Args:
        pregunta:  pregunta en lenguaje natural (igual que ejecutar_agente).
        historial: historial opcional de la sesion (memoria de corto plazo).
        origen:    etiqueta de procedencia de la traza (evaluacion | carga |
                   verificacion | api | script), para poder filtrar despues.
        caso_id:   id del caso de evaluacion (solo corridas de evaluar.py).
        variante:  variante de fraseo del caso (variabilidad de datos, IE1).

    Returns:
        dict: el registro JSONL de esta consulta (ya persistido en disco).
              Si el agente fallo, exito=False y 'error' describe la causa.
    """
    activar_observabilidad()
    TRAZADOR.preparar(origen=origen, caso_id=caso_id, variante=variante)

    try:
        agente.ejecutar_agente(pregunta, historial=historial)
    except Exception as e:  # noqa: BLE001 — en bateria, un fallo es un DATO, no un abort
        # Si la excepcion nacio DENTRO del grafo, on_chain_error ya escribio la
        # traza. Este respaldo cubre el caso rarisimo de fallar ANTES de entrar
        # al grafo (p. ej. al cargar la memoria de largo plazo).
        if TRAZADOR.ultimo_registro is None:
            registro_minimo = {
                "version_esquema": VERSION_ESQUEMA,
                "trace_id": None,
                "timestamp": _ahora_iso(),
                "origen": origen,
                "caso_id": caso_id,
                "variante": variante,
                "modelo": agente.LLM_MODEL,
                "pregunta": pregunta,
                "exito": False,
                "error": {"tipo": type(e).__name__,
                          "mensaje": str(e)[:500],
                          "donde": "fuera_del_grafo"},
                "latencia_total_s": None,
                "respuesta": None,
            }
            TRAZADOR._escribir(registro_minimo)
            TRAZADOR.ultimo_registro = registro_minimo

    return TRAZADOR.ultimo_registro
