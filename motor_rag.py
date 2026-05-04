"""
=============================================================================
MOTOR_RAG.PY - Fase 2: Motor de consultas RAG conversacional
=============================================================================
Co-piloto de Gobernanza Corporativa para directores de Cencosud S.A.

ARQUITECTURA DE LA CADENA RAG:

    Pregunta del director
            |
            v
    [1] Reformulador history-aware       <-- 1ra llamada al LLM
        Reformula la pregunta usando el historial para que sea
        autocontenida (ej: "y como cambio en 2024?" se vuelve
        "como cambio la rentabilidad operacional en 2024?").
            |
            v
    [2] Retriever sobre Atlas Vector Search
        Busca los k=6 chunks mas similares al embedding de la
        pregunta reformulada.
            |
            v
    [3] Prompt compuesto:
        - Sistema (rol director + reglas + CoT)
        - Few-Shot examples (calibracion de tono ejecutivo)
        - Historial conversacional resumido
        - Contexto recuperado (chunks con citas)
        - Pregunta original del usuario
            |
            v
    [4] LLM con streaming                 <-- 2da llamada al LLM
        Genera la respuesta token a token,
        primero "Analisis Previo", luego "Respuesta Final".
            |
            v
    [5] ConversationSummaryBufferMemory
        Persiste el turno y, si supera el limite de tokens,
        resume los turnos antiguos automaticamente.

DECISIONES DE DISENO CLAVE :

- LCEL (LangChain Expression Language) en lugar de Chains legacy: ofrece
  composicion declarativa, soporte nativo de streaming token-a-token,
  y trazabilidad automatica en LangSmith.

- ConversationSummaryBufferMemory: pese a estar deprecada en 0.3.x,
Sigue siendo funcional. Combina buffer literal de los
  turnos recientes con resumen LLM de los antiguos: equilibrio ideal entre
  fidelidad y economia de tokens en conversaciones largas sobre 1213 pags.

- Reformulacion antes de buscar: sin esto, una pregunta como "y en 2024?"
  se vectoriza tal cual, sin contexto, y recupera chunks irrelevantes.
  Una llamada extra al LLM mejora dramaticamente la calidad del retrieval.

Uso CLI:
    python motor_rag.py

Uso programatico (Fase 3 - FastAPI):
    from motor_rag import generar_respuesta_streaming
=============================================================================
"""

import os
import sys
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Nota sobre warnings de deprecacion:
# LangChain dispara un warning de
# deprecacion al instanciarla; lo silenciamos puntualmente con un context
# manager mas abajo (no a nivel modulo, para no ocultar otros warnings utiles).

# === Componentes nucleo de LangChain ===
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# === LLM y embeddings (OpenAI-compatible, redirigidos a GitHub Models) ===
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# === Vector store ===
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# === Memoria conversacional (clase legacy requerida por rubrica) ===
from langchain.memory import ConversationSummaryBufferMemory


# =============================================================================
# 1. CARGA Y VALIDACION DE CONFIGURACION
# =============================================================================
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_BASE_URL = os.getenv("GITHUB_BASE_URL", "https://models.inference.ai.azure.com")
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "copiloto_gobernanza")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documentos")
INDEX_NAME = os.getenv("INDEX_NAME", "vector_index")

# Modelo de chat. gpt-4o-mini esta en el catalogo de GitHub Models y ofrece:
#   
# este string a "gpt-4o" sin tocar el resto del codigo.
LLM_MODEL = "gpt-4o-mini"

if not GITHUB_TOKEN or not MONGO_URI:
    sys.exit(
        "[ERROR] Faltan credenciales. Verifica GITHUB_TOKEN y MONGO_URI en .env"
    )


# =============================================================================
# 2. EMBEDDINGS (mismo modelo y configuracion que ingesta.py)
# =============================================================================
# CRITICO: el modelo de embeddings DEBE ser identico al usado en ingesta.py.
# Si difiere, los vectores de la consulta y los almacenados estan en espacios
# vectoriales distintos y la similitud coseno no tiene sentido semantico.
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=GITHUB_TOKEN,
    base_url=GITHUB_BASE_URL,
    check_embedding_ctx_length=False,
    chunk_size=64,  # ver justificacion en ingesta.py
)


# =============================================================================
# 3. VECTOR STORE Y RETRIEVER (modo solo-lectura)
# =============================================================================
# Aqui NO usamos from_documents (eso re-ingestaria). Instanciamos el wrapper
# apuntando a la coleccion ya poblada por ingesta.py.
client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

vectorstore = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name=INDEX_NAME,
)

# k=6: numero de chunks recuperados por consulta. Justificacion:
#   - k pequeno (1-3): respuestas precisas pero faltan conexiones entre
#     secciones (financiero + estrategico + gobierno corporativo).
#   - k grande (10+): mas contexto pero riesgo de "lost in the middle"
#     (el LLM ignora informacion enterrada en posiciones intermedias).
#   - k=6: balance probado para documentos corporativos densos.
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6},
)


# =============================================================================
# 4. LLM CON STREAMING ACTIVADO
# =============================================================================
# streaming=True: habilita el modo token-a-token. La respuesta se entrega
# como un generador de chunks que el frontend  consumira via SSE
# en la Fase 3. Sin esto, el usuario veria una pantalla en blanco varios
# segundos y luego el texto completo de golpe.
#
# temperature=0.2: respuestas deterministicas y factuales. El proyecto exige
# "respuestas libres de alucinaciones", asi que NO queremos creatividad.
# Subir a 0.7+ aumenta variabilidad pero abre la puerta a inventar datos.
#
# max_tokens=1500: techo generoso para que el modelo tenga espacio de
# escribir el "Analisis Previo" + "Respuesta Final" sin truncamiento.
llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=GITHUB_TOKEN,
    base_url=GITHUB_BASE_URL,
    streaming=True,
    temperature=0.2,
    max_tokens=1500,
)


# =============================================================================
# 5. MEMORIA CONVERSACIONAL (ConversationSummaryBufferMemory)
# =============================================================================
# Estrategia hibrida buffer + resumen:
#   - Mantiene los ULTIMOS mensajes literales (alta fidelidad).
#   - Cuando el buffer supera max_token_limit, RESUME los mensajes antiguos
#     usando el LLM y los reemplaza por el resumen (compresion progresiva).
#
# Por que esta clase y no ConversationBufferMemory simple:
#   - Las consultas sobre Memorias Anuales (1213 paginas) generan
#     conversaciones multi-turno densas. Sin compresion, el contexto crece
#     hasta saturar la ventana del LLM y disparar costos.
#   - El resumen progresivo preserva la "intencion" del director sin
#     guardar cada palabra: alineado con un copiloto ejecutivo que sabe
#     "lo que importa" del intercambio.
#
# max_token_limit=600: equilibrio entre fidelidad y economia. Cuando el
# buffer crudo supera ~600 tokens, se gatilla el resumen automatico.
# memory_key debe coincidir con el MessagesPlaceholder del prompt principal.
# Envolvemos la instanciacion en catch_warnings para silenciar el warning
# de deprecacion sin afectar warnings utiles del resto del modulo.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    memoria = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=600,
        return_messages=True,       # devuelve list[BaseMessage], compatible con ChatPromptTemplate
        memory_key="historial",     # nombre con que el prompt accede al historial
        input_key="pregunta",       # que clave del input se considera la pregunta del usuario
    )


# =============================================================================
# 6. FEW-SHOT PROMPTING (calibracion del tono ejecutivo)
# =============================================================================
# Los ejemplos NO contienen datos reales (los reales vienen del retriever).
# Su unica funcion es ensenar al modelo:
#   - El FORMATO esperado: "Analisis Previo" + "Respuesta Final"
#   - El TONO: ejecutivo, conciso, con cifras y citas explicitas
#   - La estructura visual: markdown con bullets y enfasis
#
# Por que solo 2 ejemplos: cada ejemplo se concatena en CADA llamada,
# consumiendo tokens y latencia. 2 ejemplos bien escogidos calibran el tono
# sin inflar costos. Mas ejemplos no mejoran proporcionalmente la calidad.
ejemplos_few_shot = [
    {
        "pregunta": "¿Cuáles son los principales riesgos identificados por la compañía?",
        "respuesta": """**Análisis Previo:**
- Localizo en el contexto la sección de "Gestión Integral de Riesgos".
- Clasifico los riesgos por categoría (estratégico, financiero, operacional, regulatorio).
- Cruzo magnitud declarada con mitigantes asociados.

**Respuesta Final:**
Los riesgos prioritarios identificados por la compañía son:

1. **Volatilidad cambiaria** — exposición a divisas regionales.
   *Mitigante:* políticas de hedging declaradas.
2. **Riesgo regulatorio** — cambios en normativa de retail y financiero.
   *Mitigante:* equipo de compliance dedicado.
3. **Riesgo operacional** — disrupciones en cadena de suministro.

*Fuente: Memoria Anual, sección "Gestión Integral de Riesgos".*"""
    },
    {
        "pregunta": "Resúmeme la estrategia digital de la compañía.",
        "respuesta": """**Análisis Previo:**
- Reviso secciones "Transformación Digital" y "Estrategia Tecnológica".
- Identifico pilares declarados y métricas de avance asociadas.
- Verifico alineamiento con declaraciones del Directorio.

**Respuesta Final:**
La estrategia digital se sostiene en tres pilares declarados:

1. **Omnicanalidad** — integración de e-commerce con red física.
2. **Plataforma de datos** — unificación CRM y analítica avanzada.
3. **Eficiencia operacional** — automatización de procesos back-office.

*Fuente: Memoria Anual, capítulo "Estrategia Digital".*"""
    },
]

# Plantilla por ejemplo (lo que ve el modelo en cada par human/ai)
plantilla_ejemplo = ChatPromptTemplate.from_messages([
    ("human", "{pregunta}"),
    ("ai", "{respuesta}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=ejemplos_few_shot,
    example_prompt=plantilla_ejemplo,
)


# =============================================================================
# 7. PROMPT PRINCIPAL (Sistema + Few-Shot + Historial + Contexto + Pregunta)
# =============================================================================
# El prompt del sistema concentra:
#   - Identidad del agente (Co-piloto de Gobernanza)
#   - Restricciones de fundamentacion (anti-alucinacion)
#   - Instruccion explicita de Chain of Thought
#   - Convenciones de formato y citacion
INSTRUCCIONES_SISTEMA = """Eres un Co-piloto de Gobernanza Corporativa para los directores de Cencosud S.A.

Tu misión es ayudar a directores y comités a tomar decisiones informadas analizando \
documentos corporativos extensos (Memoria Anual, Código de Ética) y entregando \
respuestas trazables, libres de alucinaciones.

REGLAS ESTRICTAS (no negociables):

1. **Solo el contexto manda.** Responde única y exclusivamente con base en el \
CONTEXTO recuperado. Si la información no está presente o es ambigua, di literalmente: \
"No tengo información suficiente en los documentos consultados para responder con certeza."

2. **Cadena de Pensamiento (CoT) obligatoria.** Antes de la respuesta final, escribe \
una sección **Análisis Previo** donde:
   - Identifiques qué documentos y secciones del contexto son relevantes.
   - Cruces datos numéricos cuando aplique.
   - Justifiques tu razonamiento brevemente (3 a 5 viñetas).

3. **Tono ejecutivo.** La **Respuesta Final** debe ser concisa, en markdown, con \
bullets cuando ayuden, citando cifras y la fuente (ej: "*Memoria 2024, p. 78*").

4. **Trazabilidad estricta.** Toda afirmación factual debe poder mapearse a un \
fragmento del contexto. NO inventes páginas, cifras ni citas.

5. **Español formal.** El registro es ejecutivo, dirigido a un director de empresa."""

prompt_principal = ChatPromptTemplate.from_messages([
    ("system", INSTRUCCIONES_SISTEMA),
    few_shot_prompt,
    MessagesPlaceholder(variable_name="historial"),
    ("human", """CONTEXTO RELEVANTE EXTRAÍDO DE LOS DOCUMENTOS:
{contexto}

---

PREGUNTA DEL DIRECTOR:
{pregunta}

Recuerda: primero **Análisis Previo**, luego **Respuesta Final**."""),
])


# =============================================================================
# 8. REFORMULADOR HISTORY-AWARE
# =============================================================================
# Sin este paso, una pregunta como "y como cambio en 2024?" se vectoriza
# tal cual, sin contexto, y recupera chunks irrelevantes. El reformulador
# toma el historial y genera una pregunta autocontenida ANTES de la busqueda
# vectorial. Es 1 llamada extra al LLM, pero mejora dramaticamente la calidad
# del retrieval en conversaciones multi-turno.
prompt_reformular = ChatPromptTemplate.from_messages([
    ("system", """Dado un historial de conversación y la pregunta nueva del usuario, \
reformula la pregunta para que sea AUTOCONTENIDA y comprensible sin el historial.

Reglas:
- Si la pregunta ya es autocontenida, devuélvela TAL CUAL, sin cambios.
- NO la respondas, solo reformúlala.
- Mantén el lenguaje original (español).
- Devuelve solo la pregunta reformulada, sin explicaciones."""),
    MessagesPlaceholder("historial"),
    ("human", "{pregunta}"),
])

# Reformulador como sub-cadena LCEL: prompt -> LLM -> string
reformulador = prompt_reformular | llm | StrOutputParser()


# =============================================================================
# 9. HELPERS DE FORMATO Y ORQUESTACION
# =============================================================================
def formatear_contexto(documentos):
    """Convierte la lista de Documents recuperados en un string con citas
    explicitas (fuente y pagina). Esto le facilita al LLM citar correctamente
    en la Respuesta Final, cumpliendo con el requisito de trazabilidad."""
    if not documentos:
        return "(No se encontraron fragmentos relevantes en los documentos.)"

    bloques = []
    for i, doc in enumerate(documentos, 1):
        # Path -> nombre de archivo (cross-platform: Windows usa \, Linux usa /)
        ruta = doc.metadata.get("source", "?")
        archivo = Path(ruta).name if ruta != "?" else "?"
        # PyPDFLoader indexa paginas desde 0; sumamos 1 para el usuario
        pagina = doc.metadata.get("page", "?")
        if isinstance(pagina, int):
            pagina = pagina + 1
        bloques.append(
            f"[Fuente {i}] {archivo} (pág. {pagina}):\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(bloques)


def preparar_inputs(entrada):
    """Orquesta los pasos previos al prompt principal:
        1. Carga el historial actual desde la memoria.
        2. Reformula la pregunta si hay historial (history-aware).
        3. Recupera chunks relevantes del vector store.
        4. Formatea el contexto con citas.
    Devuelve el dict de inputs listo para alimentar el prompt principal.

    Args:
        entrada: dict con clave "pregunta" (str). Esta firma es la que
                 RunnableLambda espera de la cadena LCEL.
    """
    pregunta_usuario = entrada["pregunta"]
    historial = memoria.load_memory_variables({}).get("historial", [])

    # Si no hay historial, ahorramos una llamada al LLM (la primera pregunta
    # de la conversacion siempre es autocontenida por definicion).
    if historial:
        pregunta_busqueda = reformulador.invoke({
            "pregunta": pregunta_usuario,
            "historial": historial,
        }).strip()
    else:
        pregunta_busqueda = pregunta_usuario

    # Recuperacion vectorial sobre la pregunta (re)formulada
    documentos = retriever.invoke(pregunta_busqueda)
    contexto = formatear_contexto(documentos)

    # OJO: en el prompt usamos la pregunta ORIGINAL (no la reformulada),
    # porque el LLM ya tiene el historial y debe responderle al director
    # con sus propias palabras, no las de la reformulacion interna.
    return {
        "pregunta": pregunta_usuario,
        "contexto": contexto,
        "historial": historial,
    }


# =============================================================================
# 10. CADENA LCEL FINAL
# =============================================================================
# Composicion declarativa: cada `|` aplica una transformacion al input.
# El streaming se propaga automaticamente porque:
#   - llm tiene streaming=True
#   - StrOutputParser es streaming-compatible
#   - LCEL respeta el streaming end-to-end de los Runnables compatibles
cadena_rag = (
    RunnableLambda(preparar_inputs)
    | prompt_principal
    | llm
    | StrOutputParser()
)


# =============================================================================
# 11. API PUBLICA: generar_respuesta_streaming
# =============================================================================
def generar_respuesta_streaming(pregunta: str):
    """Genera la respuesta token por token y persiste el turno en memoria.

    Esta es la funcion que consumira FastAPI en la Fase 3 para reenviar
    los tokens al frontend Angular via Server-Sent Events.

    Args:
        pregunta: pregunta en lenguaje natural del director.

    Yields:
        str: trozos de texto (1 o varios tokens) a medida que el LLM los genera.
    """
    respuesta_acumulada = []

    # `.stream()` propaga el streaming desde el ultimo paso streamable de
    # la cadena. Aqui obtenemos el texto del LLM token a token (via parser).
    for chunk in cadena_rag.stream({"pregunta": pregunta}):
        respuesta_acumulada.append(chunk)
        yield chunk

    # Al cerrar el stream, persistimos el turno completo en la memoria.
    # Es importante hacerlo aqui (al final), porque la memoria debe ver la
    # respuesta tal como fue entregada al usuario.
    memoria.save_context(
        {"pregunta": pregunta},
        {"output": "".join(respuesta_acumulada)},
    )


# =============================================================================
# 12. MODO CLI (para pruebas y demos en consola)
# =============================================================================
# El bloque __main__ solo se ejecuta cuando llamas `python motor_rag.py`.
# Si en la Fase 3 importas este modulo desde FastAPI, este bloque NO corre.
if __name__ == "__main__":
    print("=" * 70)
    print("CO-PILOTO DE GOBERNANZA CORPORATIVA - Cencosud S.A.")
    print("Modo CLI. Escribe 'salir' o Ctrl+C para terminar.")
    print("=" * 70)

    while True:
        try:
            pregunta = input("\n[Director] ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[INFO] Sesión finalizada por el usuario.")
            break

        if not pregunta:
            continue
        if pregunta.lower() in ("salir", "exit", "quit"):
            print("[INFO] Sesión finalizada.")
            break

        print("\n[Co-piloto] ", end="", flush=True)
        try:
            for chunk in generar_respuesta_streaming(pregunta):
                # flush=True garantiza que cada chunk se vea en pantalla
                # inmediatamente, simulando el efecto que vera el usuario
                # final en el frontend (texto apareciendo progresivamente).
                print(chunk, end="", flush=True)
            print()  # salto de linea al terminar la respuesta
        except Exception as e:
            print(f"\n[ERROR] {type(e).__name__}: {e}")
