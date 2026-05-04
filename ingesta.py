"""
=============================================================================
INGESTA.PY - Fase 1 del Co-piloto de Gobernanza Corporativa
=============================================================================
Pipeline ETL de ingesta documental:
    PDFs (/data) -> Carga -> Chunking semantico -> Embeddings -> MongoDB Atlas

Este script se ejecuta UNA VEZ por cada lote de documentos nuevos. No forma
parte del runtime del API porque es costoso en tokens (genera embeddings de
todos los chunks) y bloqueante (procesa cientos de paginas). Separar la
ingesta del servicio de consulta es una decision arquitectonica clave:
permite escalar y desplegar ambos componentes de forma independiente.

Uso:
    python ingesta.py

Pre-requisitos:
    1. Archivo .env configurado con credenciales (ver .env.example).
    2. PDFs corporativos colocados en la carpeta /data.
    3. Indice 'vector_index' creado en Atlas Search (la definicion JSON
       se imprime al final de la ejecucion para facilitar el setup).
=============================================================================
"""

import os
import glob
import sys
from pathlib import Path
from dotenv import load_dotenv

# --- Componentes de LangChain ---
# PyPDFLoader: extrae el texto pagina por pagina conservando metadatos
# (source, page). Estos metadatos seran criticos para que el agente cite
# la pagina exacta de la Memoria Anual al responder (trazabilidad).
from langchain_community.document_loaders import PyPDFLoader

# RecursiveCharacterTextSplitter: divide el texto respetando jerarquia
# natural (parrafos -> lineas -> oraciones -> palabras). Asi evitamos
# romper ideas semanticas a mitad de oracion, problema clasico de los
# splitters por longitud fija.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# OpenAIEmbeddings: wrapper sobre la API de embeddings de OpenAI.
# Elegimos text-embedding-3-small por su excelente relacion costo/calidad
# y porque sus 1536 dimensiones son ampliamente soportadas por Atlas.
from langchain_openai import OpenAIEmbeddings

# MongoDBAtlasVectorSearch: conector oficial que sabe escribir en el
# formato exacto que el indice $vectorSearch de Atlas espera.
from langchain_mongodb import MongoDBAtlasVectorSearch

# Cliente nativo de MongoDB para operaciones administrativas
# (limpiar coleccion antes de re-ingestar).
from pymongo import MongoClient


# =============================================================================
# 1. CARGA Y VALIDACION DE CREDENCIALES
# =============================================================================
# load_dotenv lee el archivo .env y expone sus variables a os.getenv.
# Validamos ANTES de cualquier procesamiento costoso para fallar rapido
# si falta alguna credencial (fail-fast, principio de buena ingenieria).
load_dotenv()

# Credenciales del proveedor de modelos. Usamos GitHub Models (compatible con
# OpenAI vía Azure) en lugar de la API directa de OpenAI
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_BASE_URL = os.getenv("GITHUB_BASE_URL", "https://models.inference.ai.azure.com")

# Credenciales de MongoDB Atlas y nombres lógicos del store vectorial.
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "copiloto_gobernanza")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documentos")
INDEX_NAME = os.getenv("INDEX_NAME", "vector_index")

if not GITHUB_TOKEN or not MONGO_URI:
    sys.exit(
        "[ERROR] Faltan credenciales. Verifica que GITHUB_TOKEN y MONGO_URI "
        "esten definidas en tu archivo .env (usa .env.example como plantilla)."
    )


# =============================================================================
# 2. LOCALIZACION DE PDFs EN /data
# =============================================================================
# Usamos glob para detectar dinamicamente cualquier PDF presente en /data.
# Esto evita hardcodear nombres y permite agregar futuros documentos
# (Memoria 2026, nuevos codigos, etc.) sin tocar el codigo.
DATA_DIR = Path(__file__).parent / "data"
pdf_paths = sorted(glob.glob(str(DATA_DIR / "*.pdf")))

if not pdf_paths:
    sys.exit(
        f"[ERROR] No se encontraron PDFs en {DATA_DIR}.\n"
        "Coloca los documentos corporativos (memoria_2024.pdf, codigo_etica.pdf, "
        "etc.) en la carpeta /data antes de ejecutar este script."
    )

print(f"[INFO] PDFs detectados ({len(pdf_paths)}):")
for p in pdf_paths:
    print(f"       - {Path(p).name}")


# =============================================================================
# 3. CARGA DE DOCUMENTOS
# =============================================================================
# PyPDFLoader devuelve una lista de Documents (uno por pagina). Cada Document
# trae:
#   - page_content: texto extraido de la pagina
#   - metadata: {"source": ruta_pdf, "page": numero_pagina}
# Acumulamos todo en una sola lista para procesarlos en bloque.
todos_los_documentos = []
for ruta in pdf_paths:
    nombre = Path(ruta).name
    print(f"\n[INFO] Cargando: {nombre}")
    loader = PyPDFLoader(ruta)
    paginas = loader.load()
    todos_los_documentos.extend(paginas)
    print(f"       -> {len(paginas)} paginas extraidas")

print(f"\n[INFO] Total de paginas cargadas: {len(todos_los_documentos)}")


# =============================================================================
# 4. CHUNKING SEMANTICO
# =============================================================================
# RecursiveCharacterTextSplitter intenta cortar primero por separadores
# "naturales" (parrafos, luego lineas, luego oraciones) antes de cortar por
# espacios o caracteres. Esto preserva ideas completas dentro de cada chunk,
# critico para que el LLM entienda el contexto al responder.
#
# Parametros elegidos para Memorias Anuales corporativas:
#   - chunk_size=1000: suficiente para contener un parrafo ejecutivo o una
#     fila completa de tabla financiera, pero pequeno para que el retrieval
#     sea preciso (no traer contexto irrelevante).
#   - chunk_overlap=150 (~15%): solapamiento que evita perder informacion
#     que cruza la frontera entre dos chunks adyacentes (por ejemplo, una
#     conclusion que comienza al final de un chunk y termina en el siguiente).
#   - separators: orden de prioridad para los cortes. Primero parrafos
#     dobles, luego salto simple, luego punto-espacio (oracion), luego
#     espacio (palabra). El string vacio "" es el ultimo recurso (caracter).
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
    # length_function por defecto cuenta caracteres. Para mayor precision se
    # podria usar tiktoken (conteo en tokens) pero para esta escala de
    # documentos el conteo por caracteres es suficiente y mas rapido.
)

chunks = splitter.split_documents(todos_los_documentos)
print(f"[INFO] Total de chunks generados: {len(chunks)}")


# =============================================================================
# 5. CONFIGURACION DE EMBEDDINGS
# =============================================================================
# text-embedding-3-small:
#   - 1536 dimensiones (estandar OpenAI, soportado por Atlas)
#   - disponible en el catalogo de GitHub Models sin cambios de codigo
#   - rendimiento competitivo en espanol y dominio corporativo
#
# IMPORTANTE: aunque la clase se llama OpenAIEmbeddings, sobrescribimos
# api_key y base_url para redirigir las llamadas al endpoint de GitHub
# Models (Azure). Como la API es compatible con OpenAI, LangChain no
# necesita cambios adicionales: la misma clase sirve para ambos proveedores.
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=GITHUB_TOKEN,        # PAT de GitHub usado como bearer token
    base_url=GITHUB_BASE_URL,    # https://models.inference.ai.azure.com
    # check_embedding_ctx_length=False evita que LangChain intente recortar
    # con tiktoken usando el tokenizer de OpenAI; algunos proxies compatibles
    # rechazan los chunks si se reenvian ya tokenizados. Mantenemos el corte
    # por caracteres del splitter (1000 chars), muy por debajo del limite
    # del modelo (8191 tokens), asi que es seguro desactivarlo.
    check_embedding_ctx_length=False,
    # chunk_size aqui NO es el tamano del texto, sino el numero de textos
    # que LangChain agrupa por cada llamada HTTP a la API de embeddings.
    # GitHub Models impone un limite de 64.000 tokens por request, mucho
    # mas estricto que OpenAI directo. Con chunks de ~250 tokens cada uno,
    # 64 textos/batch * 250 tokens = ~16.000 tokens (margen 4x sobre el limite).
    # Si subimos a 100, algunos chunks densos podrian empujar al limite.
    chunk_size=64,
)


# =============================================================================
# 6. CONEXION A MONGODB Y CARGA VECTORIZADA
# =============================================================================
# Conectamos al cluster, seleccionamos la coleccion, y delegamos a
# MongoDBAtlasVectorSearch.from_documents() la doble tarea de:
#   (a) calcular el embedding de cada chunk (llamadas a OpenAI)
#   (b) insertarlos en MongoDB con la estructura que espera el indice
#       vectorial: { text, embedding, source, page, ... }
print("\n[INFO] Conectando a MongoDB Atlas...")
client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

# Limpiamos la coleccion antes de re-ingestar para evitar duplicados.

print(f"[INFO] Limpiando coleccion '{COLLECTION_NAME}' (re-ingesta limpia)...")
resultado = collection.delete_many({})
print(f"       -> {resultado.deleted_count} documentos previos eliminados")

print(f"[INFO] Subiendo {len(chunks)} chunks vectorizados a MongoDB Atlas...")
print(f"       (procesando en lotes de 64 chunks via GitHub Models;")
print(f"        seran ~{(len(chunks) + 63) // 64} llamadas HTTP, puede tardar varios minutos)")

vectorstore = MongoDBAtlasVectorSearch.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection=collection,
    index_name=INDEX_NAME,
)


# =============================================================================
# 7. RESUMEN FINAL Y RECORDATORIO DEL INDICE
# =============================================================================
print("\n" + "=" * 70)
print("[OK] Ingesta completada con exito")
print("=" * 70)
print(f"  Cluster URI    : {MONGO_URI.split('@')[-1].split('/')[0]}")
print(f"  Base de datos  : {DB_NAME}")
print(f"  Coleccion      : {COLLECTION_NAME}")
print(f"  Indice         : {INDEX_NAME}")
print(f"  Chunks subidos : {len(chunks)}")
print("=" * 70)

