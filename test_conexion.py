"""
TEST_CONEXION.PY - Diagnostico rapido de credenciales MongoDB
==============================================================
Script auxiliar para validar SOLAMENTE la conexion a MongoDB Atlas,
aislando este paso del resto del pipeline RAG. Si falla aqui, el
problema es de credenciales/red, no de LangChain ni de OpenAI.

Uso: python test_conexion.py
"""

import os
import sys
from urllib.parse import urlparse, quote_plus
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import OperationFailure, ServerSelectionTimeoutError

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "copiloto_gobernanza")

if not MONGO_URI:
    sys.exit("[ERROR] MONGO_URI no esta definida en .env")

# Parseamos la URI para mostrar diagnostico SIN exponer la password
# (la ocultamos con asteriscos en los logs).
parsed = urlparse(MONGO_URI)
host = parsed.hostname or "(no parseable)"
user = parsed.username or "(sin usuario)"
password_len = len(parsed.password) if parsed.password else 0

print("=" * 60)
print("DIAGNOSTICO DE CONEXION A MONGODB ATLAS")
print("=" * 60)
print(f"  Host         : {host}")
print(f"  Usuario      : {user}")
print(f"  Password len : {password_len} caracteres (oculta)")
print(f"  Base de datos: {DB_NAME}")
print("=" * 60)

# Intentamos un ping simple. Si las credenciales estan bien,
# Atlas responde { 'ok': 1.0 } en menos de un segundo.
try:
    print("\n[INFO] Conectando con timeout de 10s...")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)
    resultado = client.admin.command("ping")
    print(f"[OK] Ping exitoso: {resultado}")

    # Listamos bases de datos visibles para verificar permisos
    bases = client.list_database_names()
    print(f"[OK] Bases visibles: {bases}")

    # Probamos escritura en una coleccion temporal
    db = client[DB_NAME]
    test_col = db["_test_conexion"]
    insertado = test_col.insert_one({"test": "ok"})
    print(f"[OK] Escritura exitosa, _id: {insertado.inserted_id}")
    test_col.delete_many({})
    print("[OK] Limpieza exitosa")

    print("\n" + "=" * 60)
    print("[EXITO] Las credenciales y permisos son correctos.")
    print("        Ahora puedes ejecutar 'python ingesta.py'")
    print("=" * 60)

except OperationFailure as e:
    print(f"\n[FALLO DE AUTH] {e}")
    print("\nPosibles causas:")
    print("  1. Usuario o password incorrectos en .env")
    print("  2. Password tiene caracteres especiales sin URL-encode")
    print("     (@, :, /, ?, #, %, & deben codificarse)")
    print("  3. El usuario no tiene rol 'readWriteAnyDatabase' o similar")
    print("\n  -> Revisa: Atlas > Database Access")

except ServerSelectionTimeoutError as e:
    print(f"\n[TIMEOUT] No se pudo alcanzar el cluster: {e}")
    print("\nPosibles causas:")
    print("  1. Tu IP no esta en la lista de IPs permitidas")
    print("  2. El cluster esta apagado/pausado en Atlas")
    print("  3. La URL del cluster esta mal escrita")
    print("\n  -> Revisa: Atlas > Network Access > IP Access List")
    print("            (durante desarrollo puedes usar 0.0.0.0/0)")

except Exception as e:
    print(f"\n[ERROR INESPERADO] {type(e).__name__}: {e}")
