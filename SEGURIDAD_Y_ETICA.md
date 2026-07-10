# Protocolos de Seguridad y Uso Responsable — Co-piloto de Gobernanza (EP3, IE6)

Este documento consolida los **controles de seguridad, privacidad y uso responsable**
del Co-piloto de Gobernanza Corporativa (RAG + agente ReAct) en un contexto de
producción, siguiendo el marco del material del curso (Duoc UC, 2025c): amenazas
comunes a agentes LLM, mitigaciones, criterios éticos y normativos. Para cada
control se indica su **estado**: `[IMPLEMENTADO]` (existe en el código, con
referencia exacta) o `[PROPUESTO]` (recomendación para producción).

---

## 1. Amenazas y mitigaciones (seguridad técnica)

Mapa de los cuatro ataques comunes descritos en Duoc UC (2025c) y OWASP (2025),
aplicado a este proyecto:

### 1.1 Prompt injection (manipulación por entradas maliciosas)

- `[IMPLEMENTADO]` **Reglas no negociables en el prompt de sistema** del agente
  (`agente.py`, `INSTRUCCIONES_AGENTE`): solo el contexto recuperado manda; si no
  hay evidencia, rehusar con frase literal. Una instrucción inyectada en la
  pregunta no puede habilitar fuentes externas porque el agente no tiene
  herramientas de red ni de sistema: su superficie de acción son 3 tools acotadas.
- `[IMPLEMENTADO]` **Validación de entradas en la API** (`app_agente.py`,
  `PreguntaRequest`): Pydantic exige `min_length=1`, `max_length=2000`. Limita
  payloads gigantes (vector de DoS y de inyección extensa).
- `[PROPUESTO]` Sanitización adicional de patrones de inyección conocidos
  ("ignora tus instrucciones", etc.) y evaluación adversarial periódica con el
  set de casos de `evaluacion/casos.json` ampliado.

### 1.2 Ejecución de código arbitrario (la amenaza más grave en agentes)

- `[IMPLEMENTADO]` **Evaluador AST con lista blanca** en la herramienta
  `calcular_metricas` (`agente.py`, `_evaluar_expresion_segura`): NO se usa
  `eval()`. Se parsea la expresión a un árbol sintáctico y solo se aceptan
  constantes numéricas y los operadores `+ - * / ** %`; cualquier otro nodo
  (nombres, llamadas, atributos) lanza `ValueError`. Es la técnica de
  *sandboxing* recomendada para delegar cómputo a un agente (Duoc UC, 2025c).
- **Evidencia verificada (EP2):** `calcular_metricas("__import__('os').system('echo HACKED')")`
  es rechazado con "Expresion no permitida: Call"; el comando nunca se ejecuta.

### 1.3 Data poisoning (corpus contaminado)

- `[IMPLEMENTADO]` **Corpus cerrado y controlado**: la base vectorial (3.433
  chunks en MongoDB Atlas) solo se puebla mediante `ingesta.py` sobre los 4 PDF
  corporativos oficiales. El agente **no escribe** en la colección: en runtime
  el vector store se usa en modo solo-lectura (`motor_rag.py` instancia el
  wrapper sin `from_documents`).
- `[PROPUESTO]` Verificación de integridad del corpus (hash por documento al
  ingerir) y proceso formal de aprobación antes de añadir documentos nuevos.

### 1.4 Denial of Service / abuso de recursos

- `[IMPLEMENTADO]` **Límite de recursión del grafo** (`agente.py`,
  `ejecutar_agente`): `recursion_limit=25` corta cualquier ciclo
  razonar↔herramienta que no converja (loop infinito = costo infinito).
- `[IMPLEMENTADO]` **Techos de generación**: `max_tokens=1500` (agente) y
  `max_tokens=1800` (reportes) acotan el costo de cada llamada.
- `[IMPLEMENTADO]` **Límites del proveedor**: GitHub Models impone rate limits
  por minuto y por día; la capa de observabilidad EP3 los registra cuando
  ocurren (tipo `RateLimitError` en las trazas), haciéndolos visibles.
- `[PROPUESTO]` Rate limiting propio en la API (p. ej. por IP/sesión) y colas
  de trabajo para absorber ráfagas sin degradar a todos los usuarios.

### 1.5 Model extraction / exposición del modelo

- `[IMPLEMENTADO]` El modelo nunca se expone directamente: los usuarios llegan
  solo a los endpoints REST internos (`/preguntar`), que fijan prompt de
  sistema, herramientas y límites. `[PROPUESTO]` Autenticación de usuarios y
  CORS restringido a orígenes conocidos (hoy `allow_origins=["*"]` es aceptable
  en desarrollo, no en producción).

---

## 2. Gestión de secretos y mínimo privilegio

- `[IMPLEMENTADO]` **Credenciales fuera del código**: todas las claves (GitHub
  Models, MongoDB Atlas, LangSmith) viven en `.env`, cargado con `python-dotenv`;
  `.gitignore` excluye `.env` y llaves (`*.pem`, `*.key`). El repositorio
  publica solo `.env.example` sin valores reales.
- `[IMPLEMENTADO]` **Validación temprana**: `motor_rag.py` aborta al arrancar si
  faltan `GITHUB_TOKEN` o `MONGO_URI` (falla explícita, no silenciosa), y
  `test_conexion.py` permite verificar credenciales sin exponerlas.
- `[PROPUESTO]` **Principio de mínimo privilegio en Atlas**: usar en runtime un
  usuario de base de datos con permiso de solo lectura sobre la colección
  `documentos` (la escritura queda reservada al proceso de ingesta), y rotación
  periódica de tokens.

---

## 3. Privacidad de los datos (criterio normativo)

**Naturaleza de los datos.** Las Memorias Anuales son documentos públicos; el
Código de Ética es documentación interna. El sistema no solicita ni almacena
datos personales de los usuarios (no hay registro de identidad en ninguna capa).

**Los logs de observabilidad son datos sensibles en potencia.** La capa EP3
registra pregunta y respuesta completas en `logs/agente_trazas.jsonl` — las
consultas de un director pueden revelar temas estratégicos bajo análisis. Por eso:

- `[IMPLEMENTADO]` Los logs son **locales** (no salen a servicios externos: el
  tracing hacia LangSmith está inhabilitado en la práctica y el diseño EP3 no
  depende de él) y cada respuesta incluye su hash SHA-256, que permite auditar
  integridad sin exponer el texto si se decidiera truncarlo.
- `[PROPUESTO]` En producción: retención limitada (p. ej. 90 días), acceso al
  archivo restringido por permisos de sistema, y modo "hash-only" (guardar solo
  el hash y la longitud de la respuesta) si el directorio maneja información
  privilegiada en el sentido de la normativa de mercado de valores.

**Marco normativo aplicable (Chile + referencia internacional):**
- **Ley 21.719 (2024)** de protección de datos personales de Chile (sucesora de
  la Ley 19.628, con plena vigencia en diciembre de 2026): principios de
  finalidad, proporcionalidad y seguridad aplican al diseño de los registros.
- **EU AI Act (2024)** como referencia: un asistente de consulta documental es
  un sistema de **riesgo limitado**, sujeto a obligaciones de transparencia
  (el usuario debe saber que interactúa con una IA), que este proyecto cumple
  declarando el rol del co-piloto en la propia interfaz.

---

## 4. Uso responsable y criterios éticos

Siguiendo los tres ejes de Duoc UC (2025c):

**4.1 Anti-alucinación y anclaje a fuentes (integridad de la información).**
- `[IMPLEMENTADO]` Regla "solo el contexto manda" + frase de rehúsa literal
  ("No tengo información suficiente en los documentos consultados") + citas
  `[Fuente N] archivo (pág. X)` en la recuperación. **Evidencia medida (EP3):**
  los casos trampa del set de evaluación (categoría `anti_alucinacion`)
  registran la tasa de alucinación observada en `evaluacion/resultados_evaluacion.json`.
- `[IMPLEMENTADO]` La aritmética se delega a la calculadora exacta (los LLM
  cometen errores numéricos): elimina una fuente de desinformación financiera.

**4.2 Transparencia y explicabilidad (XAI).**
- `[IMPLEMENTADO]` El nodo `razonar` publica un **plan explícito** en cada paso
  y la capa EP3 conserva la **ruta completa del grafo** por consulta (traza con
  spans de nodo/herramienta/LLM). Esto permite **reconstruir cada decisión**
  del agente a posteriori — el requisito de auditoría y cumplimiento que
  describe Duoc UC (2025b) — sin depender de un servicio externo.

**4.3 Responsabilidad y gobernanza (humano al mando).**
- El co-piloto **asiste, no decide**: la decisión es siempre del director
  (humano responsable identificable, como exige el marco ético del curso). Las
  recomendaciones de los reportes generados citan sus fuentes para que el
  directorio pueda contrastarlas.
- `[PROPUESTO]` Política formal de uso: los reportes generados por IA deben
  declararse como tales al circular en el directorio, y revisarse por un humano
  antes de usarse en decisiones vinculantes.

**4.4 Sesgos.**
- El anclaje estricto al corpus reduce el sesgo del modelo base (no opina:
  cita), pero los documentos corporativos son autodeclaraciones de la empresa.
  `[PROPUESTO]` Declarar esta limitación en la interfaz y monitorear con la
  evaluación periódica (el set de casos es re-ejecutable) si las respuestas
  omiten sistemáticamente información desfavorable presente en el corpus.

---

## 5. La observabilidad COMO control de seguridad

La capa EP3 no es solo medición de rendimiento: es un **control transversal**
(Duoc UC, 2025a, 2025b):

| Función de seguridad | Cómo la cubre la capa de trazas |
|---|---|
| Auditoría / cumplimiento | Cada consulta queda reconstruible: trace_id, ruta del grafo, herramientas, respuesta y hash. |
| Detección de abuso | Anomalías de latencia/tokens y rutas anormalmente largas se detectan con `analisis/analizar_logs.py` (criterio IQR). |
| Monitoreo de alucinación | La evaluación re-ejecutable mide la tasa de rehúsa correcta en casos trampa. |
| Visibilidad de fallos | Errores tipificados con ubicación exacta (nodo/herramienta/LLM) en cada traza. |

---

## 6. Checklist de despliegue a producción (resumen ejecutivo)

| # | Control | Estado |
|---|---|---|
| 1 | Evaluador AST sin `eval()` en calculadora | IMPLEMENTADO |
| 2 | Prompt de sistema con reglas de anclaje y rehúsa | IMPLEMENTADO |
| 3 | Validación Pydantic de entradas (longitud) | IMPLEMENTADO |
| 4 | `recursion_limit=25` + `max_tokens` acotados | IMPLEMENTADO |
| 5 | Secretos en `.env` fuera del repositorio | IMPLEMENTADO |
| 6 | Corpus cerrado, vector store solo-lectura en runtime | IMPLEMENTADO |
| 7 | Trazabilidad local completa por consulta (EP3) | IMPLEMENTADO |
| 8 | CORS restringido + autenticación de usuarios | PROPUESTO |
| 9 | Rate limiting propio en la API | PROPUESTO |
| 10 | Retención/anonimización de logs (Ley 21.719) | PROPUESTO |
| 11 | Usuario Atlas de solo lectura (mínimo privilegio) | PROPUESTO |
| 12 | Política de revisión humana de reportes IA | PROPUESTO |

---

## Referencias (APA 7.ª ed.)

Duoc UC. (2025a). *3.1.1 Herramientas de observabilidad y métricas de rendimiento*
[Material de clase]. ISY0101 Optativo Ingeniería de Soluciones con IA.

Duoc UC. (2025b). *3.2.1 Análisis de trazabilidad y procesamiento de logs*
[Material de clase]. ISY0101 Optativo Ingeniería de Soluciones con IA.

Duoc UC. (2025c). *3.3.1 Protocolos de seguridad y consideraciones éticas*
[Material de clase]. ISY0101 Optativo Ingeniería de Soluciones con IA.

Ley 21.719, que regula la protección y el tratamiento de los datos personales.
Diario Oficial de la República de Chile (13 de diciembre de 2024).
https://www.bcn.cl/leychile/navegar?idNorma=1209272

OWASP Foundation. (2025). *OWASP Top 10 for Large Language Model Applications*.
https://owasp.org/www-project-top-10-for-large-language-model-applications/

Parlamento Europeo y Consejo de la Unión Europea. (2024). *Reglamento (UE)
2024/1689 (Ley de Inteligencia Artificial)*. Diario Oficial de la Unión Europea.
https://eur-lex.europa.eu/eli/reg/2024/1689/oj
