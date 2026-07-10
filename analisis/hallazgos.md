# Analisis de trazas del agente — EP3

Generado: 2026-07-09T16:23:04-04:00 · Fuente: `agente_trazas.jsonl` (46 trazas)

## 1. Resumen del dataset

- Trazas analizadas: **46** (desde 2026-07-09T16:03:32-04:00 hasta 2026-07-09T16:22:26-04:00)
- Por origen: {'evaluacion': 28, 'carga': 16, 'verificacion': 2}
- Tasa de exito de ejecucion: **100.0%** (46 de 46)
- Modelo: ['gpt-4o-mini']

## 2. Latencia y cuellos de botella (IE3)

- Latencia total por consulta: p50 = **7.37 s**, p95 = **19.27 s**, max = **63.79 s**
- Tiempo de espera del LLM: **90.7%** del tiempo total (383.4 s acumulados en 110 llamadas)
- Tiempo en herramientas: 13.9% del total (nota: generar_reporte llama al LLM por dentro, hay solape declarado)

| Nodo | Ejecuciones | Latencia acumulada (s) | % del tiempo |
|---|---|---|---|
| razonar | 107 | 363.8 | 86.1% |
| buscar | 48 | 38.9 | 9.2% |
| reportar | 3 | 19.8 | 4.7% |
| responder | 46 | 0.0 | 0.0% |
| calcular | 10 | 0.0 | 0.0% |

## 3. Uso de recursos (IE2)

- Tokens totales: **268,044** (prompt: 252,778 | completion: 15,266) — fuente: {'proveedor': 46}
- Tokens por consulta: media = 5,827, p95 = 12,726, max = 16,250
- Llamadas LLM por consulta: media = 2.4, max = 6
- Pasos ReAct por consulta: media = 4.7, max = 12
- Memoria del proceso (RSS): media = 134 MB, max = 136 MB | CPU medio por consulta: 0.7%
- Costo de REFERENCIA (precio lista gpt-4o-mini): **US$ 0.0471** total, US$ 0.00102 por consulta (GitHub Models free tier no factura; cifra para dimensionar escalamiento)

## 4. Errores de ejecucion (IE3)

- Sin errores de ejecucion en el periodo analizado. Los fallos de CALIDAD (criterios no cumplidos) se miden en la evaluacion (seccion 6), no aqui.

## 5. Uso de herramientas

| Herramienta | Usos | Latencia media (s) | Latencia max (s) | Fallos |
|---|---|---|---|---|
| buscar_documentos | 60 | 0.65 | 1.53 | 0 |
| calcular_metricas | 13 | 0.00 | 0.00 | 0 |
| generar_reporte | 3 | 6.61 | 7.91 | 0 |

## 6. Patrones y anomalias (IE4)

- Umbral de outlier de latencia (Q3 + 1.5*IQR): **16.4 s** -> 4 consulta(s) anomala(s)
  - `Calcula con tu calculadora la variación porcentual desde 65.2 hasta 78...` -> 63.8 s, ruta: razonar > calcular > razonar > responder
  - `Genera un reporte ejecutivo breve sobre los riesgos identificados en l...` -> 17.5 s, ruta: razonar > buscar > razonar > reportar > razonar > responder
  - `Prepara un informe ejecutivo corto para el directorio sobre los riesgo...` -> 19.9 s, ruta: razonar > buscar > razonar > reportar > razonar > responder
  - `Necesito un reporte ejecutivo resumido de los riesgos corporativos seg...` -> 20.2 s, ruta: razonar > buscar > razonar > reportar > razonar > responder
- Consultas con ruta larga (> 6 pasos): 3
  - `riesgos memoria anual 2024 cencosud cuales son...` -> 10 pasos: razonar > buscar > razonar > buscar > razonar > buscar > razonar > buscar > razonar > responder
  - `me dices los riezgos de la memoria del 2024porfavor...` -> 12 pasos: razonar > buscar > razonar > buscar > razonar > buscar > razonar > buscar > razonar > buscar > razonar > responder
  - `¿Qué información entrega la Memoria Anual 2025 sobre los resultados fi...` -> 8 pasos: razonar > buscar > razonar > buscar > razonar > buscar > razonar > responder

- Rutas del grafo observadas (patron de decision del agente):
  -  18x  `razonar > buscar > razonar > responder`
  -   7x  `razonar > responder`
  -   5x  `razonar > buscar > razonar > calcular > razonar > responder`
  -   5x  `razonar > calcular > razonar > responder`
  -   5x  `razonar > buscar > razonar > buscar > razonar > responder`
  -   3x  `razonar > buscar > razonar > reportar > razonar > responder`
  -   1x  `razonar > buscar > razonar > buscar > razonar > buscar > razonar > buscar > razonar > responder`
  -   1x  `razonar > buscar > razonar > buscar > razonar > buscar > razonar > buscar > razonar > buscar > razonar > responder`
  -   1x  `razonar > buscar > razonar > buscar > razonar > buscar > razonar > responder`

- Outliers de tokens (> 15,195): 1 consulta(s), max 16,250

## 7. Calidad (cruce con la evaluacion IE1)

- Precision global: **100.0%** sobre 27 corridas
- Por categoria: {'calculo': 1.0, 'factual': 1.0, 'anti_alucinacion': 1.0, 'reporte': 1.0, 'robustez': 1.0}
- Consistencia textual media: 0.464
- Alucinaciones en casos trampa: 0.0
- Errores por tipo: {}

## 8. HALLAZGOS (sintesis para el informe)

- **H1.** CUELLO DE BOTELLA: el nodo 'razonar' concentra el 86.1% de la latencia acumulada (363.8 s). La espera del LLM explica el 90.7% del tiempo total: la optimizacion debe apuntar a las llamadas al modelo (menos pasos, prompts mas cortos, cache), no al retrieval.
- **H2.** COLA PESADA de latencia: p95 (19.3 s) es 2.6x el p50 (7.4 s). Una fraccion de consultas degrada mucho la experiencia; ver outliers (seccion anomalias).
- **H3.** PATRON DE COSTO: los tokens de PROMPT superan 17x a los de completion. El costo esta dominado por el contexto que se reinyecta en cada paso ReAct (chunks k=6 + historial): reducir k, comprimir contexto o cachear busquedas repetidas tendria mas impacto que acortar respuestas.
- **H4.** PROCESO I/O-BOUND: CPU medio de 0.7% por consulta. El agente pasa el tiempo ESPERANDO a la API del LLM, no computando: escalar verticalmente (mas CPU) no mejoraria la latencia; si lo harian lotes concurrentes o un modelo/endpoint mas rapido.
- **H5.** HERRAMIENTA MAS COSTOSA: 'generar_reporte' con latencia media de 6.61 s por uso (3 usos).
- **H6.** ANOMALIA DE LATENCIA: 4 consulta(s) superan el umbral IQR de 16.4 s; todas corresponden a rutas con multiples busquedas o generacion de reportes (ver detalle en hallazgos.md), un patron que anticipa la degradacion al escalar.
- **H7.** PATRON DE RUTAS: se observan 9 rutas distintas del grafo; la dominante es `razonar > buscar > razonar > responder` (18 veces, 39% de las consultas exitosas).
- **H8.** ANOMALIA DE CONSUMO: 1 consulta(s) superan el umbral IQR de 15,195 tokens (max: 16,250). Corresponden a flujos multi-busqueda/reporte: candidatas a limitar k o a resumir contexto intermedio.
