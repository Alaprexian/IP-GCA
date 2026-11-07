# Arquitectura del Sistema IP-GCA — Guía Técnica de Integración

> **IP-GCA: Implicit Preference-Gated Copilot Architecture**  
> Documento técnico que detalla la función de cada módulo y cómo se integran entre sí.

---

## Tabla de Contenidos

1. [Visión General del Flujo](#1-visión-general-del-flujo)
2. [Estructura de Carpetas](#2-estructura-de-carpetas)
3. [Módulos del Código Fuente (`src/`)](#3-módulos-del-código-fuente-src)
   - [3.1 Capa de Recuperación (`retrieval/`)](#31-capa-de-recuperación-retrieval)
   - [3.2 Controlador de Enrutamiento (`controller/`)](#32-controlador-de-enrutamiento-controller)
   - [3.3 Guardrail FSM + MCP (`guardrail/`)](#33-guardrail-fsm--mcp-guardrail)
   - [3.4 Adaptación por Preferencias (`adaptation/`)](#34-adaptación-por-preferencias-adaptation)
   - [3.5 Orquestador y Generador (`copilot/`)](#35-orquestador-y-generador-copilot)
   - [3.6 Evaluación y Benchmarks (`evaluation/` + `experiments/`)](#36-evaluación-y-benchmarks-evaluation--experiments)
   - [3.7 Interfaz Gráfica (`ui/`)](#37-interfaz-gráfica-ui)
   - [3.8 Scripts Raíz de `src/`](#38-scripts-raíz-de-src)
4. [Servidores MCP (`mcp_servers/`)](#4-servidores-mcp-mcp_servers)
5. [Configuración (`configs/`)](#5-configuración-configs)
6. [Datos (`data/`)](#6-datos-data)
7. [Modelos Locales (`models/`)](#7-modelos-locales-models)
8. [Resultados (`results/`)](#8-resultados-results)
9. [Suite de Tests (`tests/`)](#9-suite-de-tests-tests)
10. [Diagrama de Integración End-to-End](#10-diagrama-de-integración-end-to-end)
11. [Dependencias entre Módulos](#11-dependencias-entre-módulos)
12. [Archivos de Proyecto Raíz](#12-archivos-de-proyecto-raíz)

---

## 1. Visión General del Flujo

Cada turno de diálogo del copiloto sigue este pipeline de **5 etapas** orquestadas por `IPGCAOrchestrator`:

```
Cliente envía Query
       │
       ▼
[1] EVALUACIÓN DE INCERTIDUMBRE (controller/uncertainty.py)
    u_ctx = α(1 - M_ret) + β(H_intent)   ← combina dispersión de scores + entropía de intent
       │
       ▼
[2] ENRUTAMIENTO DINÁMICO (controller/router.py)
    GPL-Controller → elige uno de 4 caminos según u_ctx y presupuesto de latencia
    ├── FAST_PARAMETRIC      (u_ctx < 0.25)  → genera sin RAG
    ├── HYBRID_RAG           (0.25 ≤ u < 0.65) → retrieval/engine.py completo
    ├── AGENTIC_REFLECTION   (0.65 ≤ u < 0.85) → loop ReAct con herramientas MCP
    └── ASSISTED_ABSTENTION  (u ≥ 0.85)        → deriva a operador humano
       │
       ▼
[3] INYECCIÓN DE PREFERENCIAS (adaptation/conditioning_bridge.py)
    Recupera k_pref activas de PreferenceMemory con decaimiento γ^(Δt)
    Las inyecta como contexto adicional en el prompt del LLM
       │
       ▼
[4] GENERACIÓN (copilot/generator.py)
    LLMGenerator (Qwen2.5-1.5B 4-bit cuantizado) → produce S_gen
       │
       ▼
[5] SUPERVISIÓN HITL (copilot/orchestrator.py)
    Operador edita S_gen → S_human
    EditDeltaExtractor calcula Δ_edit
    PreferenceMemory almacena nueva k_pref
    [async] LoRATrigger acumula deltas → micro-ajuste LoRA
```

---

## 2. Estructura de Carpetas

```
Proyecto_tesis/
├── configs/                    # Hiperparámetros y reglas formales FSM (YAML)
├── data/
│   ├── raw/                    # Base de conocimiento corporativa (25 documentos JSON)
│   └── mcp_resources/          # Datos CRM simulados (clientes, tickets)
├── mcp_servers/                # Servidores MCP stdio: recursos y herramientas transaccionales
├── models/                     # Pesos de modelos descargados localmente
│   ├── bge-m3/                 # Modelo de embeddings densos (BAAI/BGE-M3)
│   ├── bge-reranker-large/     # Cross-Encoder de reranking (BAAI/bge-reranker-large)
│   ├── Qwen2.5-1.5B-Instruct/  # LLM generativo local (cuantizado 4-bit)
│   └── splade-cocondenser-ensembledistil/  # Modelo de retrieval sparse (SPLADE)
├── results/                    # Salidas de experimentos, figuras, tablas LaTeX, logs
│   ├── adapters/               # Pesos de adaptadores LoRA entrenados
│   ├── figures/                # Figuras PNG de publicación (matplotlib)
│   ├── logs/                   # Logs de benchmark y demo
│   └── lora_batches/           # Datasets de deltas acumulados para micro-tuning
├── src/                        # Todo el código fuente del sistema
│   ├── adaptation/             # Módulo de aprendizaje en línea por preferencias
│   ├── controller/             # GPL-Controller: enrutamiento y estimación de incertidumbre
│   ├── copilot/                # Orquestador end-to-end y generador LLM
│   ├── evaluation/             # Métricas, benchmark dataset, matrix de ablación
│   ├── experiments/            # Experimento de evaluación LoRA (Pre vs Post)
│   ├── guardrail/              # FSM determinista y cliente MCP
│   ├── retrieval/              # Pipeline híbrido de recuperación de documentos
│   └── ui/                    # Aplicación de escritorio PyWebView (Bridge API + frontend)
└── tests/                      # Suite de 59 pruebas unitarias e integradas
```

---

## 3. Módulos del Código Fuente (`src/`)

---

### 3.1 Capa de Recuperación (`retrieval/`)

Implementa el pipeline híbrido de recuperación de conocimiento corporativo (Fase RAG).

| Archivo | Clase / Función | Función |
|---|---|---|
| `base.py` | `DocumentChunk`, `ScoredChunk` | Dataclasses base que viajan por todo el pipeline de retrieval. `DocumentChunk` contiene `chunk_id`, `title`, `content` y `metadata`. `ScoredChunk` agrega `score`, `rank` y `retrieval_method`. |
| `document_loader.py` | `DocumentLoader`, `build_index_from_file()` | Carga el archivo `corporate_knowledge_base.json` y lo convierte en `DocumentChunk`s. `build_index_from_file()` es el shortcut que usa el `HybridRetrievalEngine` al arrancar. |
| `chunker.py` | `HierarchicalChunker` | Divide documentos largos en fragmentos de tamaño controlado con solapamiento configurable. Respeta límites de párrafo/sección para preservar coherencia semántica. |
| `sparse_retriever.py` | `SparseRetriever` | Retrieval BM25 sobre tokens. Especializado en coincidencias exactas de códigos de política (ej. `POL-30D`), números de ticket, SKUs. No requiere GPU. |
| `dense_retriever.py` | `DenseRetriever` | Embeddings semánticos con `SentenceTransformer` (BGE-M3). Captura relaciones conceptuales abstractas ("¿cuánto tiempo para devoluciones?" → política de reembolso). Similarity por producto punto sobre vectores normalizados. |
| `hybrid_fusion.py` | `reciprocal_rank_fusion()`, `weighted_relative_score_fusion()` | Fusiona resultados del retriever sparse y dense. RRF (Reciprocal Rank Fusion) combina rangos sin necesidad de calibrar escalas de score. RSF pondera scores normalizados por `dense_weight`/`sparse_weight` del config. |
| `reranker.py` | `CrossEncoderReranker` | Re-clasifica el top-K candidatos mediante un Cross-Encoder (BGE-Reranker-Large) que evalúa el par `(query, passage)` conjuntamente. Mucho más preciso que los encoders bi-direccionales pero más lento (sólo se usa si el presupuesto de latencia lo permite). |
| `engine.py` | `HybridRetrievalEngine` | **Punto de entrada único** del subsistema de retrieval. Instancia y coordina `SparseRetriever`, `DenseRetriever`, `CrossEncoderReranker`. Expone `index_chunks()`, `retrieve()` y `extract_scores()` (scores para el GPL-Controller). |

**Integración:**  
`HybridRetrievalEngine` es instanciado por `IPGCAOrchestrator` al arrancar, con `kb_path` apuntando a `data/raw/corporate_knowledge_base.json`. El `AblationMatrixRunner` también lo instancia con modelos más ligeros en `fast_dev_mode`.

---

### 3.2 Controlador de Enrutamiento (`controller/`)

Implementa el **GPL-Controller** (Gated Preference-Latency Controller), el componente diferenciador central de la arquitectura.

| Archivo | Clase / Función | Función |
|---|---|---|
| `uncertainty.py` | `compute_retrieval_margin()` | Calcula el margen de dispersión entre el score top-1 y el score mínimo: `M_ret = (s1 - sk) / (s1 + ε)`. Un margen alto significa que el retrieval es confiado; un margen bajo indica ambigüedad de contexto. |
| `uncertainty.py` | `compute_intent_entropy()` | Entropía de Shannon normalizada sobre el diccionario de probabilidades de intent: `H_intent = -Σ(p·log₂p) / log₂(|intents|)`. Valor 0 = intent puro; valor 1 = ambigüedad máxima. |
| `uncertainty.py` | `evaluate_joint_uncertainty()` | Combina ambas métricas: `u_ctx = α·(1 - M_ret) + β·H_intent`. Si se provee `session_id`, aplica suavizado EMA entre turnos para modelar la convergencia contextual multi-turno. |
| `uncertainty.py` | `SessionUncertaintyTracker` | Diccionario de EMA por sesión. Mantiene `u_ctx_session(t-1)` para suavizar la señal: `u_smooth = ema_alpha·u_current + (1-ema_alpha)·u_prev`. |
| `router.py` | `GPLController` | Lee umbrales (`tau_fast`, `tau_hybrid`, `tau_agent`) y presupuestos de latencia del `config_default.yaml`. El método `route_query()` selecciona el camino óptimo en < 5 ms. Incluye **degradación por latencia**: si el presupuesto no alcanza para el camino elegido, baja automáticamente a uno más rápido. |
| `router.py` | `ExecutionPath`, `RoutingDecision` | Enum de los 4 caminos + dataclass con `selected_path`, `uncertainty_score`, `remaining_latency_budget_ms` y `rationale` textual. |

**Integración:**  
`GPLController` recibe los `retrieval_scores` que extrae el `HybridRetrievalEngine` y el diccionario `intent_probs` que proviene de la UI o del test. La decisión de enrutamiento dicta qué hace `IPGCAOrchestrator` a continuación.

---

### 3.3 Guardrail FSM + MCP (`guardrail/`)

Implementa la barrera de seguridad determinista que aísla al LLM de los sistemas transaccionales.

| Archivo | Clase / Función | Función |
|---|---|---|
| `fsm.py` | `safe_eval_condition()` | Evalúa expresiones de guardia (ej. `"approval_token != 'UNVERIFIED_TOKEN'"`) parseando el AST sin ejecutar `eval()`. Soporta comparaciones, lógica booleana y aritmética básica. Bloquea cualquier expresión no permitida. |
| `fsm.py` | `TransactionalGuardrailFSM` | Implementa la máquina de estados `M_fsm = (Q, Σ, δ, q₀, F)`. Carga estados y transiciones desde `fsm_rules.yaml`. El método `attempt_transition()` valida la secuencia de estados (precedencia), evalúa las guardas AST con los parámetros de la llamada, y registra cada transición auditada. Las herramientas que intentan saltar estados intermedios son **bloqueadas con excepción** (`TransitionGuardViolation`). |
| `fsm.py` | `TransitionRecord` | Log inmutable de cada transición: estado fuente/destino, trigger, timestamp, resultado. Alimenta la consola de auditoría de la UI. |
| `mcp_client.py` | `MCPClient` | Cliente adaptador que convierte llamadas síncronas en llamadas async al servidor MCP. Resuelve el problema del event loop activo usando `ThreadPoolExecutor`. Expone `call_tool_sync()` que el orquestador usa en el modo no-async. |

**Integración:**  
Cada vez que el LLM solicita invocar una herramienta (en `AGENTIC_REFLECTION` o cuando llega un `tool_call_request` al orquestador), el `TransactionalGuardrailFSM` valida la transición **antes** de que el `MCPClient` envíe la petición al servidor MCP. Si la validación falla, el orquestador devuelve `status: GUARDRAIL_BLOCKED` sin tocar ningún sistema externo.

**Flujo FSM de estados:**
```
IDLE → AUTHENTICATED → TICKET_LOADED → POLICY_VERIFIED → ACTION_DRAFTED → OPERATOR_APPROVED → MUTATION_EXECUTED
                                                                              ↑
                                                               [Human Gate: requiere aprobación]
```

---

### 3.4 Adaptación por Preferencias (`adaptation/`)

Implementa el **Edit-Delta Conditioning Bridge** (componente k_pref): el mecanismo de aprendizaje en línea desde las correcciones del operador.

| Archivo | Clase / Función | Función |
|---|---|---|
| `edit_extractor.py` | `EditDeltaExtractor` | Recibe `(s_gen, s_human)` y computa el delta semántico `Δ_edit`. Detecta tokens añadidos (`added_spans`), eliminados (`removed_spans`) y sustituciones. Clasifica la edición en `EditCategory` (corrección factual, expansión de contexto, cambio de tono, etc.) y calcula un `significance_score` para filtrar ediciones triviales. |
| `edit_extractor.py` | `SemanticDelta` | Dataclass resultado: `is_significant`, `edit_category`, `significance_score`, `diff_summary`, `added_spans`, `removed_spans`. |
| `preference_memory.py` | `PreferenceMemory` | Store vectorial en memoria de claves de preferencia `k_pref`. Implementa decaimiento temporal exponencial: `w_i(t) = γ^(t - t_i)`. Las claves con `weight < threshold` son eviccionadas automáticamente en cada `step_turn()`. Capacidad máxima configurable. Soporta búsqueda de preferencias relevantes por similitud de embedding. |
| `preference_memory.py` | `PreferenceKey` | Pydantic model de cada preferencia almacenada: `query`, `intent`, `s_gen`, `s_human`, `delta`, `turn_created`, `current_weight`, `vector`, `is_synced_lora`. |
| `conditioning_bridge.py` | `EditDeltaConditioningBridge` | Recupera las k_pref más relevantes de `PreferenceMemory` y las formatea como contexto adicional para inyectar en el system prompt del LLM. Ordena por peso decaído descendente y aplica un límite de tokens configurable. |
| `lora_trigger.py` | `LoRATrigger` | Acumula `SemanticDelta`s. Cuando el buffer supera `batch_update_threshold` (default: 32), dispara un micro-ajuste LoRA/PEFT sobre el LLM usando `peft` + `transformers`. Los pesos del adaptador se guardan en `results/adapters/`. Marca las k_pref usadas con `is_synced_lora=True`. |

**Integración:**  
`IPGCAOrchestrator.record_operator_feedback()` conecta estos módulos en secuencia: `EditDeltaExtractor` → `PreferenceMemory.add_preference()` → `LoRATrigger.add_delta()`. En el siguiente turno, `EditDeltaConditioningBridge` inyecta las preferencias relevantes antes de llamar al generador.

---

### 3.5 Orquestador y Generador (`copilot/`)

El núcleo ejecutivo del sistema. Coordina todos los subsistemas en cada turno de diálogo.

| Archivo | Clase / Función | Función |
|---|---|---|
| `generator.py` | `LLMGenerator` | Envuelve `AutoModelForCausalLM` de HuggingFace con carga 4-bit (BitsAndBytes). Soporta `lazy_load=True` para tests rápidos. Aplica el chat template de Qwen2.5 (`apply_chat_template`). Expone `generate(prompt, system_prompt, max_new_tokens, temperature)`. |
| `orchestrator.py` | `IPGCAOrchestrator` | **Clase central de la arquitectura**. Al inicializarse, carga y conecta todos los subsistemas: `HybridRetrievalEngine`, `GPLController`, `TransactionalGuardrailFSM`, `PreferenceMemory`, `EditDeltaConditioningBridge`, `LLMGenerator` (opcional), `LatencyTracker`. |
| `orchestrator.py` | `process_turn()` | Método principal que ejecuta el pipeline completo en un turno: (1) pre-retrieval para scores, (2) enrutamiento GPL, (3) retrieval completo según camino, (4) inyección k_pref, (5) construcción de prompt, (6) generación o template sin LLM, (7) validación FSM si hay `tool_call_request`, (8) perfilado de latencia. Retorna `CopilotTurnResponse`. |
| `orchestrator.py` | `record_operator_feedback()` | Registra la edición humana `(s_gen → s_human)`: extrae delta, almacena k_pref, agrega delta al LoRA trigger. Retorna `(SemanticDelta, a_op_score, lora_batch)`. |
| `orchestrator.py` | `CopilotTurnResponse` | Pydantic model de salida completa de un turno: `query`, `selected_path`, `s_gen`, `retrieved_chunks`, `applied_preferences`, `tool_invocations`, `latency_profile`, `uncertainty_score`. |

**Integración:**  
`IPGCAOrchestrator` es el punto de entrada para la UI (`app_api.py`), los scripts de demo (`copilot_demo.py`) y los tests de integración (`conftest.py`). El `AblationMatrixRunner` **no** usa `IPGCAOrchestrator` directamente; en su lugar reimplementa el pipeline manualmente para poder configurar con precisión qué componentes activar en cada variante A–H.

---

### 3.6 Evaluación y Benchmarks (`evaluation/` + `experiments/`)

Infraestructura de medición cuantitativa para la tesis.

| Archivo | Clase / Función | Función |
|---|---|---|
| `alignment.py` | `calculate_operator_alignment()`, `calculate_ksr()`, `calculate_dar()` | **A_op** (Operator Alignment Rate): mide qué tan cercana es S_gen a S_human via similitud léxica (1 - normalized_edit_distance). **KSR** (Knowledge Substitution Rate): proporción de spans del operador no presentes en S_gen. **DAR** (Delta Acceptance Rate): fracción de deltas aceptados sin re-edición. |
| `faithfulness.py` | `FaithfulnessEvaluator` | Mide si la respuesta generada está fundamentada en el contexto recuperado. Puede usar un modelo NLI (DeBERTa) o caer a heurística léxica (token overlap) cuando `load_model=False`. Retorna `FaithfulnessResult` con `faithfulness_score` y `hallucination_rate`. |
| `latency_tracker.py` | `LatencyTracker`, `LatencyProfile` | Profiler de microsegundos por etapa. `measure(stage_name)` es un context manager. `finish_request()` consolida un `LatencyProfile` con latencias por etapa y `total_latency_ms`. Alimenta los logs de telemetría de la UI. |
| `tool_precision.py` | `calculate_tool_precision()` | Mide la precisión de las llamadas a herramientas del LLM: `TP / (TP + FP)`. Compara las herramientas invocadas con el conjunto de herramientas válidas/necesarias para el caso. |
| `benchmark_dataset.py` | `BenchmarkSample`, `get_default_ablation_benchmark_dataset()` | Define las 28 muestras estandarizadas del benchmark de ablación. Cada muestra tiene `sample_id`, `query`, `category`, `intent_probs` y opcionalmente `expected_tools`. Las categorías cubren: `policy_faq`, `warranty_claim`, `tool_invocation`, `ambiguous_query`, `injection_attempt`. |
| `ablation_matrix.py` | `AblationMatrixRunner` | Ejecutor de la matriz 8×28 de experimentos. Para cada variante (A–H) configura el subsistema de retrieval correspondiente, desactiva selectivamente k_pref o GPL-Controller, y mide las métricas en cada muestra. `run_benchmark_matrix()` retorna un dict `{AblationVariant: VariantAggregateSummary}`. |
| `experiments/evaluate_lora_adaptation.py` | `LoRAAdaptationExperiment` | Experimento completo Pre vs Post LoRA. Genera un batch de 32 deltas sintéticos representativos, dispara el fine-tuning, y evalúa sobre 16 muestras held-out midiendo mejora en A_op y KSR. Genera `results/lora_adaptation_report.md` y `results/table_lora_adaptation.tex`. |

**Integración:**  
`run_ablation_benchmark.py` llama a `AblationMatrixRunner` y guarda resultados en `results/ablation_results.json`. `generate_figures.py` consume estos JSON para producir las figuras matplotlib. El experimento LoRA es independiente y se lanza desde `run_experiments.ps1` o directamente.

---

### 3.7 Interfaz Gráfica (`ui/`)

Aplicación de escritorio nativa PyWebView (Chromium/Edge WebView2) con arquitectura MVC.

| Archivo | Clase / Función | Función |
|---|---|---|
| `desktop_app.py` | `launch_desktop_app()` | Punto de entrada. Crea la ventana `webview.create_window()` con el `AppAPI` como bridge y sirve el frontend desde `src/ui/static/`. Arranca el `IPGCAOrchestrator` con configuración completa. |
| `app_api.py` | `AppAPI` | **Bridge Python↔JavaScript**. Todos los métodos públicos son accesibles desde JS via `window.pywebview.api.*`. Métodos principales: `process_message(query, intent_probs)` → ejecuta un turno completo; `submit_operator_feedback(s_gen, s_human, intent)` → registra corrección humana; `get_fsm_state()` → retorna estado actual y log de transiciones; `get_routing_telemetry()` → retorna último `RoutingDecision`; `get_preference_summary()` → lista k_prefs activas con sus pesos. |
| `static/index.html` | — | Frontend HTML5 de panel único con 4 secciones: **Chat Panel** (historial de turnos + input), **Routing Telemetry** (visualización del camino elegido y u_ctx), **HITL Studio** (diff en vivo S_gen vs S_human), **FSM Console** (grafo de estados y log de transiciones). Estética dark-mode con glassmorphism. |
| `static/app.js` | `AppController` | Controlador JS asíncrono. Gestiona el ciclo de UI: captura inputs, llama a `window.pywebview.api.*`, actualiza el DOM con la respuesta, renderiza el diff con `diff-match-patch`, anima los nodos FSM. |
| `static/styles.css` | — | Estilos dark glassmorphism: fondos con `backdrop-filter: blur()`, badge de camino de enrutamiento color-coded, nodos FSM con estados activos animados, diseño responsivo. |

**Integración:**  
`AppAPI` es el único consumidor de `IPGCAOrchestrator` en la UI. Mantiene la instancia del orquestador en memoria durante toda la sesión. Los eventos del operador humano (edición en HITL Studio) llaman a `submit_operator_feedback()` que internamente llama a `orchestrator.record_operator_feedback()`.

---

### 3.8 Scripts Raíz de `src/`

| Archivo | Función |
|---|---|
| `logging_config.py` | Configura el logger global con `Rich` (colores en terminal) y rotación de archivos. Importado por todos los módulos con `from src.logging_config import get_logger`. |
| `copilot_demo.py` | Demo interactiva en terminal usando `Rich` (tablas, paneles). Permite al usuario escribir queries y simular el ciclo HITL completo sin la UI gráfica. Útil para debugging rápido. |
| `retrieval_pipeline_demo.py` | Demo aislada del subsistema de retrieval. Indexa la KB, procesa 3 queries de ejemplo, imprime el camino elegido por el GPL-Controller y los fragmentos recuperados. |
| `run_ablation_benchmark.py` | CLI que instancia `AblationMatrixRunner`, ejecuta las 8 variantes sobre las 28 muestras, y persiste resultados en `results/ablation_results.json` y `results/ablation_report.md`. |
| `generate_figures.py` | Lee `results/ablation_results.json` y `results/lora_experiment_results.json`. Genera con matplotlib las 4 figuras de publicación (fig1–fig4) en `results/figures/` y las tablas `results/table_ablation_results.tex` y `results/table_lora_adaptation.tex`. |

---

## 4. Servidores MCP (`mcp_servers/`)

Los servidores MCP se ejecutan como **procesos independientes** via stdio transport. El `MCPClient` se comunica con ellos a través de pipes estándar. Esta separación garantiza que el LLM **nunca** acceda directamente a datos CRM — siempre pasa por el FSM Guardrail.

| Archivo | Nombre del Server | Rol | Herramientas Expuestas |
|---|---|---|---|
| `crm_resource_server.py` | `crm-resource-server` | **Capa de lectura** (read-only). Expone clientes y tickets como MCP Resources. El LLM puede consultar pero no mutar. | `list_resources()`, `read_resource(uri)` |
| `ticketing_tool_server.py` | `ticketing-tool-server` | **Capa de escritura** (transaccional). Cada herramienta pasa primero por el FSM. Implementa el flujo completo desde autenticación hasta ejecución del reembolso. | `authenticate_customer`, `fetch_ticket`, `verify_policy`, `draft_action`, `execute_refund`, `cancel_subscription`, `escalate_ticket` |

**Integración:**  
El `MCPClient` en `guardrail/mcp_client.py` establece conexión con estos servidores. El flujo obligatorio es: `authenticate_customer` → `fetch_ticket` → `verify_policy` → `draft_action` → [Human Gate: `OPERATOR_APPROVED`] → `execute_refund`. El FSM bloquea cualquier intento de saltar pasos.

---

## 5. Configuración (`configs/`)

| Archivo | Contenido |
|---|---|
| `config_default.yaml` | Hiperparámetros globales: dispositivo (cuda/cpu), semilla aleatoria, rutas de modelos, parámetros de retrieval (top-k, pesos RRF), umbrales del router (τ_fast=0.25, τ_hybrid=0.65, τ_agent=0.85), presupuestos de latencia, parámetros LoRA (rank, alpha, target_modules), factor de decaimiento γ=0.95. |
| `fsm_rules.yaml` | Definición formal de la FSM: lista de `states` (con `is_terminal`, `requires_human_approval`), `transitions` (con `trigger`, `source`, `destination`, `mcp_tool`, `required_params`, `guards`). Las guardas son expresiones string evaluadas por `safe_eval_condition()`. |

---

## 6. Datos (`data/`)

| Ruta | Contenido |
|---|---|
| `data/raw/corporate_knowledge_base.json` | 25 documentos corporativos estructurados en JSON: políticas de devolución, garantías, suscripciones, envíos, compliance legal. Cada documento tiene `doc_id`, `title`, `category`, `content`, `department`. Indexado por `HybridRetrievalEngine` al arrancar. |
| `data/mcp_resources/customers.json` | 5 perfiles de clientes simulados con `customer_id`, `name`, `tier`, `status`. Servidos por `crm_resource_server.py`. |
| `data/mcp_resources/tickets.json` | 5 tickets de soporte simulados con `ticket_id`, `subject`, `status`, `customer_id`. Servidos por `crm_resource_server.py`. |

---

## 7. Modelos Locales (`models/`)

Todos los modelos se ejecutan **100% localmente**. No se realizan llamadas a APIs externas en producción.

| Carpeta | Modelo HuggingFace | Uso en el sistema |
|---|---|---|
| `bge-m3/` | `BAAI/BGE-M3` (~2.1 GB) | Embeddings densos para `DenseRetriever`. Modelo multilingüe de alta calidad que soporta retrieval denso, sparse (SPLADE-style) y ColBERT simultáneamente. |
| `bge-reranker-large/` | `BAAI/bge-reranker-large` (~2.2 GB) | Cross-Encoder para `CrossEncoderReranker`. Re-clasifica candidatos con atención full entre query y passage. Se usa solo cuando el presupuesto de latencia lo permite. |
| `Qwen2.5-1.5B-Instruct/` | `Qwen/Qwen2.5-1.5B-Instruct` (~3.1 GB) | LLM generativo. Cargado con cuantización 4-bit (BitsAndBytes) para caber en GPU de 6 GB VRAM. Usado por `LLMGenerator`. |
| `splade-cocondenser-ensembledistil/` | `naver/splade-cocondenser-ensembledistil` (~438 MB) | Modelo SPLADE para `SparseRetriever`. Genera representaciones sparse aprendidas (superiores a BM25 en vocabulario técnico). |

> **Nota:** Los archivos `.gitattributes` en cada carpeta de modelo configuran Git LFS para los pesos binarios grandes.

---

## 8. Resultados (`results/`)

Todos los artefactos de salida generados por los experimentos.

| Ruta | Contenido | Generado por |
|---|---|---|
| `ablation_results.json` | Métricas agregadas de las 8 variantes × 28 muestras | `run_ablation_benchmark.py` |
| `ablation_report.md` | Reporte legible en markdown con tablas de resultados | `run_ablation_benchmark.py` |
| `table_ablation_results.tex` | Tabla LaTeX `booktabs` lista para insertar en la tesis | `generate_figures.py` |
| `lora_experiment_results.json` | Comparación Pre/Post LoRA (A_op, KSR, DAR) | `experiments/evaluate_lora_adaptation.py` |
| `lora_adaptation_report.md` | Reporte del experimento LoRA | `experiments/evaluate_lora_adaptation.py` |
| `table_lora_adaptation.tex` | Tabla LaTeX del impacto LoRA | `generate_figures.py` |
| `figures/fig1_latency_comparison.png` | Gráfica de latencia por camino de enrutamiento | `generate_figures.py` |
| `figures/fig2_metrics_radar.png` | Radar chart de métricas por variante A-H | `generate_figures.py` |
| `figures/fig3_preference_decay.png` | Curva de decaimiento γ^(Δt) de las k_pref | `generate_figures.py` |
| `figures/fig4_lora_gain.png` | Ganancia Pre→Post LoRA en A_op y KSR | `generate_figures.py` |
| `adapters/lora_batch_0001/` | Pesos del adaptador LoRA entrenado (PEFT format) | `adaptation/lora_trigger.py` |
| `lora_batches/lora_batch_0001.json` | Dataset de 32 deltas usados para el micro-tuning | `adaptation/lora_trigger.py` |
| `logs/ablation_benchmark.log` | Log estructurado de la ejecución del benchmark | `logging_config.py` |
| `logs/copilot_demo.log` | Log de la demo interactiva en terminal | `logging_config.py` |

---

## 9. Suite de Tests (`tests/`)

59 tests que validan cada subsistema de forma aislada e integrada. Pasan al 100% en < 8 minutos.

| Archivo | Tests (count) | Qué valida |
|---|---|---|
| `conftest.py` | (fixtures) | `shared_fast_orchestrator` (session-scoped, sin reranker/LLM real) y `fresh_fsm` (function-scoped). Evita reinicializar modelos entre tests. |
| `test_retrieval.py` | 4 | Sparse BM25 con código exacto, dense semántico, fusión RRF, engine end-to-end sin reranker. |
| `test_router.py` | 6 | Umbrales de enrutamiento, degradación por latencia, cálculo de entropía/margen, EMA multi-turno. |
| `test_fsm.py` | 5 | Flujo legal completo, bloqueo de transición ilegal, evaluador AST con expresiones válidas/inválidas, reset de estado. |
| `test_adaptation.py` | 11 | ExtractDelta con ediciones triviales/significativas, PreferenceMemory con decaimiento, ConditioningBridge, LoRATrigger (modo simulado). |
| `test_chunker_and_loader.py` | 8 | Carga de JSON, chunking jerárquico, límites de tamaño, metadata preservada. |
| `test_copilot_orchestrator.py` | 4 | Inicialización, turn multi-camino, loop HITL con feedback, bloqueo guardrail. |
| `test_ablation_matrix.py` | 4 | Variante A, Variante E completa, degradación de Variante H, agregado multi-variante. |
| `test_metrics.py` | 9 | Cálculo de A_op, KSR, DAR, faithfulness, hallucination_rate, latency_tracker por etapa, tool_precision. |
| `test_generator.py` | 3 | Lazy loading sin GPU, skip en ausencia de CUDA/modelo, orquestador con `enable_real_llm=False`. |
| `test_ui_api.py` | 5 | AppAPI bridge: `process_message`, `submit_feedback`, `get_fsm_state`, `get_routing_telemetry`, `get_preference_summary`. |

---

## 10. Diagrama de Integración End-to-End

```
                         ┌───────────────────────────────────────┐
                         │          Entrada del Operador          │
                         │   UI (AppAPI) ◄──► app.js / index.html │
                         └────────────────┬──────────────────────┘
                                          │  process_message(query, intent_probs)
                                          ▼
                         ┌───────────────────────────────────────┐
                         │       IPGCAOrchestrator.process_turn() │
                         │           copilot/orchestrator.py       │
                         └──┬────────┬──────────┬────────────────┘
                            │        │          │
              ┌─────────────▼──┐  ┌──▼──────────▼──────────┐
              │  HybridRetrieval│  │      GPLController       │
              │  Engine.retrieve│  │   route_query()          │
              │  retrieval/     │  │   controller/router.py   │
              └─────────────┬──┘  └──────────┬──────────────┘
                            │                 │  RoutingDecision
                            └────────┬────────┘
                                     │
              ┌──────────────────────▼──────────────────────────┐
              │            EditDeltaConditioningBridge            │
              │   get_relevant_preferences() ← PreferenceMemory   │
              │          adaptation/conditioning_bridge.py         │
              └──────────────────────┬──────────────────────────┘
                                     │  Prompt enriquecido
                                     ▼
                         ┌───────────────────────┐
                         │     LLMGenerator       │
                         │  Qwen2.5-1.5B 4-bit   │
                         │  copilot/generator.py  │
                         └──────────┬────────────┘
                                    │  S_gen
                                    ▼
                         ┌───────────────────────┐
                         │    [HITL Loop]         │
                         │  Operador edita S_gen  │
                         │  → S_human             │
                         └──────────┬────────────┘
                                    │  record_operator_feedback()
                     ┌──────────────▼──────────────────────┐
                     │          EditDeltaExtractor           │
                     │      Δ_edit = diff(S_gen, S_human)   │
                     └──────────┬──────────────────────────┘
                                │
              ┌─────────────────▼─────────────────┐
              │          PreferenceMemory           │
              │  add_preference(k_pref) + decay γt │
              └─────────────────┬─────────────────┘
                                │  [cuando acumula 32 deltas]
                                ▼
                     ┌─────────────────────┐
                     │     LoRATrigger      │
                     │   micro-tuning PEFT  │
                     │   → results/adapters │
                     └─────────────────────┘

              [Si hay tool_call_request]
                                     │
                     ┌───────────────▼──────────────┐
                     │  TransactionalGuardrailFSM    │
                     │  attempt_transition()          │
                     │    ├── valida precedencia      │
                     │    ├── evalúa guardas AST      │
                     │    └── [Human Gate si aplica]  │
                     └───────────────┬──────────────┘
                                     │  ALLOWED / BLOCKED
                                     ▼
                          ┌─────────────────────┐
                          │     MCPClient        │
                          │  call_tool_sync()    │
                          └────────┬────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │  ticketing_tool_server.py         │
                    │  (proceso MCP stdio separado)     │
                    │  authenticate → fetch → verify    │
                    │  → draft → [approved] → execute   │
                    └──────────────────────────────────┘
```

---

## 11. Dependencias entre Módulos

```
logging_config  ← [todos los módulos]

base            ← sparse_retriever, dense_retriever, hybrid_fusion, reranker, engine, chunker, document_loader
document_loader ← engine, ablation_matrix, run_ablation_benchmark, retrieval_pipeline_demo
engine          ← orchestrator, ablation_matrix, retrieval_pipeline_demo
chunker         ← document_loader

uncertainty     ← router
router          ← orchestrator, ablation_matrix, retrieval_pipeline_demo

fsm             ← orchestrator, ablation_matrix, app_api, mcp_client
mcp_client      ← orchestrator, ablation_matrix

edit_extractor  ← preference_memory, orchestrator
preference_memory ← conditioning_bridge, orchestrator, lora_trigger
conditioning_bridge ← orchestrator, ablation_matrix
lora_trigger    ← orchestrator, experiments/evaluate_lora_adaptation

generator       ← orchestrator
orchestrator    ← app_api, copilot_demo, conftest, test_copilot_orchestrator

alignment       ← ablation_matrix, evaluation/__init__
faithfulness    ← ablation_matrix, evaluation/__init__
latency_tracker ← orchestrator, ablation_matrix, retrieval_pipeline_demo, evaluation/__init__
tool_precision  ← ablation_matrix, evaluation/__init__
benchmark_dataset ← ablation_matrix, run_ablation_benchmark
ablation_matrix ← run_ablation_benchmark, test_ablation_matrix

app_api         ← desktop_app
desktop_app     ← [entry point UI]
```

---

## 12. Archivos de Proyecto Raíz

| Archivo | Función |
|---|---|
| `README.md` | Documentación de alto nivel: resumen ejecutivo, arquitectura ASCII, estructura del repositorio, protocolo experimental, resultados, instrucciones de instalación y ejecución. |
| `requirements.txt` | Dependencias Python fijadas: `torch`, `transformers`, `peft`, `sentence-transformers`, `rank-bm25`, `mcp`, `pywebview`, `pydantic`, `pyyaml`, `rich`, `pytest`, `numpy`, `matplotlib`. |
| `run_experiments.ps1` | Script PowerShell de reproducibilidad completa: crea venv, instala dependencias, corre tests, benchmark de ablación A-H, experimento LoRA, generación de figuras. Paramétrico con switches `-SkipTests`, `-RunDemo`, etc. |
| `.gitignore` | Excluye: `.venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `results/logs/`, modelos grandes en `models/` (gestionados por Git LFS). |
| `plan_de_accion.md` | Bitácora de desarrollo: registro de fases implementadas, bugs corregidos, decisiones de diseño, próximas mejoras propuestas. |
| `Estado del Arte AI Copilot.pdf` | Documento de referencia académica del estado del arte. |
| `Propuesta_Inicial.pdf` | Propuesta original de investigación de la tesis. |
| `ARCHITECTURE.md` | **Este archivo.** Documentación técnica detallada de la integración. |

---

*Para mantener actualizado, ejecutar `run_experiments.ps1` y revisar las métricas en `results/`.*
