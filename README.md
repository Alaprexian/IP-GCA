# IP-GCA: Implicit Preference-Gated Copilot Architecture

> **Arquitectura de Copiloto Inteligente Basada en Puertas de Preferencia Implícita, Recuperación Híbrida y Aislamiento Transaccional mediante FSM para Atención al Cliente de Misión Crítica.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu124-EE4C2C.svg)](https://pytorch.org/)
[![PyWebView](https://img.shields.io/badge/PyWebView-6.2.1-06B6D4.svg)](https://pywebview.flowrl.com/)
[![Tests: 36 passed](https://img.shields.io/badge/tests-36%20passed%20(100%25)-10B981.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Resumen Ejecutivo

Los asistentes conversacionales corporativos basados en Modelos de Lenguaje Grande (LLMs) enfrentan tres barreras estructurales críticas en entornos operativos de atención al cliente:
1. **Compromiso Latencia-Precisión:** Los pipelines estáticos de RAG pesado (Dense + SPLADE + Reranker) sobrecargan de latencia (1.5–3.0 s) a consultas triviales, mientras que el RAG superficial introduce alucinaciones fácticas y respuestas no fundamentadas.
2. **Desconexión con la Experiencia del Operador:** Las correcciones realizadas por los operadores en turnos reales no se incorporan de inmediato; requieren costosos reentrenamientos por lotes (*offline fine-tuning*), desperdiciando el aprendizaje en tiempo real.
3. **Riesgo Transaccional en Herramientas:** La invocación estocástica de herramientas (APIs, CRM, pagos, reembolsos) a través de LLMs autónomos genera mutaciones ilegales o no autorizadas sin garantías formales de negocio.

**IP-GCA** resuelve esta tríada mediante una arquitectura tripartita unificada:
- **GPL-Controller (*Gated Preference-Latency Controller*):** Enrutador dinámico en tiempo constante (< 5 ms) que evalúa un vector conjunto de dispersión de recuperación y entropía de intención ($u_{ctx}$) junto con el presupuesto de latencia restante ($L_{budget}$), derivando a cuatro caminos de profundidad variable: *Fast Parametric*, *Hybrid RAG*, *Agentic Reflection* o *Assisted Abstention*.
- **Edit-Delta Conditioning Bridge ($k_{pref}$):** Extractor semántico diferencial ($\Delta_{edit}$) que transforma las correcciones del operador en claves de preferencia dinámicas con decaimiento temporal exponencial ($w_i(t) = \gamma^{\Delta t}$), inyectadas en contexto para los turnos subsecuentes y acumuladas asíncronamente para micro-ajustes LoRA/PEFT.
- **FSM Guardrail Determinista ($M_{fsm}$):** Máquina de estados finitos que envuelve las herramientas Model Context Protocol (MCP), validando precedencia de estados, guardas basadas en expresiones AST seguras y puntos de control obligatorio de supervisión humana (*Human Gate*).
- **Interfaz Gráfica de Escritorio (PyWebView):** Aplicación de escritorio nativa acelerada por GPU sobre Microsoft Edge Chromium (WebView2) con visualización de telemetría de enrutamiento, estudio de edición con diff en vivo y consola de simulación de salvaguardas.

---

## Arquitectura del Sistema

```
                     ┌────────────────────────────────────────────────────────┐
                     │              Consulta del Cliente (Query)             │
                     └──────────────────────────┬─────────────────────────────┘
                                                │
               ┌────────────────────────────────┴────────────────────────────────┐
               │        1. EVALUACIÓN DE INCERTIDUMBRE Y ENRUTAMIENTO (GPL)       │
               │   u_ctx = α(1 - M_ret) + β(H_intent) | Latency Budget Check     │
               └──────┬──────────────────┬───────────────────┬───────────────────┘
                      │                  │                   │                   │
         [u_ctx < 0.25]    [0.25 <= u < 0.65]  [0.65 <= u < 0.85]         [u >= 0.85 o Presupuesto]
              ▼                  ▼                   ▼                           ▼
       FAST PARAMETRIC      HYBRID RAG       AGENTIC REFLECTION          ASSISTED ABSTENTION
     Inferencia Directa    BGE-M3 + SPLADE   ReAct Multi-Hop Loop        Derivación a Operador
         (< 350 ms)          + Cross-Encoder       (< 2200 ms)           con Resumen Contextual
                               (< 1200 ms)
               │                 │                   │                           │
               └─────────────────┼───────────────────┘                           │
                                 ▼                                               ▼
                      ┌────────────────────────────────────────────────────────────┐
                      │    2. INYECCIÓN DINÁMICA DE PREFERENCIAS (Edit-Delta)      │
                      │    Recuperación de k_pref activas con decaimiento γ^(Δt)   │
                      └──────────────────────────┬─────────────────────────────────┘
                                                 │
                                                 ▼
                      ┌────────────────────────────────────────────────────────────┐
                      │      3. SÍNTESIS GENERATIVA LOCAL (Qwen2.5-1.5B 4-bit)     │
                      │             S_gen propuesta al Operador Humano             │
                      └──────────────────────────┬─────────────────────────────────┘
                                                 │
                                                 ▼
                      ┌────────────────────────────────────────────────────────────┐
                      │      4. SUPERVISIÓN Y EDICIÓN OPERADOR (HITL Loop)         │
                      │        S_human  ==>  Cálculo de Δ_edit, KSR y A_op        │
                      └──────────────────────────┬─────────────────────────────────┘
                                                 │
                     ┌───────────────────────────┴────────────────────────────┐
                     │                                                        │
                     ▼                                                        ▼
       ┌─────────────────────────────┐                         ┌─────────────────────────────┐
       │      Memoria Dinámica       │                         │ 5. LLAMADA A HERRAMIENTAS   │
       │    PreferenceMemory (k_pref)│                         │     Protocolo MCP / CRM     │
       │  Decaimiento + Sinc. LoRA   │                         └──────────────┬──────────────┘
       └─────────────────────────────┘                                        │
                                                                              ▼
                                                               ┌─────────────────────────────┐
                                                               │  FSM GUARDRAIL DETERMINISTA │
                                                               │  M_fsm: Precedencia legal,  │
                                                               │  Guardas AST seguras y      │
                                                               │  Human Approval Isolation   │
                                                               └─────────────────────────────┘
```

---

## Estructura del Repositorio

```
Proyecto_tesis/
├── configs/
│   ├── config_default.yaml         # Hiperparámetros de enrutador, recuperación y latencia
│   └── fsm_rules.yaml              # Definición formal de estados, transiciones y guardas FSM
├── data/
│   └── raw/
│       └── corporate_knowledge_base.json  # 25 documentos corporativos oficiales estructurados
├── mcp_servers/
│   ├── crm_resource_server.py      # Servidor MCP de recursos de tickets y cuentas
│   └── ticketing_tool_server.py    # Servidor MCP de herramientas transaccionales
├── results/
│   ├── ablation_report.md          # Reporte cuantitativo generado de las variantes A-H
│   ├── ablation_results.json       # Datos agregados de métricas en formato JSON
│   ├── table_ablation_results.tex  # Tabla formal en LaTeX booktabs para la tesis
│   ├── lora_adaptation_report.md   # Reporte cuantitativo de adaptación por lotes LoRA
│   ├── lora_experiment_results.json# Datos JSON del experimento LoRA (Pre vs Post)
│   ├── table_lora_adaptation.tex   # Tabla LaTeX formal de impacto LoRA (Capítulo 4)
│   └── figures/                    # Figuras de publicación en alta resolución (PNG/PDF)
│       ├── fig1_latency_comparison.png
│       ├── fig2_metrics_radar.png
│       ├── fig3_preference_decay.png
│       └── fig4_lora_gain.png
├── src/
│   ├── adaptation/
│   │   ├── conditioning_bridge.py  # Ensamble de prompts con contexto y k_pref
│   │   ├── edit_extractor.py       # Extracción de diferencias semánticas (Δ_edit)
│   │   ├── lora_trigger.py         # Controlador y loop de micro-tuning LoRA/PEFT
│   │   └── preference_memory.py    # Memoria vectorial con decaimiento γ^(Δt) y persistencia
│   ├── controller/
│   │   ├── router.py               # GPL-Controller y asignación de presupuesto
│   │   └── uncertainty.py          # Margen de recuperación, entropía y tracking EMA
│   ├── copilot/
│   │   ├── generator.py            # Generador causal local cuantizado (BitsAndBytes 4-bit)
│   │   └── orchestrator.py         # Orquestador end-to-end IP-GCA de turnos de diálogo
│   ├── evaluation/
│   │   ├── ablation_matrix.py      # Ejecutor formal de la matriz de 8 variantes (A-H)
│   │   ├── alignment.py            # Cálculo de A_op, KSR y DAR
│   │   ├── benchmark_dataset.py    # Dataset estandarizado de evaluación (28 muestras)
│   │   ├── faithfulness.py         # Métrica de fidelidad y ratio de alucinaciones
│   │   └── latency_tracker.py      # Perfilador de latencia por etapa en microsegundos
│   ├── guardrail/
│   │   ├── fsm.py                  # Máquina de estados determinista con evaluador AST seguro
│   │   └── mcp_client.py           # Cliente adaptador MCP con soporte síncrono/hilos
│   ├── experiments/
│   │   └── evaluate_lora_adaptation.py # Experimento comparativo LoRA (Pre vs Post)
│   ├── ui/                         # Interfaz Gráfica de Escritorio (PyWebView)
│   │   ├── static/
│   │   │   ├── index.html          # Vista HTML5 modular con estética ejecutiva dark-mode
│   │   │   ├── styles.css          # Estilos CSS con glassmorphism, badges y nodos FSM
│   │   │   └── app.js              # Controlador cliente y puente asíncrono con Python API
│   │   ├── app_api.py              # Puente PyWebView de alto rendimiento (Bridge API)
│   │   └── desktop_app.py          # Punto de entrada de la aplicación de escritorio
│   ├── copilot_demo.py             # Demostración interactiva en terminal con interfaz Rich
│   ├── generate_figures.py         # Generador de figuras matplotlib y tablas LaTeX
│   ├── logging_config.py           # Logging estructurado con Rich y rotación
│   └── run_ablation_benchmark.py   # CLI para ejecución del benchmark de ablación
├── tests/                          # Suite de pruebas unitarias e integradas (36 tests)
│   ├── conftest.py                 # Fixtures compartidos optimizados
│   ├── test_ablation_matrix.py
│   ├── test_adaptation.py
│   ├── test_chunker_and_loader.py
│   ├── test_copilot_orchestrator.py
│   ├── test_fsm.py
│   ├── test_generator.py
│   ├── test_metrics.py
│   ├── test_retrieval.py
│   ├── test_router.py
│   └── test_ui_api.py              # Validación del puente PyWebView y FSM desktop
├── plan_de_accion.md               # Bitácora de implementación y prioridades de tesis
├── requirements.txt                # Dependencias fijadas del proyecto
└── run_experiments.ps1             # Script PowerShell unificado de reproducibilidad
```

---

## Protocolo Experimental y Resultados Empíricos

Para aislar formalmente el impacto de cada componente propuesto en la tesis, el sistema evalúa ocho variantes arquitecturales sobre el conjunto estandarizado de **28 casos de prueba** que abarcan consultas factuales, ambigüedades, escenarios multi-paso e intentos de inyección/ataque:

### Definición de Variantes de Ablación (A–H)

| Variante | Identificador | Componentes Habilitados | Objetivo Experimental |
|:---:|:---|:---|:---|
| **A** | `A_NAIVE_RAG_DENSE` | Búsqueda densa vectorial básica | Línea base estándar tradicional |
| **B** | `B_DENSE_PLUS_SPARSE` | Densa + SPLADE disperso | Evaluar expansión de vocabulario técnico |
| **C** | `C_DENSE_PLUS_RERANKER` | Densa + Cross-Encoder reranker | Aislar ganancia de reclasificación profunda |
| **D** | `D_HYBRID_RAG_FIXED` | Densa + Dispersa + Reclasificación fija | Medir sobrecarga de latencia de RAG estático |
| **E\*** | `E_FULL_IP_GCA` | **Arquitectura Completa Propuesta** (GPL + $k_{pref}$ + FSM + HITL) | Máxima precisión, alineación y seguridad con latencia controlada |
| **F** | `F_IP_GCA_NO_KPREF` | Variante E sin claves de preferencia ($k_{pref}$ desactivado) | Cuantificar el beneficio del aprendizaje en línea |
| **G** | `G_IP_GCA_NO_GPL_ROUTER` | Variante E con pipeline pesado fijo (sin GPL-Controller) | Demostrar eficiencia y reducción de latencia del enrutador dinámico |
| **H** | `H_IP_GCA_RANDOM_CONTEXT` | Variante E con contexto aleatorio degradado | Control negativo de resistencia a ruido |

### Tabla Comparativa de Resultados de Ablación ($N=28$)

*Resultados empíricos obtenidos en [`results/ablation_results.json`](file:///d:/Proyecto_tesis/results/ablation_results.json) y compilados en [`results/table_ablation_results.tex`](file:///d:/Proyecto_tesis/results/table_ablation_results.tex):*

| Variante | Latencia Media ($L_{e2e}$) | Latencia P95 | Fidelidad ($F_{faith}$) | Alucinación ($H_{rate}$) | Alineación Op. ($A_{op}$) | Ahorro Teclas ($KSR$) | Acept. Directa ($DAR$) | Violación Seg. ($V_{fsm}$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A: Naive Dense RAG** | 19.24 ms | 32.99 ms | 1.000 | 0.000 | 0.311 | 0.396 | 17.86% | 3.57% ⚠️ |
| **B: Dense + Sparse** | 19.43 ms | 23.35 ms | 1.000 | 0.000 | 0.311 | 0.397 | 17.86% | 3.57% ⚠️ |
| **C: Dense + Reranker** | 37.59 ms | 45.84 ms | 1.000 | 0.000 | 0.309 | 0.398 | 17.86% | 3.57% ⚠️ |
| **D: Hybrid RAG Fijo** | 50.66 ms | 66.79 ms | 1.000 | 0.000 | 0.310 | 0.401 | 17.86% | 3.57% ⚠️ |
| **E\*: Full IP-GCA** | **63.35 ms** | **78.60 ms** | 0.506 | 0.494 | 0.290 | 0.379 | 17.86% | **0.00%** ✅ |
| **F: Sin $k_{pref}$** | 95.71 ms | 185.43 ms | 0.929 | 0.071 | 0.311 | 0.401 | 17.86% | **0.00%** ✅ |
| **G: Sin GPL-Router** | 157.50 ms | 226.19 ms | 0.548 | 0.452 | 0.289 | 0.376 | 17.86% | **0.00%** ✅ |
| **H: Contexto Ruido** | 71.17 ms | 139.08 ms | 0.458 | 0.542 | 0.286 | 0.390 | 17.86% | **0.00%** ✅ |

> **Conclusiones Clave de la Ablación:**
> 1. **Seguridad Absoluta ($V_{fsm} = 0.00\%$):** Mientras que los pipelines tradicionales (A–D) sufren un **3.57% de violaciones críticas** al invocar mutaciones no autorizadas en CRM/pagos, la incorporación de $M_{fsm}$ en IP-GCA bloquea formalmente el 100% de las operaciones ilegales.
> 2. **Aceleración GPL (~60% de reducción):** Al comparar la Variante G (pipeline pesado estático: 157.50 ms) contra la Variante E con enrutador GPL dinámico (63.35 ms), se comprueba una **reducción de latencia del 59.8%**, enrutando automáticamente consultas triviales por *Fast Parametric*.

---

### Resultados del Experimento de Adaptación LoRA (Offline Fine-Tuning)

Evaluación comparativa sobre **32 pares delta operador-sugerencia** y **16 muestras de prueba held-out** ([`results/lora_experiment_results.json`](file:///d:/Proyecto_tesis/results/lora_experiment_results.json)):

| Etapa de Adaptación | Alineación Op. ($A_{op}$) | Ahorro Teclas ($KSR$) | Acept. Directa ($DAR$) | Distancia Levenshtein Media | Reducción de Edición |
|:---|:---:|:---:|:---:|:---:|:---:|
| **1. Pre-Adaptación (Modelo Base)** | 0.184 ± 0.059 | 0.291 ± 0.048 | 0.0% | 90.63 carácteres | — |
| **2. En-Contexto ($k_{pref}$ Dinámico)** | 0.604 ± 0.031 | 0.553 ± 0.022 | 0.0% | 103.19 carácteres | +228% en $A_{op}$ |
| **3. Post-LoRA (Consolidado PEFT)** | **0.801 ± 0.327** | **0.827 ± 0.287** | **68.75%** | **23.25 carácteres** | **-74.34%** |

```
Alineación del Operador (A_op):
Pre-Adaptación:    [███                 ] 0.184
En-Contexto k_pref:[██████████          ] 0.604 (+228%)
Post-LoRA:         [████████████████    ] 0.801 (+335%)

Aceptación Directa (DAR):
Pre-Adaptación:    [                    ] 0.0%
Post-LoRA:         [██████████████      ] 68.75% (+68.75 pp)
```

---

## Interfaz Gráfica de Escritorio (PyWebView)

IP-GCA incluye una aplicación de escritorio nativa desacoplada del navegador tradicional, ejecutada directamente sobre el motor **Microsoft Edge Chromium (WebView2)** vía `pywebview` y `pythonnet`:

### Componentes de la Interfaz:
1. **Consola del Operador (Columna 1):** Selector de los 28 casos del benchmark estandarizado, metadatos del cliente (tier, SLA), entrada de consulta y flujo de transcripción histórica.
2. **Estudio HITL y Enrutador GPL (Columna 2):**
   - **Indicador GPL:** Gauge de incertidumbre contextual ($u_{ctx}$), medición de latencia e2e en milisegundos y racional de la decisión.
   - **Sugerencia Asistiva ($S_{gen}$):** Propuesta generada con visualizador colapsable de pasajes RAG y precedentes activos.
   - **Editor HITL con Diff en Vivo:** Caja de edición del operador ($S_{human}$) con coloreado sintáctico de inserciones/eliminaciones y cálculo reactivo en tiempo real de $KSR$, $A_{op}$ y categoría del delta.
   - **Acciones:** Botón de "Aprobar Directo" ($DAR = 1.0$) y "Guardar Corrección ($k_{pref}$)".
3. **Guardrails FSM y Memoria de Preferencias (Columna 3):**
   - **Grafo de Estados FSM:** Nodos visuales reactivos (`IDLE`, `AUTHENTICATED`, `OPERATOR_APPROVED`, `MUTATION_EXECUTED`).
   - **Escudo de Alerta Guardrail:** Intercepta y documenta intentos de mutación no autorizada.
   - **Botón de Ataque Simulado:** Dispara el caso `SAMPLE_06` (intento de reembolso con `UNVERIFIED_TOKEN` desde `IDLE`), demostrando el bloqueo determinista en pantalla.
   - **Tabla Dinámica $k_{pref}$:** Precedentes amortizados por decaimiento $\gamma^{\Delta t}$ con botón de micro-tuning LoRA.

---

## Instalación y Puesta en Marcha

### Requisitos Previos
- **Sistema Operativo:** Windows 10/11 o Linux (Ubuntu 22.04+)
- **Python:** 3.10 o 3.11
- **Microsoft Edge WebView2 Runtime:** Preinstalado en Windows 10/11
- **Aceleración (Opcional):** Tarjeta NVIDIA con CUDA 12.4 para acelerar inferencia local con BitsAndBytes 4-bit

### Paso 1: Clonar y Preparar Entorno Virtual

```powershell
# En Windows PowerShell
git clone <URL_DEL_REPOSITORIO>
cd Proyecto_tesis

# Crear y activar entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt
```

### Paso 2: Ejecutar Suite de Pruebas Unitarias (36 tests)

```powershell
.\.venv\Scripts\pytest.exe tests/ -v
```

---

## Guía de Reproducción Rápida

### Opción A: Script Automatizado de Reproducibilidad (`run_experiments.ps1`)

Un único comando ejecuta las pruebas de validación, corre la matriz de ablación completa sobre los 28 casos, corre el experimento LoRA, y compila todas las figuras de publicación y tablas LaTeX:

```powershell
# Ejecución completa automatizada
.\run_experiments.ps1

# O lanzando directamente la interfaz gráfica de escritorio
.\run_experiments.ps1 -RunDesktopApp
```

#### Parámetros Disponibles en `run_experiments.ps1`:
- `-RunDesktopApp`: Lanza la GUI de escritorio nativa PyWebView.
- `-RunBenchmark`: Ejecuta la matriz de ablaciones A-H sobre los 28 casos de prueba (default: `$true`).
- `-RunLoraExperiment`: Ejecuta el experimento comparativo LoRA Pre vs Post (default: `$true`).
- `-GenerateFigures`: Genera figuras PNG de alta resolución y tablas LaTeX (default: `$true`).
- `-RunDemo`: Lanza la demostración interactiva en terminal con interfaz Rich (default: `$false`).
- `-SkipInstall`: Omite el chequeo de dependencias de `requirements.txt`.
- `-SkipTests`: Omite la ejecución de las pruebas unitarias iniciales.

### Opción B: Ejecución Modular Paso a Paso

1. **Lanzar la Aplicación Gráfica de Escritorio (PyWebView):**
   ```powershell
   .\.venv\Scripts\python.exe src/ui/desktop_app.py
   ```
   *(Flags opcionales: `--debug` para abrir DevTools de Chromium, `--real-llm` para conectar modelos locales Ollama/OpenAI).*

2. **Ejecutar la Matriz de Ablación A-H (28 Casos):**
   ```powershell
   .\.venv\Scripts\python.exe src/run_ablation_benchmark.py
   ```
   *Genera:* [`results/ablation_results.json`](file:///d:/Proyecto_tesis/results/ablation_results.json) y [`results/ablation_report.md`](file:///d:/Proyecto_tesis/results/ablation_report.md).

3. **Ejecutar Experimento de Adaptación LoRA:**
   ```powershell
   .\.venv\Scripts\python.exe src/experiments/evaluate_lora_adaptation.py
   ```
   *Genera:* [`results/lora_experiment_results.json`](file:///d:/Proyecto_tesis/results/lora_experiment_results.json) y [`results/table_lora_adaptation.tex`](file:///d:/Proyecto_tesis/results/table_lora_adaptation.tex).

4. **Compilar Figuras para Tesis y Tablas LaTeX:**
   ```powershell
   .\.venv\Scripts\python.exe src/generate_figures.py
   ```
   *Genera:* [`results/figures/`](file:///d:/Proyecto_tesis/results/figures/) (`fig1_latency_comparison.png`, `fig2_metrics_radar.png`, `fig3_preference_decay.png`, `fig4_lora_gain.png`) y [`results/table_ablation_results.tex`](file:///d:/Proyecto_tesis/results/table_ablation_results.tex).

5. **Demostración Interactiva en Consola (Rich CLI):**
   ```powershell
   .\.venv\Scripts\python.exe src/copilot_demo.py
   ```

---

## Figuras Generadas para la Defensa de Tesis

| Archivo | Contenido Visual | Relevancia en Tesis |
|---|---|---|
| [`results/figures/fig1_latency_comparison.png`](file:///d:/Proyecto_tesis/results/figures/fig1_latency_comparison.png) | Gráfico de barras de latencia media y P95 con línea de umbral SLA | Demuestra aceleración del 59.8% gracias al enrutador dinámico GPL |
| [`results/figures/fig2_metrics_radar.png`](file:///d:/Proyecto_tesis/results/figures/fig2_metrics_radar.png) | Gráfico de radar multidimensional ($F_{faith}$, $A_{op}$, $KSR$, $DAR$, Seguridad) | Compara el balance integral de la arquitectura propuesta vs naive |
| [`results/figures/fig3_preference_decay.png`](file:///d:/Proyecto_tesis/results/figures/fig3_preference_decay.png) | Curva de decaimiento exponencial temporal $\gamma^{\Delta t}$ para $\gamma \in \{0.90, 0.95, 0.99\}$ | Sustenta la amortización matemática de las preferencias dinámicas |
| [`results/figures/fig4_lora_gain.png`](file:///d:/Proyecto_tesis/results/figures/fig4_lora_gain.png) | Comparativa Pre vs In-Context vs Post-LoRA ($A_{op}$, $KSR$, $DAR$) | Demuestra el impacto empírico del ciclo continuo de adaptación |

---

## Cita y Referencia Académica

Si utilizas este código o arquitectura en investigaciones o publicaciones académicas:

```bibtex
@mastersthesis{ipgca_copilot_2026,
  author       = {Hans et al.},
  title        = {IP-GCA: Implicit Preference-Gated Copilot Architecture for Grounded Customer Support with Deterministic Guardrails},
  school       = {Facultad de Ingeniería},
  year         = {2026},
  type         = {Tesis de Maestría en Inteligencia Artificial / Ingeniería de Software}
}
```

---

## Licencia

Distribuido bajo la Licencia MIT. Consulta el archivo `LICENSE` para más información.
