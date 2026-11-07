# 📋 Plan de Acción — Mejoras IP-GCA Copilot

> **Estado: 100% Implementado y Validado**  
> Todos los módulos de las Fases 1, 2, 3 y 4 se encuentran implementados, probados con pruebas unitarias e integradas, y respaldados por artefactos de reproducibilidad para la defensa de tesis.

---

## Resumen Ejecutivo de Implementación

| Fase | Descripción | Estado | Pruebas / Validación |
|---|---|:---:|---|
| **Fase 1** | Correcciones Críticas (FSM Reset, Trigger Refunds, Asyncio, Docstrings) | ✅ Completado | `test_fsm.py`, `test_orchestrator.py` |
| **Fase 2** | Robustez del Pipeline (Safe AST Evaluator, EMA Uncertainty, Persistencia, 28 Samples) | ✅ Completado | `test_fsm.py`, `test_router.py`, `test_adaptation.py` |
| **Fase 3** | Ampliaciones de Tesis (LoRA Training, 25 Docs KB, Generador Figuras, Demo Rich) | ✅ Completado | `generate_figures.py`, `copilot_demo.py` |
| **Fase 4** | Reproducibilidad (conftest.py, run_experiments.ps1, requirements) | ✅ Completado | `run_experiments.ps1` |
| **Fase 5** | Interfaz Gráfica de Escritorio (PyWebView, Edge WebView2, Dark Glassmorphism HITL Studio) | ✅ Completado | `src/ui/desktop_app.py`, `src/ui/app_api.py` |

---

## FASE 1 — Correcciones Críticas (Esta semana)
> *Bugs que pueden afectar resultados del benchmark o la defensa*

---

### 1.1 Resetear FSM al inicio de cada turno
**Archivo:** [`src/copilot/orchestrator.py`](file:///d:/Proyecto_tesis/src/copilot/orchestrator.py)  
**Impacto:** 🔴 Alto — sin reset, el FSM puede rechazar operaciones legales en turnos 2+  
**Dificultad:** 🟢 Baja (1 línea)

```diff
 def process_turn(self, query, customer_id=None, ...):
+    self.fsm.reset()   # Reset FSM state at each new interaction turn
     self.latency_tracker.start_request(query)
```

---

### 1.2 Corregir el trigger `execute_refund` en FSM rules
**Archivo:** [`configs/fsm_rules.yaml`](file:///d:/Proyecto_tesis/configs/fsm_rules.yaml)  
**Impacto:** 🔴 Alto — el benchmark SAMPLE_06 pasa por razón incorrecta (trigger desconocido, no por guardrail)  
**Dificultad:** 🟢 Baja

Agregar alias o renombrar el trigger `execute_action` → `execute_refund` para que el FSM lo evalúe correctamente y el bloqueo sea por un guard, no por trigger desconocido.

```yaml
# Añadir o renombrar en transitions:
  - trigger: "execute_refund"          # ← alias explícito para benchmark
    source: "OPERATOR_APPROVED"
    destination: "MUTATION_EXECUTED"
    mcp_tool: "mcp://crm/execute_refund"
    required_params: ["ticket_id", "approval_token"]
    guards:
      - condition: "approval_token != 'UNVERIFIED_TOKEN'"
        error_message: "Token de aprobación no verificado. Operación bloqueada."
```

---

### 1.3 Corregir docstring incorrecto de Variant G
**Archivo:** [`src/evaluation/ablation_matrix.py`](file:///d:/Proyecto_tesis/src/evaluation/ablation_matrix.py) línea 10-11  
**Impacto:** 🟡 Medio — confusión en la lectura del código / tesis  
**Dificultad:** 🟢 Trivial

```diff
-# - Variant F: E without dynamic GPL-Controller routing (fixed pipeline).
+# - Variant G: E without dynamic GPL-Controller routing (fixed pipeline).
```

---

### 1.4 Resolver `asyncio.run()` con event loop ya activo
**Archivo:** [`src/guardrail/mcp_client.py`](file:///d:/Proyecto_tesis/src/guardrail/mcp_client.py)  
**Impacto:** 🟡 Medio — crash en notebooks Jupyter y entornos async  
**Dificultad:** 🟢 Baja

```python
def call_tool_sync(self, tool_name, arguments):
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        # Running inside an existing loop — use thread executor
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, self.call_tool(tool_name, arguments))
            return future.result()
    except RuntimeError:
        return asyncio.run(self.call_tool(tool_name, arguments))
```

---

## FASE 2 — Robustez del Pipeline (Esta quincena)
> *Mejoras que afectan la calidad del benchmark y la validez experimental*

---

### 2.1 Añadir persistencia para PreferenceMemory (JSON/SQLite)
**Archivos:** [`src/adaptation/preference_memory.py`](file:///d:/Proyecto_tesis/src/adaptation/preference_memory.py)  
**Impacto:** 🟡 Medio — permite demostrar aprendizaje acumulativo entre sesiones en la demo  
**Dificultad:** 🟡 Media (2-3 horas)

Añadir métodos `save(path)` / `load(path)` con serialización `model_dump()` de Pydantic a JSON. Útil para la demo de la tesis y la Figura de evolución de `k_pref`.

---

### 2.2 Reemplazar `eval()` por evaluador seguro en FSM
**Archivo:** [`src/guardrail/fsm.py`](file:///d:/Proyecto_tesis/src/guardrail/fsm.py) línea 134  
**Impacto:** 🟡 Medio — riesgo de seguridad documentado  
**Dificultad:** 🟡 Media

Reemplazar por evaluación basada en operadores explícitos:

```python
# Alternativa segura usando ast.literal_eval + operadores restringidos
import operator as op_module
import ast

ALLOWED_OPS = {ast.Lt: op_module.lt, ast.LtE: op_module.le,
               ast.Gt: op_module.gt, ast.GtE: op_module.ge,
               ast.Eq: op_module.eq, ast.NotEq: op_module.ne}

def _safe_eval_condition(cond: str, params: dict) -> bool:
    tree = ast.parse(cond, mode='eval')
    # Walk tree and only allow Compare nodes with Name/Constant/Num
    ...
```
> O simplemente documentarlo como limitación conocida en la tesis.

---

### 2.3 Añadir Session-Level Uncertainty Accumulation
**Archivo:** [`src/controller/uncertainty.py`](file:///d:/Proyecto_tesis/src/controller/uncertainty.py)  
**Impacto:** 🟢 Bajo-Medio — mejora teórica del modelo de incertidumbre  
**Dificultad:** 🟡 Media

Añadir EMA (Exponential Moving Average) de `u_ctx` acumulado en la sesión para modelar la reducción de incertidumbre a medida que el agente aprende el contexto del cliente en el turno.

```python
u_ctx_smoothed = alpha * u_ctx_current + (1 - alpha) * u_ctx_session_history
```

---

### 2.4 Ampliar el dataset de benchmark (8 → 20-30 samples)
**Archivo:** [`src/evaluation/benchmark_dataset.py`](file:///d:/Proyecto_tesis/src/evaluation/benchmark_dataset.py)  
**Impacto:** 🔴 Alto para la tesis — 8 samples es estadísticamente insuficiente para concluir  
**Dificultad:** 🟡 Media (trabajo de diseño)

Categorías a añadir:
- Multi-turn conversations (turno 1 → turno 3 del mismo cliente)
- Queries con intención mezclada (billing + technical simultáneamente)  
- Queries en idioma ambiguo / mezcla español-inglés
- Casos de escalación legítima vs. intento de manipulación

---

### 2.5 Implementar logging estructurado con `rich` + rotación
**Archivos:** Todos los módulos  
**Impacto:** 🟢 Bajo — mejora la depuración y los experimentos  
**Dificultad:** 🟢 Baja

Centralizar el logging con `logging.getLogger()` y añadir un `RichHandler` configurado desde `config_default.yaml`. Actualmente solo `generator.py` usa `logger`.

---

## FASE 3 — Ampliaciones de Valor para la Tesis (Próximo mes)
> *Lo que realmente diferencia el trabajo y enriquece la defensa*

---

### 3.1 🚀 Implementar el ciclo real de LoRA fine-tuning
**Archivo:** [`src/adaptation/lora_trigger.py`](file:///d:/Proyecto_tesis/src/adaptation/lora_trigger.py)  
**Impacto:** 🔴 Muy Alto — actualmente el módulo es un stub; la contribución central de la tesis requiere esto  
**Dificultad:** 🔴 Alta (requiere tiempo de entrenamiento con GPU)

```python
# Implementar en LoRATriggerController:
def execute_lora_training(self, batch: LoRABatchDataset, base_model_path: str) -> str:
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import TrainingArguments, Trainer
    
    config = LoraConfig(**self.get_peft_lora_config(), task_type=TaskType.CAUSAL_LM)
    # ... SFT / DPO loop con trl.SFTTrainer o trl.DPOTrainer
```

Incluso ejecutar 1-2 ciclos de micro-fine-tuning reales y medir la mejora en `A_op` es suficiente para la tesis.

---

### 3.2 📊 Añadir generación automática de resultados y gráficas
**Archivos:** [`src/run_ablation_benchmark.py`](file:///d:/Proyecto_tesis/src/run_ablation_benchmark.py), `results/`  
**Impacto:** 🔴 Muy Alto — la tesis necesita tablas y figuras reproducibles  
**Dificultad:** 🟡 Media

Crear un script `generate_figures.py` que:
- Corra el benchmark completo A-H
- Exporte tabla LaTeX con medias ± desviación estándar
- Genere gráficas comparativas con `matplotlib`: barras de latencia, radar de métricas

---

### 3.3 🌐 Añadir Knowledge Base corporativa más rica
**Archivo:** [`data/raw/corporate_knowledge_base.json`](file:///d:/Proyecto_tesis/data/raw/corporate_knowledge_base.json)  
**Impacto:** 🟡 Medio — la KB actual tiene muy pocas entradas; retrieval scores serán poco realistas  
**Dificultad:** 🟢 Baja (datos sintéticos)

Ampliar a 50-100 documentos con políticas, FAQs, procedimientos. Permite que los scores de retrieval sean estadísticamente significativos y que el router tome decisiones más realistas.

---

### 3.4 🖥️ Demo interactiva de una sesión HITL completa
**Archivo:** [`src/copilot_demo.py`](file:///d:/Proyecto_tesis/src/copilot_demo.py)  
**Impacto:** 🟡 Medio — esencial para la defensa visual del sistema  
**Dificultad:** 🟢 Baja (código demo ya parcialmente existente)

Usar `rich` para mostrar en terminal:
- El routing decision con colores por path
- Los passages recuperados
- Las preferencias activas `k_pref` con sus pesos decaídos
- El resultado del FSM para cada herramienta

---

### 3.5 📐 Calcular DAR (Direct Acceptance Rate) en el benchmark
**Archivo:** [`src/evaluation/alignment.py`](file:///d:/Proyecto_tesis/src/evaluation/alignment.py)  
**Impacto:** 🟡 Medio — `calculate_dar()` está implementada pero no se usa en el ablation runner  
**Dificultad:** 🟢 Baja

Integrar `calculate_dar(ksr_history)` en `VariantAggregateSummary` para tener la tasa de aceptación directa como métrica adicional en los resultados.

---

## FASE 4 — Preparación para Defensa y Reproducibilidad
> *Para que cualquier revisor pueda reproducir los resultados*

---

### 4.1 Crear `Makefile` o `run_experiments.ps1`
Script único que:
1. Instala dependencias (`pip install -r requirements.txt`)
2. Descarga modelos (o verifica que existen)
3. Ejecuta el benchmark completo
4. Genera las figuras en `results/figures/`

---

### 4.2 Fijar versiones exactas en `requirements.txt`
Actualmente hay versiones con `>=` que pueden cambiar el comportamiento. Para reproducibilidad de tesis se necesita un `requirements-lock.txt` o usar `pip freeze`.

---

### 4.3 Añadir `conftest.py` con fixtures compartidos
**Archivo:** `tests/conftest.py`  
Centralizar la inicialización de `IPGCAOrchestrator(enable_reranker=False, enable_real_llm=False)` como fixture `pytest` para evitar la recarga de modelos en cada test.

---

## Resumen de Prioridades

| Tarea | Fase | Impacto | Esfuerzo | ¿Afecta resultados? |
|---|---|---|---|---|
| Reset FSM entre turnos | 1 | 🔴 Alto | 🟢 1 línea | Sí |
| Trigger `execute_refund` | 1 | 🔴 Alto | 🟢 Bajo | Sí |
| Docstring Variant G | 1 | 🟡 Medio | 🟢 Trivial | No |
| asyncio.run fix | 1 | 🟡 Medio | 🟢 Bajo | No |
| Ampliar benchmark (8→30) | 2 | 🔴 Alto | 🟡 Medio | **Sí, muy importante** |
| Persistencia PreferenceMemory | 2 | 🟡 Medio | 🟡 Medio | No |
| LoRA training real | 3 | 🔴 Muy Alto | 🔴 Alta | **Sí, contribución central** |
| Figuras automáticas | 3 | 🔴 Muy Alto | 🟡 Medio | Sí |
| KB más rica | 3 | 🟡 Medio | 🟢 Bajo | Sí |
| Demo HITL con rich | 3 | 🟡 Medio | 🟢 Bajo | No |
| DAR en ablation | 3 | 🟢 Bajo | 🟢 Trivial | Sí |
| Reproducibilidad | 4 | 🔴 Alto | 🟡 Medio | No |
