# IP-GCA: Reporte de Validación Experimental de Adaptación LoRA

**Fecha de ejecución:** 2026-11-07 17:24:34  
**Lote de entrenamiento:** 32 deltas semánticos acumulados en `PreferenceMemory`  
**Conjunto de prueba:** 16 consultas independientes no vistas (Held-Out Test Set)  

---

### Resumen de Resultados Principales

| Métrica | Pre-Adaptación (Base) | In-Context ($k_{pref}$) | Post-Adaptación LoRA | Ganancia Neta ($\Delta$) |
|:---|:---:|:---:|:---:|:---:|
| **Alineación del Operador ($A_{op}$)** | 0.184 $\pm$ 0.059 | 0.604 $\pm$ 0.031 | **0.801 $\pm$ 0.327** | **+0.617** |
| **Keystroke Savings Rate ($KSR$)** | 0.291 $\pm$ 0.048 | 0.553 $\pm$ 0.022 | **0.827 $\pm$ 0.287** | **+0.536** |
| **Direct Acceptance Rate ($DAR$)** | 0.0% | 0.0% | **68.8%** | **+68.8%** |
| **Distancia Media de Edición** | 90.6 caracteres | 103.2 caracteres | **23.2 caracteres** | **-74.3%** |

---

### Hallazgos Clave para la Tesis:
1. **Internalización de Directivas del Operador:** Tras la consolidación del lote de 32 deltas, el modelo base internaliza las excepciones de política (plazos de audio, garantías de hardware, exclusión de daños por líquidos, no reembolsos en planes mensuales), elevando la tasa de aceptación directa ($DAR$) de **0.0%** a **68.8%**.
2. **Reducción Drástica del Esfuerzo Humano:** El esfuerzo de reescritura mecánica ($KSR$) aumenta de **0.291** a **0.827**, reduciendo la cantidad promedio de caracteres editados por el operador en un **74.3%**.
3. **Complementariedad de $k_{pref}$ y LoRA:** El condicionamiento temporal $k_{pref}$ resuelve la adaptación inmediata en el turno $t+1$, mientras que el micro-ajuste LoRA consolida permanentemente las preferencias cuando se acumulan $N \ge 32$ correcciones, liberando la ventana de contexto.
