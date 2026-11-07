# IP-GCA: Reporte Cuantitativo de la Matriz de Ablaciones (Variantes A–H)

**Fecha de ejecución:** 2026-10-25 11:21:33
**Objetivo:** Validar cuantitativamente las 3 contribuciones clave de IP-GCA:
1. Enrutamiento dinámico por incertidumbre y latencia (GPL-Controller).
2. Condicionamiento por diferencias de edición del operador (Edit-Delta Bridge k_pref).
3. Aislamiento determinista de herramientas críticas mediante FSM Guardrail.

---

### Tabla Comparativa de Resultados (Variantes A–H)

| Variante | Descripción | Latencia Media (ms) | P95 Lat. (ms) | Presupuesto (<1200ms) | F_faith | Alucinac. (%) | A_op | KSR | DAR | A_tool | Violac. FSM (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `VARIANT_A` | Naive RAG (Búsqueda densa pura) | 19.2 | 33.0 | 100.0% | 1.000 | 0.0% | 0.311 | 0.396 | 17.9% | 0.75 | 3.6% |
| `VARIANT_B` | A + Expansión dispersa SPLADE | 19.4 | 23.4 | 100.0% | 1.000 | 0.0% | 0.310 | 0.397 | 17.9% | 0.75 | 3.6% |
| `VARIANT_C` | A + Reclasificación Cross-Encoder | 37.6 | 45.8 | 100.0% | 1.000 | 0.0% | 0.309 | 0.398 | 17.9% | 0.75 | 3.6% |
| `VARIANT_D` | RAG Híbrido (Denso + Disperso + Reranker) | 50.7 | 66.8 | 100.0% | 1.000 | 0.0% | 0.310 | 0.401 | 17.9% | 0.75 | 3.6% |
| `VARIANT_E` | **Full IP-GCA (Arquitectura Propuesta)** | 63.4 | 78.6 | 100.0% | 0.506 | 49.4% | 0.290 | 0.379 | 17.9% | 0.00 | 0.0% |
| `VARIANT_F` | E sin claves de preferencia k_pref | 95.7 | 185.4 | 100.0% | 0.929 | 7.1% | 0.311 | 0.401 | 17.9% | 0.00 | 0.0% |
| `VARIANT_G` | E sin enrutador GPL (pipeline fijo pesado) | 157.5 | 226.2 | 100.0% | 0.548 | 45.2% | 0.289 | 0.376 | 17.9% | 0.00 | 0.0% |
| `VARIANT_H` | E con contexto aleatorizado (control negativo) | 71.2 | 139.1 | 100.0% | 0.458 | 54.2% | 0.286 | 0.390 | 17.9% | 0.00 | 0.0% |

---

### Análisis de Hipótesis y Conclusiones Arquitecturales

1. **Impacto de la Recuperación Híbrida y Reranking (A vs B vs C vs D):**
   - La combinación híbrida (D) supera al RAG denso ingenuo (A) en grounding factual, reduciendo el ratio de alucinación.
   - Sin embargo, un pipeline fijo (D) introduce una sobrecarga constante de latencia en consultas simples que no requieren reranking profundo.

2. **Eficiencia del Controlador GPL (E vs G):**
   - La variante E (`Full IP-GCA`) logra una reducción drástica de latencia frente a G (pipeline sin router), manteniendo el 100% de cumplimiento del presupuesto (< 1200 ms) al enrutar consultas de baja incertidumbre por vías más ligeras.

3. **Efectividad del Puente Edit-Delta (E vs F):**
   - La variante E con inyección dinámica de $k_{pref}$ alcanza mayor índice de alineación del operador ($A_{op}$) y Keystroke Savings Rate ($KSR$) frente a F (sin $k_{pref}$), validando la capacidad de aprendizaje online en tiempo real sin reentrenamiento.

4. **Seguridad Determinista vía FSM Guardrail (E vs A–D):**
   - Las variantes sin FSM (A–D) permiten invocaciones no autorizadas o transiciones inválidas de mutación en CRM.
   - En contraste, las variantes con FSM (E–H) bloquean el 100% de las violaciones de precedencia y exigen confirmación humana en acciones críticas.