/**
 * IP-GCA Copilot Desktop Application Frontend Controller
 * Connects the UI to the Python backend via window.pywebview.api
 */

// Global State
const state = {
  isReady: false,
  fsmState: null,
  activeQuery: "",
  activeSGen: "",
  activeIntent: "general_inquiry",
  activeCustomerId: "CUST_001",
  activeToolCall: null,
  presets: [],
  preferences: [],
  turnCounter: 1
};

// Wait for PyWebView injection or initialize fallback
window.addEventListener("pywebviewready", () => {
  console.log("PyWebView Bridge initialized.");
  state.isReady = true;
  initApp();
});

// Fallback for regular browser preview
document.addEventListener("DOMContentLoaded", () => {
  setTimeout(() => {
    if (!state.isReady && !window.pywebview) {
      console.warn("PyWebView not detected, running in standalone/mock mode.");
      state.isReady = true;
      initAppMock();
    }
  }, 1000);
});

async function initApp() {
  try {
    const initData = await window.pywebview.api.get_initial_data();
    hydrateUI(initData);
  } catch (err) {
    console.error("Error loading initial data:", err);
  }
}

function hydrateUI(data) {
  // Brand & Model
  if (data.system_info) {
    document.getElementById("stat-dense-model").textContent = data.system_info.dense_model;
    document.getElementById("stat-decay").textContent = data.system_info.preference_decay;
  }

  // Presets
  state.presets = data.presets || [];
  const presetSelect = document.getElementById("preset-select");
  presetSelect.innerHTML = '<option value="">-- Seleccionar Caso de Prueba Estandarizado --</option>';
  state.presets.forEach((p, idx) => {
    const opt = document.createElement("option");
    opt.value = p.id;
    opt.textContent = `[${p.tag}] ${p.title}`;
    presetSelect.appendChild(opt);
  });

  // FSM Nodes
  renderFSMGraph(data.fsm_state);

  // Preferences Table
  renderPreferences(data.active_preferences || []);
}

// -----------------------------------------------------------------------------
// Presets & Query Selection
// -----------------------------------------------------------------------------
function onPresetSelected() {
  const select = document.getElementById("preset-select");
  const presetId = select.value;
  if (!presetId) return;

  const preset = state.presets.find(p => p.id === presetId);
  if (!preset) return;

  document.getElementById("query-input").value = preset.query;
  state.activeIntent = Object.keys(preset.intent_probs || {})[0] || "general_inquiry";
  state.activeToolCall = preset.tool_call || null;

  // Preset hint
  const hintEl = document.getElementById("preset-hint");
  if (hintEl) {
    hintEl.textContent = `Objetivo: ${preset.description}`;
    hintEl.style.display = "block";
  }

  // Pre-fill human edit box hint
  if (preset.human_edit_hint) {
    document.getElementById("human-editor").setAttribute("data-preset-edit", preset.human_edit_hint);
  }
}

// -----------------------------------------------------------------------------
// Turn Processing
// -----------------------------------------------------------------------------
async function submitTurn() {
  const query = document.getElementById("query-input").value.trim();
  if (!query) {
    alert("Por favor ingresa una consulta para el Copilot.");
    return;
  }

  const btnSend = document.getElementById("btn-send-turn");
  btnSend.disabled = true;
  btnSend.innerHTML = "Evaluando (GPL)...";

  appendChatMessage("user", query);

  const payload = {
    query: query,
    customer_id: state.activeCustomerId,
    intent_probs: { [state.activeIntent]: 0.88, general_inquiry: 0.12 },
    tool_call: state.activeToolCall,
    human_confirmed: false,
    session_id: state.activeCustomerId
  };

  try {
    let resp;
    if (window.pywebview && window.pywebview.api) {
      resp = await window.pywebview.api.process_turn(payload);
    } else {
      resp = mockProcessTurn(payload);
    }

    if (!resp.success) {
      alert("Error del orquestador: " + (resp.error || "Desconocido"));
      return;
    }

    renderTurnResult(resp);
    appendChatMessage("assistant", resp.s_gen);

  } catch (err) {
    console.error("Error submitting turn:", err);
    alert("Error de comunicación: " + err);
  } finally {
    btnSend.disabled = false;
    btnSend.innerHTML = "Enviar al Copilot";
    // Clear one-time tool call
    state.activeToolCall = null;
  }
}

function renderTurnResult(res) {
  state.activeQuery = res.query;
  state.activeSGen = res.s_gen;

  // 1. GPL Controller Banner
  const routeEl = document.getElementById("route-badge");
  routeEl.textContent = res.selected_path;
  routeEl.className = "route-badge " + res.selected_path;

  // Uncertainty
  const uScore = res.uncertainty_score;
  document.getElementById("u-score-val").textContent = uScore.toFixed(3);
  document.getElementById("gauge-fill").style.width = Math.min(100, Math.max(5, uScore * 100)) + "%";

  // Latency
  if (res.latency_profile) {
    document.getElementById("latency-val").textContent = res.latency_profile.total_ms.toFixed(1) + " ms";
  }

  // Rationale
  document.getElementById("routing-rationale").textContent = res.routing_rationale || "";

  // 2. AI Suggestion
  document.getElementById("ai-suggestion-text").textContent = res.s_gen;

  // 3. Pre-fill HITL Editor
  const humanEditor = document.getElementById("human-editor");
  const presetEdit = humanEditor.getAttribute("data-preset-edit");
  humanEditor.value = presetEdit || res.s_gen;
  updateLiveDiff();

  // 4. Passages
  renderPassages(res.context_passages || []);

  // 5. Applied Preferences list
  renderAppliedPrefs(res.applied_preferences || []);

  // 6. Tool / Guardrail alert
  renderToolInvocations(res.tool_invocations || []);

  // 7. Update FSM Graph
  if (res.fsm_state) {
    renderFSMGraph(res.fsm_state);
  }
}

function renderPassages(passages) {
  const container = document.getElementById("passages-container");
  if (!passages.length) {
    container.innerHTML = '<span style="color: var(--text-muted); font-size:11px;">Inferencia paramétrica directa (sin pasajes RAG pesados).</span>';
    return;
  }
  container.innerHTML = passages.map((p, i) => `
    <div style="background: rgba(0,0,0,0.25); border-left: 2px solid var(--accent-cyan); padding: 6px 8px; margin-bottom: 6px; font-size: 11px; border-radius: 4px;">
      <strong style="color: var(--accent-cyan);">Chunk #${i+1}:</strong> ${escapeHtml(p)}
    </div>
  `).join("");
}

function renderAppliedPrefs(prefs) {
  const container = document.getElementById("applied-prefs-container");
  if (!prefs.length) {
    container.innerHTML = '<span style="color: var(--text-muted); font-size:11px;">Ninguna (sin precedentes o peso decaído).</span>';
    return;
  }
  container.innerHTML = prefs.map(p => `
    <span class="brand-badge" style="display:inline-block; margin: 2px; text-transform:none;">${escapeHtml(p)}</span>
  `).join("");
}

function renderToolInvocations(tools) {
  const alertEl = document.getElementById("guardrail-alert");
  if (!tools || !tools.length) {
    alertEl.style.display = "none";
    return;
  }

  const t = tools[0];
  alertEl.style.display = "block";
  if (t.status === "GUARDRAIL_BLOCKED") {
    alertEl.className = "guardrail-alert blocked";
    alertEl.innerHTML = `
      <strong style="display:block; font-size:12px; margin-bottom:2px;">🛡️ GUARDRAIL FSM: MUTACIÓN BLOQUEADA</strong>
      <strong>Herramienta:</strong> ${escapeHtml(t.tool)}<br>
      <strong>Motivo:</strong> ${escapeHtml(t.error || t.message || "Violación de Precedencia")}<br>
      <span style="font-size:10px; color:#FCA5A5;">Aislamiento determinista impidió la ejecución no autorizada en CRM.</span>
    `;
  } else if (t.status === "EXECUTED") {
    alertEl.className = "guardrail-alert executed";
    alertEl.innerHTML = `
      <strong style="display:block; font-size:12px; margin-bottom:2px;">✓ MUTACIÓN EJECUTADA CON ÉXITO</strong>
      <strong>Herramienta:</strong> ${escapeHtml(t.tool)}<br>
      <strong>Resultado:</strong> Transacción auditada por FSM y operador.
    `;
  } else {
    alertEl.className = "guardrail-alert blocked";
    alertEl.innerHTML = `<strong>Estado FSM:</strong> ${escapeHtml(t.status)} - ${escapeHtml(t.message || "")}`;
  }
}

// -----------------------------------------------------------------------------
// Human-in-the-Loop Feedback Actions
// -----------------------------------------------------------------------------
function approveDirectly() {
  const sGen = state.activeSGen;
  if (!sGen) {
    alert("No hay sugerencia activa para aprobar.");
    return;
  }
  // Approve without edits -> s_human == s_gen
  document.getElementById("human-editor").value = sGen;
  saveFeedback(true);
}

async function saveFeedback(isDirect = false) {
  const sHuman = document.getElementById("human-editor").value.trim();
  if (!sHuman) {
    alert("El texto del operador no puede estar vacío.");
    return;
  }

  const btn = document.getElementById("btn-save-hitl");
  btn.disabled = true;
  btn.textContent = "Guardando...";

  const payload = {
    query: state.activeQuery,
    s_gen: state.activeSGen,
    s_human: sHuman,
    intent: state.activeIntent
  };

  try {
    let res;
    if (window.pywebview && window.pywebview.api) {
      res = await window.pywebview.api.record_feedback(payload);
    } else {
      res = mockRecordFeedback(payload);
    }

    if (!res.success) {
      alert("Error al registrar feedback: " + res.error);
      return;
    }

    // Update metrics bar
    document.getElementById("hitl-ksr").textContent = res.delta.ksr.toFixed(3);
    document.getElementById("hitl-aop").textContent = res.a_op.toFixed(3);
    document.getElementById("hitl-cat").textContent = res.delta.category;
    document.getElementById("hitl-dar").textContent = res.is_direct_acceptance ? "100%" : "0%";

    // Render preferences
    renderPreferences(res.active_preferences || []);

    // Flash success toast
    showToast(`✓ Preferencia registrada: ${res.delta.category} (KSR=${res.delta.ksr.toFixed(2)}, A_op=${res.a_op.toFixed(2)})`);

  } catch (err) {
    console.error("Error saving feedback:", err);
    alert("Error: " + err);
  } finally {
    btn.disabled = false;
    btn.textContent = "Guardar Corrección (k_pref)";
  }
}

// Live Diff Calculation
function updateLiveDiff() {
  const sGen = state.activeSGen;
  const sHuman = document.getElementById("human-editor").value;
  const diffBox = document.getElementById("diff-output");

  if (!sGen || !sHuman) {
    diffBox.innerHTML = '<span style="color:var(--text-muted)">Sin cambios detectados.</span>';
    return;
  }

  // Quick word-level diff
  const wordsGen = sGen.split(/\s+/);
  const wordsHuman = sHuman.split(/\s+/);

  let html = "";
  let i = 0, j = 0;
  let additions = 0, deletions = 0;

  // Simple diff renderer
  const maxWords = Math.max(wordsGen.length, wordsHuman.length);
  for (let k = 0; k < maxWords; k++) {
    const wg = wordsGen[k];
    const wh = wordsHuman[k];

    if (wg === wh && wg !== undefined) {
      html += escapeHtml(wg) + " ";
    } else {
      if (wg !== undefined && wh === undefined) {
        html += `<span class="diff-removed">${escapeHtml(wg)}</span> `;
        deletions++;
      } else if (wg === undefined && wh !== undefined) {
        html += `<span class="diff-added">${escapeHtml(wh)}</span> `;
        additions++;
      } else {
        html += `<span class="diff-removed">${escapeHtml(wg)}</span> `;
        html += `<span class="diff-added">${escapeHtml(wh)}</span> `;
        additions++;
        deletions++;
      }
    }
  }

  diffBox.innerHTML = html;

  // Estimate KSR
  const maxLen = Math.max(sGen.length, sHuman.length);
  const dist = levenshtein(sGen, sHuman);
  const estKsr = maxLen > 0 ? (1.0 - dist / maxLen) : 1.0;
  document.getElementById("hitl-ksr").textContent = Math.max(0, estKsr).toFixed(3);
}

// -----------------------------------------------------------------------------
// FSM State Machine Visualization
// -----------------------------------------------------------------------------
function renderFSMGraph(fsmState) {
  if (!fsmState) return;
  state.fsmState = fsmState;

  const current = fsmState.current_state;
  document.getElementById("header-fsm-state").textContent = current;

  const container = document.getElementById("fsm-node-list");
  container.innerHTML = "";

  (fsmState.all_states || []).forEach(node => {
    const div = document.createElement("div");
    let cls = "fsm-node";
    if (node.id === current) {
      cls += " active";
    } else if (fsmState.state_history && fsmState.state_history.includes(node.id)) {
      cls += " passed";
    }
    div.className = cls;
    div.innerHTML = `
      <span>${escapeHtml(node.label)}</span>
      <span style="font-size:9px; color:var(--text-muted);">${escapeHtml(node.desc)}</span>
    `;
    container.appendChild(div);
  });
}

async function resetFSM() {
  try {
    if (window.pywebview && window.pywebview.api) {
      const res = await window.pywebview.api.reset_fsm();
      renderFSMGraph(res.fsm_state);
      showToast("FSM reseteado a IDLE.");
    }
  } catch (err) {
    console.error("Error resetting FSM:", err);
  }
}

async function testIllegalRefund() {
  // Triggers SAMPLE_06: execute_refund with UNVERIFIED_TOKEN from IDLE state
  const payload = {
    trigger: "execute_refund",
    params: { ticket_id: "TICK_001", approval_token: "UNVERIFIED_TOKEN", amount: 250.0 },
    is_human_verified: false
  };

  try {
    let res;
    if (window.pywebview && window.pywebview.api) {
      res = await window.pywebview.api.execute_fsm_transition(payload);
    } else {
      res = { success: false, message: "Transición ilegal: FSM bloqueó execute_refund desde IDLE" };
    }

    renderToolInvocations([{
      status: res.success ? "EXECUTED" : "GUARDRAIL_BLOCKED",
      tool: payload.trigger,
      error: res.message
    }]);

    if (res.fsm_state) {
      renderFSMGraph(res.fsm_state);
    }
  } catch (err) {
    console.error("Error testing refund:", err);
  }
}

// -----------------------------------------------------------------------------
// Preferences Table & LoRA Sync
// -----------------------------------------------------------------------------
function renderPreferences(prefs) {
  state.preferences = prefs || [];
  document.getElementById("stat-kpref-count").textContent = state.preferences.length;

  const tbody = document.getElementById("pref-tbody");
  if (!prefs.length) {
    tbody.innerHTML = '<tr><td colspan="4" style="color:var(--text-muted); text-align:center;">No hay deltas registrados en la sesión.</td></tr>';
    return;
  }

  tbody.innerHTML = prefs.map(p => `
    <tr>
      <td><strong>${escapeHtml(p.key_id)}</strong></td>
      <td style="max-width: 140px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${escapeHtml(p.diff_summary)}</td>
      <td><span class="pref-weight-pill">w=${p.current_weight.toFixed(2)}</span></td>
      <td>${p.is_synced_lora ? '<span style="color:var(--accent-emerald)">✓ Sync</span>' : '<span style="color:var(--accent-amber)">Pending</span>'}</td>
    </tr>
  `).join("");
}

async function triggerLoraSync() {
  const btn = document.getElementById("btn-lora-sync");
  btn.disabled = true;
  btn.textContent = "Sincronizando...";

  try {
    if (window.pywebview && window.pywebview.api) {
      const res = await window.pywebview.api.trigger_lora_sync();
      if (res.success) {
        showToast("✓ Micro-entrenamiento LoRA completado (Adaptador sincronizado).");
        renderPreferences(res.active_preferences || []);
      } else {
        alert("Aviso: " + res.message);
      }
    }
  } catch (err) {
    console.error("Error LoRA sync:", err);
    alert("Error: " + err);
  } finally {
    btn.disabled = false;
    btn.textContent = "Sincronizar LoRA";
  }
}

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------
function appendChatMessage(sender, text) {
  const stream = document.getElementById("chat-stream");
  const div = document.createElement("div");
  div.className = "chat-bubble " + sender;
  div.textContent = text;
  stream.appendChild(div);
  stream.scrollTop = stream.scrollHeight;
}

function showToast(msg) {
  const toast = document.getElementById("toast");
  toast.textContent = msg;
  toast.style.display = "block";
  setTimeout(() => { toast.style.display = "none"; }, 3500);
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function levenshtein(a, b) {
  const matrix = [];
  for (let i = 0; i <= b.length; i++) { matrix[i] = [i]; }
  for (let j = 0; j <= a.length; j++) { matrix[0][j] = j; }
  for (let i = 1; i <= b.length; i++) {
    for (let j = 1; j <= a.length; j++) {
      if (b.charAt(i - 1) === a.charAt(j - 1)) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1,
          matrix[i][j - 1] + 1,
          matrix[i - 1][j] + 1
        );
      }
    }
  }
  return matrix[b.length][a.length];
}

// Mock fallback for browser dev
function initAppMock() {
  hydrateUI({
    system_info: { dense_model: "MiniLM-L6 (Mock)", preference_decay: "γ = 0.95" },
    fsm_state: {
      current_state: "IDLE",
      all_states: [
        { id: "IDLE", label: "IDLE", desc: "Inicio" },
        { id: "AUTHENTICATED", label: "AUTHENTICATED", desc: "Validado" },
        { id: "OPERATOR_APPROVED", label: "OPERATOR_APPROVED", desc: "Aprobado" }
      ]
    },
    presets: [
      { id: "P1", tag: "RAG", title: "Consulta Devolución", query: "¿Plazo de devolución?", intent_probs: { refund: 0.9 } }
    ]
  });
}
