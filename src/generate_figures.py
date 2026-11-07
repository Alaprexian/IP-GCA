"""
Automated Research Figures and LaTeX Table Generator for IP-GCA Thesis Defense.
Produces publication-grade vector/raster figures and LaTeX tables comparing
Ablation Variants (A through H) across latency, alignment, faithfulness, and safety.
"""

import sys
import io
import json
import math
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Ensure UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless figure generation
import matplotlib.pyplot as plt
import numpy as np


def load_ablation_results(results_path: str = "results/ablation_results.json") -> Dict[str, Any]:
    """Loads benchmark ablation results from JSON."""
    p = Path(results_path)
    if not p.exists():
        raise FileNotFoundError(f"Resultados no encontrados en '{results_path}'. Ejecuta run_ablation_benchmark.py primero.")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_latex_table(data: Dict[str, Any], output_path: str = "results/table_ablation_results.tex") -> str:
    """
    Generates a publication-ready LaTeX booktabs table formatted for thesis inclusion.
    """
    variant_names = {
        "A_NAIVE_RAG_DENSE": ("Var. A", "Naive RAG (Dense only)"),
        "B_DENSE_PLUS_SPARSE": ("Var. B", "Dense + SPLADE"),
        "C_DENSE_PLUS_RERANKER": ("Var. C", "Dense + Cross-Encoder"),
        "D_HYBRID_RAG_FIXED": ("Var. D", "Hybrid RAG (Fixed pipeline)"),
        "E_FULL_IP_GCA": ("Var. E*", "\\textbf{Full IP-GCA (Propuesto)}"),
        "F_NO_PREFERENCE_KEYS": ("Var. F", "E w/o Preference Keys ($k_{pref}$)"),
        "G_NO_DYNAMIC_ROUTER": ("Var. G", "E w/o GPL Controller (Fixed heavy)"),
        "H_RANDOM_NOISE_CONTEXT": ("Var. H", "E w/ Randomized Context")
    }

    lines = [
        "% Tabla generada automáticamente para la defensa de tesis IP-GCA",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Resultados comparativos del protocolo experimental de ablación (Variantes A--H) sobre el benchmark estandarizado.}",
        "\\label{tab:ablation_matrix}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccccccc}",
        "\\toprule",
        "\\textbf{ID} & \\textbf{Descripción Arquitectural} & \\textbf{Latencia (ms)} & \\textbf{P95 (ms)} & $\\mathbf{F_{faith}}$ & \\textbf{Alucinac. (\\%)} & $\\mathbf{A_{op}}$ & \\textbf{KSR} & \\textbf{DAR (\\%)} & \\textbf{FSM Viol. (\\%)} \\\\",
        "\\midrule"
    ]

    for key, (short_id, desc) in variant_names.items():
        if key not in data:
            continue
        v = data[key]
        is_best = (key == "E_FULL_IP_GCA")
        lat = f"\\textbf{{{v['mean_latency_ms']:.1f}}}" if is_best else f"{v['mean_latency_ms']:.1f}"
        p95 = f"{v.get('p95_latency_ms', 0.0):.1f}"
        faith = f"{v.get('mean_faithfulness', 0.0):.3f}"
        halluc = f"{v.get('mean_hallucination_rate', 0.0)*100:.1f}\\%"
        aop = f"\\textbf{{{v.get('mean_operator_alignment', 0.0):.3f}}}" if is_best else f"{v.get('mean_operator_alignment', 0.0):.3f}"
        ksr = f"{v.get('mean_ksr', 0.0):.3f}"
        dar = f"{v.get('direct_acceptance_rate', 0.0)*100:.1f}\\%"
        fsm_viol = f"\\textbf{{{v.get('fsm_safety_violation_rate', 0.0)*100:.1f}\\%}}" if v.get('fsm_safety_violation_rate', 0.0) == 0 else f"{v.get('fsm_safety_violation_rate', 0.0)*100:.1f}\\%"

        row = f"{short_id} & {desc} & {lat} & {p95} & {faith} & {halluc} & {aop} & {ksr} & {dar} & {fsm_viol} \\\\"
        lines.append(row)

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\vspace{2mm}",
        "\\raggedright \\footnotesize{* Variante E corresponde a la arquitectura propuesta con GPL-Controller, Edit-Delta Bridge y FSM Guardrail.}",
        "\\end{table*}"
    ])

    tex_content = "\n".join(lines)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(tex_content)
    return tex_content


def plot_latency_comparison(data: Dict[str, Any], output_path: str = "results/figures/fig1_latency_comparison.png") -> None:
    """Bar chart comparing mean and P95 latency across variants."""
    labels = ["A (Naive)", "B (+Sparse)", "C (+Rerank)", "D (Hybrid)", "E* (IP-GCA)", "F (No k_pref)", "G (No GPL)", "H (Random)"]
    keys = [
        "A_NAIVE_RAG_DENSE", "B_DENSE_PLUS_SPARSE", "C_DENSE_PLUS_RERANKER", "D_HYBRID_RAG_FIXED",
        "E_FULL_IP_GCA", "F_NO_PREFERENCE_KEYS", "G_NO_DYNAMIC_ROUTER", "H_RANDOM_NOISE_CONTEXT"
    ]

    mean_lats = [data.get(k, {}).get("mean_latency_ms", 0) for k in keys]
    p95_lats = [data.get(k, {}).get("p95_latency_ms", 0) for k in keys]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    # Highlight proposed Variant E
    colors_mean = ['#2b5c8f' if i != 4 else '#1b9e77' for i in range(len(labels))]
    colors_p95 = ['#75aadb' if i != 4 else '#66c2a5' for i in range(len(labels))]

    bars1 = ax.bar(x - width/2, mean_lats, width, label='Latencia Media', color=colors_mean, edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width/2, p95_lats, width, label='P95 Latencia', color=colors_p95, edgecolor='black', linewidth=0.8)

    # Latency budget threshold line
    ax.axhline(1200, color='#d95f02', linestyle='--', linewidth=1.5, label='Presupuesto SLA (1200 ms)')

    ax.set_ylabel('Latencia de Inferencia (ms)', fontsize=11, fontweight='bold')
    ax.set_title('Comparativa de Latencia por Variante Arquitectural (Protocolo A--H)', fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    ax.legend(frameon=True, facecolor='white', edgecolor='none')
    ax.grid(axis='y', linestyle=':', alpha=0.6)

    # Label values on bars
    for bar in bars1:
        yval = bar.get_height()
        if yval > 0:
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.0f}", ha='center', va='bottom', fontsize=7.5)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()


def plot_radar_metrics(data: Dict[str, Any], output_path: str = "results/figures/fig2_metrics_radar.png") -> None:
    """Radar chart comparing key architectural dimensions between Naive RAG, Hybrid RAG, and IP-GCA."""
    categories = [
        'Alineación Op. (A_op)',
        'Ahorro Pulsaciones (KSR)',
        'Aceptación Directa (DAR)',
        'Seguridad FSM (1 - Viol)',
        'Eficiencia Latencia (1/L)',
        'Fidelidad Grounding (F_faith)'
    ]
    N = len(categories)

    def extract_vector(var_key: str) -> List[float]:
        v = data.get(var_key, {})
        aop = min(1.0, v.get("mean_operator_alignment", 0.0) / 0.8)
        ksr = min(1.0, v.get("mean_ksr", 0.0) / 0.8)
        dar = v.get("direct_acceptance_rate", 0.0)
        safety = 1.0 - v.get("fsm_safety_violation_rate", 0.0)
        # Latency efficiency: 100ms -> 1.0, 1000ms -> 0.1
        lat_eff = max(0.05, min(1.0, 100.0 / max(1.0, v.get("mean_latency_ms", 100.0))))
        faith = v.get("mean_faithfulness", 0.0)
        return [aop, ksr, dar, safety, lat_eff, faith]

    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True), dpi=300)

    # Variants to compare
    to_compare = [
        ("A_NAIVE_RAG_DENSE", "Naive RAG (Var A)", '#d95f02', '--'),
        ("D_HYBRID_RAG_FIXED", "Hybrid RAG Fijo (Var D)", '#7570b3', '-.'),
        ("E_FULL_IP_GCA", "Full IP-GCA (Var E*)", '#1b9e77', '-')
    ]

    for key, label, color, line_style in to_compare:
        values = extract_vector(key)
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle=line_style, label=label, color=color)
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9, fontweight='semibold')
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7.5, color='gray')
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), frameon=True)
    ax.set_title("Comparativa Multidimensional de Desempeño", fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()


def plot_preference_decay(output_path: str = "results/figures/fig3_preference_decay.png") -> None:
    """Plots temporal decay curves of preference keys w_i(t) = gamma^(delta_t)."""
    turns = np.arange(0, 45, 1)
    gammas = [0.90, 0.95, 0.98]
    colors = ['#e41a1c', '#377eb8', '#4daf4a']

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

    for g, c in zip(gammas, colors):
        weights = [math.pow(g, t) for t in turns]
        ax.plot(turns, weights, label=f"$\\gamma = {g}$ ({'Rápido' if g == 0.90 else 'Por defecto' if g == 0.95 else 'Persistente'})",
                color=c, linewidth=2.0)

    # Eviction threshold
    ax.axhline(0.05, color='black', linestyle=':', linewidth=1.2, label='Umbral de desalojo ($\\tau_{evict} = 0.05$)')

    ax.set_xlabel('Turnos transcurridos desde la corrección del operador ($\\Delta t$)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Peso asignado al Delta de Preferencia ($w_i$)', fontsize=10, fontweight='bold')
    ax.set_title('Dinámica del Decaimiento Temporal Exponencial de Preferencias ($k_{pref}$)', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 42)
    ax.legend(frameon=True, facecolor='white')

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()


def generate_all_artifacts(results_path: str = "results/ablation_results.json"):
    """Entrypoint to generate all thesis evaluation artifacts."""
    print(f"\n[1/4] Cargando resultados experimentales desde {results_path}...")
    data = load_ablation_results(results_path)

    print("[2/4] Generando tabla LaTeX formal (table_ablation_results.tex)...")
    tex = generate_latex_table(data, "results/table_ablation_results.tex")
    print("  -> Guardado en results/table_ablation_results.tex")

    print("[3/4] Generando figuras de alta resolución (matplotlib)...")
    plot_latency_comparison(data, "results/figures/fig1_latency_comparison.png")
    print("  -> Guardado: results/figures/fig1_latency_comparison.png")

    plot_radar_metrics(data, "results/figures/fig2_metrics_radar.png")
    print("  -> Guardado: results/figures/fig2_metrics_radar.png")

    plot_preference_decay("results/figures/fig3_preference_decay.png")
    print("  -> Guardado: results/figures/fig3_preference_decay.png")

    print("\n[4/4] Todos los artefactos de la tesis fueron generados con éxito en results/figures/ y results/.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de figuras y tablas para tesis IP-GCA.")
    parser.add_argument("--results", default="results/ablation_results.json", help="Ruta a ablation_results.json")
    args = parser.parse_args()

    generate_all_artifacts(args.results)
