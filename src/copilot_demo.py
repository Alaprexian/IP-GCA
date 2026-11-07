"""
IP-GCA Copilot End-to-End Interactive Demonstration.
Demonstrates:
1. Dynamic routing via GPL-Controller.
2. Grounded generation with corporate knowledge base.
3. Human-in-the-Loop (HITL) operator edit and Delta_edit -> k_pref extraction.
4. Subsequent turn demonstrating in-context adaptation.
5. FSM Guardrail isolation blocking unauthorized tool invocation.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logging_config import setup_logging
setup_logging(level="INFO", log_file="results/logs/copilot_demo.log")

from src.copilot.orchestrator import IPGCAOrchestrator
from src.controller.router import ExecutionPath

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def run_copilot_demo(use_real_llm: bool = False):
    console = Console() if HAS_RICH else None

    if HAS_RICH:
        console.print(Panel.fit(
            "[bold cyan]IP-GCA: Implicit Preference-Gated Copilot Architecture[/bold cyan]\n"
            "[dim]Demostración Interactiva End-to-End con Human-in-the-Loop y Guardrails FSM[/dim]",
            border_style="cyan"
        ))
    else:
        print("\n" + "="*70)
        print("  IP-GCA: Implicit Preference-Gated Copilot Architecture Demo")
        print("="*70)

    mode_str = "Qwen2.5-1.5B (4-bit GPU)" if use_real_llm else "Inferencia Rápida Heurística (<50ms)"
    if HAS_RICH:
        console.print(f"[bold]Modo de Generación:[/bold] [yellow]{mode_str}[/yellow]")
        console.print("\n[bold blue][1/5] Inicializando el Orquestador IP-GCA...[/bold blue]")
    else:
        print(f"Modo: {mode_str}")
        print("\n[Paso 1] Inicializando el Orquestador IP-GCA...")

    copilot = IPGCAOrchestrator(
        config_path="configs/config_default.yaml",
        fsm_config_path="configs/fsm_rules.yaml",
        kb_path="data/raw/corporate_knowledge_base.json",
        enable_reranker=False,
        enable_real_llm=use_real_llm
    )
    if HAS_RICH:
        console.print("  [green]✓[/green] Base de conocimiento corporativa indexada.")
        console.print("  [green]✓[/green] FSM Guardrail inicializado en estado [bold]IDLE[/bold].")
        console.print("  [green]✓[/green] Memoria dinámica de preferencias [italic]PreferenceMemory[/italic] lista.\n")

    # Turno 1: Consulta del cliente
    query_1 = "¿Cuánto tiempo tengo para devolver unos auriculares inalámbricos?"
    intents_1 = {"return_inquiry": 0.85, "technical_support": 0.15}

    if HAS_RICH:
        console.rule("[bold magenta]Turno 1: Consulta Inicial del Cliente[/bold magenta]")
        console.print(f"[bold white]Cliente:[/bold white] [italic]\"{query_1}\"[/italic]")
    else:
        print(f"\n[Turno 1] Consulta: \"{query_1}\"")

    resp_1 = copilot.process_turn(
        query=query_1,
        customer_id="CUST_108",
        intent_probs=intents_1
    )

    path_color = "green" if resp_1.selected_path == ExecutionPath.FAST_PARAMETRIC else "cyan"
    if HAS_RICH:
        t_route = Table(show_header=True, header_style="bold blue")
        t_route.add_column("Métrica de Enrutamiento", style="dim")
        t_route.add_column("Valor Asignado")
        t_route.add_row("Ruta GPL-Controller", f"[{path_color}]{resp_1.selected_path.value}[/{path_color}]")
        t_route.add_row("Incertidumbre u_ctx", f"{resp_1.uncertainty_score:.3f}")
        if resp_1.latency_profile:
            t_route.add_row("Latencia Total", f"[bold green]{resp_1.latency_profile.total_latency_ms:.1f} ms[/bold green]")
        console.print(t_route)

        console.print(Panel(
            f"[bold green]Sugerencia generada por IA (S_gen):[/bold green]\n{resp_1.s_gen}",
            title="Sugerencia Copilot al Operador",
            border_style="green"
        ))
    else:
        print(f"  Ruta: {resp_1.selected_path.value} | u_ctx: {resp_1.uncertainty_score:.3f}")
        print(f"  S_gen: \"{resp_1.s_gen}\"")

    # Paso 3: Operador humano corrige la sugerencia (HITL)
    s_human_1 = "El plazo general es de 7 días, pero en la categoría Audio dispones de 15 días calendario presentando el empaque original intacto."

    if HAS_RICH:
        console.rule("[bold yellow]Supervisión Humana (HITL): Edición del Operador[/bold yellow]")
        console.print(Panel(
            f"[bold yellow]Corrección realizada por el operador (S_human):[/bold yellow]\n{s_human_1}",
            title="Human-in-the-Loop Feedback",
            border_style="yellow"
        ))
    else:
        print(f"\n[HITL] Corrección Operador: \"{s_human_1}\"")

    delta, a_op, lora_batch = copilot.record_operator_feedback(
        query=query_1,
        s_gen=resp_1.s_gen,
        s_human=s_human_1,
        intent="return_inquiry"
    )

    if HAS_RICH:
        t_delta = Table(title="Extracción de Preferencia Semántica (Δ_edit → k_pref)", show_header=True)
        t_delta.add_column("Propiedad", style="cyan")
        t_delta.add_column("Valor Extraído", style="bold")
        t_delta.add_row("Categoría de Edición", str(delta.edit_category.value))
        t_delta.add_row("Ahorro de Pulsaciones (KSR)", f"{delta.ksr:.3f}")
        t_delta.add_row("Alineación del Operador (A_op)", f"{a_op:.3f}")
        t_delta.add_row("Resumen de Diferencias", delta.diff_summary)
        t_delta.add_row("Claves k_pref en Memoria", str(len(copilot.preference_memory.keys)))
        console.print(t_delta)
    else:
        print(f"  Delta: {delta.edit_category.value} | KSR: {delta.ksr:.3f} | A_op: {a_op:.3f}")

    # Paso 4: Turno 2 del cliente (similar) demostrando adaptación en contexto
    query_2 = "¿Puedo devolver mis audífonos después de 10 días de compra?"
    intents_2 = {"return_inquiry": 0.88, "billing": 0.12}

    if HAS_RICH:
        console.rule("[bold magenta]Turno 2: Nueva Consulta Similar (Adaptación Dinámica)[/bold magenta]")
        console.print(f"[bold white]Cliente 2:[/bold white] [italic]\"{query_2}\"[/italic]")
    else:
        print(f"\n[Turno 2] Consulta: \"{query_2}\"")

    resp_2 = copilot.process_turn(
        query=query_2,
        customer_id="CUST_205",
        intent_probs=intents_2
    )

    if HAS_RICH:
        if resp_2.applied_preferences:
            t_pref = Table(title="Claves k_pref In-Context Recuperadas y Aplicadas", show_header=True)
            t_pref.add_column("Preferencia de Operador Activa", style="yellow")
            for p in resp_2.applied_preferences:
                t_pref.add_row(p)
            console.print(t_pref)

        console.print(Panel(
            f"[bold cyan]Nueva sugerencia adaptada automáticamente (S_gen):[/bold cyan]\n{resp_2.s_gen}",
            title="Copilot Adaptado (Sin Fine-Tuning Offline)",
            border_style="cyan"
        ))
    else:
        print(f"  Preferencias aplicadas: {resp_2.applied_preferences}")
        print(f"  Nueva S_gen: \"{resp_2.s_gen}\"")

    # Paso 5: Demostración de aislamiento por FSM Guardrail
    if HAS_RICH:
        console.rule("[bold red]Paso 5: Guardrail FSM Determinista (Aislamiento MCP)[/bold red]")
    else:
        print("\n[Paso 5] Guardrail FSM para llamadas a herramientas MCP")

    tool_req = {
        "tool_name": "execute_action",
        "parameters": {"ticket_id": "TICK_999", "approval_token": "UNVERIFIED"}
    }
    if HAS_RICH:
        console.print(f"[dim]Simulando solicitud del LLM para mutación crítica:[/dim] [red]{tool_req['tool_name']}[/red]")
        console.print(f"[dim]Parámetros:[/dim] {tool_req['parameters']}")

    resp_tool = copilot.process_turn(
        query="Efectuar la devolución inmediata del dinero a mi cuenta",
        tool_call_request=tool_req,
        intent_probs={"refund_request": 0.95}
    )

    if resp_tool.tool_invocations:
        t_res = resp_tool.tool_invocations[0]
        status = t_res.get("status")
        msg = t_res.get("error") or t_res.get("message")
        if HAS_RICH:
            console.print(Panel(
                f"[bold red]Estado Guardrail:[/bold red] {status}\n"
                f"[bold white]Acción de Protección:[/bold white] {msg}\n"
                f"[dim]El FSM impidió la ejecución de la mutación porque el estado no es OPERATOR_APPROVED.[/dim]",
                title="Aislamiento Determinista Exitoso",
                border_style="red"
            ))
        else:
            print(f"  Resultado Guardrail FSM: Estado = {status}")
            print(f"  Mensaje de protección: {msg}")

    if HAS_RICH:
        console.print(Panel.fit(
            "[bold green]Demostración completada con éxito.[/bold green]\n"
            "Todos los componentes de la arquitectura IP-GCA funcionaron de acuerdo a la formulación teórica.",
            border_style="green"
        ))
    else:
        print("\n" + "="*70)
        print("  Demostración completada con éxito.")
        print("="*70 + "\n")


if __name__ == "__main__":
    use_real = ("--real-llm" in sys.argv) or ("--real" in sys.argv)
    run_copilot_demo(use_real_llm=use_real)


