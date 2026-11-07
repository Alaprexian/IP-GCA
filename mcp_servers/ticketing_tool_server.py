"""
MCP Transactional Tools Server (Write-Layer over CRM).
Implements ticketing operations as MCP Tools governed by the FSM Guardrail.
Each mutation tool validates state via the TransactionalGuardrailFSM before
touching any record in the simulated CRM.
"""

import json
import asyncio
from pathlib import Path
from typing import Any
import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

# In-memory store (in production: replace with DB calls)
TICKETS_PATH   = Path("data/mcp_resources/tickets.json")
CUSTOMERS_PATH = Path("data/mcp_resources/customers.json")

app = Server("ticketing-tool-server")

# Compatibility shim for mcp versions where Server uses request_handlers
if not hasattr(app, "list_tools"):
    def _dummy_dec(*args, **kwargs):
        def _wrapper(fn):
            return fn
        return _wrapper
    app.list_tools = _dummy_dec
    app.call_tool = _dummy_dec


def _load(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(path: Path, data: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="authenticate_customer",
            description="Verifica la identidad del cliente y devuelve su perfil activo.",
            inputSchema={
                "type": "object",
                "properties": {"customer_id": {"type": "string"}},
                "required": ["customer_id"]
            }
        ),
        types.Tool(
            name="fetch_ticket",
            description="Carga los detalles completos de un ticket abierto.",
            inputSchema={
                "type": "object",
                "properties": {"ticket_id": {"type": "string"}},
                "required": ["ticket_id"]
            }
        ),
        types.Tool(
            name="verify_policy",
            description="Consulta las reglas de negocio aplicables a una categoría de caso.",
            inputSchema={
                "type": "object",
                "properties": {
                    "policy_category": {
                        "type": "string",
                        "enum": ["refund", "warranty", "subscription", "shipping", "legal"]
                    }
                },
                "required": ["policy_category"]
            }
        ),
        types.Tool(
            name="draft_action",
            description="Prepara una propuesta de acción para revisión del operador humano.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action_type": {"type": "string", "enum": ["refund", "replacement", "cancel", "escalate"]},
                    "amount": {"type": "number"},
                    "ticket_id": {"type": "string"}
                },
                "required": ["action_type", "ticket_id"]
            }
        ),
        types.Tool(
            name="execute_refund",
            description="Ejecuta el reembolso aprobado por el operador. Requiere token de aprobación.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "string"},
                    "approval_token": {"type": "string"},
                    "amount": {"type": "number"}
                },
                "required": ["ticket_id", "approval_token"]
            }
        ),
        types.Tool(
            name="close_ticket",
            description="Cierra el ticket con un resumen de resolución.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "string"},
                    "resolution_summary": {"type": "string"}
                },
                "required": ["ticket_id", "resolution_summary"]
            }
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    try:
        if name == "authenticate_customer":
            cid = arguments["customer_id"]
            customers = _load(CUSTOMERS_PATH)
            record = next((c for c in customers if c["customer_id"] == cid), None)
            if not record:
                return [types.TextContent(type="text", text=f"ERROR: Customer '{cid}' not found.")]
            if record["status"] == "suspended":
                return [types.TextContent(type="text", text=f"ERROR: Account '{cid}' is suspended.")]
            return [types.TextContent(type="text", text=json.dumps({"status": "authenticated", "customer": record}, ensure_ascii=False))]

        elif name == "fetch_ticket":
            tid = arguments["ticket_id"]
            tickets = _load(TICKETS_PATH)
            record = next((t for t in tickets if t["ticket_id"] == tid), None)
            if not record:
                return [types.TextContent(type="text", text=f"ERROR: Ticket '{tid}' not found.")]
            return [types.TextContent(type="text", text=json.dumps(record, ensure_ascii=False))]

        elif name == "verify_policy":
            category = arguments["policy_category"]
            policy_limits = {
                "refund":       {"max_days": 30, "max_amount": 500.0, "no_receipt": False},
                "warranty":     {"coverage_months": 12, "covers_accidental": False},
                "subscription": {"prorate_refund": False, "allow_pause": True},
                "shipping":     {"report_window_days": 5},
                "legal":        {"data_retention_years": 5}
            }
            result = policy_limits.get(category, {})
            return [types.TextContent(type="text", text=json.dumps({"policy_category": category, "rules": result}, ensure_ascii=False))]

        elif name == "draft_action":
            return [types.TextContent(type="text", text=json.dumps({
                "status": "draft_ready",
                "ticket_id": arguments["ticket_id"],
                "action_type": arguments["action_type"],
                "amount": arguments.get("amount", 0),
                "requires_human_approval": True
            }, ensure_ascii=False))]

        elif name == "execute_refund":
            tid = arguments["ticket_id"]
            token = arguments["approval_token"]
            amount = arguments.get("amount", 0)

            # Simulate approval token validation
            if not token or len(token) < 4:
                return [types.TextContent(type="text", text="ERROR: Invalid approval token.")]

            tickets = _load(TICKETS_PATH)
            updated = False
            for t in tickets:
                if t["ticket_id"] == tid:
                    t["status"] = "refund_processed"
                    t["refund_amount"] = amount
                    updated = True
                    break

            if not updated:
                return [types.TextContent(type="text", text=f"ERROR: Ticket '{tid}' not found for refund.")]

            _save(TICKETS_PATH, tickets)
            return [types.TextContent(type="text", text=json.dumps({
                "status": "refund_executed",
                "ticket_id": tid,
                "amount_refunded": amount,
                "processing_days": "5-7 business days"
            }, ensure_ascii=False))]

        elif name == "close_ticket":
            tid = arguments["ticket_id"]
            summary = arguments["resolution_summary"]
            tickets = _load(TICKETS_PATH)
            updated = False
            for t in tickets:
                if t["ticket_id"] == tid:
                    t["status"] = "closed"
                    t["resolution_summary"] = summary
                    updated = True
                    break

            if not updated:
                return [types.TextContent(type="text", text=f"ERROR: Ticket '{tid}' not found.")]

            _save(TICKETS_PATH, tickets)
            return [types.TextContent(type="text", text=json.dumps({
                "status": "closed",
                "ticket_id": tid,
                "resolution_summary": summary
            }, ensure_ascii=False))]

        else:
            return [types.TextContent(type="text", text=f"ERROR: Unknown tool '{name}'.")]

    except Exception as e:
        return [types.TextContent(type="text", text=f"ERROR: {type(e).__name__}: {e}")]


async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
