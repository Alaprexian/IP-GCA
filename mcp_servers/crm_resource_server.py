"""
MCP Resources Server (Read-Only CRM Layer).
Exposes customer and ticket data as MCP Resources via stdio transport.
The LLM can read context here but CANNOT mutate any record.
"""

import json
import asyncio
from pathlib import Path
import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

CUSTOMERS_PATH = Path("data/mcp_resources/customers.json")
TICKETS_PATH   = Path("data/mcp_resources/tickets.json")


def _load(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


app = Server("crm-resource-server")

# Compatibility shim for mcp versions where Server uses request_handlers
if not hasattr(app, "list_resources"):
    def _dummy_dec(*args, **kwargs):
        def _wrapper(fn):
            return fn
        return _wrapper
    app.list_resources = _dummy_dec
    app.read_resource = _dummy_dec


@app.list_resources()
async def list_resources() -> list[types.Resource]:
    customers = _load(CUSTOMERS_PATH)
    tickets   = _load(TICKETS_PATH)

    resources: list[types.Resource] = []

    for c in customers:
        resources.append(types.Resource(
            uri=f"mcp://crm/customers/{c['customer_id']}",
            name=f"Customer: {c['name']}",
            description=f"Profile for {c['name']} — tier: {c['tier']}, status: {c['status']}",
            mimeType="application/json"
        ))

    for t in tickets:
        resources.append(types.Resource(
            uri=f"mcp://crm/tickets/{t['ticket_id']}",
            name=f"Ticket: {t['ticket_id']}",
            description=t["subject"],
            mimeType="application/json"
        ))

    return resources


@app.read_resource()
async def read_resource(uri: types.AnyUrl) -> str:
    uri_str = str(uri)

    if uri_str.startswith("mcp://crm/customers/"):
        customer_id = uri_str.split("/")[-1]
        customers = _load(CUSTOMERS_PATH)
        record = next((c for c in customers if c["customer_id"] == customer_id), None)
        if record is None:
            raise ValueError(f"Customer '{customer_id}' not found.")
        return json.dumps(record, ensure_ascii=False, indent=2)

    elif uri_str.startswith("mcp://crm/tickets/"):
        ticket_id = uri_str.split("/")[-1]
        tickets = _load(TICKETS_PATH)
        record = next((t for t in tickets if t["ticket_id"] == ticket_id), None)
        if record is None:
            raise ValueError(f"Ticket '{ticket_id}' not found.")
        return json.dumps(record, ensure_ascii=False, indent=2)

    raise ValueError(f"Unknown resource URI: {uri_str}")


async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
