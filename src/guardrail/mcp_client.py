"""
In-process MCP client adapter for testing and orchestration.
Wraps MCP tool calls directly without requiring a running subprocess server,
enabling integration tests and the main orchestrator to invoke tools programmatically.
"""

import json
from typing import Any, Dict, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_servers.ticketing_tool_server import call_tool as _server_call_tool


class MCPClient:
    """
    Thin synchronous wrapper around MCP tool calls for use within the IP-GCA orchestrator.
    In production, replace the inner call with a proper MCP stdio/HTTP client.
    """

    def __init__(self):
        self._call_log: list = []

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Invokes an MCP tool and returns the parsed JSON result.
        Logs each call for FSM and audit traceability.
        """
        raw_result = await _server_call_tool(tool_name, arguments)
        content = raw_result[0].text if raw_result else ""

        self._call_log.append({
            "tool": tool_name,
            "arguments": arguments,
            "raw_response": content
        })

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"error": content}

    def call_tool_sync(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for use in non-async contexts.
        Handles the case where an event loop is already running
        (e.g. Jupyter notebooks, FastAPI handlers) by delegating to a
        background thread instead of calling asyncio.run() directly.
        """
        import asyncio
        import concurrent.futures

        try:
            asyncio.get_running_loop()
            # An event loop is already running — delegate to a separate thread
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    asyncio.run, self.call_tool(tool_name, arguments)
                )
                return future.result()
        except RuntimeError:
            # No running loop — safe to call asyncio.run() directly
            return asyncio.run(self.call_tool(tool_name, arguments))

    def get_call_log(self) -> list:
        return list(self._call_log)
