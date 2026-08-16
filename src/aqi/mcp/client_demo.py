"""Tiny MCP client that exercises our MCP server over the protocol.

Spawns ``aqi.mcp.server`` as a subprocess (stdio transport), lists its tools,
and calls a couple of them — the same round-trip an MCP client like Claude
Desktop performs. Purely local and read-only; safe to run anytime.

    python -m aqi.mcp.client_demo
"""
from __future__ import annotations

import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=["-m", "aqi.mcp.server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("MCP tools exposed:", [t.name for t in tools.tools])

            print("\n-- call list_cities --")
            res = await session.call_tool("list_cities", {})
            print(res.content[0].text[:160], "...")

            print("\n-- call get_forecast(Lahore) --")
            res = await session.call_tool("get_forecast", {"city": "Lahore"})
            print(res.content[0].text[:400])

            print("\n-- call get_history_summary(Faisalabad) --")
            res = await session.call_tool("get_history_summary", {"city": "Faisalabad"})
            print(res.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
