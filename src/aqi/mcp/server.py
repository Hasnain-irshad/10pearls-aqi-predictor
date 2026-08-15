"""Model Context Protocol (MCP) server for the Pearls AQI Predictor.

Exposes our real forecast + history tools over MCP, so any MCP client — Claude
Desktop, an IDE, or our own advisor — can query live Pakistani air-quality data
in a standardized way. This is the genuine "MCP" artifact: the same tool
functions the in-app advisor uses (``aqi.tools``), published over the protocol.

Run it (stdio transport):
    python -m aqi.mcp.server

Register with Claude Desktop by adding to its MCP config:
    {
      "mcpServers": {
        "pearls-aqi": {"command": "python", "args": ["-m", "aqi.mcp.server"]}
      }
    }
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer  # MCP 2.0 high-level server (was FastMCP in 1.x)

from aqi import tools as aqi_tools

mcp = MCPServer(name="pearls-aqi", version="0.1.0")


@mcp.tool()
def list_cities() -> list[dict]:
    """List every Pakistani city the system can forecast (name, province, coords)."""
    return aqi_tools.list_cities()


@mcp.tool()
def get_forecast(city: str) -> dict:
    """Current AQI plus the 3-day forecast and hazard alert for a city."""
    return aqi_tools.get_forecast(city)


@mcp.tool()
def get_history_summary(city: str) -> dict:
    """Long-run historical AQI statistics for a city (mean, record high, seasonality)."""
    return aqi_tools.get_history_summary(city)


if __name__ == "__main__":
    mcp.run()  # stdio transport
