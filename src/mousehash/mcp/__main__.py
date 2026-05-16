"""Stdio launcher for the MouseHash FastMCP server.

Invoked as `python -m mousehash.mcp` (see `.mcp.json` at the repo root).
"""

from __future__ import annotations

from mousehash.mcp.server import mcp


if __name__ == "__main__":
    mcp.run()
