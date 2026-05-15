# dandi_mcp

A minimal [FastMCP](https://github.com/jlowin/fastmcp) server that exposes
MouseHash's DANDI parsing and RSA tools over MCP, so Claude Code can drive the
workflow conversationally.

## Tools

| Tool | What it does |
|------|--------------|
| `inspect_dandiset` | Summarize a dandiset's metadata from the DANDI API. |
| `analyze_dandiset` | Fetch a representative NWB and build its role manifest in one step. |
| `parse_nwb_manifest` | Build a role manifest for a local NWB file. |
| `show_role_manifest` | Pretty-print a persisted role manifest. |
| `suggest_analyses` | Rank analysis tools by readiness for a manifest. |
| `run_rsa` | Run spike-trial Representational Similarity Analysis on an RSA-ready manifest. |

## Install

The `mousehash` package plus `dandi`/`pynwb` are already in the project venv;
only `fastmcp` is extra:

```bash
/home/maria/mousehash/.venv/bin/pip install -r dandi_mcp/requirements.txt
```

## Register with Claude Code

The repo's `.mcp.json` already points Claude Code at this server:

```json
{
  "mcpServers": {
    "mousehash-dandi": {
      "command": "/home/maria/mousehash/.venv/bin/python",
      "args": ["/home/maria/mousehash/dandi_mcp/server.py"]
    }
  }
}
```

Restart Claude Code in the repo root and run `/mcp` to confirm `mousehash-dandi`
is connected.

## Typical chain

`inspect_dandiset` → `analyze_dandiset` (returns a `manifest_path`) →
`suggest_analyses` on that path → if `run_rsa` shows as ready, `run_rsa` on the
same `manifest_path`. RSA artifacts land under
`RSAAnalysis/outputs/spike_trial_rsa/`.
