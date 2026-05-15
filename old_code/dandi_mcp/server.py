"""Minimal FastMCP server exposing MouseHash DANDI + RSA tools.

Wraps the repo's existing plain-Python tool functions so Claude Code can drive
the DANDI parsing + RSA workflow over MCP. The heavy lifting lives in
``mousehash.agents.dandi_agent.agent_tools`` and ``RSAAnalysis.spike_trial_rsa``;
this module only adapts signatures and registers the tools.

Run directly (``python dandi_mcp/server.py``) for stdio transport, which is how
Claude Code launches it via ``.mcp.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Repo root on sys.path so the top-level ``RSAAnalysis`` package imports
# regardless of the working directory Claude Code launches us from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastmcp import FastMCP

from mousehash.agents.dandi_agent.agent_tools import (
    analyze_dandiset,
    inspect_dandiset,
    parse_nwb_manifest,
    show_role_manifest,
    suggest_analyses,
)
from RSAAnalysis.spike_trial_rsa import DEFAULT_OUTPUT_DIR, run_spike_trial_rsa

mcp = FastMCP("mousehash-dandi")

# The DANDI tools already have MCP-friendly signatures (str/int args, descriptive
# docstrings) and return JSON strings, so register them as-is.
mcp.tool()(inspect_dandiset)
mcp.tool()(analyze_dandiset)
mcp.tool()(parse_nwb_manifest)
mcp.tool()(show_role_manifest)
mcp.tool()(suggest_analyses)


@mcp.tool()
def run_rsa(
    manifest_path: str,
    output_dir: str = "",
    item_column: str = "",
    align_column: str = "",
    response_window_s: float = 0.4,
    distance_metric: str = "correlation",
    rsa_correlation: str = "spearman",
    n_permutations: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Run spike-trial Representational Similarity Analysis on a role manifest.

    The manifest must be RSA-ready: produce one with ``analyze_dandiset`` (from a
    dandiset id) or ``parse_nwb_manifest`` (from a local NWB), then check
    ``suggest_analyses`` lists ``run_rsa`` as ready. Builds a trial x unit spike
    count matrix, computes a neural RDM, compares it to a categorical target RDM
    via a permutation test, and writes ``.npy`` artifacts + ``summary.json`` under
    the output directory.

    Args:
        manifest_path: Path to an EvidenceBackedRoleManifest JSON.
        output_dir: Where to write artifacts. Empty uses the repo default
            (``RSAAnalysis/outputs/spike_trial_rsa``).
        item_column: Trial table column for categorical labels. Empty = auto-pick.
        align_column: Trial table column for the response window start. Empty =
            auto-pick.
        response_window_s: Seconds after the alignment time to count spikes in.
        distance_metric: Neural RDM metric: "correlation", "euclidean", or
            "cosine".
        rsa_correlation: RDM comparison statistic: "spearman" or "pearson".
        n_permutations: Permutations for the null distribution / p-value.
        seed: RNG seed for the permutation test.

    Returns:
        The RSA summary dict (rsa_statistic, p_value, n_trials, n_units,
        label_counts, meta_features, and artifact paths).
    """
    return run_spike_trial_rsa(
        Path(manifest_path).expanduser(),
        output_dir=Path(output_dir).expanduser() if output_dir else DEFAULT_OUTPUT_DIR,
        item_column=item_column or None,
        align_column=align_column or None,
        response_window_s=response_window_s,
        distance_metric=distance_metric,
        rsa_correlation=rsa_correlation,
        n_permutations=n_permutations,
        seed=seed,
    )


if __name__ == "__main__":
    mcp.run()
