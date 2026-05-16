"""Server-level registration smoke test.

Imports the live `mcp` instance and asserts that every tool / resource / prompt
we declared is actually registered. Uses FastMCP 3.x async listing API.
"""

from __future__ import annotations

import asyncio

import pytest


EXPECTED_TOOL_NAMES = {
    # Phase 1a (skeleton + readiness)
    "allen_list_datasets",
    "allen_build_manifest",
    "get_manifest",
    "list_runnable_tools",
    "explain_tool_readiness",
    # Phase 1b (compute wrappers)
    "list_views",
    "inspect_view",
    "extract_vit_features",
    "extract_jpeg_sizes",
    "extract_event_responses",
    "run_pca",
    "run_nmf",
    "compare_jpeg_by_animate_inanimate",
    "generate_structure_report",
    "run_allen_natural_scenes_v0",
    # Decoders + BlahML driver
    "decode_animate_inanimate",
    "decoder_next_question",
}

EXPECTED_RESOURCE_URIS = {
    "mousehash://targets",
    "mousehash://targets/allen/datasets",
}

EXPECTED_RESOURCE_TEMPLATE_URIS = {
    "mousehash://manifests/{manifest_id}",
    "mousehash://tools/{tool_name}/contract",
}

EXPECTED_PROMPT_NAMES = {
    "explain_dataset_readiness",
    "design_analysis_plan",
    "design_decoder_fit",
}


def _run(coro):
    return asyncio.run(coro)


class TestServerRegistration:
    def test_mcp_instance_importable(self):
        from mousehash.mcp.server import mcp
        assert mcp is not None
        assert mcp.name == "mousehash"

    def test_all_expected_tools_registered(self):
        from mousehash.mcp.server import mcp
        tools = _run(mcp.list_tools())
        names = {t.name for t in tools}
        assert EXPECTED_TOOL_NAMES <= names, (
            f"missing tools: {EXPECTED_TOOL_NAMES - names} (registered: {sorted(names)})"
        )

    def test_all_expected_prompts_registered(self):
        from mousehash.mcp.server import mcp
        prompts = _run(mcp.list_prompts())
        names = {p.name for p in prompts}
        assert EXPECTED_PROMPT_NAMES <= names, (
            f"missing prompts: {EXPECTED_PROMPT_NAMES - names}"
        )

    def test_fixed_resources_registered(self):
        from mousehash.mcp.server import mcp
        resources = _run(mcp.list_resources())
        uris = {str(r.uri) for r in resources}
        for expected in EXPECTED_RESOURCE_URIS:
            assert expected in uris, f"missing resource: {expected} (have {sorted(uris)})"

    def test_templated_resources_registered(self):
        from mousehash.mcp.server import mcp
        templates = _run(mcp.list_resource_templates())
        uris = {t.uri_template for t in templates}
        for expected in EXPECTED_RESOURCE_TEMPLATE_URIS:
            assert expected in uris, f"missing resource template: {expected} (have {sorted(uris)})"

    def test_tool_schemas_present(self):
        """Each tool should have a JSON schema FastMCP derived from its type hints."""
        from mousehash.mcp.server import mcp
        tools = _run(mcp.list_tools())
        for t in tools:
            assert t.parameters is not None
            # FastMCP 3.x stores the input schema on `parameters` as a JSON Schema dict.
            assert "properties" in t.parameters or t.parameters.get("type") == "object"

    def test_tool_descriptions_from_docstrings(self):
        """Tool descriptions should be lifted from the function docstrings."""
        from mousehash.mcp.server import mcp
        tools = _run(mcp.list_tools())
        by_name = {t.name: t for t in tools}
        assert "Allen Brain Observatory" in by_name["allen_list_datasets"].description
        assert "manifest" in by_name["allen_build_manifest"].description.lower()
