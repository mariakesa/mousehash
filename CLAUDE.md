# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Orienting fact

The repo was reset on 2026-05-15. **All prior implementation (DataJoint pipelines, smolagents DANDI agent, BlahML, RSAAnalysis) was moved into `old_code/` as reference only.** The live tree is being rebuilt to match a new design:

> **`mousehash_fastmcp_architecture.md` at the repo root is the authoritative design document.** Read its section 17 (MVP sequence) before adding code — every new module should map to a phase in that plan.

Do not import from `old_code/`. Do not point new code at the old paths.

## Where things are right now

Only **Phase 1 (library spine)** has been built. Anything not on this list does not exist yet:

- `src/mousehash/core/` — pydantic v2 types: `RoleBundle` / `RoleEvidence`, `RoleManifest` / `DatasetRef`, `AnalysisView` / `AnalysisViewKind`, `ToolContract` / `check_manifest_satisfies`, `Artifact` / `ArtifactKind`, `ids.py` (`stable_hash`, NewType IDs), `errors.py`.
- `src/mousehash/targets/base.py` — `TargetAdapter` Protocol + `DatasetQuery` / `DatasetMetadata` / `ResourceRef` / `LocalResource`.

There are **no targets implemented**, **no FastMCP server**, **no tests**, **no scripts**, and **no `artifacts/` or `schema/` modules** yet. The architecture doc lists them; they're future phases.

## Environment

- Python venv: `/home/maria/mousehash/.venv` (Python 3.13). Always invoke explicitly: `.venv/bin/python …`. Editable install is already in place (`pip install -e .`).
- The core install is intentionally light: `pydantic`, `PyYAML`, `typing-extensions`. **Heavy ecosystem deps are extras**: `.[mcp]`, `.[dandi]`, `.[allen]`, `.[ibl]`, `.[science]`, `.[provenance]`, `.[dev]`. Do not promote anything into the base `dependencies` list — adapters and tools should be importable only when their extra is installed.
- No `.env` is required to import the core. The old `MOUSEHASH_DATA_ROOT` / `DJ_*` variables described in `.env.example` are stale relative to the rebuild and not consumed by current code.

## Stale config to be aware of (do not trust without updating)

- `.mcp.json` still points at `dandi_mcp/server.py`, which now lives under `old_code/`. The new FastMCP server (Phase 2+) will live at `src/mousehash/mcp/server.py` per the architecture doc §3 — update `.mcp.json` when it's built.
- `.env.example` describes the prior DataJoint setup.
- `README.md` is a single quote and has no engineering content.

## Architectural rules (enforce these when adding code)

The architecture doc names six invariants. They are easy to violate by accident and they matter:

1. **Targets ingest, roles describe, transformations build views, tools consume views, artifacts remember.** A `TargetAdapter` must not analyze data; analysis tools must not take raw target objects.
2. **Tools consume `AnalysisView` objects, not raw arrays or NWB handles.** `AnalysisView.kind` (an `AnalysisViewKind` enum) is part of the tool contract — never let a tool secretly accept a different kind.
3. **Manifests are evidence-backed.** Every role on a `RoleBundle` carries a `list[RoleEvidence]` with `status`, `confidence`, `source`, `path`. Don't introduce code paths that set role status without evidence.
4. **Readiness is declarative.** Use `ToolContract` + `check_manifest_satisfies()` to decide whether a tool can run. Do not inline ad-hoc checks in tool implementations.
5. **IDs are content-addressed.** `stable_hash()` (sha256, sorted JSON, 12-char digest) drives `manifest_id`, `view_id.lineage_hash`, etc. Preserve that — agents rely on equal inputs producing equal IDs.
6. **DatasetRef is canonical in `core/manifests.py`.** `targets/__init__.py` re-exports it for ergonomics; don't define a parallel one.

## Coding idioms in this codebase

- All scientific data types are **pydantic v2 `BaseModel`** (not stdlib dataclasses). This gives free JSON round-trip for MCP responses and YAML round-trip via `RoleManifest.to_yaml()` / `from_yaml()`. Keep new core types as `BaseModel` subclasses.
- Errors are specific (`RoleMissingError`, `ViewKindMismatchError`, `ContractViolationError`, `ManifestParseError`) so the future MCP server can translate them to structured failure responses instead of leaking tracebacks.
- The default of *no comments* applies — comment only when the *why* is non-obvious (a hidden constraint, an invariant, an interop quirk).
- When you need to write a "*module purpose*" line at the top of a file, keep it to one short docstring that says what role this file plays in the architecture, not what each function does.

## Smoke-testing the spine

There is no test suite yet. To sanity-check changes to `core/` or `targets/base.py`, run an ad-hoc Python session that exercises the public surface — build a `DatasetRef`, `RoleBundle` with evidence, `RoleManifest.new(...)`, YAML round-trip it, check a `ToolContract` against the manifest with `check_manifest_satisfies(...)`, and construct an `AnalysisView.new(...)` to verify `lineage_hash` is deterministic. Once Phase 1 sees real consumers, a `tests/` tree should land alongside Phase 2.

## What to do when asked to build the next phase

The architecture doc's MVP sequence is the source of truth:

- **Phase 2 (DANDI scanner)**: `targets/dandi/{client,nwb_inspector,manifest}.py` + `mcp/server.py` exposing `dandi_build_role_manifest` and `list_runnable_tools`. Install `.[dandi]` and `.[mcp]` extras.
- **Phase 3 (Allen v0 rebuild)**: port the working logic from `old_code/src/mousehash/tools/allen/` and `old_code/src/mousehash/tools/representations/` into `targets/allen/` + `tools/` shape. The old code is good reference for AllenSDK quirks; the *shape* should follow the new architecture.
- **Phase 4+**: shared `transformations/`, shared `tools/`, IBL adapter.

Don't skip ahead. The spine intentionally has no `schema/` (DataJoint provenance) because that's a persistence decision that should follow a working MCP loop, not precede it.
