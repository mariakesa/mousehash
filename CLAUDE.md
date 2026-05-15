# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Orienting fact

The repo was reset on 2026-05-15. **All prior implementation (DataJoint pipelines, smolagents DANDI agent, BlahML, RSAAnalysis) was moved into `old_code/` as reference only.** The live tree is being rebuilt to match a new design:

> **`mousehash_fastmcp_architecture.md` at the repo root is the authoritative design document.** Read its section 17 (MVP sequence) before adding code — every new module should map to a phase in that plan.

Do not import from `old_code/`. Do not point new code at the old paths.

## Where things are now

**Built (Phase 1 spine + Phase 3 Allen vertical):**

- `src/mousehash/core/` — pydantic v2 types: `RoleBundle` / `RoleEvidence`, `RoleManifest` / `DatasetRef`, `AnalysisView` / `AnalysisViewKind`, `ToolContract` / `check_manifest_satisfies`, `Artifact` / `ArtifactKind`, `ids.py` (`stable_hash`, NewType IDs), `errors.py`.
- `src/mousehash/artifacts/` — `paths.py` (lazy env-resolved roots), `io.py` (JSON/NPY/HTML/CSV), `hashes.py` (`sha1_file`).
- `src/mousehash/targets/base.py` — `TargetAdapter` Protocol + query/metadata/resource dataclasses.
- `src/mousehash/targets/allen/` — full Allen adapter: `client.py`, `stimuli.py`, `manifest.py`, `loaders.py`, `adapter.py`.
- `src/mousehash/transformations/` — `feature_extraction.py` (ViT-ImageNet → AnalysisView), `labeling.py` (top1, animate/inanimate, ImageNet 1000 labels).
- `src/mousehash/tools/factor_models/` — `pca.py` + `nmf.py` with declared `ToolContract`s, consuming `OBSERVATION_BY_FEATURE` views.
- `src/mousehash/tools/reports/structure_discovery.py` — combined PCA/NMF HTML reports + index page.
- `src/mousehash/pipelines/allen_natural_scenes.py` — `run_allen_natural_scenes_v0()` end-to-end recipe.
- `scripts/run_allen_v0.py` — CLI entry.
- `tests/` — 127 tests, 9 files, runs in ~3 seconds.

**Not built yet (deliberately):**

- `src/mousehash/mcp/` — FastMCP server. The whole point of the rebuild, but the tools-and-types layer had to exist first. `.mcp.json` is intentionally empty until this lands.
- `src/mousehash/targets/dandi/` — Phase 2 in the architecture doc. The DANDI scanner is the flagship MVP because it stress-tests the role-bundle abstraction against heterogeneous NWB files.
- `src/mousehash/targets/ibl/` — Phase 6.
- `src/mousehash/schema/` — DataJoint or alternative persistence. Deferred until the MCP loop is real.

## Environment

- Python venv: `/home/maria/mousehash/.venv` (Python 3.13). Always invoke explicitly: `.venv/bin/python …`. Editable install is in place (`pip install -e .`).
- **Core install is intentionally light**: `pydantic`, `PyYAML`, `python-dotenv`, `typing-extensions`. Heavy ecosystem deps are extras: `.[mcp]`, `.[dandi]`, `.[allen]`, `.[ibl]`, `.[science]`, `.[ml]` (torch+transformers), `.[viz]` (plotly), `.[provenance]` (datajoint), `.[dev]`. The convenience extra `.[allen_v0]` bundles everything needed to run the natural-scenes pipeline.
- **Paths are env-driven and lazy.** `mousehash.artifacts.paths` reads `MOUSEHASH_DATA_ROOT` (required) plus optional subroot overrides on first call, cached. `import mousehash` does NOT require `.env`; only calling a path helper does.
- `.env` schema is in `.env.example`. Allen's BrainObservatoryCache manifest path is read from either `ALLEN_MANIFEST_PATH` (preferred) or `ALLEN_DATA` (legacy alias).

## Common commands

```bash
# Tests — fast suite, no live AllenSDK/ViT (everything heavy is mocked).
.venv/bin/python -m pytest tests/
.venv/bin/python -m pytest tests/test_pipeline_v0.py -v
.venv/bin/python -m pytest tests/test_contracts.py::TestCheckManifestSatisfies::test_any_of_satisfied_by_one

# End-to-end Allen v0 pipeline — needs .[allen_v0] + a .env with ALLEN_MANIFEST_PATH (or ALLEN_DATA).
.venv/bin/python scripts/run_allen_v0.py --scene-set-id allen_natural_scenes_v1

# Reinstall after editing pyproject.toml.
.venv/bin/pip install -e .
```

## Architectural rules (enforce these when adding code)

1. **Targets ingest, roles describe, transformations build views, tools consume views, artifacts remember.** A `TargetAdapter` must not analyze data; analysis tools must not take raw target objects.
2. **Tools consume `AnalysisView` objects, not raw arrays or NWB handles.** `AnalysisView.kind` is part of the tool contract — `run_pca` / `run_nmf` raise `ViewKindMismatchError` on the wrong kind. Preserve that pattern in new tools.
3. **Manifests are evidence-backed.** Every role on a `RoleBundle` carries a `list[RoleEvidence]` with `status`, `confidence`, `source`, `path`.
4. **Readiness is declarative.** Use `ToolContract` + `check_manifest_satisfies()` to decide whether a tool can run. Do not inline ad-hoc role checks in tool bodies.
5. **IDs are content-addressed.** `stable_hash()` (sha256, sorted JSON, 12-char digest) drives `manifest_id`, `view_id.lineage_hash`, decomposition output directories. Same inputs → same IDs.
6. **`DatasetRef` is canonical in `core/manifests.py`.** `targets/__init__.py` re-exports it for ergonomics; don't define a parallel one.

## Coding idioms

- All scientific data types are **pydantic v2 `BaseModel`**, not stdlib dataclasses. Free JSON round-trip + YAML round-trip via `RoleManifest.to_yaml()` / `from_yaml()`.
- Errors are specific (`RoleMissingError`, `ViewKindMismatchError`, `ContractViolationError`, `MissingEnvError`, `ManifestParseError`, `AllenSDKMissingError`) so the future MCP server can translate them to structured failure responses.
- **Heavy deps are imported lazily inside functions**, not at module top. This lets `import mousehash.targets.allen` work without `.[allen]` installed (and similarly for ml/viz).
- **No env reads at import time.** All env consumption goes through `mousehash.artifacts.paths` (or `targets.allen.client.resolve_manifest_path` for the Allen manifest), which load `.env` on first call.
- Default to no comments. Comment only when the *why* is non-obvious.

## Testing conventions

- `tests/conftest.py::_isolate_mousehash_env` is **autouse**. It strips every `MOUSEHASH_*` and `ALLEN_*` env var per test AND monkeypatches the inner `load_dotenv` call to a no-op, so the user's real `.env` cannot leak into tests.
- The `data_root_tmp` fixture gives a test a fresh tmp-path-backed `MOUSEHASH_DATA_ROOT` (subroots default under it).
- **AllenSDK and ViT are always mocked.** `fetch_natural_scene_template` and `run_vit_on_frames` are monkeypatched with deterministic stubs. `sklearn` and `plotly` are NOT mocked — they're cheap and worth exercising.
- A subtle wrinkle: `load_dotenv()` searches upward from the **caller's `__file__`**, not cwd. `chdir` alone is not enough to escape the repo's `.env` — you have to monkeypatch the `load_dotenv` import explicitly.

## What to build next

The natural sequence per architecture doc §17:

- **Phase 4 (FastMCP server)**: `src/mousehash/mcp/server.py` exposing `allen_build_role_manifest`, `list_runnable_tools`, `materialize_analysis_view`, `run_pca`, `run_nmf`, `generate_structure_discovery_report` over MCP. Wires `.mcp.json` back up.
- **Phase 2 (DANDI scanner)**: `targets/dandi/{client,nwb_inspector,manifest,adapter}.py`. The architecture doc's flagship MVP because it stress-tests the role-bundle abstraction against heterogeneous NWB files. Install `.[dandi]`.
- **Phase 6 (IBL adapter)**: `targets/ibl/`. Completes the three-target story. Install `.[ibl]`.

Don't skip the architecture doc when in doubt — it has worked examples and a tool registry shape that the readiness engine will eventually consume.
