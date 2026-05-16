# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Orienting fact

The repo was reset on 2026-05-15. **All prior implementation (DataJoint pipelines, smolagents DANDI agent, BlahML, RSAAnalysis) was moved into `old_code/` as reference only.** The live tree is being rebuilt to match a new design:

> **`mousehash_fastmcp_architecture.md` at the repo root is the authoritative design document.** Read its section 17 (MVP sequence) before adding code — every new module should map to a phase in that plan.

Do not import from `old_code/`. Do not point new code at the old paths.

## Where things are now

**Built (Phase 1 spine + Phase 3 Allen vertical + cache):**

- `src/mousehash/core/` — pydantic v2 types: `RoleBundle` / `RoleEvidence`, `RoleManifest` / `DatasetRef`, `AnalysisView` / `AnalysisViewKind`, `ToolContract` / `check_manifest_satisfies`, `Artifact` / `ArtifactKind`, `ids.py` (`stable_hash`, NewType IDs), `errors.py`.
- `src/mousehash/artifacts/` — `paths.py` (lazy env-resolved roots), `io.py` (JSON/NPY/HTML/CSV), `hashes.py` (`sha1_file`), **`cache.py` (`ComputationSpec` + `cached_computation` content-addressed cache)**.
- `src/mousehash/targets/base.py` — `TargetAdapter` Protocol + query/metadata/resource dataclasses.
- `src/mousehash/targets/allen/` — full Allen adapter: `client.py`, `stimuli.py`, `manifest.py`, `loaders.py`, `adapter.py`.
- `src/mousehash/transformations/` — `feature_extraction.py` (ViT-ImageNet → AnalysisView, cached), `image_compression.py` (JPEG byte sizes at multiple qualities → AnalysisView, cached), `labeling.py` (top1, animate/inanimate, ImageNet 1000 labels).
- `src/mousehash/tools/factor_models/` — `pca.py` + `nmf.py` with declared `ToolContract`s. On the `cached_computation` pattern: return `(AnalysisView, summary)`, idempotent on identical inputs.
- `src/mousehash/tools/comparison/` — two-group statistical comparison of any `OBSERVATION_BY_FEATURE` view against a binary label vector. Welch's t + Mann-Whitney U + Cohen's d + Bonferroni-corrected min-p, plus an interactive plotly grouped boxplot and a plain-English summary string. `GROUP_COMPARISON_CONTRACT` is in the readiness registry. Cached via `cached_computation`.
- `src/mousehash/tools/scheduling/` — `schedule_comparison.py`: per-session block-permutation diagnostics + pairwise cross-session agreement + donor breakdown, consumes a `PRESENTATION_TABLE` view from `extract_stimulus_schedule_view`. Plotly heatmap output, plain-English summary that directly answers "same order each trial? / same schedule for all animals?". `STIMULUS_SCHEDULE_CONTRACT` is in the readiness registry. Cached via `cached_computation`.
- `src/mousehash/transformations/stimulus_schedule.py` — `extract_stimulus_schedule_view`: pulls `boc.get_ophys_experiment_data(sid).get_stimulus_table(stim)` for every Allen session with the stimulus, persists per-session `frame`/`start`/`end` arrays + donor/container metadata as a `PRESENTATION_TABLE` AnalysisView. Materializes the manifest's `time_organization` role.
- `src/mousehash/tools/reports/structure_discovery.py` — combined PCA/NMF HTML reports + index page.
- `src/mousehash/pipelines/allen_natural_scenes.py` — `run_allen_natural_scenes_v0()` end-to-end recipe (ViT + JPEG + PCA + NMF + report).
- `src/mousehash/mcp/` — FastMCP server with **16 tools** registered:
  - Targets + manifests + readiness: `allen_list_datasets`, `allen_build_manifest`, `get_manifest`, `list_runnable_tools`, `explain_tool_readiness`.
  - Views: `list_views`, `inspect_view`.
  - Transformations: `extract_vit_features`, `extract_jpeg_sizes`, `extract_stimulus_schedule`.
  - Analysis: `run_pca`, `run_nmf`, `compare_jpeg_by_animate_inanimate`, `analyze_stimulus_schedule`.
  - Reports: `generate_structure_report`.
  - All-in-one pipeline: `run_allen_natural_scenes_v0`.

  Plus 4 `@mcp.resource` entries (`mousehash://targets`, `mousehash://targets/allen/datasets`, `mousehash://manifests/{id}`, `mousehash://tools/{name}/contract`) and 2 `@mcp.prompt` templates (`explain_dataset_readiness`, `design_analysis_plan`). Launch via `python -m mousehash.mcp` (wired in `.mcp.json`). Each tool wrapper is `@mcp_safe`-decorated so `MouseHashError` subclasses surface as structured `{error, type, details}` JSON.
- `scripts/run_allen_v0.py` — CLI entry.
- `tests/` — 281 tests across 25 files, runs in ~6 seconds.

**Not built yet (deliberately):**

- **HTTP transport for MCP.** Stdio-only for now (Claude Code's default).
- `src/mousehash/targets/dandi/` — Phase 2 in the architecture doc. The DANDI scanner is the flagship MVP because it stress-tests the role-bundle abstraction against heterogeneous NWB files.
- `src/mousehash/targets/ibl/` — Phase 6.
- `src/mousehash/schema/` — DataJoint or alternative persistence. Deferred until the MCP loop sees real load.
- **Multi-target transformation dispatch.** `mcp/transformation_tools.py::_materialize_image_stack` is currently Allen-only. When DANDI / IBL land, dispatch on `manifest.dataset.target` to the right adapter.

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

## The cache pattern (use this for every new transformation)

`artifacts/cache.py` is the canonical way to persist intermediate computations. Every transformation should be:

1. **Build a `ComputationSpec`** with `family` (e.g. `"representations"`, `"compression"`), `scope` (the dataset/scene_set id), `name` (the tool name), `parameters` (everything that changes the output), and `input_fingerprints` (e.g. `fingerprint_array(frames)`). Use `label` for free-form tags — `label` does NOT participate in the cache hash.
2. **Call `cached_computation(spec, compute)`** with a `compute(out_dir)` callback that writes data files into `out_dir` and returns `(view, summary)`. The cache writes `spec.json` + `view.json` + `summary.json` automatically. Same spec → same cache dir under `<artifact_root>/<family>/<scope>/<spec_hash>/`. Second call is a cache hit.
3. **Include `input_fingerprint[:12]` as the first entry of `view.transformation_lineage`** so the resulting `view.lineage_hash` (and therefore `view.view_id`) reflects the input data, not just the structural shape. Without this, two views with different inputs but the same lineage strings end up with identical view_ids — see `extract_vit_features_view` and `extract_jpeg_size_view` for the pattern.

Idempotence is the default — re-running a pipeline with identical inputs is a no-op. **PCA and NMF are now on this pattern too**: they take a source `AnalysisView`, return `(output_view, summary)`, and cache under `<artifact_root>/decompositions/<source_view.lineage_hash>/<spec_hash>/`.

## The MCP layer (how to add a tool)

The MCP server lives in `src/mousehash/mcp/`. It's a thin **registration + adapter** layer over the Python library — it never adds new analysis logic. Launch: `python -m mousehash.mcp` (the entry point is `mcp/__main__.py`, the registration is in `mcp/server.py`).

To add a new MCP tool:

1. Pick the right `mcp/<domain>_tools.py` file (target / manifest / view / transformation / analysis / report / pipeline) or add a new one.
2. Write a wrapper. **Args**: `str`, `int`, `float`, `bool`, `list[...]` only — no `Path`, no `Optional[T]` outside string sentinels, no `np.ndarray`, no `tuple`, no pydantic models at the boundary. For optional paths, use `Path(arg).expanduser() if arg else None`.
3. **Return** a plain dict whose values are JSON-serializable. Lift `manifest_id` / `view_id` / `artifact_path` to top-level keys when present. Pydantic models go through `.model_dump(mode="json")`.
4. Decorate with `@mcp_safe` (from `mcp/errors.py`) outermost. It catches `MouseHashError` subclasses and returns `{"error", "type", "details"}` JSON; other exceptions bubble.
5. Register in `mcp/server.py` with `mcp.tool()(your_fn)`. Tool name = function name; the server's `mousehash` namespace separates them in Claude's view.
6. Add a smoke test in `tests/test_mcp_<domain>_tools.py` that monkeypatches the core call and checks the wrapper's signature shape + JSON-serializable return.

Resources go in `mcp/resources.py` as `mcp.resource("mousehash://...")(fn)` callbacks (return JSON strings). Prompts go in `mcp/prompts.py` as `mcp.prompt()(fn)` string-returning functions. Both are **read-only** — never mutate state in a resource or prompt handler.

**View-by-id lookup**: `mcp/views.py::find_view_by_id` walks `artifact_root().rglob("view.json")` and matches `view.view_id`. O(n) over the cache; fine for v0 (low view cardinality). Upgrade to an index file written inside `cached_computation` if cardinality exceeds ~1000.

**Adding a new tool to the readiness registry**: in `mcp/manifest_tools.py`, append the new `ToolContract` to `_CONTRACTS`. `list_runnable_tools` and `explain_tool_readiness` will pick it up automatically.

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
