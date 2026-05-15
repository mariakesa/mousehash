"""Batch driver for the standardized spike-trial RSA prototype."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from RSAAnalysis.spike_trial_rsa import DEFAULT_OUTPUT_DIR, run_spike_trial_rsa


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_READY_JSON = Path("/media/maria/notsudata/MousehashManifests/rsa_ready_experiments.json")


def _load_ready_matches(ready_json: Path) -> list[dict[str, Any]]:
    payload = json.loads(ready_json.read_text(encoding="utf-8"))
    return list(payload.get("matches", []))


def _meta_row(summary: dict[str, Any]) -> dict[str, Any]:
    meta = dict(summary.get("meta_features", {}))
    meta.update(
        {
            "dandiset_id": summary.get("dandiset_id"),
            "asset_id": summary.get("asset_id"),
            "manifest_path": summary.get("manifest_path"),
            "nwb_path": summary.get("nwb_path"),
            "item_column": summary.get("item_column"),
            "align_column": summary.get("align_column"),
            "response_window_s": summary.get("response_window_s"),
            "distance_metric": summary.get("distance_metric"),
            "rsa_correlation": summary.get("rsa_correlation"),
            "n_permutations": summary.get("n_permutations"),
            "output_dir": summary.get("output_dir"),
        }
    )
    return meta


def _write_progress(
    output_dir: Path,
    *,
    ready_json: Path,
    total: int,
    processed: int,
    results: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    current_manifest: str | None,
) -> None:
    progress_path = output_dir / "batch_progress.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(
        json.dumps(
            {
                "ready_json": str(ready_json),
                "n_input_matches": total,
                "n_processed": processed,
                "n_results": len(results),
                "n_skipped": len(skipped),
                "n_errors": len(errors),
                "current_manifest": current_manifest,
                "last_result_manifest": results[-1]["manifest_path"] if results else None,
                "last_skip_manifest": skipped[-1]["manifest_path"] if skipped else None,
                "last_error_manifest": errors[-1]["manifest_path"] if errors else None,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready-json", type=Path, default=DEFAULT_READY_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument("--response-window-s", type=float, default=0.4)
    parser.add_argument("--distance-metric", choices=["correlation", "euclidean", "cosine"], default="correlation")
    parser.add_argument("--rsa-correlation", choices=["spearman", "pearson"], default="spearman")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    matches = _load_ready_matches(args.ready_json)
    if args.max_experiments is not None:
        matches = matches[: args.max_experiments]

    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for index, match in enumerate(matches, start=1):
        manifest_path = Path(match["manifest_path"])
        started_at = time.perf_counter()
        logger.info("Starting %d/%d %s", index, len(matches), manifest_path.name)
        _write_progress(
            args.output_dir,
            ready_json=args.ready_json,
            total=len(matches),
            processed=index - 1,
            results=results,
            skipped=skipped,
            errors=errors,
            current_manifest=str(manifest_path),
        )
        try:
            summary = run_spike_trial_rsa(
                manifest_path,
                output_dir=args.output_dir,
                response_window_s=args.response_window_s,
                distance_metric=args.distance_metric,
                rsa_correlation=args.rsa_correlation,
                n_permutations=args.n_permutations,
                seed=args.seed,
            )
            results.append(summary)
            logger.info(
                "Finished %s (rsa=%.4f, p=%.4f, elapsed=%.2fs)",
                summary.get("dandiset_id"),
                summary.get("rsa_statistic"),
                summary.get("p_value"),
                time.perf_counter() - started_at,
            )
        except ValueError as exc:
            skipped.append({"manifest_path": str(manifest_path), "reason": str(exc)})
            logger.info(
                "Skipped %s (%s, elapsed=%.2fs)",
                manifest_path.name,
                exc,
                time.perf_counter() - started_at,
            )
        except Exception as exc:
            logger.exception("Failed on %s", manifest_path)
            errors.append({"manifest_path": str(manifest_path), "error": str(exc)})
        _write_progress(
            args.output_dir,
            ready_json=args.ready_json,
            total=len(matches),
            processed=index,
            results=results,
            skipped=skipped,
            errors=errors,
            current_manifest=None,
        )

    summary_path = args.output_dir / "batch_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    meta_rows = [_meta_row(result) for result in results]
    pd.DataFrame(meta_rows).to_csv(args.output_dir / "meta_analysis_table.csv", index=False)
    (args.output_dir / "meta_analysis_table.json").write_text(
        json.dumps(meta_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "ready_json": str(args.ready_json),
                "n_input_matches": len(matches),
                "n_results": len(results),
                "n_skipped": len(skipped),
                "n_errors": len(errors),
                "meta_analysis_csv": str(args.output_dir / "meta_analysis_table.csv"),
                "meta_analysis_json": str(args.output_dir / "meta_analysis_table.json"),
                "results": results,
                "skipped": skipped,
                "errors": errors,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote batch summary to %s", summary_path)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())