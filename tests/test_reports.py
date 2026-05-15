"""Tests for tools/reports/structure_discovery.py.

Reports produce HTML files; we verify they're non-empty, contain expected
section markers, and that the index page references the sub-reports.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from mousehash.artifacts.io import save_json, save_npy
from mousehash.core.ids import DatasetId, TargetName
from mousehash.core.manifests import DatasetRef, RoleManifest
from mousehash.core.role_bundle import (
    RoleBundle,
    RoleConfidence,
    RoleStatus,
    StimuliRole,
)
from mousehash.tools.reports.structure_discovery import (
    build_nmf_report,
    build_pca_report,
    generate_structure_discovery_report,
)


def _stub_images(tmp_path: Path, n: int = 6) -> list[Path]:
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    rng = np.random.default_rng(0)
    for i in range(n):
        p = img_dir / f"scene_{i:04d}.png"
        arr = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
        Image.fromarray(arr).save(p)
        paths.append(p)
    return paths


def _stub_pca_artifacts(out: Path, n: int, k: int) -> dict:
    rng = np.random.default_rng(1)
    scores = rng.normal(size=(n, k)).astype(np.float32)
    components = rng.normal(size=(k, 1000)).astype(np.float32)
    evr = np.sort(rng.dirichlet(np.ones(k)))[::-1]
    save_npy(out / "scores.npy", scores)
    save_npy(out / "components.npy", components)
    save_json(out / "component_stats.json", {
        "explained_variance_ratio": evr.tolist(),
        "singular_values": (evr * 100).tolist(),
        "cumulative_variance": np.cumsum(evr).tolist(),
    })
    summary = {
        "method": "pca", "n_components": k, "n_images": n,
        "explained_variance_ratio_total": float(evr.sum()),
        "artifacts": {
            "scores": str(out / "scores.npy"),
            "components": str(out / "components.npy"),
            "component_stats": str(out / "component_stats.json"),
        },
    }
    return summary


def _stub_nmf_artifacts(out: Path, n: int, k: int) -> dict:
    rng = np.random.default_rng(2)
    scores = np.abs(rng.normal(size=(n, k))).astype(np.float32)
    components = np.abs(rng.normal(size=(k, 1000))).astype(np.float32)
    save_npy(out / "scores.npy", scores)
    save_npy(out / "components.npy", components)
    save_json(out / "component_stats.json", {"reconstruction_err": 0.05, "n_iter": 10})
    return {
        "method": "nmf", "n_components": k, "n_images": n,
        "reconstruction_err": 0.05, "temperature": 1.0,
        "artifacts": {
            "scores": str(out / "scores.npy"),
            "components": str(out / "components.npy"),
            "component_stats": str(out / "component_stats.json"),
        },
    }


class TestBuildPCAReport:
    def test_html_written(self, tmp_path: Path):
        image_paths = _stub_images(tmp_path, n=5)
        rng = np.random.default_rng(0)
        out = build_pca_report(
            scores=rng.normal(size=(5, 3)).astype(np.float32),
            components=rng.normal(size=(3, 1000)).astype(np.float32),
            explained_variance_ratio=np.array([0.5, 0.3, 0.2]),
            animate_inanimate=np.array([1, 0, 1, 0, 1], dtype=np.int8),
            image_paths=image_paths,
            output_path=tmp_path / "pca.html",
            title="Stub PCA",
        )
        assert out.exists()
        text = out.read_text()
        assert "Stub PCA" in text
        assert "Explained variance per PC" in text
        # Plotly inlined exactly once
        assert text.count("plotly.min.js") <= 1

    def test_handles_one_component(self, tmp_path: Path):
        image_paths = _stub_images(tmp_path, n=4)
        rng = np.random.default_rng(0)
        out = build_pca_report(
            scores=rng.normal(size=(4, 1)).astype(np.float32),
            components=rng.normal(size=(1, 1000)).astype(np.float32),
            explained_variance_ratio=np.array([1.0]),
            animate_inanimate=np.array([1, 0, 1, 0], dtype=np.int8),
            image_paths=image_paths,
            output_path=tmp_path / "pca1.html",
        )
        assert out.read_text().strip() != ""


class TestBuildNMFReport:
    def test_html_written(self, tmp_path: Path):
        image_paths = _stub_images(tmp_path, n=5)
        rng = np.random.default_rng(0)
        out = build_nmf_report(
            scores=np.abs(rng.normal(size=(5, 3))).astype(np.float32),
            components=np.abs(rng.normal(size=(3, 1000))).astype(np.float32),
            animate_inanimate=np.array([1, 0, 1, 0, 1], dtype=np.int8),
            image_paths=image_paths,
            output_path=tmp_path / "nmf.html",
            reconstruction_err=0.123,
            title="Stub NMF",
        )
        assert out.exists()
        text = out.read_text()
        assert "Stub NMF" in text
        assert "reconstruction err=0.1230" in text or "0.123" in text
        assert "heatmap" in text.lower()


class TestGenerateStructureDiscoveryReport:
    def test_writes_index_pca_nmf(self, data_root_tmp: Path, tmp_path: Path):
        image_paths = _stub_images(tmp_path, n=6)
        catalog = [{"image_idx": i, "image_path": str(p)} for i, p in enumerate(image_paths)]
        pca_dir = tmp_path / "pca_out"
        nmf_dir = tmp_path / "nmf_out"
        pca_dir.mkdir()
        nmf_dir.mkdir()
        pca_summary = _stub_pca_artifacts(pca_dir, n=6, k=3)
        nmf_summary = _stub_nmf_artifacts(nmf_dir, n=6, k=3)

        manifest = RoleManifest.new(
            dataset=DatasetRef(target=TargetName("allen"), dataset_id=DatasetId("test"),
                                label="test dataset"),
            roles=RoleBundle(stimuli=StimuliRole(status=RoleStatus.PRESENT, confidence=RoleConfidence.HIGH)),
        )

        bundle = generate_structure_discovery_report(
            manifest=manifest,
            pca_summary=pca_summary,
            nmf_summary=nmf_summary,
            image_catalog=catalog,
            animate_inanimate=np.array([1, 0, 1, 0, 1, 0], dtype=np.int8),
        )
        for key in ("index", "pca", "nmf"):
            p = Path(bundle["reports"][key])
            assert p.exists()
            assert p.stat().st_size > 0

    def test_index_references_subreports(self, data_root_tmp: Path, tmp_path: Path):
        image_paths = _stub_images(tmp_path, n=5)
        catalog = [{"image_idx": i, "image_path": str(p)} for i, p in enumerate(image_paths)]
        pca_dir = tmp_path / "pca"
        nmf_dir = tmp_path / "nmf"
        pca_dir.mkdir()
        nmf_dir.mkdir()
        pca_summary = _stub_pca_artifacts(pca_dir, n=5, k=2)
        nmf_summary = _stub_nmf_artifacts(nmf_dir, n=5, k=2)
        manifest = RoleManifest.new(
            dataset=DatasetRef(target=TargetName("allen"), dataset_id=DatasetId("t2")),
        )
        bundle = generate_structure_discovery_report(
            manifest=manifest,
            pca_summary=pca_summary,
            nmf_summary=nmf_summary,
            image_catalog=catalog,
            animate_inanimate=np.array([1, 0, 1, 0, 1], dtype=np.int8),
        )
        index_text = Path(bundle["reports"]["index"]).read_text()
        assert "pca_report.html" in index_text
        assert "nmf_report.html" in index_text
        assert manifest.manifest_id in index_text

    def test_writes_bundle_summary_json(self, data_root_tmp: Path, tmp_path: Path):
        image_paths = _stub_images(tmp_path, n=5)
        catalog = [{"image_idx": i, "image_path": str(p)} for i, p in enumerate(image_paths)]
        pca_dir = tmp_path / "pca"
        nmf_dir = tmp_path / "nmf"
        pca_dir.mkdir()
        nmf_dir.mkdir()
        pca_summary = _stub_pca_artifacts(pca_dir, n=5, k=2)
        nmf_summary = _stub_nmf_artifacts(nmf_dir, n=5, k=2)
        manifest = RoleManifest.new(
            dataset=DatasetRef(target=TargetName("allen"), dataset_id=DatasetId("t3")),
        )
        bundle = generate_structure_discovery_report(
            manifest=manifest,
            pca_summary=pca_summary,
            nmf_summary=nmf_summary,
            image_catalog=catalog,
            animate_inanimate=np.array([1, 0, 1, 0, 1], dtype=np.int8),
        )
        out_dir = Path(bundle["output_dir"])
        assert (out_dir / "report_bundle.json").exists()
