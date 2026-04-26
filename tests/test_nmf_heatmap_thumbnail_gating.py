from __future__ import annotations

from pathlib import Path

import numpy as np

from mousehash.tools.reports.nmf_html import build_nmf_report


def test_nmf_heatmap_omits_renderable_indices_for_zero_scores(tmp_path: Path) -> None:
    output_path = tmp_path / "nmf_report.html"

    build_nmf_report(
        scores=np.array([[0.2, 0.05], [0.1, 0.11]]),
        components=np.array([[0.8, 0.2], [0.1, 0.9]]),
        animate_inanimate=np.array([1, 0], dtype=np.int8),
        image_paths=[tmp_path / "img0.png", tmp_path / "img1.png"],
        output_path=output_path,
        class_labels=["class0", "class1"],
        image_thumbs=["thumb0", "thumb1"],
        heatmap_thumb_threshold=0.1,
    )

    html = output_path.read_text(encoding="utf-8")

    assert '"customdata":[[0,null],[null,1]]' in html
    assert '"customdata":[[0,0],[1,1]]' not in html
    assert "typeof idx!=='number'||!Number.isFinite(idx)" in html
    assert "HEATMAP_THUMB_THRESHOLD=0.1" in html
    assert "p.z<=HEATMAP_THUMB_THRESHOLD" in html
    assert "thumbnail threshold \\u003e 0.1" in html