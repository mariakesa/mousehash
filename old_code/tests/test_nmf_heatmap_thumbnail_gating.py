from __future__ import annotations

from pathlib import Path

import numpy as np

from mousehash.tools.reports.nmf_html import build_nmf_report


def test_nmf_heatmap_embeds_component_thumbnail_tiles(tmp_path: Path) -> None:
    output_path = tmp_path / "nmf_report.html"
    scores = np.array([
        [0.012, 0.005],
        [0.030, 0.011],
        [0.020, 0.012],
        [0.070, 0.013],
        [0.050, 0.014],
        [0.060, 0.015],
        [0.090, 0.016],
        [0.080, 0.017],
        [0.100, 0.018],
        [0.110, 0.019],
    ])

    build_nmf_report(
        scores=scores,
        components=np.array([[0.8, 0.2], [0.1, 0.9]]),
        animate_inanimate=np.array([1, 0] * 5, dtype=np.int8),
        image_paths=[tmp_path / f"img{i}.png" for i in range(10)],
        output_path=output_path,
        class_labels=["class0", "class1"],
        image_thumbs=[f"thumb{i}" for i in range(10)],
        heatmap_thumb_threshold=0.01,
    )

    html = output_path.read_text(encoding="utf-8")

    assert 'HEATMAP_COMPONENT_INDICES=[[9, 8, 6, 7, 3, 5, 4, 1], [9, 8, 7, 6, 5, 4, 3, 2]]' in html
    assert 'hover-thumb-grid' in html
    assert 'grid-template-columns:repeat(4,minmax(0,72px));gap:6px;' in html
    assert 'showGrid(evt,getHeatmapComponentIndex(p),HEATMAP_COMPONENT_INDICES[getHeatmapComponentIndex(p)])' in html
    assert "typeof idx!=='number'||!Number.isFinite(idx)" in html
    assert "HEATMAP_THUMB_THRESHOLD=0.01" in html
    assert "HEATMAP_THUMB_MAX_IMAGES=8" in html
    assert "thumbnail tile threshold \\u003e 0.01, max 8" in html