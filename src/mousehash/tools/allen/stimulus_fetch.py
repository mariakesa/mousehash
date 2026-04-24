from __future__ import annotations

from pathlib import Path

import numpy as np


def fetch_natural_scene_template(manifest_path: Path) -> np.ndarray:
    """Return (n_images, height, width) uint8 array of natural scene stimuli.

    Downloads session data on first call if not cached under manifest_path's
    parent directory.  Requires allensdk to be installed.
    """
    try:
        from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    except ImportError as e:
        raise ImportError(
            "allensdk is required for Allen ingestion.\n"
            "Install it with: pip install allensdk"
        ) from e

    boc = BrainObservatoryCache(manifest_file=str(manifest_path))
    exps = boc.get_ophys_experiments(stimuli=["natural_scenes"])
    if not exps:
        raise ValueError(
            f"No natural-scene experiments found in manifest at {manifest_path}"
        )

    # Any session with natural scenes carries the same template; use the first.
    exp_id = exps[0]["id"]
    dataset = boc.get_ophys_experiment_data(exp_id)
    template = dataset.get_stimulus_template("natural_scenes")
    return template  # (n_images, H, W) uint8
