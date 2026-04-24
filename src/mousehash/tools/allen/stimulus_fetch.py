from __future__ import annotations

from pathlib import Path

import numpy as np


NATURAL_SCENES_EXPERIMENT_ID = 501559087


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

    exp_ids = {exp["id"] for exp in exps}
    if NATURAL_SCENES_EXPERIMENT_ID not in exp_ids:
        raise ValueError(
            "Configured natural-scenes experiment "
            f"{NATURAL_SCENES_EXPERIMENT_ID} is unavailable in manifest at {manifest_path}"
        )

    # Match the original ProjectionSort pipeline, which used SESSION_B.
    exp_id = NATURAL_SCENES_EXPERIMENT_ID
    dataset = boc.get_ophys_experiment_data(exp_id)
    template = dataset.get_stimulus_template("natural_scenes")
    return template  # (n_images, H, W) uint8
