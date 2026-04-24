from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def run_pca(
    logits: np.ndarray,
    n_components: int = 10,
    normalize_input: bool = True,
) -> dict:
    """Fit PCA on logits and return scores, components, and per-component stats.

    Args:
        logits:          (n_images, n_classes) float32 array of raw classifier logits.
        n_components:    Number of PCA components to retain.
        normalize_input: If True, z-score each feature before PCA (recommended
                         for logits so dominant classes don't dominate variance).

    Returns:
        Dict with keys:
            ``scores``           – (n_images, n_components) float32 projections.
            ``components``       – (n_components, n_classes) float32 component matrix.
            ``explained_variance_ratio`` – (n_components,) float64 array.
            ``singular_values``  – (n_components,) float64 array.
            ``mean``             – (n_classes,) mean used for centering (post-scaling).
    """
    X = logits.astype(np.float64)

    if normalize_input:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        logger.info("Normalized logits: mean=%.3f std=%.3f", scaler.mean_.mean(), scaler.scale_.mean())

    pca = PCA(n_components=n_components, random_state=0)
    scores = pca.fit_transform(X).astype(np.float32)

    logger.info(
        "PCA: %d components explain %.1f%% variance",
        n_components,
        pca.explained_variance_ratio_.sum() * 100,
    )

    return dict(
        scores=scores,
        components=pca.components_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_,
        singular_values=pca.singular_values_,
        mean=pca.mean_.astype(np.float32),
    )
