from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import NMF

logger = logging.getLogger(__name__)


def run_nmf(
    probabilities: np.ndarray,
    n_components: int = 10,
) -> dict:
    """Fit NMF on softmax probabilities and return scores, components, and stats.

    NMF on probabilities recovers nonnegative additive semantic mixture
    structure: each component is an additive semantic theme, and each image's
    score vector shows how much of each theme it contains.

    Args:
        probabilities: (n_images, n_classes) float32 softmax probabilities.
                       Must be non-negative (guaranteed by softmax).
        n_components:  Number of NMF components.

    Returns:
        Dict with keys:
            ``scores``            – (n_images, n_components) float32 (the W matrix).
            ``components``        – (n_components, n_classes) float32 (the H matrix).
            ``reconstruction_err``– scalar float, Frobenius reconstruction error.
            ``n_iter``            – number of iterations run.
    """
    X = probabilities.astype(np.float64)

    nmf = NMF(
        n_components=n_components,
        init="nndsvda",   # deterministic init; better than random for small n
        random_state=0,
        max_iter=500,
    )
    scores = nmf.fit_transform(X).astype(np.float32)

    logger.info(
        "NMF: %d components, reconstruction error=%.4f, %d iterations",
        n_components,
        nmf.reconstruction_err_,
        nmf.n_iter_,
    )

    return dict(
        scores=scores,
        components=nmf.components_.astype(np.float32),
        reconstruction_err=float(nmf.reconstruction_err_),
        n_iter=int(nmf.n_iter_),
    )
