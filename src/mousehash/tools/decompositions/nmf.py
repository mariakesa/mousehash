from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import NMF

logger = logging.getLogger(__name__)


def apply_probability_temperature(
    probabilities: np.ndarray,
    temperature: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Apply temperature smoothing/sharpening to an existing probability matrix.

    This is equivalent to applying softmax(logits / temperature) if the
    probabilities originally came from softmax(logits), up to numerical issues.

    Args:
        probabilities: Array of shape (n_samples, n_classes). Rows should sum to 1.
        temperature:
            1.0 = unchanged
            >1.0 = smoother / more diluted
            <1.0 = sharper / more concentrated
        eps: Small value to avoid log/zero issues.

    Returns:
        Temperature-adjusted probabilities with rows summing to 1.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    P = probabilities.astype(np.float64)

    if np.any(P < 0):
        raise ValueError("probabilities must be non-negative")

    # Avoid exact zeros before exponentiation.
    P = np.clip(P, eps, None)

    # p_T ∝ p^(1/T)
    P_temp = P ** (1.0 / temperature)

    row_sums = P_temp.sum(axis=1, keepdims=True)

    if np.any(row_sums <= 0):
        raise ValueError("temperature-adjusted probabilities have invalid row sums")

    P_temp = P_temp / row_sums

    return P_temp


def run_nmf(
    probabilities: np.ndarray,
    n_components: int = 10,
    init: str = "nndsvda",
    solver: str = "cd",
    beta_loss: str = "frobenius",
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: int = 0,
    alpha_W: float = 0.0,
    alpha_H: float = 0.0,
    l1_ratio: float = 0.0,
    temperature: float = 1.0,
) -> dict:
    """Fit NMF on softmax probabilities and return scores, components, and stats.

    NMF on probabilities recovers nonnegative additive semantic mixture
    structure: each component is an additive semantic theme, and each image's
    score vector shows how much of each theme it contains.

    Args:
        probabilities: (n_images, n_classes) softmax probabilities. Rows should
                       sum to 1 and all entries must be non-negative.
        n_components:  Number of NMF components.
        init, solver, beta_loss, max_iter, tol, random_state, alpha_W, alpha_H,
        l1_ratio:      Forwarded directly to ``sklearn.decomposition.NMF``.
        temperature:   Temperature applied to the input probabilities before
                       fitting (T>1 smooths, T<1 sharpens, T=1 is a no-op).

    Returns:
        Dict with keys:
            ``scores``            – (n_images, n_components) float32 (the W matrix).
            ``components``        – (n_components, n_classes) float32 (the H matrix).
            ``reconstruction_err``– scalar float, Frobenius reconstruction error.
            ``n_iter``            – number of iterations run.
    """
    if temperature != 1.0:
        X = apply_probability_temperature(probabilities, temperature)
    else:
        X = probabilities.astype(np.float64)

    nmf = NMF(
        n_components=n_components,
        init=init,
        solver=solver,
        beta_loss=beta_loss,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        l1_ratio=l1_ratio,
    )
    scores = nmf.fit_transform(X).astype(np.float32)

    logger.info(
        "NMF: %d components, T=%.3f, reconstruction error=%.4f, %d iterations",
        n_components,
        temperature,
        nmf.reconstruction_err_,
        nmf.n_iter_,
    )

    return dict(
        scores=scores,
        components=nmf.components_.astype(np.float32),
        reconstruction_err=float(nmf.reconstruction_err_),
        n_iter=int(nmf.n_iter_),
    )
