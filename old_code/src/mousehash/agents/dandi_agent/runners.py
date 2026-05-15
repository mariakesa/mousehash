"""Runner stubs for the MVP-14 DANDI tools.

These functions exist so the BlahML validator can resolve
``tool_binding.python_function`` paths in ``blahml/manifests/dandi_*.yaml``.
Actual scientific implementations land in MVP 2; the stubs raise
``NotImplementedError`` rather than silently no-op so a premature call is
obvious.
"""

from __future__ import annotations

from typing import Any


def _todo(name: str) -> Any:
    raise NotImplementedError(
        f"{name} is a stub for the MouseHash DANDI agent MVP-2 implementation."
    )


def generate_raster_plot(*args: Any, **kwargs: Any) -> Any:
    return _todo("generate_raster_plot")


def generate_psth_plot(*args: Any, **kwargs: Any) -> Any:
    return _todo("generate_psth_plot")


def run_pca(*args: Any, **kwargs: Any) -> Any:
    return _todo("run_pca")


def run_trial_averaged_pca(*args: Any, **kwargs: Any) -> Any:
    return _todo("run_trial_averaged_pca")


def fit_logistic_decoder(*args: Any, **kwargs: Any) -> Any:
    return _todo("fit_logistic_decoder")


def cv_decoding_eval(*args: Any, **kwargs: Any) -> Any:
    return _todo("cv_decoding_eval")


def fit_ridge_encoding(*args: Any, **kwargs: Any) -> Any:
    return _todo("fit_ridge_encoding")


def cv_encoding_eval(*args: Any, **kwargs: Any) -> Any:
    return _todo("cv_encoding_eval")


def compute_rsm(*args: Any, **kwargs: Any) -> Any:
    return _todo("compute_rsm")


def run_rsa(*args: Any, **kwargs: Any) -> Any:
    return _todo("run_rsa")


def compute_noise_correlations(*args: Any, **kwargs: Any) -> Any:
    return _todo("compute_noise_correlations")


def run_permutation_test(*args: Any, **kwargs: Any) -> Any:
    return _todo("run_permutation_test")


def generate_latent_trajectory_plot(*args: Any, **kwargs: Any) -> Any:
    return _todo("generate_latent_trajectory_plot")


def generate_structure_discovery_report(*args: Any, **kwargs: Any) -> Any:
    return _todo("generate_structure_discovery_report")
