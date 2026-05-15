"""Linear factor models: PCA, NMF, FA, GPFA, etc."""

from mousehash.tools.factor_models.nmf import NMF_CONTRACT, apply_probability_temperature, run_nmf
from mousehash.tools.factor_models.pca import PCA_CONTRACT, run_pca

__all__ = [
    "NMF_CONTRACT",
    "PCA_CONTRACT",
    "apply_probability_temperature",
    "run_nmf",
    "run_pca",
]
