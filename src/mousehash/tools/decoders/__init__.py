"""Decoder model tools.

Decoders consume an `AnalysisView` (typically OBSERVATION_BY_FEATURE) plus a
labels vector and produce a metric-table view describing classification
performance.
"""

from mousehash.tools.decoders.logistic_decoder import (
    LOGISTIC_DECODER_CONTRACT,
    run_logistic_decoder,
)

__all__ = ["LOGISTIC_DECODER_CONTRACT", "run_logistic_decoder"]
