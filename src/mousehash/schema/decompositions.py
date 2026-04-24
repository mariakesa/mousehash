from __future__ import annotations

import datajoint as dj

from mousehash.config import DB_PREFIX
from mousehash.schema.representations import StimulusRepresentation

schema = dj.Schema(f"{DB_PREFIX}_decompositions")


@schema
class DecompositionSpec(dj.Lookup):
    definition = """
    decomposition_spec_id: varchar(128)
    ---
    method: varchar(64)
    input_kind: varchar(64)
    n_components: int
    normalize_input: bool
    mode: varchar(64)
    """


@schema
class StimulusDecomposition(dj.Manual):
    definition = """
    -> StimulusRepresentation
    -> DecompositionSpec
    ---
    scores_path: varchar(1024)
    components_path: varchar(1024)
    component_stats_path: varchar(1024)
    summary_path: varchar(1024)
    created_at=CURRENT_TIMESTAMP: timestamp
    """