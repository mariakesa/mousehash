from __future__ import annotations

import datajoint as dj

from mousehash.config import DB_PREFIX
from mousehash.schema.decompositions import StimulusDecomposition

schema = dj.Schema(f"{DB_PREFIX}_reports")


@schema
class StimulusDecompositionReport(dj.Manual):
    definition = """
    -> StimulusDecomposition
    report_type: varchar(64)
    ---
    report_path: varchar(1024)
    summary_path: varchar(1024)
    created_at=CURRENT_TIMESTAMP: timestamp
    """