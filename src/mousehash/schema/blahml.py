from __future__ import annotations

import datajoint as dj

from mousehash.config import DB_PREFIX

schema = dj.Schema(f"{DB_PREFIX}_blahml")


@schema
class ToolRunSpec(dj.Manual):
    """Audit row for a BlahML-resolved tool invocation.

    Captures *what was asked* (manifest identity + sha256), *what got
    answered* (parameters_json, input_artifacts_json), and *why each
    parameter was chosen* (question_trace_json). The manifest YAML itself
    is the source of truth in git; ``manifest_sha256`` lets later audits
    detect drift.
    """

    definition = """
    tool_run_spec_id: varchar(64)
    ---
    tool_id: varchar(64)
    manifest_version: varchar(32)
    manifest_sha256: char(64)
    parameters_json: longblob
    question_trace_json: longblob
    input_artifacts_json: longblob
    created_at: datetime
    created_by: varchar(64)
    """
