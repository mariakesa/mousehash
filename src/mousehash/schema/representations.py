from __future__ import annotations

import datajoint as dj

from mousehash.config import DB_PREFIX
from mousehash.schema.stimuli import StimulusSet

schema = dj.Schema(f"{DB_PREFIX}_representations")


@schema
class RepresentationSpec(dj.Lookup):
    definition = """
    representation_spec_id: varchar(128)
    ---
    model_name: varchar(255)
    model_family: varchar(128)
    feature_space: varchar(128)
    preprocessing: varchar(128)
    batch_size: int
    device: varchar(32)
    """


@schema
class AnimateInanimateRule(dj.Lookup):
    definition = """
    rule_id: varchar(128)
    ---
    rule_name: varchar(255)
    threshold_max_class_idx: int
    description='': varchar(1024)
    """


@schema
class StimulusRepresentation(dj.Manual):
    definition = """
    -> StimulusSet
    -> RepresentationSpec
    -> AnimateInanimateRule
    ---
    n_images: int
    n_classes: int
    logits_path: varchar(1024)
    probabilities_path: varchar(1024)
    top1_path: varchar(1024)
    animate_inanimate_path: varchar(1024)
    summary_path: varchar(1024)
    created_at=CURRENT_TIMESTAMP: timestamp
    """