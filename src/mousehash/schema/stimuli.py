from __future__ import annotations

import datajoint as dj

from mousehash.config import DJ_HOST, DJ_PORT, DJ_USER, DJ_PASS, DB_PREFIX

dj.config["database.host"] = DJ_HOST
dj.config["database.port"] = DJ_PORT
dj.config["database.user"] = DJ_USER
dj.config["database.password"] = DJ_PASS

schema = dj.Schema(f"{DB_PREFIX}_stimuli")


@schema
class AllenNaturalSceneSet(dj.Manual):
    definition = """
    scene_set_id: varchar(128)
    ---
    dataset_name: varchar(255)
    stimulus_name: varchar(255)
    n_images: int
    notes='': varchar(1024)
    created_at=CURRENT_TIMESTAMP: timestamp
    """


@schema
class AllenNaturalSceneImage(dj.Manual):
    definition = """
    -> AllenNaturalSceneSet
    image_idx: int
    ---
    image_path: varchar(1024)
    height=null: int
    width=null: int
    image_sha1='': varchar(40)
    """