from dandi.dandiapi import DandiAPIClient

import json
import re
import time
from pathlib import Path
from collections import Counter

import pandas as pd
from tqdm.auto import tqdm


OUT_DIR = Path("mousehash_dandi_scan")
OUT_DIR.mkdir(exist_ok=True)

MAX_DANDISETS = None        # set to e.g. 50 for test
SLEEP_SECONDS = 0.05
SAVE_EVERY = 50


MOUSEHASH_ONTOLOGY = {
    "neural_data": [
        "spikes",
        "lfp",
        "eeg",
        "ecog",
        "calcium",
        "photometry",
        "images",
        "intracellular_recording",
        "voltage_imaging",
    ],
    "stimuli": {
        "sensory": [
            "visual",
            "auditory",
            "tactile",
            "odor",
            "gustatory",
            "thermal",
            "vestibular",
        ],
        "interventions": [
            "optogenetic",
            "electrical",
            "pharmacological",
            "anesthesia",
            "stimulation",
            "surgical",
        ],
    },
    "behavior": [
        "choices",
        "reaction_times",
        "pose",
        "locomotion",
        "pupil",
        "kinematics",
        "behavioral_events",
        "behavioral_states",
        "licking",
        "reward",
        "whisking",
        "eye_movements",
    ],
    "conditions": [
        "task_labels",
        "trial_labels",
        "experimental_groups",
        "brain_states",
        "session_phases",
        "stimulus_labels",
        "animal_groups",
    ],
    "time_organization": [
        "continuous_time",
        "trials",
        "epochs",
        "events",
        "frames",
        "alignment_rules",
        "timestamps",
        "sampling_rate",
    ],
    "metadata": [
        "subject",
        "species",
        "genotype",
        "sex",
        "age",
        "strain",
        "session",
        "brain_area",
        "probe/electrode/imaging_plane",
        "acquisition_device",
        "preprocessing_info",
        "data_standard",
        "license",
        "contributors",
        "protocol",
        "anatomy",
        "number_of_subjects",
        "number_of_files",
        "number_of_bytes",
    ],
}


# This was missing in your pasted script.
ROLE_RULES = [
    # Neural data
    {
        "role": "neural_data.spikes",
        "patterns": [
            r"\bUnits\b",
            r"spike sorting",
            r"\bspike(s)?\b",
            r"extracellular electrophysiology",
            r"electrophysiological approach",
        ],
        "status": "present",
        "confidence": "high",
    },
    {
        "role": "neural_data.lfp",
        "patterns": [
            r"\bLFP\b",
            r"local field potential",
        ],
        "status": "present",
        "confidence": "high",
    },
    {
        "role": "neural_data.eeg",
        "patterns": [
            r"\bEEG\b",
            r"electroencephalography",
        ],
        "status": "present",
        "confidence": "high",
    },
    {
        "role": "neural_data.ecog",
        "patterns": [
            r"\bECoG\b",
            r"electrocorticography",
        ],
        "status": "present",
        "confidence": "high",
    },
    {
        "role": "neural_data.calcium",
        "patterns": [
            r"ophys",
            r"two[- ]?photon",
            r"2[- ]?photon",
            r"calcium",
            r"Fluorescence",
            r"DfOverF",
            r"PlaneSegmentation",
            r"RoiResponseSeries",
            r"ImageSegmentation",
        ],
        "status": "present",
        "confidence": "high",
    },
    {
        "role": "neural_data.photometry",
        "patterns": [
            r"photometry",
            r"fiber photometry",
        ],
        "status": "present",
        "confidence": "high",
    },
    {
        "role": "neural_data.images",
        "patterns": [
            r"ImageSeries",
            r"TwoPhotonSeries",
            r"OnePhotonSeries",
            r"microscopy",
        ],
        "status": "likely",
        "confidence": "medium",
    },
    {
        "role": "neural_data.intracellular_recording",
        "patterns": [
            r"icephys",
            r"intracellular",
            r"patch clamp",
            r"current clamp",
            r"voltage clamp",
        ],
        "status": "present",
        "confidence": "high",
    },

    # Stimuli
    {
        "role": "stimuli.sensory.visual",
        "patterns": [
            r"visual",
            r"image",
            r"movie",
            r"natural scene",
            r"grating",
            r"drifting",
        ],
        "status": "likely",
        "confidence": "medium",
    },
    {
        "role": "stimuli.sensory.auditory",
        "patterns": [
            r"auditory",
            r"sound",
            r"tone",
            r"vocalization",
        ],
        "status": "likely",
        "confidence": "medium",
    },
    {
        "role": "stimuli.sensory.tactile",
        "patterns": [
            r"tactile",
            r"touch",
            r"whisker stimulation",
            r"vibrotactile",
        ],
        "status": "likely",
        "confidence": "medium",
    },
    {
        "role": "stimuli.sensory.odor",
        "patterns": [
            r"odor",
            r"olfactory",
            r"smell",
        ],
        "status": "likely",
        "confidence": "medium",
    },
    {
        "role": "stimuli.interventions.optogenetic",
        "patterns": [
            r"optogenetic",
            r"optogenetics",
            r"\bopto\b",
            r"photostimulation",
        ],
        "status": "likely",
        "confidence": "medium",
    },
    {
        "role": "stimuli.interventions.electrical",
        "patterns": [
            r"electrical stimulation",
            r"electrical stimulus",
            r"stimulation electrode",
        ],
        "status": "likely",
        "confidence": "medium",
    },
    {
        "role": "stimuli.interventions.pharmacological",
        "patterns": [
            r"pharmacological",
            r"drug",
            r"agonist",
            r"antagonist",
            r"injection",
        ],
        "status": "likely",
        "confidence": "medium",
    },
    {
        "role": "stimuli.interventions.anesthesia",
        "patterns": [
            r"anesthesia",
            r"anaesthesia",
            r"anesthetic",
            r"isoflurane",
            r"ketamine",
        ],
        "status": "likely",
        "confidence": "medium",
    },
    {
        "role": "stimuli.interventions.surgical",
        "patterns": [
            r"surgical technique",
            r"surgery",
            r"implant",
        ],
        "status": "likely",
        "confidence": "low",
    },

    # Behavior
    {
        "role": "behavior.behavioral_events",
        "patterns": [
            r"BehavioralEvents",
            r"behavioral events",
            r"behavioral technique",
            r"behavioral approach",
            r"lick",
            r"reward",
            r"lever",
            r"poke",
        ],
        "status": "present",
        "confidence": "high",
    },
    {
        "role": "behavior.behavioral_states",
        "patterns": [
            r"BehavioralEpochs",
            r"behavioral states",
            r"sleep",
            r"wake",
            r"arousal",
        ],
        "status": "likely",
        "confidence": "medium",
    },
    {
        "role": "behavior.choices",
        "patterns": [
            r"choice",
            r"decision",
            r"left choice",
            r"right choice",
            r"response side",
        ],
        "status": "likely",
        "confidence": "medium",
    },
    {
        "role": "behavior.reaction_times",
        "patterns": [
            r"reaction time",
            r"response time",
            r"latency",
        ],
        "status": "likely",
        "confidence": "medium",
    },
    {
        "role": "behavior.pose",
        "patterns": [
            r"pose",
            r"PoseEstimation",
            r"DeepLabCut",
            r"SLEAP",
            r"keypoint",
        ],
        "status": "present",
        "confidence": "high",
    },
    {
        "role": "behavior.locomotion",
        "patterns": [
            r"locomotion",
            r"running",
            r"wheel",
            r"speed",
            r"velocity",
        ],
        "status": "likely",
        "confidence": "medium",
    },
    {
        "role": "behavior.pupil",
        "patterns": [
            r"pupil",
            r"eye tracking",
            r"EyeTracking",
        ],
        "status": "likely",
        "confidence": "medium",
    },
    {
        "role": "behavior.kinematics",
        "patterns": [
            r"kinematic",
            r"movement",
            r"motion",
            r"reaching",
            r"trajectory",
        ],
        "status": "likely",
        "confidence": "medium",
    },
    {
        "role": "behavior.licking",
        "patterns": [
            r"lick",
            r"licking",
            r"lickometer",
        ],
        "status": "likely",
        "confidence": "medium",
    },
    {
        "role": "behavior.reward",
        "patterns": [
            r"reward",
            r"water delivery",
            r"reinforcement",
        ],
        "status": "likely",
        "confidence": "medium",
    },
    {
        "role": "behavior.whisking",
        "patterns": [
            r"whisk",
            r"whisker",
            r"vibrissa",
        ],
        "status": "likely",
        "confidence": "medium",
    },

    # Conditions
    {
        "role": "conditions.task_labels",
        "patterns": [
            r"task",
            r"trial type",
            r"condition",
            r"delay response",
            r"go/no-go",
            r"discrimination",
        ],
        "status": "likely",
        "confidence": "medium",
    },
    {
        "role": "conditions.trial_labels",
        "patterns": [
            r"\btrials\b",
            r"\btrial\b",
            r"trial labels",
        ],
        "status": "likely",
        "confidence": "medium",
    },
    {
        "role": "conditions.experimental_groups",
        "patterns": [
            r"experimental group",
            r"control group",
            r"treatment group",
        ],
        "status": "likely",
        "confidence": "low",
    },
    {
        "role": "conditions.brain_states",
        "patterns": [
            r"brain state",
            r"sleep state",
            r"anesthesia state",
        ],
        "status": "likely",
        "confidence": "medium",
    },
    {
        "role": "conditions.session_phases",
        "patterns": [
            r"session phase",
            r"baseline",
            r"delay period",
            r"response period",
            r"epoch",
        ],
        "status": "likely",
        "confidence": "medium",
    },

    # Time organization
    {
        "role": "time_organization.continuous_time",
        "patterns": [
            r"TimeSeries",
            r"timestamps",
            r"sampling rate",
            r"ElectricalSeries",
            r"RoiResponseSeries",
            r"BehavioralEvents",
            r"\bUnits\b",
        ],
        "status": "likely",
        "confidence": "medium",
    },
    {
        "role": "time_organization.trials",
        "patterns": [
            r"\btrials\b",
            r"\btrial\b",
        ],
        "status": "likely",
        "confidence": "medium",
    },
    {
        "role": "time_organization.epochs",
        "patterns": [
            r"epoch",
            r"interval",
            r"BehavioralEpochs",
        ],
        "status": "likely",
        "confidence": "medium",
    },
    {
        "role": "time_organization.events",
        "patterns": [
            r"event",
            r"BehavioralEvents",
            r"stimulus onset",
            r"lick",
            r"reward",
        ],
        "status": "likely",
        "confidence": "medium",
    },
    {
        "role": "time_organization.frames",
        "patterns": [
            r"frame",
            r"frames",
            r"movie",
            r"ImageSeries",
            r"TwoPhotonSeries",
            r"OnePhotonSeries",
        ],
        "status": "likely",
        "confidence": "medium",
    },
    {
        "role": "time_organization.timestamps",
        "patterns": [
            r"timestamps",
            r"time stamps",
        ],
        "status": "likely",
        "confidence": "medium",
    },

    # Metadata
    {
        "role": "metadata.subject",
        "patterns": [
            r"subject",
            r"numberOfSubjects",
        ],
        "status": "present",
        "confidence": "high",
    },
    {
        "role": "metadata.species",
        "patterns": [
            r"species",
            r"House mouse",
            r"Mus musculus",
            r"NCBITaxon",
        ],
        "status": "present",
        "confidence": "high",
    },
    {
        "role": "metadata.genotype",
        "patterns": [
            r"genotype",
            r"strain",
            r"transgenic",
            r"\bcre\b",
        ],
        "status": "likely",
        "confidence": "medium",
    },
    {
        "role": "metadata.brain_area",
        "patterns": [
            r"Anatomy",
            r"brain area",
            r"cortex",
            r"hippocampus",
            r"thalamus",
            r"striatum",
            r"UBERON",
        ],
        "status": "present",
        "confidence": "high",
    },
    {
        "role": "metadata.probe/electrode/imaging_plane",
        "patterns": [
            r"ElectrodeGroup",
            r"electrode",
            r"probe",
            r"imaging plane",
            r"ImagingPlane",
            r"PlaneSegmentation",
        ],
        "status": "likely",
        "confidence": "high",
    },
    {
        "role": "metadata.acquisition_device",
        "patterns": [
            r"Device",
            r"acquisition device",
            r"microscope",
            r"camera",
        ],
        "status": "likely",
        "confidence": "medium",
    },
    {
        "role": "metadata.preprocessing_info",
        "patterns": [
            r"spike sorting technique",
            r"processed",
            r"preprocessing",
            r"segmentation",
            r"Suite2p",
            r"CaImAn",
        ],
        "status": "likely",
        "confidence": "medium",
    },
    {
        "role": "metadata.data_standard",
        "patterns": [
            r"Neurodata Without Borders",
            r"\bNWB\b",
        ],
        "status": "present",
        "confidence": "high",
    },
    {
        "role": "metadata.license",
        "patterns": [
            r"CC-BY",
            r"CC0",
            r"license",
        ],
        "status": "present",
        "confidence": "high",
    },
]


STATUS_RANK = {"absent": 0, "unknown": 1, "likely": 2, "present": 3}
CONF_RANK = {"low": 1, "medium": 2, "high": 3}


def blank_role_entry():
    return {
        "status": "unknown",
        "confidence": "low",
        "evidence": [],
        "sources": [],
    }


def make_blank_manifest():
    m = {}
    for family, value in MOUSEHASH_ONTOLOGY.items():
        if isinstance(value, dict):
            m[family] = {}
            for subfamily, terms in value.items():
                m[family][subfamily] = {
                    term: blank_role_entry() for term in terms
                }
        else:
            m[family] = {
                term: blank_role_entry() for term in value
            }
    return {"mousehash_roles": m}


def get_nested(d, path):
    node = d
    for part in path.split("."):
        node = node[part]
    return node


def mark(manifest, role, status, confidence, source, evidence):
    entry = get_nested(manifest["mousehash_roles"], role)

    if STATUS_RANK[status] >= STATUS_RANK[entry["status"]]:
        entry["status"] = status
    if CONF_RANK[confidence] >= CONF_RANK[entry["confidence"]]:
        entry["confidence"] = confidence

    if source not in entry["sources"]:
        entry["sources"].append(source)
    if evidence not in entry["evidence"]:
        entry["evidence"].append(evidence)


def safe_jsonable(x):
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {str(k): safe_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [safe_jsonable(v) for v in x]
    if hasattr(x, "model_dump"):
        try:
            return safe_jsonable(x.model_dump())
        except Exception:
            pass
    if hasattr(x, "dict"):
        try:
            return safe_jsonable(x.dict())
        except Exception:
            pass
    return str(x)


def flatten_text(obj):
    return json.dumps(safe_jsonable(obj), default=str, ensure_ascii=False)


def field_values_from_assets_summary(summary):
    values = []

    def add(source, value):
        if value is None:
            return
        if isinstance(value, list):
            for v in value:
                add(source, v)
        elif isinstance(value, dict):
            if "name" in value:
                values.append((source + ".name", str(value["name"])))
            if "identifier" in value:
                values.append((source + ".identifier", str(value["identifier"])))
            if "schemaKey" in value:
                values.append((source + ".schemaKey", str(value["schemaKey"])))
            for k, v in value.items():
                if k not in {"name", "identifier", "schemaKey"}:
                    add(source + "." + k, v)
        else:
            values.append((source, str(value)))

    for key, value in (summary or {}).items():
        add(f"assetsSummary.{key}", value)

    return values


def metadata_values_for_matching(raw_meta):
    values = []

    summary = raw_meta.get("assetsSummary", {}) or {}
    values.extend(field_values_from_assets_summary(summary))

    for key in [
        "name",
        "description",
        "citation",
        "license",
        "keywords",
        "protocol",
        "studyTarget",
    ]:
        if key in raw_meta:
            values.append((key, flatten_text(raw_meta.get(key))))

    for item in raw_meta.get("about", []) or []:
        values.append(("about", flatten_text(item)))
        if isinstance(item, dict):
            for k in ["name", "identifier", "schemaKey"]:
                if k in item:
                    values.append((f"about.{k}", str(item[k])))

    return values


def apply_rules(raw_meta):
    manifest = make_blank_manifest()
    evidence_rows = []

    values = metadata_values_for_matching(raw_meta)

    for source, value in values:
        for rule in ROLE_RULES:
            for pat in rule["patterns"]:
                if re.search(pat, value, flags=re.IGNORECASE):
                    mark(
                        manifest=manifest,
                        role=rule["role"],
                        status=rule["status"],
                        confidence=rule["confidence"],
                        source=source,
                        evidence=f"matched pattern {pat!r} in value {value[:250]!r}",
                    )
                    evidence_rows.append(
                        {
                            "role": rule["role"],
                            "status": rule["status"],
                            "confidence": rule["confidence"],
                            "source": source,
                            "pattern": pat,
                            "value": value[:500],
                        }
                    )

    summary = raw_meta.get("assetsSummary", {}) or {}

    if summary.get("numberOfSubjects") is not None:
        mark(
            manifest,
            "metadata.number_of_subjects",
            "present",
            "high",
            "assetsSummary.numberOfSubjects",
            f"numberOfSubjects={summary.get('numberOfSubjects')}",
        )

    if summary.get("numberOfFiles") is not None:
        mark(
            manifest,
            "metadata.number_of_files",
            "present",
            "high",
            "assetsSummary.numberOfFiles",
            f"numberOfFiles={summary.get('numberOfFiles')}",
        )

    if summary.get("numberOfBytes") is not None:
        mark(
            manifest,
            "metadata.number_of_bytes",
            "present",
            "high",
            "assetsSummary.numberOfBytes",
            f"numberOfBytes={summary.get('numberOfBytes')}",
        )

    return manifest, pd.DataFrame(evidence_rows)


def summarize_manifest_presence(manifest):
    rows = []

    def walk(prefix, node):
        if (
            isinstance(node, dict)
            and {"status", "confidence", "evidence", "sources"} <= set(node.keys())
        ):
            rows.append(
                {
                    "role": prefix,
                    "status": node["status"],
                    "confidence": node["confidence"],
                    "n_evidence": len(node["evidence"]),
                    "sources": "; ".join(node["sources"][:5]),
                }
            )
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(f"{prefix}.{k}" if prefix else k, v)

    walk("", manifest["mousehash_roles"])
    return pd.DataFrame(rows)


def suggest_schema_extensions_from_unmapped_values(raw_records, top_n=100):
    all_values = []

    for rec in raw_records:
        raw_meta = rec.get("raw_metadata", {})
        summary = raw_meta.get("assetsSummary", {}) or {}
        vals = field_values_from_assets_summary(summary)

        for source, value in vals:
            all_values.append((source, value))

    mapped_patterns = [pat for rule in ROLE_RULES for pat in rule["patterns"]]

    unmapped = []
    for source, value in all_values:
        matched = any(
            re.search(pat, value, flags=re.IGNORECASE)
            for pat in mapped_patterns
        )
        if not matched:
            unmapped.append((source, value))

    counter = Counter(unmapped)
    rows = [
        {"source": source, "value": value, "count": count}
        for (source, value), count in counter.most_common(top_n)
    ]
    return pd.DataFrame(rows)


def crawl_dandi():
    records = []
    errors = []

    with DandiAPIClient.for_dandi_instance("dandi") as client:
        dandisets_iter = client.get_dandisets()

        for i, dandiset in enumerate(tqdm(dandisets_iter, desc="Scanning Dandisets")):
            if MAX_DANDISETS is not None and i >= MAX_DANDISETS:
                break

            try:
                dandiset_id = (
                    getattr(dandiset, "identifier", None)
                    or getattr(dandiset, "dandiset_id", None)
                )
                version_id = getattr(dandiset, "version_id", None) or "unknown"

                raw_meta = dandiset.get_raw_metadata()
                raw_meta = safe_jsonable(raw_meta)

                manifest, evidence_df = apply_rules(raw_meta)
                role_summary_df = summarize_manifest_presence(manifest)

                records.append(
                    {
                        "dandiset_id": raw_meta.get("identifier", dandiset_id),
                        "id": raw_meta.get("id"),
                        "version": raw_meta.get("version", version_id),
                        "name": raw_meta.get("name"),
                        "license": raw_meta.get("license"),
                        "url": raw_meta.get("url"),
                        "assetsSummary": raw_meta.get("assetsSummary", {}),
                        "raw_metadata": raw_meta,
                        "manifest": manifest,
                        "evidence": evidence_df.to_dict(orient="records"),
                        "role_summary": role_summary_df.to_dict(orient="records"),
                    }
                )

            except Exception as e:
                errors.append(
                    {
                        "index": i,
                        "dandiset_repr": repr(dandiset),
                        "error": repr(e),
                    }
                )

            if (i + 1) % SAVE_EVERY == 0:
                tmp_path = OUT_DIR / "partial_dandi_mousehash_scan.json"
                tmp_path.write_text(
                    json.dumps(records, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                err_path = OUT_DIR / "partial_dandi_mousehash_errors.json"
                err_path.write_text(
                    json.dumps(errors[:50], indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

            if SLEEP_SECONDS:
                time.sleep(SLEEP_SECONDS)

    return records, errors


def build_tables(records):
    dandiset_rows = []

    for rec in records:
        summary = rec.get("assetsSummary") or {}

        dandiset_rows.append(
            {
                "dandiset_id": rec.get("dandiset_id"),
                "id": rec.get("id"),
                "version": rec.get("version"),
                "name": rec.get("name"),
                "license": json.dumps(rec.get("license"), ensure_ascii=False),
                "url": rec.get("url"),
                "numberOfFiles": summary.get("numberOfFiles"),
                "numberOfBytes": summary.get("numberOfBytes"),
                "numberOfSubjects": summary.get("numberOfSubjects"),
                "species": "; ".join(
                    [
                        x.get("name", "")
                        for x in summary.get("species", []) or []
                        if isinstance(x, dict)
                    ]
                ),
                "variableMeasured": "; ".join(summary.get("variableMeasured", []) or []),
                "approach": "; ".join(
                    [
                        x.get("name", "")
                        for x in summary.get("approach", []) or []
                        if isinstance(x, dict)
                    ]
                ),
                "measurementTechnique": "; ".join(
                    [
                        x.get("name", "")
                        for x in summary.get("measurementTechnique", []) or []
                        if isinstance(x, dict)
                    ]
                ),
                "dataStandard": "; ".join(
                    [
                        x.get("name", "")
                        for x in summary.get("dataStandard", []) or []
                        if isinstance(x, dict)
                    ]
                ),
            }
        )

    dandisets_df = pd.DataFrame(dandiset_rows)

    role_presence_rows = []

    for rec in records:
        for row in rec["role_summary"]:
            if row["status"] in {"present", "likely"}:
                role_presence_rows.append(
                    {
                        "dandiset_id": rec["dandiset_id"],
                        "name": rec["name"],
                        "role": row["role"],
                        "status": row["status"],
                        "confidence": row["confidence"],
                        "n_evidence": row["n_evidence"],
                    }
                )

    role_presence_df = pd.DataFrame(role_presence_rows)

    if role_presence_df.empty:
        role_matrix = pd.DataFrame(columns=["dandiset_id", "name"])
        role_counts = pd.DataFrame(
            columns=["role", "status", "confidence", "n_dandisets"]
        )
    else:
        role_matrix = (
            role_presence_df.assign(value=1)
            .pivot_table(
                index=["dandiset_id", "name"],
                columns="role",
                values="value",
                aggfunc="max",
                fill_value=0,
            )
            .reset_index()
        )

        role_counts = (
            role_presence_df.groupby(["role", "status", "confidence"])
            .size()
            .reset_index(name="n_dandisets")
            .sort_values("n_dandisets", ascending=False)
        )

    unmapped_df = suggest_schema_extensions_from_unmapped_values(records, top_n=200)

    return dandisets_df, role_presence_df, role_matrix, role_counts, unmapped_df


def save_outputs(records, errors, dandisets_df, role_presence_df, role_matrix, role_counts, unmapped_df):
    full_json_path = OUT_DIR / "dandi_mousehash_scan_full.json"
    full_json_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    errors_path = OUT_DIR / "dandi_mousehash_scan_errors.json"
    errors_path.write_text(
        json.dumps(errors, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    dandisets_df.to_csv(OUT_DIR / "dandisets_summary.csv", index=False)
    role_presence_df.to_csv(OUT_DIR / "mousehash_role_presence_long.csv", index=False)
    role_matrix.to_csv(OUT_DIR / "mousehash_role_matrix.csv", index=False)
    role_counts.to_csv(OUT_DIR / "mousehash_role_counts.csv", index=False)
    unmapped_df.to_csv(OUT_DIR / "unmapped_assets_summary_values.csv", index=False)

    extension_report = {
        "current_mousehash_ontology": MOUSEHASH_ONTOLOGY,
        "top_unmapped_assets_summary_values": unmapped_df.head(100).to_dict(
            orient="records"
        ),
        "role_counts": role_counts.to_dict(orient="records"),
        "notes": [
            "Unmapped values are not automatically schema extensions.",
            "High-frequency unmapped variableMeasured values are the best candidates.",
            "BehavioralEvents should be added separately from behavioral_states.",
            "ECoG and intracellular_recording are likely useful neural_data extensions.",
            "licking, reward, whisking, and eye_movements are likely useful behavior extensions.",
        ],
    }

    (OUT_DIR / "mousehash_schema_extension_report.json").write_text(
        json.dumps(extension_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return full_json_path, errors_path


def print_candidate_sets(role_matrix):
    if role_matrix.empty:
        print("No role matrix produced.")
        return

    def has_role(role):
        if role not in role_matrix.columns:
            return pd.Series(False, index=role_matrix.index)
        return role_matrix[role].astype(bool)

    spikes_plus_behavior = role_matrix[
        has_role("neural_data.spikes")
        & (
            has_role("behavior.behavioral_events")
            | has_role("behavior.choices")
            | has_role("behavior.locomotion")
            | has_role("behavior.pose")
        )
    ]

    calcium_plus_behavior = role_matrix[
        has_role("neural_data.calcium")
        & (
            has_role("behavior.behavioral_events")
            | has_role("behavior.locomotion")
            | has_role("behavior.pupil")
            | has_role("behavior.pose")
        )
    ]

    visual_candidates = role_matrix[
        has_role("stimuli.sensory.visual")
        & (
            has_role("neural_data.spikes")
            | has_role("neural_data.calcium")
        )
    ]

    print("\nSpikes + behavior candidates:")
    print(spikes_plus_behavior[["dandiset_id", "name"]].head(50))

    print("\nCalcium + behavior candidates:")
    print(calcium_plus_behavior[["dandiset_id", "name"]].head(50))

    print("\nVisual neural candidates:")
    print(visual_candidates[["dandiset_id", "name"]].head(50))


def main():
    records, errors = crawl_dandi()

    print(f"\n{len(records)} records, {len(errors)} errors")

    if errors:
        print("\nFirst error:")
        print(json.dumps(errors[0], indent=2, ensure_ascii=False))

    if not records:
        print("\nNo records collected. Stopping before pandas tables.")
        print(f"Errors saved under: {OUT_DIR}")
        return

    (
        dandisets_df,
        role_presence_df,
        role_matrix,
        role_counts,
        unmapped_df,
    ) = build_tables(records)

    full_json_path, errors_path = save_outputs(
        records,
        errors,
        dandisets_df,
        role_presence_df,
        role_matrix,
        role_counts,
        unmapped_df,
    )

    print("\nDandisets summary:")
    print(dandisets_df.head())

    print("\nRole presence:")
    print(role_presence_df.head())

    print("\nRole matrix:")
    print(role_matrix.head())

    print("\nRole counts:")
    print(role_counts.head(50))

    print("\nUnmapped values:")
    print(unmapped_df.head(100))

    print_candidate_sets(role_matrix)

    print("\nSaved:")
    print(full_json_path)
    print(errors_path)


if __name__ == "__main__":
    main()