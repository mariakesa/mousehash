#!/usr/bin/env python3
"""
MouseHash NWB Manifest Parser

Infers a MouseHash-style role manifest from a single NWB file.
Designed for DANDI/NWB files such as Dandiset 000011 ALM behavior+ecephys+ogen data.

Example:
    python parse_mousehash_nwb_manifest.py \
      /home/maria/mousehash/src/DANDI-agent/dandi_nwb_cache/light_data/000011/0.220126.1907/sub-291064/sub-291064_ses-20150907_behavior+ecephys+ogen.nwb

Optional:
    pip install pynwb h5py pyyaml

Outputs:
    - JSON manifest by default
    - YAML manifest with --format yaml
    - summary of discovered NWB paths and role evidence

Design:
    This parser is conservative. It infers only what it can support from the NWB file itself.
    Paper-derived knowledge such as "anterior pole -> lick left" is not assumed unless a matching
    trial column, interval table, or event stream is found in the file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import h5py
except Exception as exc:  # pragma: no cover
    h5py = None

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

try:
    from pynwb import NWBHDF5IO
except Exception:  # pragma: no cover
    NWBHDF5IO = None


# -----------------------------
# Role vocabulary / heuristics
# -----------------------------

SPIKE_HINTS = {"units", "spike_times", "spike_times_index", "spike_event_series", "spikeeventseries"}
LFP_HINTS = {"lfp", "electricalseries", "electrical_series"}
BEHAVIOR_EVENT_HINTS = {
    "lick", "licks", "reward", "pole", "go", "cue", "stim", "photostim", "photostimulus",
    "trial", "response", "choice", "movement", "touch", "whisk", "whisker", "delay", "sample",
}
CHOICE_HINTS = {"choice", "lick", "response", "correct", "hit", "miss", "left", "right"}
REACTION_TIME_HINTS = {"reaction", "latency", "movement", "response_time", "response latency", "rt"}
TACTILE_HINTS = {"pole", "whisk", "whisker", "touch", "tactile"}
OPTO_HINTS = {
    "opto", "photo", "photostim", "photostimulus", "laser", "light", "ogen", "optogen",
    "photoinhibition", "photoactivation", "silencing", "activation",
}
ALM_HINTS = {"alm", "anterior lateral motor", "motor cortex"}
HEMISPHERE_HINTS = {"left", "right", "ipsi", "contra", "contralateral", "ipsilateral", "bilateral", "hemisphere"}
PHASE_HINTS = {"sample", "delay", "response", "epoch", "phase", "period"}
EVENT_HINTS = {"pole", "go", "cue", "lick", "stim", "photostim", "trial", "response", "reward"}

HEAVY_DATA_HINTS = {"ElectricalSeries", "LFP", "TwoPhotonSeries", "OnePhotonSeries", "ImageSeries"}


@dataclass
class EvidenceItem:
    role: str
    label: str
    source: str
    reason: str


class ManifestParser:
    def __init__(self, nwb_path: str, max_dataset_preview: int = 8) -> None:
        self.nwb_path = str(Path(nwb_path).expanduser())
        self.max_dataset_preview = max_dataset_preview
        self.evidence: List[EvidenceItem] = []
        self.paths: List[str] = []
        self.path_types: Dict[str, str] = {}
        self.path_attrs: Dict[str, Dict[str, Any]] = {}
        self.object_names: Set[str] = set()
        self.trial_columns: List[str] = []
        self.intervals: List[str] = []
        self.processing_modules: List[str] = []
        self.acquisition_objects: List[str] = []
        self.stimulus_objects: List[str] = []
        self.devices: List[str] = []
        self.electrode_groups: List[str] = []
        self.subject: Dict[str, Any] = {}
        self.session_metadata: Dict[str, Any] = {}
        self.units_summary: Dict[str, Any] = {}
        self.file_summary: Dict[str, Any] = {}

    # ---------- Utilities ----------

    @staticmethod
    def norm(text: Any) -> str:
        return str(text).lower().replace("_", " ").replace("-", " ")

    @staticmethod
    def compact(text: Any) -> str:
        return str(text).lower().replace("_", "").replace("-", "").replace(" ", "")

    @staticmethod
    def contains_any(text: Any, hints: Iterable[str]) -> bool:
        t = ManifestParser.norm(text)
        c = ManifestParser.compact(text)
        return any(h.lower() in t or h.lower().replace("_", "").replace(" ", "") in c for h in hints)

    def add_evidence(self, role: str, label: str, source: str, reason: str) -> None:
        self.evidence.append(EvidenceItem(role=role, label=label, source=source, reason=reason))

    def unique_sources_for(self, role: str, label: Optional[str] = None) -> List[str]:
        out = []
        for e in self.evidence:
            if e.role == role and (label is None or e.label == label):
                out.append(e.source)
        return sorted(set(out))

    def labels_for(self, role: str) -> List[str]:
        return sorted(set(e.label for e in self.evidence if e.role == role))

    # ---------- HDF5 structural scan ----------

    def scan_hdf5(self) -> None:
        if h5py is None:
            raise RuntimeError("h5py is required for structural scanning. Install with: pip install h5py")
        if not os.path.exists(self.nwb_path):
            raise FileNotFoundError(self.nwb_path)

        file_size = os.path.getsize(self.nwb_path)
        self.file_summary["path"] = self.nwb_path
        self.file_summary["size_bytes"] = file_size
        self.file_summary["size_mb"] = round(file_size / 1024**2, 3)

        with h5py.File(self.nwb_path, "r") as f:
            def visitor(name: str, obj: Any) -> None:
                path = "/" + name if not name.startswith("/") else name
                self.paths.append(path)
                self.object_names.add(Path(path).name)
                self.path_types[path] = type(obj).__name__

                attrs = {}
                for k, v in obj.attrs.items():
                    try:
                        if hasattr(v, "tolist"):
                            v = v.tolist()
                        if isinstance(v, bytes):
                            v = v.decode("utf-8", errors="replace")
                        attrs[str(k)] = v if isinstance(v, (str, int, float, bool, list, dict, type(None))) else str(v)
                    except Exception:
                        attrs[str(k)] = str(v)
                if attrs:
                    self.path_attrs[path] = attrs

                lower_path = self.norm(path)
                compact_path = self.compact(path)

                # Primary neural data roles
                if "/units" in lower_path or any(h in compact_path for h in ["spiketimes", "spikeeventseries"]):
                    self.add_evidence("neural_data", "spikes", path, "Found /units or spike timing related dataset/group.")

                if self.contains_any(path, LFP_HINTS):
                    self.add_evidence("neural_data", "lfp_or_raw_ephys_present", path, "Found LFP/ElectricalSeries-like object; may be heavy raw data.")

                # Behavior and events
                if self.contains_any(path, BEHAVIOR_EVENT_HINTS):
                    if self.contains_any(path, CHOICE_HINTS):
                        self.add_evidence("behavior", "choices", path, "Path/name suggests choice, lick, response, correctness, or left/right behavior.")
                    if self.contains_any(path, REACTION_TIME_HINTS):
                        self.add_evidence("behavior", "reaction_times_or_movement_timing", path, "Path/name suggests reaction latency, movement timing, or response time.")
                    if "behavior" in lower_path or self.contains_any(path, {"lick", "reward", "movement", "running", "position"}):
                        self.add_evidence("behavior", "behavioral_events_or_timeseries", path, "Path/name suggests behavioral events or time series.")

                # Stimuli
                if self.contains_any(path, TACTILE_HINTS):
                    self.add_evidence("stimuli", "tactile", path, "Path/name suggests pole/whisker/touch/tactile stimulus.")
                if self.contains_any(path, OPTO_HINTS):
                    self.add_evidence("stimuli", "optogenetic_intervention", path, "Path/name suggests optogenetic stimulation/inhibition/activation.")

                # Conditions
                if self.contains_any(path, CHOICE_HINTS | {"trial type", "trial_type", "condition", "correct", "error"}):
                    self.add_evidence("conditions", "trial_labels", path, "Path/name suggests trial labels, choice labels, correctness, or condition labels.")
                if self.contains_any(path, OPTO_HINTS | HEMISPHERE_HINTS):
                    self.add_evidence("conditions", "perturbation_labels", path, "Path/name suggests optogenetic or hemispheric perturbation labels.")
                if self.contains_any(path, PHASE_HINTS):
                    self.add_evidence("conditions", "session_phases", path, "Path/name suggests sample/delay/response/epoch phase information.")

                # Time organization
                if "/trials" in lower_path or "intervals/trials" in lower_path:
                    self.add_evidence("time_organization", "trials", path, "Found trials interval table or trial-related path.")
                if self.contains_any(path, PHASE_HINTS):
                    self.add_evidence("time_organization", "epochs", path, "Found sample/delay/response/epoch-like path.")
                if self.contains_any(path, EVENT_HINTS):
                    self.add_evidence("time_organization", "events", path, "Found event-like path/name.")
                if any(token in lower_path for token in ["timestamps", "start_time", "stop_time", "starting_time"]):
                    self.add_evidence("time_organization", "timestamps", path, "Found timestamps/start_time/stop_time/starting_time.")

                # Metadata
                if "/general/subject" in lower_path:
                    self.add_evidence("metadata", "subject", path, "Found subject metadata group/path.")
                if "/general/devices" in lower_path or "/general/extracellular_ephys" in lower_path:
                    self.add_evidence("metadata", "recording_device_or_electrodes", path, "Found device/electrode metadata.")
                if "electrodes" in lower_path or "electrode_group" in lower_path or "electrodegroup" in compact_path:
                    self.add_evidence("metadata", "electrode_group", path, "Found electrode/electrode-group metadata.")
                if self.contains_any(path, ALM_HINTS):
                    self.add_evidence("metadata", "brain_area_ALM", path, "Path/name/attrs suggest ALM/anterior lateral motor cortex.")
                if self.contains_any(path, HEMISPHERE_HINTS):
                    self.add_evidence("metadata", "hemisphere", path, "Path/name suggests hemisphere or ipsi/contra/bilateral annotation.")

            f.visititems(visitor)

            # Top-level NWB/session attrs if present
            for key in ["nwb_version", "session_description", "identifier", "session_start_time", "experiment_description", "institution", "lab"]:
                if key in f.attrs:
                    val = f.attrs[key]
                    if isinstance(val, bytes):
                        val = val.decode("utf-8", errors="replace")
                    self.session_metadata[key] = str(val)

            # Basic known groups
            for group_name in ["acquisition", "stimulus", "processing", "intervals", "general", "units", "trials"]:
                if group_name in f:
                    self.add_evidence("nwb_structure", group_name, f"/{group_name}", f"Top-level NWB group /{group_name} exists.")

            self._summarize_hdf5_known_tables(f)

    def _read_dataset_preview(self, ds: Any) -> Any:
        try:
            if not hasattr(ds, "shape"):
                return None
            if len(ds.shape) == 0:
                val = ds[()]
            elif ds.shape[0] == 0:
                return []
            else:
                val = ds[: min(ds.shape[0], self.max_dataset_preview)]
            if hasattr(val, "tolist"):
                val = val.tolist()
            if isinstance(val, bytes):
                val = val.decode("utf-8", errors="replace")
            if isinstance(val, list):
                out = []
                for item in val:
                    if isinstance(item, bytes):
                        out.append(item.decode("utf-8", errors="replace"))
                    else:
                        out.append(item if isinstance(item, (str, int, float, bool, type(None))) else str(item))
                return out
            return val if isinstance(val, (str, int, float, bool, type(None))) else str(val)
        except Exception:
            return None

    def _summarize_hdf5_known_tables(self, f: Any) -> None:
        # Units summary
        if "units" in f:
            units = f["units"]
            self.units_summary["present"] = True
            for name in units.keys():
                obj = units[name]
                if hasattr(obj, "shape"):
                    self.units_summary[name] = {"shape": list(obj.shape), "dtype": str(obj.dtype)}
            if "id" in units and hasattr(units["id"], "shape"):
                self.units_summary["n_units"] = int(units["id"].shape[0])
            elif "spike_times_index" in units and hasattr(units["spike_times_index"], "shape"):
                self.units_summary["n_units_estimated_from_spike_times_index"] = int(units["spike_times_index"].shape[0])
            if "spike_times" in units and hasattr(units["spike_times"], "shape"):
                self.units_summary["n_spikes_total"] = int(units["spike_times"].shape[0])
        else:
            self.units_summary["present"] = False

        # Intervals and trial columns
        if "intervals" in f:
            self.intervals = sorted(list(f["intervals"].keys()))
            for interval_name in self.intervals:
                self.add_evidence("time_organization", "epochs_or_intervals", f"/intervals/{interval_name}", "Found NWB interval table.")

        if "intervals/trials" in f:
            trials = f["intervals/trials"]
            self.trial_columns = sorted(list(trials.keys()))
            for col in self.trial_columns:
                source = f"/intervals/trials/{col}"
                if col in ["start_time", "stop_time"]:
                    self.add_evidence("time_organization", "trials", source, "Trial table has start/stop times.")
                if self.contains_any(col, CHOICE_HINTS | {"trial_type", "trial type", "correct", "error"}):
                    self.add_evidence("conditions", "trial_labels", source, "Trial column looks like a trial/choice/correctness label.")
                if self.contains_any(col, REACTION_TIME_HINTS):
                    self.add_evidence("behavior", "reaction_times_or_movement_timing", source, "Trial column looks like latency/movement timing.")
                if self.contains_any(col, OPTO_HINTS | HEMISPHERE_HINTS):
                    self.add_evidence("conditions", "perturbation_labels", source, "Trial column looks like opto/hemisphere perturbation label.")
                if self.contains_any(col, TACTILE_HINTS):
                    self.add_evidence("stimuli", "tactile", source, "Trial column looks like tactile/pole/whisker stimulus annotation.")
                if self.contains_any(col, PHASE_HINTS):
                    self.add_evidence("conditions", "session_phases", source, "Trial column looks like task phase annotation.")

        # Other top-level object names
        for group_name, target_list in [
            ("processing", self.processing_modules),
            ("acquisition", self.acquisition_objects),
            ("stimulus", self.stimulus_objects),
            ("general/devices", self.devices),
            ("general/extracellular_ephys", self.electrode_groups),
        ]:
            if group_name in f:
                try:
                    target_list.extend(sorted(list(f[group_name].keys())))
                except Exception:
                    pass

    # ---------- Optional PyNWB enriched scan ----------

    def scan_pynwb(self) -> None:
        if NWBHDF5IO is None:
            self.file_summary["pynwb_available"] = False
            return

        self.file_summary["pynwb_available"] = True
        try:
            with NWBHDF5IO(self.nwb_path, mode="r", load_namespaces=True) as io:
                nwbfile = io.read()

                self.session_metadata.update({
                    "session_description": getattr(nwbfile, "session_description", None),
                    "identifier": getattr(nwbfile, "identifier", None),
                    "session_start_time": str(getattr(nwbfile, "session_start_time", None)),
                    "experiment_description": getattr(nwbfile, "experiment_description", None),
                    "institution": getattr(nwbfile, "institution", None),
                    "lab": getattr(nwbfile, "lab", None),
                    "keywords": list(getattr(nwbfile, "keywords", []) or []),
                })

                subject = getattr(nwbfile, "subject", None)
                if subject is not None:
                    for key in ["subject_id", "species", "genotype", "sex", "age", "description", "strain"]:
                        val = getattr(subject, key, None)
                        if val is not None:
                            self.subject[key] = str(val)
                    self.add_evidence("metadata", "subject", "/general/subject", "PyNWB found subject object.")

                if getattr(nwbfile, "units", None) is not None:
                    try:
                        self.units_summary["n_units_pynwb"] = len(nwbfile.units.id[:])
                    except Exception:
                        pass
                    self.add_evidence("neural_data", "spikes", "/units", "PyNWB found Units table.")

                # Trials DataFrame columns if possible
                if getattr(nwbfile, "trials", None) is not None:
                    try:
                        df = nwbfile.trials.to_dataframe()
                        self.trial_columns = sorted(set(self.trial_columns) | set(map(str, df.columns)))
                        self.file_summary["n_trials"] = int(len(df))
                        for col in df.columns:
                            self._infer_from_column_values(str(col), df[col].head(20).tolist(), f"/intervals/trials/{col}")
                    except Exception as exc:
                        self.file_summary["trials_dataframe_error"] = str(exc)

                # Devices/electrode groups metadata text can reveal ALM/hemisphere/opto
                for container_name, container in [
                    ("devices", getattr(nwbfile, "devices", {}) or {}),
                    ("electrode_groups", getattr(nwbfile, "electrode_groups", {}) or {}),
                    ("ogen_sites", getattr(nwbfile, "ogen_sites", {}) or {}),
                ]:
                    try:
                        for name, obj in container.items():
                            text = " ".join([str(name), str(getattr(obj, "description", "")), str(getattr(obj, "location", ""))])
                            source = f"/general/{container_name}/{name}"
                            if self.contains_any(text, ALM_HINTS):
                                self.add_evidence("metadata", "brain_area_ALM", source, "PyNWB metadata text suggests ALM.")
                            if self.contains_any(text, HEMISPHERE_HINTS):
                                self.add_evidence("metadata", "hemisphere", source, "PyNWB metadata text suggests hemisphere.")
                            if self.contains_any(text, OPTO_HINTS):
                                self.add_evidence("metadata", "photoinhibition_or_optogenetic_protocol", source, "PyNWB metadata text suggests optogenetic protocol.")
                    except Exception:
                        pass

        except Exception as exc:
            self.file_summary["pynwb_error"] = str(exc)

    def _infer_from_column_values(self, col: str, values: Sequence[Any], source: str) -> None:
        text = col + " " + " ".join(map(str, values[:10]))
        if self.contains_any(text, TACTILE_HINTS):
            self.add_evidence("stimuli", "tactile", source, "Trial column/value text suggests tactile pole/whisker stimulus.")
        if self.contains_any(text, {"anterior", "posterior", "left", "right", "lick"}):
            self.add_evidence("conditions", "trial_labels", source, "Trial column/value text suggests anterior/posterior/left/right/lick labels.")
        if self.contains_any(text, OPTO_HINTS):
            self.add_evidence("stimuli", "optogenetic_intervention", source, "Trial column/value text suggests optogenetic intervention.")
            self.add_evidence("conditions", "perturbation_labels", source, "Trial column/value text suggests optogenetic perturbation labels.")
        if self.contains_any(text, HEMISPHERE_HINTS):
            self.add_evidence("metadata", "hemisphere", source, "Trial column/value text suggests hemisphere metadata.")
            self.add_evidence("conditions", "perturbation_labels", source, "Trial column/value text suggests ipsi/contra/bilateral perturbation labels.")
        if self.contains_any(text, REACTION_TIME_HINTS):
            self.add_evidence("behavior", "reaction_times_or_movement_timing", source, "Trial column/value text suggests reaction/movement timing.")
        if self.contains_any(text, PHASE_HINTS):
            self.add_evidence("conditions", "session_phases", source, "Trial column/value text suggests sample/delay/response phase.")

    # ---------- Build MouseHash manifest ----------

    def build_manifest(self) -> Dict[str, Any]:
        manifest: Dict[str, Any] = {
            "source_file": self.nwb_path,
            "file_summary": self.file_summary,
            "mousehash_roles": {
                "neural_data": [],
                "stimuli": {
                    "sensory": [],
                    "interventions": [],
                },
                "behavior": [],
                "conditions": {},
                "time_organization": [],
                "metadata": [],
            },
            "nwb_summary": {
                "session_metadata": {k: v for k, v in self.session_metadata.items() if v not in [None, "None", ""]},
                "subject": self.subject,
                "units": self.units_summary,
                "trial_columns": self.trial_columns,
                "intervals": self.intervals,
                "processing_modules": self.processing_modules,
                "acquisition_objects": self.acquisition_objects,
                "stimulus_objects": self.stimulus_objects,
                "devices": self.devices,
                "electrode_groups": self.electrode_groups,
            },
            "evidence": [asdict(e) for e in self.evidence],
            "role_sources": {},
            "warnings": [],
        }

        roles = manifest["mousehash_roles"]

        if self.labels_for("neural_data"):
            if self.unique_sources_for("neural_data", "spikes"):
                roles["neural_data"].append("spikes")
            if self.unique_sources_for("neural_data", "lfp_or_raw_ephys_present"):
                roles["neural_data"].append("lfp_or_raw_ephys_present")
                manifest["warnings"].append("This file appears to contain LFP/ElectricalSeries-like objects; it may not be lightweight spike-only data.")

        if self.unique_sources_for("stimuli", "tactile"):
            roles["stimuli"]["sensory"].append("tactile")
        if self.unique_sources_for("stimuli", "optogenetic_intervention"):
            # We infer generic optogenetic intervention from file structure; inhibition/activation subtype needs labels/metadata.
            roles["stimuli"]["interventions"].append("optogenetic intervention")

        for label in ["choices", "reaction_times_or_movement_timing", "behavioral_events_or_timeseries"]:
            if self.unique_sources_for("behavior", label):
                roles["behavior"].append(label)

        condition_map = {
            "trial_labels": "trial_labels",
            "perturbation_labels": "perturbation_labels",
            "session_phases": "session_phases",
        }
        for label, out_key in condition_map.items():
            sources = self.unique_sources_for("conditions", label)
            if sources:
                roles["conditions"][out_key] = {"inferred": True, "sources": sources[:20]}

        for label in ["trials", "epochs", "epochs_or_intervals", "events", "timestamps"]:
            if self.unique_sources_for("time_organization", label):
                roles["time_organization"].append(label)

        for label in [
            "subject", "recording_device_or_electrodes", "electrode_group", "brain_area_ALM",
            "hemisphere", "photoinhibition_or_optogenetic_protocol",
        ]:
            if self.unique_sources_for("metadata", label):
                roles["metadata"].append(label)

        # De-duplicate while preserving order
        roles["neural_data"] = list(dict.fromkeys(roles["neural_data"]))
        roles["stimuli"]["sensory"] = list(dict.fromkeys(roles["stimuli"]["sensory"]))
        roles["stimuli"]["interventions"] = list(dict.fromkeys(roles["stimuli"]["interventions"]))
        roles["behavior"] = list(dict.fromkeys(roles["behavior"]))
        roles["time_organization"] = list(dict.fromkeys(roles["time_organization"]))
        roles["metadata"] = list(dict.fromkeys(roles["metadata"]))

        # Role source index for debugging/agent inspection
        for role in sorted(set(e.role for e in self.evidence)):
            manifest["role_sources"][role] = defaultdict(list)  # type: ignore[assignment]
        role_sources: Dict[str, Dict[str, List[str]]] = {}
        for e in self.evidence:
            role_sources.setdefault(e.role, {}).setdefault(e.label, []).append(e.source)
        manifest["role_sources"] = {
            role: {label: sorted(set(sources))[:50] for label, sources in labels.items()}
            for role, labels in role_sources.items()
        }

        # Honest caveats
        if "optogenetic intervention" in roles["stimuli"]["interventions"]:
            manifest["warnings"].append(
                "The parser inferred generic optogenetic intervention. Distinguishing photoinhibition vs photoactivation requires explicit labels/metadata in the file or paper-level knowledge."
            )
        if roles["stimuli"]["sensory"] == []:
            manifest["warnings"].append("No sensory stimulus type was confidently inferred from internal NWB names/metadata.")
        if "spikes" not in roles["neural_data"]:
            manifest["warnings"].append("No /units or spike timing structure was detected; verify that this is a spike-sorted NWB.")
        if not roles["conditions"]:
            manifest["warnings"].append("No condition labels were confidently inferred; inspect trial columns and interval tables manually.")

        return manifest

    def parse(self, use_pynwb: bool = True) -> Dict[str, Any]:
        self.scan_hdf5()
        if use_pynwb:
            self.scan_pynwb()
        return self.build_manifest()


def write_output(manifest: Dict[str, Any], output_path: Optional[str], fmt: str) -> None:
    text: str
    if fmt == "yaml":
        if yaml is None:
            raise RuntimeError("PyYAML is required for --format yaml. Install with: pip install pyyaml")
        text = yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True)
    else:
        text = json.dumps(manifest, indent=2, ensure_ascii=False)

    if output_path:
        out = Path(output_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"[saved] {out}")
    else:
        print(text)


def default_output_path(nwb_path: str, fmt: str) -> str:
    suffix = ".yaml" if fmt == "yaml" else ".json"
    p = Path(nwb_path).expanduser()
    return str(p.with_name(p.stem + ".mousehash_manifest" + suffix))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Infer a MouseHash role manifest from one NWB file.")
    parser.add_argument("nwb_path", help="Path to a local .nwb file")
    parser.add_argument("--output", "-o", default=None, help="Output manifest path. Defaults to next to NWB unless --stdout.")
    parser.add_argument("--stdout", action="store_true", help="Print manifest to stdout instead of saving.")
    parser.add_argument("--format", choices=["json", "yaml"], default="json", help="Output format")
    parser.add_argument("--no-pynwb", action="store_true", help="Use only h5py structural scan, skip PyNWB enriched parsing.")
    parser.add_argument("--max-dataset-preview", type=int, default=8, help="Small preview length for scalar/table inference.")
    args = parser.parse_args(argv)

    nwb_path = str(Path(args.nwb_path).expanduser())
    parser_obj = ManifestParser(nwb_path, max_dataset_preview=args.max_dataset_preview)
    manifest = parser_obj.parse(use_pynwb=not args.no_pynwb)

    output = None if args.stdout else (args.output or default_output_path(nwb_path, args.format))
    write_output(manifest, output, args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
