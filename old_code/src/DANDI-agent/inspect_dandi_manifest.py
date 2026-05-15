#!/usr/bin/env python3
"""
MouseHash DANDI Manifest Inspector

Usage:
    python inspect_dandi_manifest.py 000006 --version draft --max-assets 5
    python inspect_dandi_manifest.py 000006 --version draft --max-assets 5 --probe-nwb

Optional:
    pip install pynwb h5py

What it does:
    1. Connects to DANDI.
    2. Fetches Dandiset metadata.
    3. Lists assets.
    4. Infers MouseHash roles from DANDI/NWB metadata.
    5. Optionally opens small remote NWB handles and inspects top-level structure.
    6. Writes a JSON manifest.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dandi.dandiapi import DandiAPIClient


# -----------------------------
# MouseHash role evidence model
# -----------------------------

@dataclass
class RoleEvidence:
    status: str = "unknown"       # present | likely | absent | unknown
    confidence: str = "low"       # high | medium | low
    sources: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)


def empty_manifest() -> Dict[str, Any]:
    return {
        "mousehash_roles": {
            "neural_data": {
                "spikes": RoleEvidence(),
                "lfp": RoleEvidence(),
                "eeg": RoleEvidence(),
                "calcium": RoleEvidence(),
                "photometry": RoleEvidence(),
                "images": RoleEvidence(),
            },
            "stimuli": {
                "sensory": {
                    "visual": RoleEvidence(),
                    "auditory": RoleEvidence(),
                    "tactile": RoleEvidence(),
                    "odor": RoleEvidence(),
                },
                "interventions": {
                    "optogenetic": RoleEvidence(),
                    "electrical": RoleEvidence(),
                    "pharmacological": RoleEvidence(),
                    "anesthesia": RoleEvidence(),
                },
            },
            "behavior": {
                "choices": RoleEvidence(),
                "reaction_times": RoleEvidence(),
                "pose": RoleEvidence(),
                "locomotion": RoleEvidence(),
                "pupil": RoleEvidence(),
                "kinematics": RoleEvidence(),
                "behavioral_states": RoleEvidence(),
            },
            "conditions": {
                "task_labels": RoleEvidence(),
                "trial_labels": RoleEvidence(),
                "experimental_groups": RoleEvidence(),
                "brain_states": RoleEvidence(),
                "session_phases": RoleEvidence(),
            },
            "time_organization": {
                "continuous_time": RoleEvidence(),
                "trials": RoleEvidence(),
                "epochs": RoleEvidence(),
                "events": RoleEvidence(),
                "frames": RoleEvidence(),
                "alignment_rules": RoleEvidence(),
            },
            "metadata": {
                "subject": RoleEvidence(),
                "species": RoleEvidence(),
                "genotype": RoleEvidence(),
                "session": RoleEvidence(),
                "brain_area": RoleEvidence(),
                "probe/electrode/imaging_plane": RoleEvidence(),
                "acquisition_device": RoleEvidence(),
                "preprocessing_info": RoleEvidence(),
            },
        }
    }


def mark(
    manifest: Dict[str, Any],
    path: str,
    status: str,
    confidence: str,
    source: str,
    evidence: str,
) -> None:
    """
    Update one manifest role by dotted path.

    Example:
        mark(manifest, "neural_data.spikes", "present", "high", "/units", "NWB Units table found")
        mark(manifest, "stimuli.sensory.visual", "likely", "medium", "metadata", "visual keyword found")
    """
    node = manifest["mousehash_roles"]
    parts = path.split(".")
    for part in parts:
        node = node[part]

    if isinstance(node, RoleEvidence):
        # Upgrade status/confidence if stronger evidence arrives.
        status_rank = {"absent": 0, "unknown": 1, "likely": 2, "present": 3}
        conf_rank = {"low": 1, "medium": 2, "high": 3}

        if status_rank.get(status, 0) >= status_rank.get(node.status, 0):
            node.status = status
        if conf_rank.get(confidence, 0) >= conf_rank.get(node.confidence, 0):
            node.confidence = confidence

        if source not in node.sources:
            node.sources.append(source)
        if evidence not in node.evidence:
            node.evidence.append(evidence)
    else:
        raise TypeError(f"Manifest path does not point to RoleEvidence: {path}")


# -----------------------------
# Metadata utilities
# -----------------------------

def safe_jsonable(x: Any) -> Any:
    """
    Convert DANDI/Pydantic-ish objects to JSON-friendly values.
    """
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {str(k): safe_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [safe_jsonable(v) for v in x]

    # Pydantic v2
    if hasattr(x, "model_dump"):
        try:
            return safe_jsonable(x.model_dump())
        except Exception:
            pass

    # Pydantic v1 / dandischema helpers
    if hasattr(x, "dict"):
        try:
            return safe_jsonable(x.dict())
        except Exception:
            pass

    if hasattr(x, "json_dict"):
        try:
            return safe_jsonable(x.json_dict())
        except Exception:
            pass

    return str(x)


def flatten_text(obj: Any) -> str:
    """
    Make one lowercase searchable string from nested metadata.
    """
    return json.dumps(safe_jsonable(obj), default=str).lower()


def get_asset_path(asset: Any) -> str:
    return str(getattr(asset, "path", "") or "")


def get_asset_size(asset: Any, metadata: Any = None) -> Optional[int]:
    for attr in ["size", "blob_size"]:
        value = getattr(asset, attr, None)
        if isinstance(value, int):
            return value

    md = safe_jsonable(metadata)
    if isinstance(md, dict):
        for key in ["contentSize", "size"]:
            value = md.get(key)
            if isinstance(value, int):
                return value

    return None


# -----------------------------
# Rule-based role inference
# -----------------------------

def infer_from_dandiset_metadata(manifest: Dict[str, Any], metadata: Any) -> None:
    text = flatten_text(metadata)

    # Metadata roles
    if "subject" in text:
        mark(manifest, "metadata.subject", "likely", "medium", "dandiset_metadata", "subject keyword found")

    if "species" in text or "mus musculus" in text or "mouse" in text:
        mark(manifest, "metadata.species", "likely", "medium", "dandiset_metadata", "species keyword found")

    if "genotype" in text or "strain" in text:
        mark(manifest, "metadata.genotype", "likely", "medium", "dandiset_metadata", "genotype/strain keyword found")

    # Modality-ish roles
    if any(k in text for k in ["extracellular electrophysiology", "ecephys", "spike", "unit"]):
        mark(manifest, "neural_data.spikes", "likely", "medium", "dandiset_metadata", "ephys/spike/unit keyword found")

    if any(k in text for k in ["lfp", "local field potential"]):
        mark(manifest, "neural_data.lfp", "likely", "medium", "dandiset_metadata", "LFP keyword found")

    if any(k in text for k in ["ophys", "two-photon", "2-photon", "calcium imaging", "fluorescence"]):
        mark(manifest, "neural_data.calcium", "likely", "medium", "dandiset_metadata", "ophys/calcium keyword found")

    if any(k in text for k in ["photometry", "fiber photometry"]):
        mark(manifest, "neural_data.photometry", "likely", "medium", "dandiset_metadata", "photometry keyword found")

    # Stimuli
    if any(k in text for k in ["visual", "image", "movie", "natural scene", "grating"]):
        mark(manifest, "stimuli.sensory.visual", "likely", "medium", "dandiset_metadata", "visual stimulus keyword found")

    if any(k in text for k in ["auditory", "sound", "tone"]):
        mark(manifest, "stimuli.sensory.auditory", "likely", "medium", "dandiset_metadata", "auditory stimulus keyword found")

    if any(k in text for k in ["optogenetic", "optogenetics", "opto"]):
        mark(manifest, "stimuli.interventions.optogenetic", "likely", "medium", "dandiset_metadata", "optogenetic keyword found")

    if any(k in text for k in ["electrical stimulation", "stimulation electrode"]):
        mark(manifest, "stimuli.interventions.electrical", "likely", "medium", "dandiset_metadata", "electrical stimulation keyword found")

    # Behavior
    if any(k in text for k in ["running", "locomotion", "wheel"]):
        mark(manifest, "behavior.locomotion", "likely", "medium", "dandiset_metadata", "running/locomotion keyword found")

    if any(k in text for k in ["pupil", "eye tracking"]):
        mark(manifest, "behavior.pupil", "likely", "medium", "dandiset_metadata", "pupil/eye keyword found")

    if any(k in text for k in ["choice", "decision"]):
        mark(manifest, "behavior.choices", "likely", "medium", "dandiset_metadata", "choice keyword found")

    if any(k in text for k in ["reaction time", "response time"]):
        mark(manifest, "behavior.reaction_times", "likely", "medium", "dandiset_metadata", "reaction-time keyword found")


def infer_from_asset_metadata(manifest: Dict[str, Any], asset_path: str, metadata: Any) -> None:
    text = flatten_text(metadata)
    source = f"asset_metadata:{asset_path}"

    if asset_path.endswith(".nwb"):
        mark(manifest, "metadata.session", "likely", "medium", source, "NWB asset found")

    if any(k in text for k in ["units", "spike_times", "spikes", "ecephys"]):
        mark(manifest, "neural_data.spikes", "likely", "medium", source, "units/spike/ecephys metadata found")

    if any(k in text for k in ["electricalseries", "electrodes", "extracellular"]):
        mark(manifest, "metadata.probe/electrode/imaging_plane", "likely", "medium", source, "electrode/electrical metadata found")

    if any(k in text for k in ["lfp", "local field potential"]):
        mark(manifest, "neural_data.lfp", "likely", "medium", source, "LFP metadata found")

    if any(k in text for k in ["twophotonseries", "ophys", "fluorescence", "plane segmentation", "imagingplane"]):
        mark(manifest, "neural_data.calcium", "likely", "medium", source, "ophys/calcium metadata found")
        mark(manifest, "metadata.probe/electrode/imaging_plane", "likely", "medium", source, "imaging plane metadata found")

    if any(k in text for k in ["trials", "trial"]):
        mark(manifest, "time_organization.trials", "likely", "medium", source, "trial metadata found")
        mark(manifest, "conditions.trial_labels", "likely", "medium", source, "trial labels may exist")

    if any(k in text for k in ["epochs", "intervals"]):
        mark(manifest, "time_organization.epochs", "likely", "medium", source, "epoch/interval metadata found")

    if any(k in text for k in ["stimulus", "stimuli", "presentation"]):
        mark(manifest, "time_organization.events", "likely", "medium", source, "stimulus/event metadata found")

    if any(k in text for k in ["image", "movie", "frame", "visual"]):
        mark(manifest, "stimuli.sensory.visual", "likely", "medium", source, "visual/frame/image metadata found")
        mark(manifest, "time_organization.frames", "likely", "medium", source, "frame-like metadata found")

    if any(k in text for k in ["running", "wheel", "locomotion"]):
        mark(manifest, "behavior.locomotion", "likely", "medium", source, "locomotion metadata found")

    if any(k in text for k in ["pupil", "eye"]):
        mark(manifest, "behavior.pupil", "likely", "medium", source, "pupil/eye metadata found")

    if any(k in text for k in ["pose", "deeplabcut", "sleap"]):
        mark(manifest, "behavior.pose", "likely", "medium", source, "pose tracking metadata found")


# -----------------------------
# Optional NWB structure probe
# -----------------------------

def probe_nwb_structure(manifest: Dict[str, Any], asset: Any) -> Dict[str, Any]:
    """
    Try to open a remote NWB asset and inspect its high-level structure.

    This is intentionally lightweight. It does not read big arrays.
    """
    asset_path = get_asset_path(asset)
    source = f"nwb_probe:{asset_path}"
    result = {
        "path": asset_path,
        "opened": False,
        "error": None,
        "top_level": [],
        "nwb_summary": {},
    }

    try:
        from pynwb import NWBHDF5IO
    except Exception as e:
        result["error"] = f"pynwb not available: {e}"
        return result

    try:
        # DANDI RemoteBlobAsset supports sparse remote reads through fsspec.
        with asset.as_readable().open() as f:
            with NWBHDF5IO(file=f, mode="r", load_namespaces=True) as io:
                nwbfile = io.read()
                result["opened"] = True

                # Top-level-ish NWB object summaries
                result["nwb_summary"] = {
                    "identifier": getattr(nwbfile, "identifier", None),
                    "session_description": getattr(nwbfile, "session_description", None),
                    "session_start_time": str(getattr(nwbfile, "session_start_time", None)),
                    "has_units": nwbfile.units is not None,
                    "num_units": len(nwbfile.units.id[:]) if nwbfile.units is not None else 0,
                    "num_trials": len(nwbfile.trials.id[:]) if nwbfile.trials is not None else 0,
                    "acquisition_keys": list(nwbfile.acquisition.keys()),
                    "processing_keys": list(nwbfile.processing.keys()),
                    "stimulus_keys": list(nwbfile.stimulus.keys()),
                    "interval_keys": list(nwbfile.intervals.keys()),
                }

                # Stronger evidence from actual NWB object presence
                mark(manifest, "metadata.session", "present", "high", source, "NWB file opened")
                mark(manifest, "time_organization.continuous_time", "present", "high", source, "NWB time-based structure exists")

                if nwbfile.subject is not None:
                    mark(manifest, "metadata.subject", "present", "high", source, "nwbfile.subject found")
                    if getattr(nwbfile.subject, "species", None):
                        mark(manifest, "metadata.species", "present", "high", source, "nwbfile.subject.species found")
                    if getattr(nwbfile.subject, "genotype", None):
                        mark(manifest, "metadata.genotype", "present", "high", source, "nwbfile.subject.genotype found")

                if nwbfile.units is not None and len(nwbfile.units.id[:]) > 0:
                    mark(manifest, "neural_data.spikes", "present", "high", source, "nwbfile.units table found with rows")

                if nwbfile.trials is not None and len(nwbfile.trials.id[:]) > 0:
                    mark(manifest, "time_organization.trials", "present", "high", source, "nwbfile.trials table found with rows")
                    mark(manifest, "conditions.trial_labels", "likely", "medium", source, "trial table may contain condition columns")

                if len(nwbfile.intervals.keys()) > 0:
                    mark(manifest, "time_organization.epochs", "present", "high", source, "nwbfile.intervals found")

                if len(nwbfile.stimulus.keys()) > 0:
                    mark(manifest, "stimuli.sensory.visual", "likely", "medium", source, "nwbfile.stimulus found")
                    mark(manifest, "time_organization.events", "likely", "medium", source, "stimulus objects imply events/presentations")

                # Inspect acquisition names for rough modality clues
                names = " ".join(list(nwbfile.acquisition.keys()) + list(nwbfile.processing.keys())).lower()

                if any(k in names for k in ["lfp", "electricalseries", "ecephys"]):
                    mark(manifest, "neural_data.lfp", "likely", "medium", source, "LFP/electrical acquisition-like key found")

                if any(k in names for k in ["twophoton", "ophys", "fluorescence", "df/f", "roi"]):
                    mark(manifest, "neural_data.calcium", "likely", "medium", source, "ophys/calcium-like key found")

                if any(k in names for k in ["running", "locomotion", "wheel"]):
                    mark(manifest, "behavior.locomotion", "present", "high", source, "locomotion-like processing/acquisition key found")

                if any(k in names for k in ["pupil", "eye"]):
                    mark(manifest, "behavior.pupil", "present", "high", source, "pupil/eye-like processing/acquisition key found")

                if any(k in names for k in ["pose", "deeplabcut", "sleap"]):
                    mark(manifest, "behavior.pose", "present", "high", source, "pose-like processing/acquisition key found")

    except Exception as e:
        result["error"] = repr(e)

    return result


# -----------------------------
# Valid view suggestions
# -----------------------------

def get_role(manifest: Dict[str, Any], path: str) -> RoleEvidence:
    node = manifest["mousehash_roles"]
    for part in path.split("."):
        node = node[part]
    return node


def is_present_or_likely(manifest: Dict[str, Any], path: str) -> bool:
    return get_role(manifest, path).status in {"present", "likely"}


def suggest_valid_views_and_tools(manifest: Dict[str, Any]) -> Dict[str, Any]:
    valid_views = []
    valid_tools = []
    blocked_tools = []

    has_spikes = is_present_or_likely(manifest, "neural_data.spikes")
    has_trials = is_present_or_likely(manifest, "time_organization.trials")
    has_trial_labels = is_present_or_likely(manifest, "conditions.trial_labels")
    has_visual = is_present_or_likely(manifest, "stimuli.sensory.visual")
    has_locomotion = is_present_or_likely(manifest, "behavior.locomotion")

    if has_spikes:
        valid_views.append({
            "name": "spike_count_matrix",
            "requires": ["neural_data.spikes", "time_organization.continuous_time"],
            "transformations": ["extract_units_table", "bin_spikes"],
        })
        valid_tools.extend(["raster_plot", "psth_plot", "pca_on_neural_activity"])

    if has_spikes and has_trials:
        valid_views.append({
            "name": "trial_time_neuron_tensor",
            "requires": ["neural_data.spikes", "time_organization.trials"],
            "transformations": ["extract_units_table", "extract_trials_table", "align_to_trial_start", "bin_spikes", "make_trial_time_neuron_tensor"],
        })
        valid_tools.extend(["trial_aligned_raster", "trial_aligned_psth", "latent_trajectory_plot"])

    if has_spikes and has_trials and has_trial_labels:
        valid_views.append({
            "name": "condition_neuron_matrix",
            "requires": ["neural_data.spikes", "time_organization.trials", "conditions.trial_labels"],
            "transformations": ["make_trial_time_neuron_tensor", "aggregate_by_condition", "make_condition_neuron_matrix"],
        })
        valid_tools.extend(["trial_averaged_pca", "representational_similarity_matrix", "signal_correlations"])

    if has_spikes and has_visual and has_trials:
        valid_views.append({
            "name": "stimulus_aligned_response_matrix",
            "requires": ["neural_data.spikes", "stimuli.sensory.visual", "time_organization.trials"],
            "transformations": ["align_to_stimulus_onset", "extract_response_window", "aggregate_response_window"],
        })
        valid_tools.extend(["stimulus_response_summary", "visual_stimulus_psth"])

    if has_spikes and has_locomotion:
        valid_views.append({
            "name": "neural_behavior_aligned_matrix",
            "requires": ["neural_data.spikes", "behavior.locomotion", "time_organization.continuous_time"],
            "transformations": ["align_behavior_to_neural_time", "bin_spikes", "resample_behavior"],
        })
        valid_tools.extend(["behavior_correlation", "locomotion_regression"])

    if not (has_spikes and has_visual and has_trials):
        blocked_tools.append({
            "name": "ridge_encoding_model",
            "missing_or_uncertain": [
                p for p in [
                    "neural_data.spikes",
                    "stimuli.sensory.visual",
                    "time_organization.trials",
                ]
                if not is_present_or_likely(manifest, p)
            ],
        })

    if not (has_spikes and has_trial_labels):
        blocked_tools.append({
            "name": "logistic_decoder",
            "missing_or_uncertain": [
                p for p in [
                    "neural_data.spikes",
                    "conditions.trial_labels",
                ]
                if not is_present_or_likely(manifest, p)
            ],
        })

    return {
        "valid_analysis_views": valid_views,
        "valid_tools": sorted(set(valid_tools)),
        "blocked_tools": blocked_tools,
    }


# -----------------------------
# Serialization
# -----------------------------

def manifest_to_jsonable(obj: Any) -> Any:
    if isinstance(obj, RoleEvidence):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: manifest_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [manifest_to_jsonable(v) for v in obj]
    return safe_jsonable(obj)


# -----------------------------
# Main
# -----------------------------

def inspect_dandi(
    dandiset_id: str,
    version: Optional[str],
    max_assets: int,
    probe_nwb: bool,
) -> Dict[str, Any]:
    manifest = empty_manifest()

    output: Dict[str, Any] = {
        "dandiset_id": dandiset_id,
        "version": version,
        "manifest_fit_mode": "metadata_only_plus_optional_nwb_probe" if probe_nwb else "metadata_only",
        "dandiset_metadata_summary": {},
        "assets_scanned": [],
        "nwb_probe_results": [],
        "role_manifest": None,
        "analysis_suggestions": None,
    }

    with DandiAPIClient.for_dandi_instance("dandi") as client:
        dandiset = client.get_dandiset(dandiset_id, version)

        try:
            dandiset_metadata = dandiset.get_raw_metadata()
        except Exception:
            # Fallback for some client versions
            dandiset_metadata = dandiset.get_metadata()

        dandiset_metadata_json = safe_jsonable(dandiset_metadata)
        output["dandiset_metadata_summary"] = {
            "name": dandiset_metadata_json.get("name") if isinstance(dandiset_metadata_json, dict) else None,
            "description": dandiset_metadata_json.get("description") if isinstance(dandiset_metadata_json, dict) else None,
            "license": dandiset_metadata_json.get("license") if isinstance(dandiset_metadata_json, dict) else None,
            "keywords": dandiset_metadata_json.get("keywords") if isinstance(dandiset_metadata_json, dict) else None,
        }

        infer_from_dandiset_metadata(manifest, dandiset_metadata_json)

        assets = []
        for i, asset in enumerate(dandiset.get_assets()):
            if i >= max_assets:
                break
            assets.append(asset)

        for asset in assets:
            path = get_asset_path(asset)

            try:
                asset_metadata = asset.get_raw_metadata()
            except Exception:
                try:
                    asset_metadata = asset.get_metadata()
                except Exception as e:
                    asset_metadata = {"metadata_error": repr(e)}

            asset_metadata_json = safe_jsonable(asset_metadata)

            asset_record = {
                "path": path,
                "identifier": str(getattr(asset, "identifier", "")),
                "asset_type": str(getattr(asset, "asset_type", "")),
                "size": get_asset_size(asset, asset_metadata_json),
                "metadata_keys": sorted(asset_metadata_json.keys()) if isinstance(asset_metadata_json, dict) else [],
            }
            output["assets_scanned"].append(asset_record)

            infer_from_asset_metadata(manifest, path, asset_metadata_json)

            if probe_nwb and path.endswith(".nwb"):
                probe_result = probe_nwb_structure(manifest, asset)
                output["nwb_probe_results"].append(probe_result)

    output["role_manifest"] = manifest_to_jsonable(manifest)
    output["analysis_suggestions"] = suggest_valid_views_and_tools(manifest)

    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dandiset_id", help="DANDI Dandiset ID, e.g. 000006")
    parser.add_argument("--version", default=None, help="Dandiset version, e.g. draft or 0.230629.1955. Defaults to latest published or draft.")
    parser.add_argument("--max-assets", type=int, default=10, help="Maximum number of assets to scan.")
    parser.add_argument("--probe-nwb", action="store_true", help="Try to open remote NWB files and inspect structure. Requires pynwb.")
    parser.add_argument("--out", default=None, help="Output JSON path.")
    args = parser.parse_args()

    result = inspect_dandi(
        dandiset_id=args.dandiset_id,
        version=args.version,
        max_assets=args.max_assets,
        probe_nwb=args.probe_nwb,
    )

    out_path = args.out or f"mousehash_manifest_{args.dandiset_id}.json"
    Path(out_path).write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\nWrote manifest fit to: {out_path}\n")

    suggestions = result["analysis_suggestions"]
    print("Valid analysis views:")
    for view in suggestions["valid_analysis_views"]:
        print(f"  ✓ {view['name']}")

    print("\nValid tools:")
    for tool in suggestions["valid_tools"]:
        print(f"  ✓ {tool}")

    print("\nBlocked tools:")
    for tool in suggestions["blocked_tools"]:
        missing = ", ".join(tool["missing_or_uncertain"])
        print(f"  ✗ {tool['name']} — missing/uncertain: {missing}")


if __name__ == "__main__":
    main()