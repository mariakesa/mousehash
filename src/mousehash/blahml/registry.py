from __future__ import annotations

import importlib.resources
from dataclasses import dataclass
from pathlib import Path

from mousehash.blahml.hashing import manifest_sha256
from mousehash.blahml.loader import load_manifest
from mousehash.blahml.models import BlahManifest


@dataclass(frozen=True)
class RegisteredManifest:
    manifest: BlahManifest
    path: Path
    sha256: str


class ManifestRegistry:
    """Discovers and caches all BlahML manifests shipped with the package.

    Manifests are loaded once at construction; mutating the YAML on disk
    after that point requires building a new registry.
    """

    def __init__(self, manifest_dir: Path | None = None):
        self._dir = Path(manifest_dir) if manifest_dir else self._default_dir()
        self._by_id: dict[str, RegisteredManifest] = {}
        self._load_all()

    @staticmethod
    def _default_dir() -> Path:
        return Path(
            importlib.resources.files("mousehash.blahml").joinpath("manifests")
        )

    def _load_all(self) -> None:
        from mousehash.blahml.validator import validate_manifest_structure

        for yaml_path in sorted(self._dir.glob("*.yaml")):
            manifest = load_manifest(yaml_path)
            validate_manifest_structure(manifest)
            sha = manifest_sha256(yaml_path)
            if manifest.id in self._by_id:
                raise ValueError(
                    f"Duplicate manifest id {manifest.id!r} "
                    f"(seen at {self._by_id[manifest.id].path} and {yaml_path})"
                )
            self._by_id[manifest.id] = RegisteredManifest(
                manifest=manifest, path=yaml_path, sha256=sha
            )

    def by_id(self, tool_id: str) -> RegisteredManifest:
        if tool_id not in self._by_id:
            raise KeyError(
                f"No BlahML manifest with id {tool_id!r}. "
                f"Known ids: {sorted(self._by_id)}"
            )
        return self._by_id[tool_id]

    def by_workflow_family(self, family: str) -> list[RegisteredManifest]:
        return [
            r for r in self._by_id.values() if r.manifest.workflow_family == family
        ]

    def all_ids(self) -> list[str]:
        return sorted(self._by_id)
