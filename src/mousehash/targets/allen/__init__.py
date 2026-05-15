"""Allen Brain Observatory adapter.

Exposes the natural-scenes stimulus set, presentation timing, and the
identity catalog as a MouseHash `RoleManifest` + `RoleBundle` so downstream
target-agnostic tools can run without ever importing AllenSDK.
"""

from mousehash.targets.allen.adapter import AllenAdapter

__all__ = ["AllenAdapter"]
