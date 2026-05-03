"""Test-suite bootstrap.

`mousehash.config` reads required env vars (DJ_USER, DJ_HOST, DATA_ROOT, ...)
at import time, and several schema modules instantiate ``dj.Schema(...)`` at
module load. The project's `.env` lives at the repo root; load it here so
running pytest from any cwd works.
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")
