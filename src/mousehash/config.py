from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


DATA_ROOT = Path(_required("MOUSEHASH_DATA_ROOT")).expanduser().resolve()
ALLEN_MANIFEST_PATH = Path(_required("ALLEN_DATA")).expanduser().resolve()

DJ_HOST = _required("DJ_HOST")
DJ_PORT = int(os.getenv("DJ_PORT", "3306"))
DJ_USER = _required("DJ_USER")
DJ_PASS = _required("DJ_PASS")
DB_PREFIX = os.getenv("MOUSEHASH_DB_PREFIX", "mousehash")