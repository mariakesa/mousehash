from dandi.dandiapi import DandiAPIClient

import json
import time
from pathlib import Path
from tqdm.auto import tqdm


OUT_DIR = Path("dandi_metadata_dump")
OUT_DIR.mkdir(exist_ok=True)

OUT_JSON = OUT_DIR / "all_dandiset_metadata.json"
ERRORS_JSON = OUT_DIR / "all_dandiset_metadata_errors.json"
PARTIAL_JSON = OUT_DIR / "partial_all_dandiset_metadata.json"

MAX_DANDISETS = None   # set to e.g. 20 for testing
SAVE_EVERY = 25
SLEEP_SECONDS = 0.05


def safe_jsonable(x):
    """
    Convert DANDI/Pydantic-ish objects into plain JSON-compatible objects.
    """
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


def get_dandiset_metadata(dandiset):
    """
    Try a few ways of getting raw metadata.
    """
    if hasattr(dandiset, "get_raw_metadata"):
        return dandiset.get_raw_metadata()

    if hasattr(dandiset, "get_metadata"):
        return dandiset.get_metadata()

    raise RuntimeError("Dandiset object has no get_raw_metadata() or get_metadata().")


def crawl_all_dandiset_metadata():
    records = []
    errors = []

    with DandiAPIClient.for_dandi_instance("dandi") as client:
        dandisets_iter = client.get_dandisets()

        for i, dandiset in enumerate(tqdm(dandisets_iter, desc="Fetching DANDI metadata")):
            if MAX_DANDISETS is not None and i >= MAX_DANDISETS:
                break

            try:
                raw_meta = get_dandiset_metadata(dandiset)
                raw_meta = safe_jsonable(raw_meta)

                record = {
                    "crawl_index": i,
                    "dandiset_id": raw_meta.get("identifier"),
                    "id": raw_meta.get("id"),
                    "version": raw_meta.get("version"),
                    "name": raw_meta.get("name"),
                    "url": raw_meta.get("url"),
                    "metadata": raw_meta,
                }

                records.append(record)

            except Exception as e:
                errors.append({
                    "crawl_index": i,
                    "dandiset_repr": repr(dandiset),
                    "error": repr(e),
                })

            if (i + 1) % SAVE_EVERY == 0:
                PARTIAL_JSON.write_text(
                    json.dumps(records, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                ERRORS_JSON.write_text(
                    json.dumps(errors, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

            if SLEEP_SECONDS:
                time.sleep(SLEEP_SECONDS)

    OUT_JSON.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    ERRORS_JSON.write_text(
        json.dumps(errors, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print(f"Saved metadata records: {len(records)}")
    print(f"Saved errors: {len(errors)}")
    print(f"Metadata JSON: {OUT_JSON}")
    print(f"Errors JSON: {ERRORS_JSON}")

    if errors:
        print("\nFirst error:")
        print(json.dumps(errors[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    crawl_all_dandiset_metadata()