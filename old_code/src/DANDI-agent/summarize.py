import json
from pathlib import Path


JSON_PATH = Path("dandi_metadata_dump/all_dandiset_metadata.json")


def load_records(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def extract_name_or_value(x):
    """
    DANDI fields like approach and measurementTechnique are usually lists of dicts:
      {"name": "...", "schemaKey": "..."}
    variableMeasured is usually a list of strings.
    """
    if isinstance(x, dict):
        return x.get("name") or x.get("identifier") or json.dumps(x, ensure_ascii=False)
    return str(x)


def unique_assets_summary_values(records, field_name: str):
    values = set()

    for rec in records:
        metadata = rec.get("metadata", {}) or {}
        assets_summary = metadata.get("assetsSummary", {}) or {}

        field_value = assets_summary.get(field_name, [])

        if field_value is None:
            continue

        if not isinstance(field_value, list):
            field_value = [field_value]

        for item in field_value:
            values.add(extract_name_or_value(item))

    return sorted(values)


def main():
    records = load_records(JSON_PATH)

    approach_values = unique_assets_summary_values(records, "approach")
    variable_measured_values = unique_assets_summary_values(records, "variableMeasured")
    measurement_technique_values = unique_assets_summary_values(records, "measurementTechnique")

    print("\napproach =")
    print(repr(approach_values))

    print("\nvariableMeasured =")
    print(repr(variable_measured_values))

    print("\nmeasurementTechnique =")
    print(repr(measurement_technique_values))


if __name__ == "__main__":
    main()