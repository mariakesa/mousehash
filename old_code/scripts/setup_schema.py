import traceback

modules = [
    "mousehash.schema.stimuli",
    "mousehash.schema.representations",
    "mousehash.schema.decompositions",
    "mousehash.schema.reports",
]

for module_name in modules:
    print(f"\n=== Importing {module_name} ===")
    try:
        __import__(module_name)
        print(f"OK: {module_name}")
    except Exception as e:
        print(f"FAILED: {module_name}")
        print(f"{type(e).__name__}: {e}")
        traceback.print_exc()
        raise