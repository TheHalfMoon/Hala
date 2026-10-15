"""Export only original training families; dev/test and benchmark paths are denied."""
import argparse
import hashlib
import json
from pathlib import Path

from .factory import load_frozen, training_records


def export_training(manifest, path):
    records = training_records(manifest)
    # Only source fixtures are accepted. Evaluation reports have another schema
    # and cannot silently become training examples.
    payload = ("\n".join(json.dumps(c, ensure_ascii=False, sort_keys=True) for c in records) + "\n").encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "  " + path.name + "\n", encoding="ascii")
    return {"path": str(path), "records": len(records), "family_count": len({c["family_id"] for c in records}), "sha256": digest, "split": "train", "contains_evaluation_trajectories": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_training(load_frozen(), args.output), sort_keys=True))


if __name__ == "__main__":
    main()
