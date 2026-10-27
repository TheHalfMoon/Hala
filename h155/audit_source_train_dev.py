"""H155 owner-only independent file, image SHA and typed label audit."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from h154.miniwob_source_probe import TASKS
from h155.action_contract import from_neural_response
from h155.source_train_dev import SCHEMA, SPLITS, excluded_images


def audit(owner):
    owner = Path(owner)
    folder = owner / "private-h155" / "source-train-dev-v1"
    raw = (folder / "manifest.json").read_bytes()
    doc = json.loads(raw)
    rows = doc.get("private_cases")
    if (doc.get("schema") != SCHEMA or
        doc.get("status") != "PRIVATE_ACTION_TRAIN_DEV_SOURCE_COMPLETE" or
        doc.get("train") != 48 or doc.get("dev") != 12 or
        doc.get("dev_is_not_blind") is not True or
        doc.get("model_inference_calls") != 0 or
        doc.get("optimizer_steps") != 0 or
        doc.get("h140_blind_neural_calls") != 0 or
        not isinstance(rows, list) or len(rows) != 60):
        raise ValueError("H155 original private TRAIN/DEV manifest invalid")
    banned = excluded_images(owner)
    sha_seen = set()
    seeds = set()
    counter = Counter()
    for case in rows:
        if (set(case) != {
            "split", "task", "action_family", "seed", "image",
            "image_sha256", "instruction", "training_supervision",
            "reference_reward_positive", "model_inference_calls"} or
            case["split"] not in SPLITS or
            case["task"] not in TASKS or
            case["action_family"] != TASKS[case["task"]] or
            not isinstance(case["seed"], int) or
            case["seed"] in seeds or
            case["reference_reward_positive"] is not True or
            case["model_inference_calls"] != 0):
            raise ValueError("H155 original source case schema, identity or reward invalid")
        name = case["image"]
        if (not isinstance(name, str) or
            name != Path(name).name or not name.endswith(".png")):
            raise ValueError("Unsafe H155 private image path")
        data = (folder / "images" / name).read_bytes()
        expected = hashlib.sha256(data).hexdigest()
        if (expected != case["image_sha256"] or
            expected in sha_seen or expected in banned):
            raise ValueError("H155 original source SHA duplicate, leak or substitution")
        if not isinstance(case["instruction"], str) or not case["instruction"]:
            raise ValueError("Missing visible original user instruction")
        # Structured parser only; NO neural call. Label stays private.
        from_neural_response(json.dumps(case["training_supervision"]),
            task=case["task"], source="h155-new-trained-model")
        seeds.add(case["seed"])
        sha_seen.add(expected)
        counter[(case["split"], case["task"])] += 1
    if counter != {(split, task): count for split, count in SPLITS.items()
                   for task in TASKS}:
        raise ValueError("H155 source train/dev strata are not frozen/balanced")
    return {
        "schema": "hala.h155.owner-private-train-dev-independent-audit.v1",
        "source_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "status": "PASS_PRIVATE_ORIGINAL_ACTION_SOURCE_AUDIT",
        "training_samples": 48, "dev_samples": 12,
        "fresh_unique_sha": len(sha_seen),
        "previous_h154_media_excluded": len(banned),
        "validated_typed_training_actions": len(rows),
        "model_inference_calls": 0, "optimizer_steps": 0,
        "h140_blind_calls": 0,
        "model_accuracy_measured": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner-root", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.owner_root), indent=2, sort_keys=True))
