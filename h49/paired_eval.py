"""Offline independent scoring of genuine *externally produced* H49 paired raw model runs.

This scorer NEVER executes a model, synthesizes attempts, or claims blind TEST.
It refuses incomplete/forged source bindings, duplicate trials and mismatched
hardware/prompts/checkpoints before reporting any outcome.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from h49.audit import inspect
from h49.factory import WIDTH, HEIGHT, sha
from training.artifacts import strict_json

SCHEMA = "hala.h49.paired-raw-inference.v1"
MODEL = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
ARMS = ("baseline", "adapter")
FIELDS = {"case_id", "language", "image_sha256", "arm", "raw",
          "error", "elapsed_seconds", "output_tokens"}
META = {"schema", "arm", "model_id", "model_revision",
        "processor_revision", "prompt_sha256", "dataset_sha256",
        "source_sha256", "capture_sha256", "gpu_name", "precision",
        "provider", "provider_run_id", "adapter_sha256",
        "raw_sha256", "status"}
MAX_RAW_BYTES = 256 * 1024


def public_prompt(goal):
    if type(goal) is not str or not goal or len(goal) > 1000:
        raise ValueError("Bounded actual H49 public goal required")
    return ("Select the single intended visible control from this screenshot. "
            "Return only one JSON window-pixel click proposal; do not act.\n"
            "Screenshot dimensions: 1024 by 640.\nGoal: " + goal)


def digest_prompt(goals):
    # Bind both arms to the exact same ordered H49 instruction/goal sequence.
    return sha("".join(public_prompt(goal) + "\n" for goal in goals).encode("utf-8"))


def _strict(raw):
    def pairs(items):
        found = {}
        for key, value in items:
            if key in found:
                raise ValueError("Duplicate JSON output key")
            found[key] = value
        return found
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite output")))


def parse_action(raw):
    if type(raw) is not str or len(raw.encode("utf-8")) > 2048:
        raise ValueError("Missing/oversized model response")
    action = _strict(raw)
    if action == {"abstain": True}:
        return None
    if (type(action) is not dict or
        set(action) != {"type", "point", "coordinate_space",
                        "image_scale", "element_ref_hint"} or
        action["type"] != "click" or action["coordinate_space"] != "window" or
        type(action["image_scale"]) is not int or action["image_scale"] != 1 or
        action["element_ref_hint"] != "visual_target" or
        type(action["point"]) is not list or len(action["point"]) != 2 or
        any(type(v) is not int for v in action["point"]) or
        not (0 <= action["point"][0] < WIDTH and
             0 <= action["point"][1] < HEIGHT)):
        raise ValueError("Not exact SFT-compatible window pixel click")
    return tuple(action["point"])


def bind(point, controls):
    x, y = point
    matches = []
    for control in controls:
        left, top, width, height = control["bounds"]
        if left <= x < left + width and top <= y < top + height:
            matches.append(control["key"])
    return matches


def judge(attempt, case, controls):
    if attempt["error"] is not None:
        return "inference_error"
    try:
        point = parse_action(attempt["raw"])
    except (ValueError, TypeError, OverflowError, RecursionError, UnicodeError):
        return "invalid_json_or_schema"
    if point is None:
        return "abstain"
    matched = bind(point, controls)
    if not matched:
        return "unbound"
    if len(matched) != 1:
        return "ambiguous"
    return "correct" if matched[0] == case["target_key"] else "wrong_target"


def _read(path, supplied_sha):
    if not re.fullmatch("[0-9a-f]{64}", str(supplied_sha)):
        raise ValueError("External original evidence SHA256 required")
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_RAW_BYTES:
        raise ValueError("Regular bounded original evidence required")
    data = path.read_bytes()
    if sha(data) != supplied_sha:
        raise ValueError("External original evidence bytes differ")
    return data


def _load_arm(root, arm, meta_path, raw_path, meta_sha, raw_sha,
              source, dev_rows, goal_sha):
    metadata = strict_json(_read(meta_path, meta_sha).decode("utf-8"))
    if type(metadata) is not dict or set(metadata) != META:
        raise ValueError("Exact complete external provider metadata required")
    expected = {
        "schema": SCHEMA, "arm": arm, "model_id": MODEL,
        "model_revision": REVISION, "processor_revision": REVISION,
        "source_sha256": source["source_sha256"],
        "capture_sha256": source["capture_sha256"],
        "dataset_sha256": source["dataset_sha256"],
        "prompt_sha256": goal_sha,
        "precision": "fp16", "status": "completed",
        "adapter_sha256": None if arm == "baseline" else metadata["adapter_sha256"],
        "raw_sha256": raw_sha,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Model/source/prompt/raw/adapter binding mismatch")
    if (metadata["provider"] not in ("colab", "linux_cuda") or
        type(metadata["provider_run_id"]) is not str or not metadata["provider_run_id"] or
        type(metadata["gpu_name"]) is not str or not metadata["gpu_name"] or
        (arm == "adapter" and not re.fullmatch("[0-9a-f]{64}",
                                                 str(metadata["adapter_sha256"])))):
        raise ValueError("Missing genuine provider and versioned adapter identification")
    data = _read(raw_path, raw_sha)
    attempts = []
    lines = data.splitlines()
    if len(lines) != len(dev_rows):
        raise ValueError("Exactly one actual model record for every public DEV case")
    for index, (raw_line, case) in enumerate(zip(lines, dev_rows)):
        record = strict_json(raw_line.decode("utf-8"))
        if type(record) is not dict or set(record) != FIELDS:
            raise ValueError("Unexpected raw model record fields")
        if (record["case_id"] != case["id"] or record["arm"] != arm or
            record["language"] != case["language"] or
            record["image_sha256"] != case["image_sha256"]):
            raise ValueError("Reordered/duplicate/wrong-model case binding")
        if (type(record["elapsed_seconds"]) not in (float, int) or
            not math.isfinite(record["elapsed_seconds"]) or
            record["elapsed_seconds"] < 0 or
            type(record["output_tokens"]) is not int or
            record["output_tokens"] < 0 or record["output_tokens"] > 1024 or
            (record["error"] is not None and
             (type(record["error"]) is not str or
              not 1 <= len(record["error"]) <= 100 or record["raw"] is not None)) or
            (record["error"] is None and type(record["raw"]) is not str)):
            raise ValueError("Invalid actual model attempt telemetry")
        attempts.append(record)
    return metadata, attempts


def score(root, baseline_meta, baseline_raw, baseline_meta_sha, baseline_raw_sha,
          adapter_meta, adapter_raw, adapter_meta_sha, adapter_raw_sha):
    root = Path(root)
    projections, manifest = inspect(root)
    model = strict_json(projections["model_inputs.json"].decode("utf-8"))
    evaluator = strict_json(projections["evaluator_only.json"].decode("utf-8"))
    dev = [r for r in model["cases"] if r["split"] == "dev"]
    oracle = {r["id"]: r for r in evaluator["cases"] if r["split"] == "dev"}
    if (len(dev) != 32 or len(oracle) != 32 or
        sum(r["language"] == "ar" for r in dev) != 16 or
        sum(r["language"] == "en" for r in dev) != 16):
        raise ValueError("Original H49 public DEV case universe changed")
    for r in dev:
        if r["id"] not in oracle or r["image_sha256"] != oracle[r["id"]]["image_sha256"]:
            raise ValueError("Model-only and independent evaluator identities differ")
    training_goals = {row["id"]: row["goal"] for row in model["cases"]
                      if row["split"] == "train"}
    sft_rows = [strict_json(line) for line in projections["authoring_sft.jsonl"]
                .decode("utf-8").splitlines()]
    if (len(sft_rows) != len(training_goals) or
        any(row["id"] not in training_goals or
            row["prompt"] != public_prompt(training_goals[row["id"]])
            for row in sft_rows)):
        raise ValueError("Actual trained SFT prompt differs from paired inference protocol")
    prompt_hash = digest_prompt([row["goal"] for row in dev])
    metadata = {}
    attempts = {}
    for arm, mp, rp, mh, rh in (
        ("baseline", baseline_meta, baseline_raw, baseline_meta_sha, baseline_raw_sha),
        ("adapter", adapter_meta, adapter_raw, adapter_meta_sha, adapter_raw_sha),
    ):
        metadata[arm], attempts[arm] = _load_arm(
            root, arm, mp, rp, mh, rh, manifest, dev, prompt_hash)
    # Paired conditions: no hardware/model/prompt/processor/source confounding.
    for field in ("gpu_name", "provider", "model_id", "model_revision", "processor_revision",
                  "precision", "prompt_sha256", "source_sha256", "capture_sha256"):
        if metadata["baseline"][field] != metadata["adapter"][field]:
            raise ValueError("Actual paired comparison conditions mismatch: " + field)
    source = strict_json((root / "capture.json").read_text("utf-8"))
    controls = {record["id"]: record["controls"] for record in source["records"]}
    result = {"schema": "hala.h49.public-dev-paired-score.v1",
              "source_sha256": manifest["source_sha256"],
              "capture_sha256": manifest["capture_sha256"],
              "training_data_sha256": manifest["dataset_sha256"],
              "raw_sha256": {arm: metadata[arm]["raw_sha256"] for arm in ARMS},
              "model_id": MODEL, "revision": REVISION,
              "adapter_sha256": metadata["adapter"]["adapter_sha256"],
              "real_model_provenance_independently_verified": False,
              "public_dev_only": True, "independent_blind_test": False,
              "quality_qualified": False,
              "total_tasks": len(dev), "counts": {}, "paired": {},
              "per_case": []}
    for arm in ARMS:
        outcomes = [judge(attempt, oracle[case["id"]], controls[case["id"]])
                    for case, attempt in zip(dev, attempts[arm])]
        if len(outcomes) != len(dev):
            raise ValueError("Missing attempt during independent score")
        result["counts"][arm] = {
            lang: dict(Counter(outcome for case, outcome in zip(dev, outcomes)
                               if case["language"] == lang),
                       denominator=sum(case["language"] == lang for case in dev))
            for lang in ("ar", "en", "all")
        }
        # Explicit all-language aggregate (not a copied unbound 'all' bucket).
        result["counts"][arm]["all"] = dict(Counter(outcomes), denominator=len(dev))
        result["counts"][arm]["all"]["latency_seconds"] = sum(
            attempt["elapsed_seconds"] for attempt in attempts[arm])
        result["counts"][arm]["all"]["output_tokens"] = sum(
            attempt["output_tokens"] for attempt in attempts[arm])
        result.setdefault("_outcomes", {})[arm] = outcomes
    for i,case in enumerate(dev):
        a = result["_outcomes"]["baseline"][i]
        b = result["_outcomes"]["adapter"][i]
        result["per_case"].append({"id": case["id"], "language": case["language"],
                                   "family": case["family"], "baseline": a, "adapter": b})
    for lang in ("ar", "en", "all"):
        pairs = [r for r in result["per_case"] if lang == "all" or r["language"] == lang]
        result["paired"][lang] = {
            "denominator": len(pairs),
            "adapter_only_correct": sum(r["baseline"] != "correct" and r["adapter"] == "correct"
                                        for r in pairs),
            "baseline_only_correct": sum(r["baseline"] == "correct" and r["adapter"] != "correct"
                                         for r in pairs),
            "both_correct": sum(r["baseline"] == "correct" and r["adapter"] == "correct"
                                for r in pairs),
            "neither_correct": sum(r["baseline"] != "correct" and r["adapter"] != "correct"
                                   for r in pairs),
        }
    result.pop("_outcomes")
    return result


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="h49/assets")
    p.add_argument("--baseline-meta", required=True)
    p.add_argument("--baseline-raw", required=True)
    p.add_argument("--baseline-meta-sha", required=True)
    p.add_argument("--baseline-raw-sha", required=True)
    p.add_argument("--adapter-meta", required=True)
    p.add_argument("--adapter-raw", required=True)
    p.add_argument("--adapter-meta-sha", required=True)
    p.add_argument("--adapter-raw-sha", required=True)
    p.add_argument("--output", required=True)
    args = vars(p.parse_args())
    dest = Path(args.pop("output"))
    result = score(args.pop("root"), args.pop("baseline_meta"), args.pop("baseline_raw"),
                   args.pop("baseline_meta_sha"), args.pop("baseline_raw_sha"),
                   args.pop("adapter_meta"), args.pop("adapter_raw"),
                   args.pop("adapter_meta_sha"), args.pop("adapter_raw_sha"))
    with dest.open("xb") as f:
        f.write((json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())
    print("H49_PUBLIC_DEV_NOT_BLIND", result["paired"]["all"],
          "MODEL_PROVENANCE_NOT_INDEPENDENTLY_VERIFIED", flush=True)

if __name__ == "__main__":
    main()
