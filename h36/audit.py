"""Fail-closed paired H36 synthetic DEV raw and image replay.

Raw hash is supplied separately by the operator; metadata is an observation,
not proof of GPU identity or independent holdout.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from h34.audit import bounded_read, image_manifest, strict_json
from h34.corpus import build_cases, case_sha256
from h34.protocol import score_attempt
from .protocol import ARMS, MODEL, PROTOCOL, RAW_FIELDS, REVISION, plan, plan_sha, prompt_sha, sha_bytes


def audit(output: Path, expected_raw_sha256: str) -> dict:
    output = Path(output)
    if type(expected_raw_sha256) is not str or not re.fullmatch("[0-9a-f]{64}", expected_raw_sha256):
        raise ValueError("external raw SHA256 required")
    raw_bytes = bounded_read(output / "raw_attempts.jsonl", 200_000)
    if sha_bytes(raw_bytes) != expected_raw_sha256:
        raise ValueError("H36 raw journal digest mismatch")
    if raw_bytes and not raw_bytes.endswith(b"\n"):
        raise ValueError("partial journal record")
    cases = build_cases()
    tasks = plan()
    if bounded_read(output / "cases.sha256", 65) != (case_sha256() + "\n").encode():
        raise ValueError("changed H34 frozen case digest")
    if bounded_read(output / "plan.sha256", 65) != (plan_sha() + "\n").encode():
        raise ValueError("changed H36 prospective trial order")
    images = image_manifest(output, cases)
    if len(images) != 32:
        raise ValueError("all 32 SHA-bound images required")
    rows = [strict_json(line) for line in raw_bytes.decode("utf-8").splitlines()]
    if len(rows) > len(tasks):
        raise ValueError("extra trial records")
    indexed = {case["id"]: (case, image) for case, image in zip(cases, images)}
    counts = {arm: {"denominator": 32, "attempted": 0, "parse_valid": 0, "target_hit": 0,
                    "wrong_target": 0, "gate_valid_unique": 0, "abstain": 0, "error": 0}
              for arm in ARMS}
    grouped = {arm: {lang: {"denominator": 16, "target_hit": 0, "gate_valid_unique": 0}
                      for lang in ("en", "ar")} for arm in ARMS}
    outcomes = {}
    for index, trial in enumerate(tasks):
        arm = trial["arm"]
        if index >= len(rows):
            counts[arm]["error"] += 1
            continue
        record = rows[index]
        case, image = indexed[trial["case_id"]]
        if type(record) is not dict or set(record) != RAW_FIELDS:
            raise ValueError("raw schema differs")
        expected = {"case_id": trial["case_id"], "language": case["language"],
                    "arm": arm, "prompt_sha256": prompt_sha(arm),
                    "image_sha256": image["sha256"]}
        if any(record[key] != value or type(record[key]) is not str for key, value in expected.items()):
            raise ValueError("raw trial, order, prompt, or image binding differs")
        raw, error = record["raw"], record["error"]
        if error is None:
            if type(raw) is not str or len(raw.encode("utf-8")) > 2048:
                raise ValueError("invalid raw response")
        elif (type(error) is not str or not re.fullmatch("[A-Za-z_][A-Za-z0-9_]{0,79}", error)
              or raw is not None):
            raise ValueError("invalid error record")
        outcome = score_attempt(case, raw, error)
        counts[arm]["attempted"] += 1
        for key in ("parse_valid", "target_hit", "wrong_target", "gate_valid_unique", "abstain", "error"):
            counts[arm][key] += bool(outcome.get(key))
        for key in ("target_hit", "gate_valid_unique"):
            grouped[arm][case["language"]][key] += bool(outcome.get(key))
        outcomes[(case["id"], arm)] = outcome
    paired = {"denominator": 32, "both_attempted": 0, "both_correct": 0,
              "only_numeric_correct": 0, "only_no_numeric_correct": 0,
              "neither_correct": 0}
    for case in cases:
        a = outcomes.get((case["id"], ARMS[0]))
        b = outcomes.get((case["id"], ARMS[1]))
        if a is None or b is None:
            continue
        paired["both_attempted"] += 1
        left, right = bool(a["target_hit"]), bool(b["target_hit"])
        paired["both_correct"] += left and right
        paired["only_numeric_correct"] += left and not right
        paired["only_no_numeric_correct"] += right and not left
        paired["neither_correct"] += not left and not right
    metadata = strict_json(bounded_read(output / "run_metadata.json", 65536))
    if (type(metadata) is not dict or metadata.get("protocol") != PROTOCOL
            or metadata.get("plan_sha256") != plan_sha()
            or metadata.get("source_commit") is None
            or not isinstance(metadata.get("source_commit"), str)
            or not re.fullmatch("[a-f0-9]{40}", metadata["source_commit"])
            or metadata.get("model") != MODEL or metadata.get("revision") != REVISION
            or metadata.get("split") != "exposed-synthetic-dev"
            or metadata.get("actions_executed") is not False):
        raise ValueError("run metadata identity invalid")
    return {"schema": "hala.h36.paired_dev_replay.v1", "raw_sha256": expected_raw_sha256,
            "case_sha256": case_sha256(), "plan_sha256": plan_sha(),
            "source_commit_observed": metadata["source_commit"],
            "attempted_records": len(rows), "complete": len(rows) == 64,
            "arms": counts, "languages": grouped, "paired": paired,
            "exposed_dev_only": True, "gpu_observations_not_attestation": True,
            "model_qualified": False, "real_screenshot_qualified": False,
            "native_actions_executed": False, "independent_test_qualified": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-raw-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(Path(args.output), args.expected_raw_sha256),
                     sort_keys=True, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
