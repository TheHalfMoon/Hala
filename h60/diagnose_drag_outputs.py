"""H60 honest diagnostic over real raw public-dev model drag outputs.

The frozen public-dev scorers accept only a bare JSON object. This diagnostic
additionally reports what a tolerant reader would see, so that a *format*
failure (for example a markdown fence) is never silently reported as a
*grounding* failure and never silently reported as a success.

It never modifies the raw model strings and it never reads a public evaluation
or hidden test source. It produces a receipt, not a qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "hala.h60.public-dev-drag-output-diagnostic.v1"
FENCE = "```json\n"


def strict_json(text: str):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("Duplicate JSON key")
            value[key] = item
        return value

    def constant(name):
        raise ValueError("Non finite JSON constant")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def _parse(raw: str, *, tolerant: bool):
    if not isinstance(raw, str) or len(raw) > 4096:
        return None, "unbounded_or_not_text"
    text = raw
    if tolerant:
        stripped = raw.strip()
        if stripped.startswith(FENCE) and stripped.endswith("```"):
            text = stripped[len(FENCE) : -3].strip()
    try:
        value = strict_json(text)
    except (ValueError, RecursionError, TypeError):
        return None, "not_json"
    if (
        not isinstance(value, dict)
        or set(value) != {"type", "start_point", "end_point", "coordinate_space", "ordered"}
        or value["type"] != "drag"
        or value["coordinate_space"] != "source_image_pixels"
        or type(value["ordered"]) is not bool
    ):
        return None, "schema_mismatch"
    for name in ("start_point", "end_point"):
        point = value[name]
        if (
            not isinstance(point, list)
            or len(point) != 2
            or type(point[0]) is not int
            or not 0 <= point[0] < 1024
            or type(point[1]) is not int
            or not 0 <= point[1] < 640
        ):
            return None, "point_out_of_range"
    return value, "ok"


def diagnose(dev: str, root: Path, raw_attempts: Path) -> dict:
    root = Path(root)
    capture = json.loads((root / "capture.json").read_text("utf8"))
    attempts = json.loads(Path(raw_attempts).read_text("utf8"))
    expected = {case["id"]: case for case in capture["cases"]}
    cases = attempts.get("cases", [])
    if set(expected) != {case["id"] for case in cases}:
        raise ValueError("Raw attempt set does not match the frozen public DEV scene set")
    buckets: dict[str, dict] = {}
    failures: dict[str, int] = {}
    for case in cases:
        truth = expected[case["id"]]
        label = truth.get("domain", "slider") + "/" + truth["language"]
        bucket = buckets.setdefault(
            label,
            dict(
                n=0,
                strict_valid=0,
                tolerant_valid=0,
                ordered_semantics_valid=0,
                start_hit=0,
                end_hit=0,
                geometrically_correct=0,
                degenerate_start_equals_end=0,
            ),
        )
        bucket["n"] += 1
        strict, _ = _parse(case["raw_output"], tolerant=False)
        if strict is not None:
            bucket["strict_valid"] += 1
        tolerant, reason = _parse(case["raw_output"], tolerant=True)
        if tolerant is None:
            failures[reason] = failures.get(reason, 0) + 1
            continue
        bucket["tolerant_valid"] += 1
        if tolerant["start_point"] == tolerant["end_point"]:
            bucket["degenerate_start_equals_end"] += 1
        if tolerant["ordered"] is not truth.get("ordered", True):
            continue
        bucket["ordered_semantics_valid"] += 1
        start, end = truth["start_point"], truth["end_point"]
        tolerance = 8 if dev == "h55" else 22
        close = lambda a, b: math.dist(a, b) < tolerance  # noqa: E731
        forward = close(tolerant["start_point"], start) and close(tolerant["end_point"], end)
        backward = close(tolerant["start_point"], end) and close(tolerant["end_point"], start)
        bucket["start_hit"] += int(close(tolerant["start_point"], start))
        bucket["end_hit"] += int(close(tolerant["end_point"], end))
        good = forward or (not truth.get("ordered", True) and backward)
        bucket["geometrically_correct"] += int(good)
    total = {
        key: sum(bucket[key] for bucket in buckets.values())
        for key in (
            "n",
            "strict_valid",
            "tolerant_valid",
            "ordered_semantics_valid",
            "start_hit",
            "end_hit",
            "geometrically_correct",
            "degenerate_start_equals_end",
        )
    }
    return {
        "schema": SCHEMA,
        "dev_set": dev,
        "source_sha256": hashlib.sha256((root / "source.json").read_bytes()).hexdigest(),
        "raw_attempts_sha256": hashlib.sha256(Path(raw_attempts).read_bytes()).hexdigest(),
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "revision": "89644892e4d85e24eaac8bacfd4f463576704203",
        "evidence": "real_inference",
        "adapters_loaded": False,
        "model_training_executed": False,
        "by_domain_language": buckets,
        "totals": total,
        "parse_failures": failures,
        "public_dev_only": True,
        "blind_test": False,
        "real_browser_model_task_success_verified": False,
        "independent_blind_test_completed": False,
        "official_benchmark_comparison_qualified": False,
        "notes": (
            "strict_valid counts outputs the frozen public-dev scorer accepts as a bare "
            "JSON object; tolerant_valid additionally accepts a markdown fenced object so a "
            "format failure is not reported as a grounding failure."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", required=True, choices=("h55", "h58"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--raw-attempts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    receipt = diagnose(args.dev, Path(args.root), Path(args.raw_attempts))
    Path(args.out).write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8"
    )
    print(json.dumps(receipt["totals"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
