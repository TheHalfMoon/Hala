"""H63 paired-image intervention audit.

Unlike H62's static target permutation, this gate requires *recorded separate
model executions* for reference pixels, the same-pixel repeat control and a
different-pixel intervention. It cannot itself attest that a model was run;
the producer must retain independent input-byte, prompt and model receipts.
It provides conservative evidence of visual sensitivity, not proof of reliable
computer use or causal identification in the presence of unmeasured confounders.

Input: JSONL; each row has case_id, reference, repeat and intervention arms.
Every arm has image_sha256, model_sha256, prompt_sha256, target_box,
point (or null), valid_output. The reference/repeat image must be identical;
the intervention image must differ. All prompts and model hashes must match.
Target boxes must be non-overlapping to make a constant point ineligible.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SCHEMA = "hala.h63.paired-image-intervention.v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SUFFICIENT = "PAIRED_VISUAL_SENSITIVITY_OBSERVED"
NO_CHANGE = "NO_MEASURED_VISUAL_SENSITIVITY"
UNSTABLE = "UNSTABLE_REPEAT_CONTROL"
INCONCLUSIVE = "INCONCLUSIVE"
TOO_SMALL = "INSUFFICIENT_PAIRED_EVIDENCE"


def _digest(value, label):
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase sha256 hex digest")
    return value


def _box(value, label):
    if not isinstance(value, list) or len(value) != 4 or any(
        isinstance(n, bool) or not isinstance(n, (int, float)) for n in value
    ):
        raise ValueError(f"{label} must be a numeric xyxy list")
    x1, y1, x2, y2 = value
    if not (0 <= x1 < x2 and 0 <= y1 < y2):
        raise ValueError(f"{label} is not a valid positive xyxy box")
    return tuple(value)


def _overlap(a, b):
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def _point(value, label):
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2 or any(
        isinstance(n, bool) or not isinstance(n, (int, float)) for n in value
    ):
        raise ValueError(f"{label} must be [x,y] or null")
    return tuple(value)


def _hit(point, box):
    return bool(point is not None and box[0] <= point[0] < box[2]
                and box[1] <= point[1] < box[3])


def _arm(row, name):
    arm = row.get(name)
    if not isinstance(arm, dict):
        raise ValueError(f"missing {name} inference arm")
    return {
        "image": _digest(arm.get("image_sha256"), f"{name}.image_sha256"),
        "model": _digest(arm.get("model_sha256"), f"{name}.model_sha256"),
        "prompt": _digest(arm.get("prompt_sha256"), f"{name}.prompt_sha256"),
        "box": _box(arm.get("target_box"), f"{name}.target_box"),
        "point": _point(arm.get("point"), f"{name}.point"),
        "valid": arm.get("valid_output") is True,
    }


def evaluate_pairs(rows, *, minimum=8):
    """Fail on non-paired or unverifiable metadata; count bad answers as misses."""
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 2:
        raise ValueError("minimum must be an integer >= 2")
    seen = set()
    results = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {index} must be an object")
        case = row.get("case_id")
        if not isinstance(case, str) or not case.strip() or case in seen:
            raise ValueError(f"row {index} has missing or duplicate case_id")
        seen.add(case)
        ref = _arm(row, "reference")
        repeat = _arm(row, "repeat")
        swapped = _arm(row, "intervention")
        if ref["image"] != repeat["image"] or ref["image"] == swapped["image"]:
            raise ValueError(f"{case}: need same-image repeat and different-image intervention")
        if len({ref["model"], repeat["model"], swapped["model"]}) != 1:
            raise ValueError(f"{case}: model checkpoint changes between arms")
        if len({ref["prompt"], repeat["prompt"], swapped["prompt"]}) != 1:
            raise ValueError(f"{case}: instruction/prompt changes between arms")
        if ref["box"] != repeat["box"] or _overlap(ref["box"], swapped["box"]):
            raise ValueError(f"{case}: repeat targets differ or intervention targets overlap")
        ref_ok = ref["valid"] and _hit(ref["point"], ref["box"])
        swap_ok = swapped["valid"] and _hit(swapped["point"], swapped["box"])
        stable = (ref["valid"] == repeat["valid"] and
                  ref["point"] == repeat["point"])
        changed = ref["valid"] and swapped["valid"] and ref["point"] != swapped["point"]
        results.append({
            "case_id": case, "reference_hit": ref_ok,
            "intervention_hit": swap_ok, "repeat_stable": stable,
            "prediction_changed": changed,
            "paired_success": bool(ref_ok and swap_ok and stable and changed),
        })
    n = len(results)
    score = lambda field: sum(1 for item in results if item[field])
    paired = score("paired_success")
    stability = score("repeat_stable")
    changes = score("prediction_changed")
    summary = {
        "schema": SCHEMA, "pairs": n, "minimum_pairs": minimum,
        "paired_successes": paired,
        "repeat_stable_pairs": stability,
        "prediction_changed_pairs": changes,
        "reference_hits": score("reference_hit"),
        "intervention_hits": score("intervention_hit"),
        "paired_success_rate": paired / n if n else 0.0,
        "repeat_stability_rate": stability / n if n else 0.0,
        "prediction_change_rate": changes / n if n else 0.0,
        "evidence": "paired_separate_model_inference_with_same_pixel_repeat",
        "claim_limit": (
            "Evidence is conditional on externally verified input pixels, "
            "identical prompt/model/decoding settings and genuine separate "
            "inference receipts; it does not establish task success or SOTA."
        ),
        "results": results,
    }
    if n < minimum:
        summary["verdict"] = TOO_SMALL
    elif stability / n < .95:
        summary["verdict"] = UNSTABLE
    elif paired / n >= .75:
        summary["verdict"] = SUFFICIENT
    elif changes / n <= .10:
        summary["verdict"] = NO_CHANGE
    else:
        summary["verdict"] = INCONCLUSIVE
    return summary


def load_jsonl(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"line {number}: {error.msg}") from error
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", type=Path, help="actual separate paired inference receipts as JSONL")
    parser.add_argument("--minimum", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = evaluate_pairs(load_jsonl(args.pairs), minimum=args.minimum)
    content = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(content, encoding="utf8")
    print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
