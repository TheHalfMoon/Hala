"""SHA-bound retrospective analysis of the retained H34 DEV model responses.

No Kaggle sessions, image generation, model calls, or native actions occur here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from h34.audit import audit
from h34.corpus import build_cases
from h34.geometry import contains
from h34.protocol import parse_raw, score_attempt


def classify(case: dict, raw: str | None, error: str | None = None) -> dict:
    """Fail-closed classification with an explicit independent score agreement."""
    baseline = score_attempt(case, raw, error)
    point = None
    if error is not None:
        category = "inference_error"
    elif not baseline["parse_valid"]:
        category = "parse_invalid"
    else:
        point = parse_raw(raw, case["width"], case["height"])
        if point is None:
            category = "abstain"
        else:
            matches = [c for c in case["controls"] if contains(c["bounds"], point)]
            if not matches:
                category = "off_controls"
            elif len(matches) > 1:
                category = "overlapping_controls"
            elif matches[0]["label"] == case["expected_label"]:
                category = "hit"
            else:
                category = "wrong_control"
    if (baseline["target_hit"] != (category == "hit")
            or bool(baseline.get("gate_valid_unique", False)) !=
            (category in ("hit", "wrong_control"))):
        raise ValueError("H35 classification disagrees with original H34 scorer")
    result = {"class": category, "target_hit": baseline["target_hit"],
              "gate_valid_unique": bool(baseline.get("gate_valid_unique", False)),
              "point_px": None, "distance_to_target_box_px": None,
              "delta_to_target_center_px": None}
    if point is not None:
        px, py = point
        target = next(c for c in case["controls"] if c["is_expected"])
        x, y, w, h = target["bounds"]
        sx = min(max(px, x), x+w)
        sy = min(max(py, y), y+h)
        result.update(point_px=[round(px, 4), round(py, 4)],
                      distance_to_target_box_px=round(math.hypot(px-sx, py-sy), 4),
                      delta_to_target_center_px=[round(px-(x+w/2), 4),
                                                 round(py-(y+h/2), 4)])
    return result


def analyze(directory: Path, expected_raw_sha256: str) -> dict:
    """Require full H34 raw, image hash, PNG, corpus and order verification first."""
    original = audit(Path(directory), expected_raw_sha256)
    cases = build_cases()
    records = original["raw_attempts"]
    counts = Counter()
    groups = {axis: defaultdict(Counter) for axis in ("language", "layout", "domain")}
    rows = []
    for index, case in enumerate(cases):
        detail = (classify(case, records[index]["raw"], records[index]["error"])
                  if index < len(records) else
                  {"class": "not_attempted", "target_hit": False,
                   "gate_valid_unique": False, "point_px": None,
                   "distance_to_target_box_px": None,
                   "delta_to_target_center_px": None})
        row = {"case_id": case["id"], "language": case["language"],
               "layout": case["layout"], "domain": case["domain"], **detail}
        rows.append(row)
        counts[row["class"]] += 1
        for axis in groups:
            groups[axis][row[axis]][row["class"]] += 1
    if sum(counts.values()) != original["metrics"]["denominator"]:
        raise ValueError("H35 unconditional denominator changed")
    for language, metrics in original["metrics"]["languages"].items():
        if sum(r["target_hit"] for r in rows if r["language"] == language) != metrics["target_hit"]:
            raise ValueError("H35 replay disagrees with H34 language scores")
    classes = ("hit", "wrong_control", "off_controls", "overlapping_controls",
               "abstain", "parse_invalid", "inference_error", "not_attempted")
    return {"schema": "hala.h35.h34_retained_failure_forensics.v1",
            "raw_sha256": original["raw_sha256"], "case_sha256": original["case_sha256"],
            "attempted_records": original["attempted_records"], "denominator": len(cases),
            "class_counts": dict(sorted(counts.items())),
            "groups": {axis: {name: {category: tally[category] for category in classes}
                              for name, tally in sorted(values.items())}
                       for axis, values in groups.items()},
            "cases": rows, "replayed_h34_integrity": True,
            "retrospective_dev_only": True, "new_model_inference": False,
            "independent_test_qualified": False, "model_qualified": False,
            "native_actions_executed": False}


def main():
    parser = argparse.ArgumentParser(description="SHA-anchored H34 DEV failure taxonomy")
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--expected-raw-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(Path(args.evidence), args.expected_raw_sha256),
                     ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
