"""H252 native screenshot action diagnostics — invented CPU inputs only.

This scorer measures parsed action agreement; it does NOT infer GUI success.
NEVER use this entry point with previously scored H173 OOD or owner-private gold.
No file loading, no model inference, no optimizer, no GPU, no admission.
"""
from __future__ import annotations

import json
from math import hypot
from collections import Counter

from h199.native_action_sft import KINDS, parse_actions, require

RADIUS_PX = 12.0
SCROLL_DELTA_TOLERANCE_PX = 120
MEASURABLE_KINDS = frozenset(KINDS - {"abstain"})
MAX_SYNTHETIC_CASES = 32


def _close_point(a: list, b: list) -> bool:
    """A geometric diagnostic, NEVER a verified action outcome."""
    return hypot(a[0] - b[0], a[1] - b[1]) <= RADIUS_PX


def diagnose_prediction(prediction_json: str, reference_json: str) -> dict:
    """Strictly compare ONE predicted action to ONE original/reference action.

    Return only non-sensitive booleans and the reference action *kind*.
    Never return text, points, prediction JSON, task, gold or source hashes.
    H173 scored OOD must not be provided here (even if parseable).
    """
    reference = parse_actions(reference_json, training_target=True)["actions"][0]
    kind = reference["kind"]
    require(kind in MEASURABLE_KINDS, "H252 reference must be a non-abstain action")
    try:
        predicted = parse_actions(prediction_json)["actions"][0]
    except (TypeError, ValueError):
        return {"kind": kind, "valid": False, "abstained": False,
                "kind_match": False, "exact": False, "near_point": None,
                "direction_match": None, "scroll_delta_near": None}

    pred_kind = predicted["kind"]
    abstained = pred_kind == "abstain"
    same = pred_kind == kind
    point: bool | None = None
    direction: bool | None = None
    magnitude: bool | None = None
    if same:
        if kind in ("click", "key_press", "scroll"):
            point = _close_point(predicted["point"], reference["point"])
        elif kind == "drag":
            point = (_close_point(predicted["start"], reference["start"])
                     and _close_point(predicted["end"], reference["end"]))
        if kind == "scroll":
            direction = ((predicted["dy"] > 0) == (reference["dy"] > 0))
            magnitude = abs(predicted["dy"] - reference["dy"]) <= SCROLL_DELTA_TOLERANCE_PX
    return {"kind": kind, "valid": True, "abstained": abstained,
            "kind_match": same, "exact": same and predicted == reference,
            "near_point": point, "direction_match": direction,
            "scroll_delta_near": magnitude}


def synthetic_contract_summary(pairs: list[tuple[str, str]]) -> dict:
    """Aggregate an explicitly INVENTED fixture list; never read private gold.

    Input ordering: (prediction_json, reference_json). Bounded examples
    prevent unwitting processing of private/historical OOD corpora. This is
    diagnostic agreement only, NOT actual inference or model performance.
    """
    require(type(pairs) is list and 1 <= len(pairs) <= MAX_SYNTHETIC_CASES,
            "H252 only 1..32 invented, synthetic CPU pairs")
    stats: list[dict] = []
    for row in pairs:
        require(type(row) is tuple and len(row) == 2
                and all(type(value) is str for value in row),
                "invented prediction/reference pair must be two JSON strings")
        stats.append(diagnose_prediction(row[0], row[1]))
    kinds = Counter(s["kind"] for s in stats)
    def count(key):
        return sum(s[key] is True for s in stats)
    spatial = [s for s in stats if s["near_point"] is not None]
    scroll = [s for s in stats if s["direction_match"] is not None]
    # No source content, raw user task, screenshot, gold, reference coordinate,
    # prediction text or per-case rows in this public-safe report.
    return {
        "status": "H252_INVENTED_SYNTHETIC_ACTION_DIAGNOSTICS_ONLY",
        "diagnostic_cases": len(stats),
        "reference_kind_counts": {k: kinds[k] for k in sorted(kinds)},
        "valid_action_count": count("valid"),
        "abstain_count": count("abstained"),
        "kind_match_count": count("kind_match"),
        "exact_action_count": count("exact"),
        "spatial_comparison_cases": len(spatial),
        "spatial_within_12px_count": sum(s["near_point"] is True for s in spatial),
        "scroll_comparison_cases": len(scroll),
        "scroll_direction_match_count": sum(s["direction_match"] is True for s in scroll),
        "scroll_delta_within_120px_count": sum(
            s["scroll_delta_near"] is True for s in scroll
        ),
        "actual_browser_task_success_measured": False,
        "actual_model_predictions_measured": False,
        "model_calls": 0,
        "optimizer_steps": 0,
        "train_admitted": False,
        "dev_admitted": False,
        "ood_evaluated": False,
        "new_model_gain_proven": False,
    }


def _encoded(item: dict) -> str:
    return json.dumps({"actions": [item]}, ensure_ascii=False, separators=(",", ":"))


def invented_cpu_example() -> dict:
    """Five INVENTED action labels only. No browser/media/ML invocation."""
    references = (
        {"kind": "click", "point": [320, 260]},
        {"kind": "drag", "start": [100, 120], "end": [250, 210]},
        {"kind": "key_press", "point": [410, 315], "key": "ArrowDown"},
        {"kind": "type_text", "text": "أضف مهمة جديدة"},
        {"kind": "scroll", "point": [480, 600], "dy": 480},
    )
    return synthetic_contract_summary([
        (_encoded(item), _encoded(item)) for item in references
    ])


if __name__ == "__main__":
    print(json.dumps(invented_cpu_example(), sort_keys=True))
