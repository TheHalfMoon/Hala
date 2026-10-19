"""Strict raw response parser and independently recomputed DEV metrics."""
from __future__ import annotations

import json
import math

from .geometry import bind_point


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key")
        out[key] = value
    return out


def parse_raw(raw: str, width: int, height: int):
    if type(raw) is not str or len(raw.encode("utf-8")) > 2048:
        raise ValueError("raw response invalid or oversized")
    try:
        obj = json.loads(raw, object_pairs_hook=_pairs,
                         parse_constant=lambda token: (_ for _ in ()).throw(ValueError("non-finite JSON")))
    except (RecursionError, OverflowError) as exc:
        raise ValueError("bounded JSON nesting/number invalid") from exc
    if type(obj) is not dict or set(obj) - {"point", "abstain"} or len(obj) != 1:
        raise ValueError("expected exactly point or abstain")
    if "abstain" in obj:
        if obj["abstain"] is not True:
            raise ValueError("abstain must be true")
        return None
    point = obj["point"]
    if type(point) is not list or len(point) != 2:
        raise ValueError("point must contain two normalized coordinates")
    # Check the range before converting an arbitrary JSON integer to a float.
    if any(type(v) not in (int, float) or v < 0 or v > 1 or not math.isfinite(v) for v in point):
        raise ValueError("point coordinates must be finite normalized numbers")
    return (point[0] * width, point[1] * height)


def score_attempt(case: dict, raw: str | None, error: str | None = None) -> dict:
    if error is not None:
        return {"parse_valid": False, "target_hit": False, "wrong_target": False,
                "abstain": False, "error": True}
    try:
        point = parse_raw(raw, case["width"], case["height"])
    except (ValueError, TypeError, json.JSONDecodeError, UnicodeError):
        return {"parse_valid": False, "target_hit": False, "wrong_target": False,
                "abstain": False, "error": False}
    if point is None:
        return {"parse_valid": True, "target_hit": False, "wrong_target": False,
                "abstain": True, "error": False}
    bound = bind_point(point, case["controls"], case["controls"], case["width"], case["height"])
    if bound is None:
        return {"parse_valid": True, "target_hit": False, "wrong_target": False,
                "abstain": False, "error": False, "gate_valid_unique": False}
    correct = bound["label"] == case["expected_label"]
    return {"parse_valid": True, "target_hit": correct, "wrong_target": not correct,
            "abstain": False, "error": False, "gate_valid_unique": True}


def summarize(cases: list[dict], outcomes: list[dict]) -> dict:
    if len(cases) != len(outcomes):
        raise ValueError("one outcome required per frozen case")
    result = {"denominator": len(cases), "languages": {}}
    for lang in ("en", "ar"):
        rows = [(c, o) for c, o in zip(cases, outcomes) if c["language"] == lang]
        metrics = {key: sum(bool(o.get(key)) for _, o in rows) for key in
                   ("parse_valid", "target_hit", "wrong_target", "abstain", "error", "gate_valid_unique")}
        metrics["denominator"] = len(rows)
        metrics["observed_control_count"] = sum(len(case["controls"]) for case, _ in rows)
        result["languages"][lang] = metrics
    return result
