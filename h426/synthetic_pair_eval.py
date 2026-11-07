"""H426 sealed-format metrics on self-authored synthetic GUI DEV predictions.

This is a read-only scorer for externally supplied predictions; it does NOT
run Qwen, read owner-private original data, TRAIN, or qualify independent OOD.
It reuses the exact frozen H425 decoder/scoring tolerance and original target.
"""
from __future__ import annotations

import json

from h426.h425_loss_bridge import H425_SOURCE, _load_source_module
from h426.synthetic_drag_curriculum import DOMAINS, generate

MAX_RAW_PREDICTION_CHARS = 8192


def score_dev_pair(index: int, domain: str, predictions: list[dict]):
    """Score exactly one synthetic DEV screenshot's two distinct EN/AR tasks.

    Caller supplies model responses; never return raw text or original labels.
    Strict id/sha pairing prevents scoring a prediction on the wrong image.
    """
    if type(predictions) is not list or len(predictions) != 2:
        raise ValueError("H426 synthetic dev requires exactly two paired predictions")
    if domain not in DOMAINS or type(domain) is not str:
        raise ValueError("H426 synthetic dev domain unregistered")
    source = generate("dev", index, domain)
    candidates = {row["id"]: row for row in source["actions"]}
    if len(candidates) != 2:
        raise ValueError("H426 synthetic generated pair identifiers ambiguous")
    h425 = _load_source_module("h426_synthetic_dev_original_h425", H425_SOURCE)
    expected_keys = {"id", "image_sha256", "raw_prediction"}
    reported = {}
    for p in predictions:
        if (type(p) is not dict or set(p) != expected_keys
            or type(p.get("id")) is not str or p["id"] not in candidates
            or p["id"] in reported
            or p.get("image_sha256") != source["image_sha256"]
            or type(p.get("raw_prediction")) is not str
            or len(p["raw_prediction"]) > MAX_RAW_PREDICTION_CHARS):
            raise ValueError("H426 synthetic DEV prediction identity or shape drift")
        reference = candidates[p["id"]]
        gold = reference["target"]
        scored = h425.score_predicted(p["raw_prediction"], gold)
        decoded = h425.parse_drag(p["raw_prediction"])
        if bool(scored["valid"]) != (decoded is not None):
            raise ValueError("H426 synthetic DEV H425 parser/scorer mismatch")
        if decoded is None:
            start_ok, end_ok = False, False
        else:
            tol = h425.TOL_NORM
            def near(v, goal):
                return all(abs(v[j] - goal[j]) <= tol for j in (0, 1))
            start_ok = near(decoded["start_point"], gold["start_point"])
            end_ok = near(decoded["end_point"], gold["end_point"])
        # Original generated actions are always ordered; strict original
        # H425 scorer is the final authority for aggregate correctness.
        if gold["ordered"] is not True:
            raise ValueError("H426 invented source gold ordered flag changed")
        if decoded is not None and decoded["ordered"] is True:
            if bool(scored["correct"]) != (start_ok and end_ok):
                raise ValueError("H426 H425 strict endpoint correctness disagreement")
        reported[p["id"]] = {
            "valid": bool(scored["valid"]),
            "correct": bool(scored["correct"]),
            "start_near": start_ok,
            "end_near": end_ok,
            "error": scored["error"],
            "language": reference["language"],
        }
    if set(reported) != set(candidates):
        raise ValueError("H426 synthetic DEV must score both paired instructions")
    result = {a["language"]: reported[a["id"]] for a in source["actions"]}
    return {
        "schema": "hala.h426.synthetic-dev-paired-errors.v1",
        "split": "dev", "domain": domain,
        "source_family": source["family"],
        "source_image_sha256": source["image_sha256"],
        "actions": 2,
        "correct_actions": sum(int(v["correct"]) for v in result.values()),
        "both_actions_correct": all(v["correct"] for v in result.values()),
        "per_language": result,
        "author_generated_source_only": True,
        "new_model_inference_executed": False,
        "independent_external_ood": False,
        "model_release_qualified": False,
    }


def compare_synthetic_dev(index, domain, baseline, candidate):
    """Paired within-image comparison, not independent real-world OOD."""
    old = score_dev_pair(index, domain, baseline)
    new = score_dev_pair(index, domain, candidate)
    if (old["source_image_sha256"] != new["source_image_sha256"]
        or old["source_family"] != new["source_family"]):
        raise ValueError("H426 candidate and H425 baseline evaluated different sources")
    details = {}
    for language in ("en", "ar"):
        earlier = old["per_language"][language]
        later = new["per_language"][language]
        details[language] = {
            "old_correct": earlier["correct"],
            "new_correct": later["correct"],
            "new_start_near": later["start_near"],
            "new_end_near": later["end_near"],
            "end_only_error": later["start_near"] and not later["end_near"],
            "start_only_error": later["end_near"] and not later["start_near"],
            "both_points_error": (not later["start_near"] and not later["end_near"]),
        }
    return {
        "schema": "hala.h426.synthetic-dev-model-comparison.v1",
        "domain": domain,
        "source_family": old["source_family"],
        "source_image_sha256": old["source_image_sha256"],
        "baseline_correct": old["correct_actions"],
        "candidate_correct": new["correct_actions"],
        "baseline_both_correct": old["both_actions_correct"],
        "candidate_both_correct": new["both_actions_correct"],
        "per_language": details,
        "synthetic_predictions_supplied_by_caller": True,
        "trained_or_queried_model_here": False,
        "independent_external_ood": False,
        "model_release_qualified": False,
    }
