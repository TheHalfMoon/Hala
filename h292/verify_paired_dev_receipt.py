"""H292: independent source-only closed receipt for real private 3-GUI LoRA.

An accepted JSON alone cannot prove a real Kaggle run: first check
owner-private terminal status, immutable submitted version, then download
ONLY the exact allowed small filename. Never publish raw private JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from h280.verify_private_native_aggregate import (
    EvidenceRejected, decode_bounded, exact_keys, need, flag, count,
)
from h288.verify_private_lora_receipt import closed_report as verify_queue
from h292.private_paired_routeboard_dev import (
    COMPOUND_REPORT, COMPOUND_STATUS, DEV_SOURCE_SHA,
    DEV_SOURCE_BLOB, DEV_INDICES,
)

TOP = frozenset({
    "status", "original_queueboard_research",
    "original_dev_source_commit", "original_dev_html_blob",
    "original_dev_exposed_not_blind_ood",
    "dev_scenes_preregistered",
    "dev_model_frames_distinct_and_train_queue_disjoint",
    "paired_dev_screenshots_identical",
    "dev_base", "dev_adapted",
    "dev_base_successes", "dev_adapted_successes",
    "dev_paired_delta", "official_train_dev_ood_admitted",
    "generalized_hala_skill_proven",
    "raw_predictions_images_points_gold_or_weights_saved",
})
ROW = frozenset({
    "scene", "language", "model_generation", "grammar_call",
    "strict", "kind", "native_click", "task_success",
})
DEV_SCENES = ("dev_00", "dev_01", "dev_04", "dev_06", "dev_07", "dev_10")
DEV_LANGUAGES = ("en", "en", "en", "ar", "ar", "ar")


def verify_object(obj: object) -> dict:
    data = exact_keys(obj, TOP, "H292 unexpected/missing private top-level fields")
    need(type(data["status"]) is str and
         data["status"] == COMPOUND_STATUS,
         "H292 private original paired research status mismatch")
    for key, expected in (
        ("original_dev_source_commit", DEV_SOURCE_SHA),
        ("original_dev_html_blob", DEV_SOURCE_BLOB),
    ):
        need(type(data[key]) is str and data[key] == expected,
             "H292 source-owned DEV revision/blob mismatch")
    count(data["dev_scenes_preregistered"], len(DEV_INDICES),
          "H292 preregistered source-DEV case count mismatch")
    for field in (
        "original_dev_exposed_not_blind_ood",
        "dev_model_frames_distinct_and_train_queue_disjoint",
        "paired_dev_screenshots_identical",
    ):
        flag(data[field], True, "H292 paired source-DEV identity proof missing")
    for field in (
        "official_train_dev_ood_admitted",
        "generalized_hala_skill_proven",
        "raw_predictions_images_points_gold_or_weights_saved",
    ):
        flag(data[field], False, "H292 false readiness/private output claim")
    # This verifier is independently frozen in H288 and refuses fabricated
    # original TRAIN, 8 changed LoRA optimizer steps and original Queueboard.
    original = verify_queue(data["original_queueboard_research"])
    scores = {}
    for phase, field in (("base", "dev_base"), ("adapted", "dev_adapted")):
        rows = data[field]
        need(type(rows) is list and len(rows) == len(DEV_SCENES),
             "H292 exactly six real original RouteBoard DEV source cases")
        strict = clicks = hits = 0
        for i, item in enumerate(rows):
            row = exact_keys(item, ROW, "H292 unknown raw/sensitive DEV row")
            need(type(row["scene"]) is str and
                 row["scene"] == DEV_SCENES[i] and
                 type(row["language"]) is str and
                 row["language"] == DEV_LANGUAGES[i],
                 "H292 original DEV scene ordering/language drift")
            count(row["model_generation"], 1,
                  "H292 must record a true image-conditioned generation")
            count(row["grammar_call"], 1,
                  "H292 must attest model-time strict generation grammar")
            for field_name in ("strict", "native_click", "task_success"):
                need(type(row[field_name]) is bool,
                     "H292 invalid nonboolean native source outcome")
            kind = row["kind"]
            need(kind is None or type(kind) is str
                 and kind in ("click", "abstain"),
                 "H292 only one real click or abstain")
            need((row["strict"] or
                  (kind is None and not row["native_click"]
                   and not row["task_success"]))
                 and (not row["strict"] or kind in ("click", "abstain"))
                 and (not row["native_click"] or
                      (row["strict"] and kind == "click"))
                 and (not row["task_success"] or
                      (row["native_click"] and kind == "click")),
                 "H292 impossible/fabricated native task success")
            strict += int(row["strict"])
            clicks += int(row["native_click"])
            hits += int(row["task_success"])
        total_field = "dev_base_successes" if phase == "base" else "dev_adapted_successes"
        count(data[total_field], hits, "H292 source-DEV task total mismatch")
        scores[phase] = {
            "grammar_valid": strict, "native_clicks": clicks,
            "task_successes": hits,
        }
    delta = scores["adapted"]["task_successes"] - scores["base"]["task_successes"]
    count(data["dev_paired_delta"], delta,
          "H292 real RouteBoard DEV paired delta mismatch")
    return {
        "status": "H292_REAL_MODEL_RESEARCH_DEV_NOT_OOD",
        "verified_research_optimizer_steps":
            original["real_adapter_optimizer_steps"],
        "source_train_candidate_count":
            original["validated_original_train_candidates"],
        "original_queueboard": original["paired"],
        "queueboard_paired_delta": original["paired_delta"],
        "original_routeboard": scores,
        "routeboard_paired_delta": delta,
        "routeboard_original_cases": len(DEV_SCENES),
        "original_dev_source_public_exposed": True,
        "official_train_dev_admitted": False,
        "blind_ood_proven": False,
        "generalizable_gain_proven": False,
    }


def verify_bytes(raw: bytes) -> dict:
    return verify_object(decode_bounded(raw))


def verify_private_file(path: Path) -> dict:
    need(isinstance(path, Path) and path.name == COMPOUND_REPORT
         and path.is_file() and not path.is_symlink(),
         "H292 only exact whitelisted private research basename")
    try:
        n = path.stat().st_size
        need(250 <= n <= 8500,
             "H292 privacy-limited private aggregate size not allowed")
        return verify_bytes(path.read_bytes())
    except OSError:
        raise EvidenceRejected("H292 owner-private research receipt inaccessible") from None


def main() -> None:
    parser = argparse.ArgumentParser(description="H292 safe owner-private research receipt only")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        safe = verify_private_file(args.report)
    except EvidenceRejected as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps(safe, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()
