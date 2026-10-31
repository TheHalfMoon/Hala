"""H288 — only closed, categorical evidence from PRIVATE H287 T4 LoRA pilot.

This parser cannot prove that an arbitrary JSON was produced by a real
Kaggle run. Independently authenticate owner-private kernel status,
submitted source version and exact filename BEFORE running it.

Never print unknown values or raw model text, coordinates, screenshot,
gold, weights or gradients. All errors are hard-coded labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from h280.verify_private_native_aggregate import (
    EvidenceRejected, decode_bounded, exact_keys, hardware_valid,
    need, flag, count,
)

REPORT_NAME = "hala-h287-private-shiftdesk-crossfamily-lora-aggregate.json"
STATUS = "H287_REAL_SHIFT_TRAIN_QUEUE_RESEARCH_NOT_ADMITTED_NOT_OOD"
SOURCE_COMMIT = "76a4ebb204d825b280fd4d332ad75591c5f13007"
TRAIN_BLOB = "f0ac89b2be6b6cc5200ce4a220dbf0c2e00aeaac"
EVAL_BLOB = "3cf452e84a75b017905bc14b04a8571411929891"
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
ORIGINAL_SCENES = ("en_priority_q118", "ar_all_q142")
TOP = frozenset({
    "status", "source_commit", "train_original_blob", "eval_original_blob",
    "model_id", "model_revision", "hardware",
    "original_train_candidates_verified", "research_train_optimizer_steps",
    "changed_lora_tensors_each_step", "coordinate_supervision_each_step",
    "paired_original_eval_scenes", "paired_pretrained", "paired_adapted",
    "paired_pretrained_successes", "paired_adapted_successes", "paired_delta",
    "same_exact_eval_screens_before_after",
    "train_eval_exact_png_disjoint",
    "research_source_exposed_not_blind_ood",
    "official_train_dev_ood_admitted",
    "generalizable_hala_improvement_proven",
    "adapter_weights_published",
    "raw_predictions_images_points_or_gold_published",
})
ROW = frozenset({
    "scene", "model_generation", "grammar_call", "strict", "kind",
    "native_click", "task_success",
})


def closed_report(data: object) -> dict:
    report = exact_keys(data, TOP, "H287 receipt unknown/missing top-level fields")
    need(type(report["status"]) is str and report["status"] == STATUS,
         "H287 closed report status mismatch")
    for field, expected in (
        ("source_commit", SOURCE_COMMIT),
        ("train_original_blob", TRAIN_BLOB),
        ("eval_original_blob", EVAL_BLOB),
        ("model_id", MODEL_ID),
        ("model_revision", MODEL_REVISION),
    ):
        need(type(report[field]) is str and report[field] == expected,
             "H287 pinned provenance mismatch")
    hardware_valid(report["hardware"])
    count(report["original_train_candidates_verified"], 24,
          "H287 must source-verify 24 original browser training scenes")
    count(report["research_train_optimizer_steps"], 8,
          "H287 requires eight genuine adapter updates")
    count(report["paired_original_eval_scenes"], 2,
          "H287 exactly two paired research GUI scenes")
    for key in (
        "changed_lora_tensors_each_step", "coordinate_supervision_each_step",
        "same_exact_eval_screens_before_after", "train_eval_exact_png_disjoint",
        "research_source_exposed_not_blind_ood",
    ):
        flag(report[key], True, "H287 positive source/optimizer evidence missing")
    for key in (
        "official_train_dev_ood_admitted",
        "generalizable_hala_improvement_proven",
        "adapter_weights_published",
        "raw_predictions_images_points_or_gold_published",
    ):
        flag(report[key], False, "H287 unauthorized admission/gain/output claim")

    accepted = {}
    summaries = {}
    for name, field in (("base", "paired_pretrained"),
                        ("adapted", "paired_adapted")):
        rows = report[field]
        need(type(rows) is list and len(rows) == 2,
             "H287 paired case count not exactly two")
        safe = []
        successes = 0
        strict_count = 0
        clicks = 0
        for i, item in enumerate(rows):
            row = exact_keys(item, ROW, "H287 private raw/missing row fields")
            need(type(row["scene"]) is str
                 and row["scene"] == ORIGINAL_SCENES[i],
                 "H287 source research scene drift")
            count(row["model_generation"], 1,
                  "H287 one real image-conditioned model generation per scene")
            count(row["grammar_call"], 1,
                  "H287 one model-time grammar call per scene")
            for flag_name in ("strict", "native_click", "task_success"):
                need(type(row[flag_name]) is bool,
                     "H287 integer/nonbool model outcome")
            kind = row["kind"]
            need(kind is None or type(kind) is str
                 and kind in ("click", "abstain"),
                 "H287 only exactly one click or abstain")
            need((row["strict"] or
                  (kind is None and not row["native_click"]
                   and not row["task_success"]))
                 and (not row["strict"] or kind in ("click", "abstain"))
                 and (not row["native_click"] or
                      (row["strict"] and kind == "click"))
                 and (not row["task_success"] or
                      (row["native_click"] and kind == "click")),
                 "H287 fabricated model format/native action/task success")
            strict_count += int(row["strict"])
            clicks += int(row["native_click"])
            successes += int(row["task_success"])
            safe.append({
                "scene": ORIGINAL_SCENES[i],
                "strict": row["strict"], "kind": kind,
                "native_click": row["native_click"],
                "task_success": row["task_success"],
            })
        count(report["paired_pretrained_successes"
                     if name == "base" else
                     "paired_adapted_successes"], successes,
              "H287 individual source-state total mismatch")
        accepted[name] = safe
        summaries[name] = {
            "strict": strict_count, "native_clicks": clicks,
            "task_successes": successes,
        }

    delta = summaries["adapted"]["task_successes"] - summaries["base"]["task_successes"]
    count(report["paired_delta"], delta,
          "H287 paired delta does not match true counted source states")
    return {
        "family": "H287_ORIGINAL_SHIFT_TRAIN_QUEUE_EVAL_RESEARCH_NOT_OOD",
        "real_adapter_optimizer_steps": 8,
        "validated_original_train_candidates": 24,
        "paired_source_cases": 2,
        "paired": summaries,
        "paired_delta": delta,
        "rows": accepted,
        "official_training_admitted": False,
        "blind_ood_established": False,
        "generalizable_model_gain_established": False,
    }


def verify_bytes(raw: bytes) -> dict:
    return closed_report(decode_bounded(raw))


def verify_private_file(path: Path) -> dict:
    need(isinstance(path, Path) and path.name == REPORT_NAME
         and path.is_file() and not path.is_symlink(),
         "H288 exact original tiny private report basename required")
    try:
        size = path.stat().st_size
        need(100 <= size <= 4096, "H288 owner-private report byte bound exceeded")
        raw = path.read_bytes()
    except OSError:
        raise EvidenceRejected("H288 private receipt inaccessible") from None
    return verify_bytes(raw)


def main() -> None:
    cli = argparse.ArgumentParser(description="H287 closed privacy-safe T4 receipt")
    cli.add_argument("--report", required=True, type=Path)
    args = cli.parse_args()
    try:
        summary = verify_private_file(args.report)
    except EvidenceRejected as exc:
        cli.exit(2, str(exc) + "\n")
    print(json.dumps(summary, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()
