"""H283 private H282 ClinicGrid Qwen receipt verifier (source-only).

This is strict *parsing*, not proof that a private GPU actually ran.
Never accept unrecognized raw text, coordinates, images, oracle labels,
model weights, or unqualified performance/generalization claims.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from h280.verify_private_native_aggregate import (
    EvidenceRejected, MODEL_ID, MODEL_REV, decode_bounded,
    exact_keys, hardware_valid, need, flag, count,
)

ORIGINAL_SOURCE_SHA = "ab6d4c77ee89a6abfb4375edbba4f1a2108db445"
ORIGINAL_SOURCE_BLOB = "dacdfaf5c4b9fe9ed22624109c917f7e72bf4edb"
REPORT = "hala-h282-private-clinicgrid-native-aggregate.json"
STATUS = "H282_REAL_PRETRAINED_CLINICGRID_FIRSTPARTY_NOT_OOD"
SCENES = ("en_urgent_c214", "ar_all_c308")

REPORT_KEYS = frozenset((
    "status", "source_commit", "source_html_git_blob", "model_id",
    "model_revision", "hardware", "real_pretrained_image_generations",
    "model_time_grammar_generate_calls", "rows",
    "firstparty_task_successes", "optimizer_updates", "independent_blind_ood",
    "official_train_dev_ood_admitted", "adapted_model_improvement_proven",
    "raw_model_text_points_screenshots_or_weights_published",
))
ROW_KEYS = frozenset((
    "scene", "real_generations", "grammar_generate_calls",
    "screenshot_integrity_attested", "strict_normalized_json",
    "native_dispatched", "firstparty_task_success", "native_kind",
    "all_source_visits_checked",
))


def verify_report(document: object) -> dict:
    d = exact_keys(document, REPORT_KEYS, "H282 report fields")
    need(type(d["status"]) is str and d["status"] == STATUS,
         "H282 report version drift")
    need(type(d["source_commit"]) is str
         and d["source_commit"] == ORIGINAL_SOURCE_SHA
         and type(d["source_html_git_blob"]) is str
         and d["source_html_git_blob"] == ORIGINAL_SOURCE_BLOB,
         "H282 original source provenance mismatch")
    need(type(d["model_id"]) is str and d["model_id"] == MODEL_ID
         and type(d["model_revision"]) is str
         and d["model_revision"] == MODEL_REV,
         "H282 pretrained model provenance mismatch")
    hardware_valid(d["hardware"])
    count(d["real_pretrained_image_generations"], 2,
          "H282 exactly two real image-conditioned generations required")
    count(d["model_time_grammar_generate_calls"], 2,
          "H282 two model-time grammar callbacks required")
    count(d["optimizer_updates"], 0, "H282 must not train a model")
    flag(d["independent_blind_ood"], False,
         "H282 previously exposed first-party research is not blind OOD")
    flag(d["official_train_dev_ood_admitted"], False,
         "H282 has no approved TRAIN/DEV manifest")
    flag(d["adapted_model_improvement_proven"], False,
         "H282 pretrained-only research is not adaptation gain")
    flag(d["raw_model_text_points_screenshots_or_weights_published"], False,
         "H282 raw model/private media publication prohibited")
    rows = d["rows"]
    need(type(rows) is list and len(rows) == 2,
         "H282 exactly two original scenes required")

    scores = {"valid": 0, "dispatched": 0, "task_successes": 0,
              "state_oracle_attested": 0}
    safe_rows = []
    for index, original in enumerate(rows):
        row = exact_keys(original, ROW_KEYS, "H282 row contains raw/unknown fields")
        need(type(row["scene"]) is str and row["scene"] == SCENES[index],
             "H282 original scene registration changed")
        count(row["real_generations"], 1, "H282 original scene real generation")
        count(row["grammar_generate_calls"], 1, "H282 true LMFE model call")
        for field in ("strict_normalized_json", "native_dispatched",
                      "firstparty_task_success", "screenshot_integrity_attested"):
            need(type(row[field]) is bool, "H282 integer-as-boolean forbidden")
        valid = row["strict_normalized_json"]
        kind = row["native_kind"]
        need(kind is None or type(kind) is str and kind in ("click", "abstain"),
             "H282 unsupported model action kind")
        need(row["screenshot_integrity_attested"] is valid,
             "H282 screenshot lock not independently attested")
        if valid:
            count(row["all_source_visits_checked"], 6,
                  "H282 full six-visit postaction state unverified")
            need(kind in ("click", "abstain"),
                 "H282 strict valid output missing native kind")
            scores["state_oracle_attested"] += 1
        else:
            need(row["all_source_visits_checked"] is None and kind is None
                 and row["native_dispatched"] is False
                 and row["firstparty_task_success"] is False,
                 "H282 rejected output cannot claim action/oracle")
        need(not row["native_dispatched"]
             or (valid and kind == "click"),
             "H282 no real click from malformed/abstain text")
        need(not row["firstparty_task_success"]
             or (row["native_dispatched"] and kind == "click"),
             "H282 no task success without real native click")
        scores["valid"] += int(valid)
        scores["dispatched"] += int(row["native_dispatched"])
        scores["task_successes"] += int(row["firstparty_task_success"])
        safe_rows.append({
            "scene": SCENES[index],
            "strict": valid,
            "screenshot_locked": row["screenshot_integrity_attested"],
            "full_source_oracle_checked": row["all_source_visits_checked"] == 6,
            "kind": kind,
            "native_click": row["native_dispatched"],
            "task_success": row["firstparty_task_success"],
        })
    count(d["firstparty_task_successes"], scores["task_successes"],
          "H282 real source task success total mismatch")

    return {
        "family": "H282_CLINICGRID_PRETRAINED_PUBLIC_RESEARCH_NOT_OOD",
        "real_generations": 2,
        "model_time_grammar_calls": 2,
        "counts": scores,
        "rows": safe_rows,
        "trained_adapter_gain": False,
        "independent_blind_ood": False,
        "approved_training_data": False,
    }


def verify_bytes(raw: bytes) -> dict:
    return verify_report(decode_bounded(raw))


def verify_local_file(path: Path) -> dict:
    need(isinstance(path, Path) and path.name == REPORT
         and path.is_file() and not path.is_symlink(),
         "H282 invalid private report path")
    try:
        size = path.stat().st_size
        need(100 <= size <= 4096, "H282 private report exceeds safe bounds")
        content = path.read_bytes()
    except OSError:
        raise EvidenceRejected("H283 private report inaccessible") from None
    return verify_bytes(content)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read one owner-private sanitized ClinicGrid Qwen receipt")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        out = verify_local_file(args.report)
    except EvidenceRejected as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps(out, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()
