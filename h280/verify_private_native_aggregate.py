"""H280: fail-closed tiny owner-private H279 GPU aggregate verifier.

This is an *evidence-reading tool*, not a model, a scored blind benchmark,
a private prediction viewer, a dataset admission tool, or a success claim.
It refuses unknown fields, duplicate JSON keys, non-JSON constants and
unattested/provenance-drifted summaries; prints only hard-coded safe
counts and categorical source-owned scene labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MAX_BYTES = 4096
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_REV = "89644892e4d85e24eaac8bacfd4f463576704203"
STOCKROOM_SHA = "53d7fb25e113f8f2cd337bb4aebeafac026d1a22"
QUEUEBOARD_SHA = "c61409e0b0178227180f0d8e05c67f5a210d01f9"
QUEUEBOARD_BLOB = "3cf452e84a75b017905bc14b04a8571411929891"

STOCKROOM_NAME = "hala-h279-private-relative-native-aggregate.json"
QUEUEBOARD_NAME = "hala-h279-private-queueboard-native-aggregate.json"
STOCKROOM_STATUS = "H279_PRIVATE_PRETRAINED_NATIVE_RELATIVE_FIRSTPARTY_NOT_OOD"
QUEUEBOARD_STATUS = "H279_REAL_PRETRAINED_QUEUEBOARD_FIRSTPARTY_NOT_OOD"
STOCKROOM_SCENES = ("ar_sorted_tape", "ar_sorted_gloves")
QUEUEBOARD_SCENES = ("en_priority_q118", "ar_all_q142")
STOCKROOM_TOP = frozenset({
    "status", "pinned_original_source_sha", "pinned_model_id",
    "pinned_model_revision", "hardware", "real_pretrained_image_generations",
    "rows", "strict_relative_actions", "native_actions_dispatched",
    "first_plus_target_hits", "actual_firstparty_task_success",
    "real_optimizer_updates", "no_raw_predictions_points_or_images_saved",
    "weights_or_adapters_published", "independent_blind_ood",
    "official_train_dev_ood_admitted", "generalizable_hala_skill_proven",
})
STOCKROOM_ROW = frozenset({
    "scene", "real_model_generation_count",
    "real_token_filter_generate_calls", "original_screenshot_unchanged",
    "format_class", "strict_relative_action_valid", "native_action_kind",
    "native_action_dispatched", "first_plus_target_hit",
    "firstparty_inventory_task_success", "other_rows_unchanged",
})
QUEUEBOARD_TOP = frozenset({
    "status", "source_commit", "source_html_git_blob", "model_id",
    "model_revision", "hardware", "real_pretrained_image_generations",
    "model_time_grammar_generate_calls", "rows",
    "firstparty_task_successes", "optimizer_updates", "independent_blind_ood",
    "official_train_dev_ood_admitted", "model_improvement_proven",
    "raw_model_text_points_screenshots_or_weights_published",
})
QUEUEBOARD_ROW = frozenset({
    "scene", "real_generations", "grammar_generate_calls",
    "strict_normalized_json", "native_dispatched",
    "firstparty_task_success", "native_kind",
})
FORMAT_CLASSES = frozenset({
    "STRICT_EXACT_VALID", "NON_STRING", "EMPTY", "OVER_128_CHARS",
    "MARKDOWN_FENCE", "INCOMPLETE_OR_UNCLOSED_JSON",
    "NOT_JSON_OR_NONCANONICAL_PREFIX", "JSON_DIFFERENT_TOP_LEVEL_KEYS",
    "JSON_DIFFERENT_ACTION_COUNT", "JSON_CLICK_INVALID_RELATIVE_POINT",
    "JSON_VALID_CLICK_NONCANONICAL_SYNTAX",
    "JSON_VALID_ABSTAIN_NONCANONICAL_SYNTAX", "JSON_OTHER_ACTION",
})


class EvidenceRejected(ValueError):
    """Safe, constant error labels only: never echo untrusted data."""


def need(ok: bool, label: str) -> None:
    if not ok:
        raise EvidenceRejected("H280 " + label)


def exact_keys(data: object, allowed: frozenset[str], label: str) -> dict:
    need(type(data) is dict and data.keys() == allowed, label)
    return data


def flag(value: object, expected: bool, label: str) -> None:
    need(type(value) is bool and value is expected, label)


def count(value: object, expected: int, label: str) -> None:
    need(type(value) is int and value == expected, label)


def _unique(pairs: list[tuple[str, object]]) -> dict:
    seen = {}
    for key, val in pairs:
        if key in seen:
            raise EvidenceRejected("H280 duplicate JSON keys")
        seen[key] = val
    return seen


def _constant(_: str) -> None:
    raise EvidenceRejected("H280 nonstandard JSON constant")


def decode_bounded(raw: bytes) -> dict:
    need(type(raw) is bytes and 100 <= len(raw) <= MAX_BYTES,
         "report size outside strict bound")
    try:
        doc = json.loads(raw.decode("utf-8", errors="strict"),
                         object_pairs_hook=_unique, parse_constant=_constant)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise EvidenceRejected("H280 malformed JSON") from None
    need(type(doc) is dict, "top-level report not object")
    return doc


def hardware_valid(value: object) -> None:
    obj = exact_keys(value, frozenset({"free_t4", "devices"}), "GPU fields")
    flag(obj["free_t4"], True, "not verified free T4")
    need(type(obj["devices"]) is int and obj["devices"] in (1, 2),
         "invalid T4 count")


def _no_release_claims(doc: dict, *, stockroom: bool) -> None:
    flag(doc["independent_blind_ood"], False, "blind OOD claim forbidden")
    flag(doc["official_train_dev_ood_admitted"], False,
         "official data admission claim forbidden")
    if stockroom:
        flag(doc["generalizable_hala_skill_proven"], False,
             "generalized skill claim forbidden")
        flag(doc["weights_or_adapters_published"], False,
             "adapter publication claim forbidden")
        flag(doc["no_raw_predictions_points_or_images_saved"], True,
             "raw media retention claim forbidden")
        count(doc["real_optimizer_updates"], 0, "unexpected optimizer updates")
    else:
        flag(doc["model_improvement_proven"], False,
             "model improvement claim forbidden")
        flag(doc["raw_model_text_points_screenshots_or_weights_published"],
             False, "private prediction publication claim forbidden")
        count(doc["optimizer_updates"], 0, "unexpected optimizer updates")


def stockroom_summary(doc: dict) -> dict:
    obj = exact_keys(doc, STOCKROOM_TOP, "Stockroom top-level keys")
    need(obj["status"] == STOCKROOM_STATUS
         and obj["pinned_original_source_sha"] == STOCKROOM_SHA
         and obj["pinned_model_id"] == MODEL_ID
         and obj["pinned_model_revision"] == MODEL_REV,
         "Stockroom pinned provenance")
    hardware_valid(obj["hardware"])
    count(obj["real_pretrained_image_generations"], 2,
          "Stockroom real generations")
    _no_release_claims(obj, stockroom=True)
    rows = obj["rows"]
    need(type(rows) is list and len(rows) == 2, "Stockroom scene count")
    totals = dict(strict=0, dispatched=0, target_hits=0, task_successes=0)
    safe_rows = []
    for i, entry in enumerate(rows):
        row = exact_keys(entry, STOCKROOM_ROW, "Stockroom row keys")
        need(row["scene"] == STOCKROOM_SCENES[i], "Stockroom scene order")
        count(row["real_model_generation_count"], 1, "Stockroom true generation")
        count(row["real_token_filter_generate_calls"], 1,
              "Stockroom model-time grammar")
        flag(row["original_screenshot_unchanged"], True,
             "Stockroom screenshot changed")
        need(type(row["format_class"]) is str
             and row["format_class"] in FORMAT_CLASSES,
             "Stockroom format category")
        for field in ("strict_relative_action_valid", "native_action_dispatched",
                      "first_plus_target_hit", "firstparty_inventory_task_success",
                      "other_rows_unchanged"):
            need(type(row[field]) is bool, "Stockroom non-boolean state")
        need(row["strict_relative_action_valid"] ==
             (row["format_class"] == "STRICT_EXACT_VALID"),
             "Stockroom inconsistent grammar/validation")
        need(row["native_action_kind"] in (None, "click", "abstain")
             and (row["native_action_dispatched"] is False
                  or (row["strict_relative_action_valid"]
                      and row["native_action_kind"] == "click"))
             and (row["first_plus_target_hit"] is False
                  or (row["native_action_dispatched"]
                      and row["native_action_kind"] == "click"))
             and (row["firstparty_inventory_task_success"] is False
                  or (row["first_plus_target_hit"]
                      and row["other_rows_unchanged"])),
             "Stockroom impossible action/score")
        totals["strict"] += int(row["strict_relative_action_valid"])
        totals["dispatched"] += int(row["native_action_dispatched"])
        totals["target_hits"] += int(row["first_plus_target_hit"])
        totals["task_successes"] += int(row["firstparty_inventory_task_success"])
        safe_rows.append({
            "scene": STOCKROOM_SCENES[i],
            "category": row["format_class"],
            "kind": row["native_action_kind"],
            "native_click": row["native_action_dispatched"],
            "target_hit": row["first_plus_target_hit"],
            "task_success": row["firstparty_inventory_task_success"],
        })
    for key, reported in (
        ("strict", "strict_relative_actions"),
        ("dispatched", "native_actions_dispatched"),
        ("target_hits", "first_plus_target_hits"),
        ("task_successes", "actual_firstparty_task_success"),
    ):
        count(obj[reported], totals[key], "Stockroom inconsistent totals")
    return {
        "family": "H279_STOCKROOM_PRETRAINED_ORIGINAL_NOT_OOD",
        "real_generations": 2,
        "grammar_calls": 2,
        "counts": totals,
        "rows": safe_rows,
        "adapter_improvement_proven": False,
        "independent_blind_ood": False,
    }


def queueboard_summary(doc: dict) -> dict:
    obj = exact_keys(doc, QUEUEBOARD_TOP, "Queueboard top-level keys")
    need(obj["status"] == QUEUEBOARD_STATUS
         and obj["source_commit"] == QUEUEBOARD_SHA
         and obj["source_html_git_blob"] == QUEUEBOARD_BLOB
         and obj["model_id"] == MODEL_ID
         and obj["model_revision"] == MODEL_REV,
         "Queueboard pinned provenance")
    hardware_valid(obj["hardware"])
    count(obj["real_pretrained_image_generations"], 2,
          "Queueboard true generations")
    count(obj["model_time_grammar_generate_calls"], 2,
          "Queueboard grammar calls")
    _no_release_claims(obj, stockroom=False)
    rows = obj["rows"]
    need(type(rows) is list and len(rows) == 2, "Queueboard scene count")
    safe_rows = []
    native_count = success_count = strict_count = 0
    for i, entry in enumerate(rows):
        row = exact_keys(entry, QUEUEBOARD_ROW, "Queueboard row keys")
        need(row["scene"] == QUEUEBOARD_SCENES[i], "Queueboard scene order")
        count(row["real_generations"], 1, "Queueboard true generation")
        count(row["grammar_generate_calls"], 1, "Queueboard true grammar")
        for field in ("strict_normalized_json", "native_dispatched",
                      "firstparty_task_success"):
            need(type(row[field]) is bool, "Queueboard non-boolean outcome")
        need(row["native_kind"] in (None, "click", "abstain")
             and (row["strict_normalized_json"]
                  or (row["native_kind"] is None
                      and not row["native_dispatched"]
                      and not row["firstparty_task_success"]))
             and (not row["native_dispatched"]
                  or (row["strict_normalized_json"]
                      and row["native_kind"] == "click"))
             and (not row["firstparty_task_success"]
                  or (row["native_dispatched"]
                      and row["native_kind"] == "click")),
             "Queueboard impossible action/score")
        native_count += int(row["native_dispatched"])
        success_count += int(row["firstparty_task_success"])
        strict_count += int(row["strict_normalized_json"])
        safe_rows.append({
            "scene": QUEUEBOARD_SCENES[i],
            "strict": row["strict_normalized_json"],
            "kind": row["native_kind"],
            "native_click": row["native_dispatched"],
            "task_success": row["firstparty_task_success"],
        })
    count(obj["firstparty_task_successes"], success_count,
          "Queueboard total mismatch")
    return {
        "family": "H279_QUEUEBOARD_PRETRAINED_ORIGINAL_NOT_OOD",
        "real_generations": 2,
        "grammar_calls": 2,
        "counts": {"strict": strict_count, "dispatched": native_count,
                   "task_successes": success_count},
        "rows": safe_rows,
        "adapter_improvement_proven": False,
        "independent_blind_ood": False,
    }


def validate_bytes(raw: bytes, *, family: str) -> dict:
    doc = decode_bounded(raw)
    if family == "stockroom":
        return stockroom_summary(doc)
    if family == "queueboard":
        return queueboard_summary(doc)
    raise EvidenceRejected("H280 unregistered original family")


def validate_local_file(path: Path, *, family: str) -> dict:
    need(type(path) is Path and family in ("stockroom", "queueboard"),
         "invalid requested input")
    expected = STOCKROOM_NAME if family == "stockroom" else QUEUEBOARD_NAME
    need(path.name == expected and path.is_file() and not path.is_symlink(),
         "invalid private report path")
    try:
        size = path.stat().st_size
        need(100 <= size <= MAX_BYTES, "invalid file size")
        data = path.read_bytes()
    except OSError:
        raise EvidenceRejected("H280 report file inaccessible") from None
    return validate_bytes(data, family=family)


def main() -> None:
    parser = argparse.ArgumentParser(description=(
        "Validate owner-private H279 aggregate; NO source media or predictions"))
    parser.add_argument("--family", choices=("stockroom", "queueboard"),
                        required=True)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = validate_local_file(args.report, family=args.family)
    except EvidenceRejected as exc:
        # Exceptions contain only hard-coded identifiers; never user data.
        parser.exit(2, str(exc) + "\n")
    print(json.dumps(result, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()
