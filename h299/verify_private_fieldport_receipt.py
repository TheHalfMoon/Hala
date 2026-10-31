"""H299: independent close-schema PRIVATE original two-stage Qwen receipt auditor.

This parser authenticates only the claimed bounded schema/consistency,
NOT the running Kaggle identity by itself. Verify exact owner-private
kernel status and immutable submitted code BEFORE using an allowlisted
download. Only safe EN/AR category counts may be printed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from h280.verify_private_native_aggregate import (
    EvidenceRejected, decode_bounded, exact_keys,
    need, flag, count, hardware_valid,
)
from h299.private_pretrained_fieldport_pilot import (
    SOURCE_SHA,SOURCE_BLOB,MODEL_ID,MODEL_REV,STATUS,DIAG_STATUS,
    REPORT,DIAGNOSTIC_REPORT,STAGES,
)

TOP = frozenset({
    "status","source_commit","original_html_git_blob","model_id",
    "model_revision","hardware","source_research_scenes",
    "real_model_visual_generations","model_time_grammar_generate_calls",
    "rows","original_firstparty_task_successes",
    "pretrained_no_optimizer_steps","source_public_previously_exposed",
    "official_train_dev_ood_admitted","independent_blind_ood",
    "trained_adapter_gain_proven",
    "raw_model_output_media_points_gold_or_weights_saved",
})
ROW = frozenset({
    "language","generations","grammar_calls","screenshot_steps",
    "focus_clicks","type_calls","abstained","task_success",
    "all_four_source_fields_checked",
})
DIAG = frozenset({
    "status","failure_stage","exception_class","source_commit",
    "model_id","model_revision","verified_model_skill_scores",
    "verified_optimizer_steps","official_train_dev_admitted",
    "blind_ood_proven",
    "raw_exception_text_model_outputs_media_points_saved",
})
ERROR_CLASSES = frozenset({
    "ValueError","RuntimeError","OSError","ImportError",
    "ModuleNotFoundError","CalledProcessError",
    "FileNotFoundError","OutOfMemoryError","OTHER_ERROR_CLASS",
})


def verify_success(data: object) -> dict:
    doc=exact_keys(data,TOP,"H299 unknown PRIVATE/extra aggregate field")
    need(type(doc["status"]) is str and doc["status"]==STATUS,
         "wrong original pretrained research schema")
    for key,pinned in (
        ("source_commit",SOURCE_SHA),
        ("original_html_git_blob",SOURCE_BLOB),
        ("model_id",MODEL_ID),
        ("model_revision",MODEL_REV),
    ):
        need(type(doc[key]) is str and doc[key]==pinned,
             "H299 frozen original source/model pin drift")
    hardware_valid(doc["hardware"])
    count(doc["source_research_scenes"],2,
          "H299 exactly EN+AR original tasks required")
    count(doc["pretrained_no_optimizer_steps"],0,
          "pretrained cannot falsely claim training")
    for field,expected in (
        ("source_public_previously_exposed",True),
        ("official_train_dev_ood_admitted",False),
        ("independent_blind_ood",False),
        ("trained_adapter_gain_proven",False),
        ("raw_model_output_media_points_gold_or_weights_saved",False),
    ):
        flag(doc[field],expected,
             "H299 false source/blind/skill/private media assertion")
    rows=doc["rows"]
    need(type(rows) is list and len(rows)==2,
         "exact EN+AR original source-owned result rows")
    total_gen=hits=0
    categorical=[]
    for i,item in enumerate(rows):
        row=exact_keys(item,ROW,
                       "H299 source model row contains private text/point")
        need(type(row["language"]) is str
             and row["language"]==("en" if i==0 else "ar"),
             "H299 frozen EN+AR order changed")
        n=row["generations"]
        need(type(n) is int and n in (1,2),
             "H299 actual first/second visual generation missing")
        count(row["grammar_calls"],n,
              "H299 actual generated token-filter call count mismatch")
        count(row["screenshot_steps"],n,
              "H299 fresh screenshot per original model turn needed")
        need(type(row["focus_clicks"]) is int
             and row["focus_clicks"] in (0,1)
             and type(row["type_calls"]) is int
             and row["type_calls"] in (0,1),
             "H299 native focus/type counters invalid")
        for name in ("abstained","task_success",
                     "all_four_source_fields_checked"):
            need(type(row[name]) is bool,
                 "H299 strict firstparty source-state booleans")
        flag(row["all_four_source_fields_checked"],True,
             "H299 source-owned 4-field oracle absent")
        need((n==1)==(row["focus_clicks"]==0)
             and row["type_calls"]<=row["focus_clicks"]
             and (not row["task_success"]
                  or (n==2 and row["focus_clicks"]==1
                      and row["type_calls"]==1 and not row["abstained"]))
             and (not row["abstained"] or not row["task_success"]),
             "H299 fabricated source success or inconsistent native steps")
        total_gen+=n
        hits+=int(row["task_success"])
        categorical.append({
            "language":row["language"],
            "real_model_visual_generations":n,
            "native_focus_click_dispatched":bool(row["focus_clicks"]),
            "native_type_dispatched":bool(row["type_calls"]),
            "abstained":row["abstained"],
            "firstparty_task_success":row["task_success"],
        })
    count(doc["real_model_visual_generations"],total_gen,
          "H299 total real image generations drift")
    count(doc["model_time_grammar_generate_calls"],total_gen,
          "H299 total prospective LMFE grammar generation drift")
    count(doc["original_firstparty_task_successes"],hits,
          "H299 original source task state success total drift")
    return {
        "status":"H299_STRICT_SOURCE_MODEL_RESEARCH_RECEIPT_ACCEPTED",
        "original_research_cases":2,
        "model_time_pretrained_generations":total_gen,
        "true_model_time_grammar_calls":total_gen,
        "genuine_firstparty_task_successes":hits,
        "per_language":categorical,
        "owner_private_authentication_external_required":True,
        "public_source_exposed":True,
        "official_train_dev_admitted":False,
        "blind_ood_proven":False,
        "adapted_model_gain_proven":False,
    }


def verify_failure(data: object) -> dict:
    doc=exact_keys(data,DIAG,
                   "H299 unexpected private error/gold fields")
    need(type(doc["status"]) is str and doc["status"]==DIAG_STATUS,
         "H299 not original bounded failure stage receipt")
    need(type(doc["failure_stage"]) is str
         and doc["failure_stage"] in STAGES,
         "H299 invalid categorical failure stage")
    need(type(doc["exception_class"]) is str
         and doc["exception_class"] in ERROR_CLASSES,
         "H299 failure cannot include raw exception message")
    for field,pinned in (
        ("source_commit",SOURCE_SHA),
        ("model_id",MODEL_ID),
        ("model_revision",MODEL_REV),
    ):
        need(type(doc[field]) is str and doc[field]==pinned,
             "H299 wrong failure-source/model identity")
    count(doc["verified_optimizer_steps"],0,
          "H299 pretrained no verified optimizer")
    for f in (
        "verified_model_skill_scores",
        "official_train_dev_admitted",
        "blind_ood_proven",
        "raw_exception_text_model_outputs_media_points_saved",
    ):
        flag(doc[f],False,"H299 unknown score/privacy error")
    return {
        "status":"H299_STRICT_PRIVATE_STAGE_ONLY_NO_MODEL_SCORE",
        "failure_stage":doc["failure_stage"],
        "exception_class":doc["exception_class"],
        "verified_native_model_score":False,
        "official_train_dev_admitted":False,
        "independent_blind_ood":False,
    }


def verify_private(path: Path, *, diagnostic: bool=False) -> dict:
    expected=DIAGNOSTIC_REPORT if diagnostic else REPORT
    need(isinstance(path,Path) and path.name==expected
         and path.is_file() and not path.is_symlink(),
         "H299 exact whitelisted private owner-only basename required")
    try:
        return (verify_failure if diagnostic else verify_success)(
            decode_bounded(path.read_bytes()))
    except OSError:
        raise EvidenceRejected("H299 private report unreadable") from None


def main() -> None:
    p=argparse.ArgumentParser(description="H299 private bounded model research receipt ONLY")
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--diagnostic",action="store_true")
    args=p.parse_args()
    try:
        safe=verify_private(args.report,diagnostic=args.diagnostic)
    except EvidenceRejected as err:
        p.exit(2,str(err)+"\n")
    print(json.dumps(safe,sort_keys=True,ensure_ascii=True))


if __name__=="__main__":
    main()
