"""H158 independently audited true cross-app neural native task outcomes."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

from h158.audit_source import audit as audit_source
from h158.gpu_infer import MODEL_SHAS
from h158.original_todomvc_source import COUNTS
from h158.replay_neural_native import OUT,LOCK

CASE_FIELDS={"role","task","language","image_sha","valid","abstained",
             "click_sent","original_app_task_success"}


def paired(rows,cases):
    if not isinstance(rows,list) or len(rows)!=32:
        raise ValueError("H158 independent app needs 32 actual native episodes")
    expected={(c["image_sha256"],c["task"],c["language"]) for c in cases}
    if len(expected)!=16:
        raise ValueError("H158 second-app 16 original screenshots required")
    by={}
    for row in rows:
        if (not isinstance(row,dict) or set(row)!=CASE_FIELDS or
            row["role"] not in ("before","after") or
            not all(type(row[key]) is bool for key in (
                "valid","abstained","click_sent","original_app_task_success")) or
            row["original_app_task_success"] and (
                not row["click_sent"] or not row["valid"] or row["abstained"]) or
            row["abstained"] and row["click_sent"]):
            raise ValueError("H158 actual cross-app native task receipt invalid")
        key=(row["image_sha"],row["task"],row["language"])
        if key not in expected or (key,row["role"]) in by:
            raise ValueError("H158 model predicted unrelated or duplicate original scene")
        by[(key,row["role"])]=row
    if len(by)!=32:
        raise ValueError("H158 model arm missing from matched original cases")
    totals=Counter()
    groups={}
    for case in cases:
        key=(case["image_sha256"],case["task"],case["language"])
        before=by[(key,"before")]["original_app_task_success"]
        after=by[(key,"after")]["original_app_task_success"]
        totals["before"]+=int(before)
        totals["after"]+=int(after)
        totals["new_wins"]+=int(after and not before)
        totals["regressions"]+=int(before and not after)
        totals["both_pass"]+=int(before and after)
        totals["both_fail"]+=int(not before and not after)
        name=(case["task"],case["language"])
        group=groups.setdefault(name,Counter())
        group["n"]+=1
        group["before"]+=int(before)
        group["after"]+=int(after)
    if {key:val["n"] for key,val in groups.items()}!={
        (task,lang):n for task,g in COUNTS.items() for lang,n in g.items()}:
        raise ValueError("H158 bilingual external application strata drift")
    d=totals["new_wins"]+totals["regressions"]
    m=min(totals["new_wins"],totals["regressions"])
    p=min(1.0,2*sum(math.comb(d,k) for k in range(m+1))/(2**d)) if d else 1.0
    return {
        "paired_new_app_tasks":16,
        "before_actual_app_success":totals["before"],
        "after_actual_app_success":totals["after"],
        "new_wins":totals["new_wins"],"regressions":totals["regressions"],
        "both_pass":totals["both_pass"],"both_fail":totals["both_fail"],
        "paired_sign_two_sided_p":p,
        "per_task_language":{f"{t}/{l}":dict(v) for (t,l),v in sorted(groups.items())},
        "original_app_has_only_memory_storage":True,
        "evaluation_is_external_app_not_original_training_app":True,
    }


def audit(owner):
    owner=Path(owner)
    source=audit_source(owner)
    folder=owner/"private-h158"
    report=json.loads((folder/OUT).read_text("utf8"))
    lock=json.loads((folder/LOCK).read_text("utf8"))
    summary=report.get("summary")
    if (not isinstance(summary,dict) or
        summary.get("schema")!="hala.h158.private-genuine-second-app-native-reward.v1" or
        summary.get("status")!="H158_COMPLETED_32_REAL_UNTRAINED_NEW_APP_MODEL_ACTIONS" or
        summary.get("frozen_source_manifest_sha256")!=source["manifest_sha256"] or
        summary.get("baseline_adapter_sha")!=MODEL_SHAS["before"] or
        summary.get("trained_adapter_sha")!=MODEL_SHAS["after"] or
        summary.get("native_browser_trials")!=32 or
        summary.get("new_original_task_images")!=16 or
        summary.get("app_state_persistent_after_reload") is not False or
        summary.get("new_app_inference_only") is not True or
        summary.get("optimizer_steps")!=0 or
        summary.get("h140_blind_calls")!=0 or
        summary.get("paid_gpu") is not False or
        lock.get("source_manifest_sha256")!=source["manifest_sha256"] or
        lock.get("model_sha")!=MODEL_SHAS):
        raise ValueError("H158 source, model and native app receipts inconsistent")
    data=json.loads((folder/"todomvc-source-once-v1"/"manifest.json").read_text("utf8"))
    result=paired(report.get("private_cases"),data["private_cases"])
    if (result["before_actual_app_success"]!=summary["before_actual_task_success"] or
        result["after_actual_app_success"]!=summary["after_actual_task_success"]):
        raise ValueError("H158 native task aggregate contradicts private case outcomes")
    return {"status":"H158_AUDITED_DIFFERENT_APP_NATIVE_MODEL_RESULT",
            "source_sha256":source["manifest_sha256"],
            "baseline_lora_sha":MODEL_SHAS["before"],
            "trained_lora_sha":MODEL_SHAS["after"],
            **result,
            "Hala35B_unchanged":True,
            "h140_blind_calls":0,
            "paid_gpu":False}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--owner-root",required=True)
    x=parser.parse_args()
    print(json.dumps(audit(x.owner_root),sort_keys=True,indent=2))
