"""Independent H161 owner-private 48 genuine model-native DEV outcome audit."""
import argparse
import json
import math
from collections import Counter,defaultdict
from pathlib import Path
from h161.replay_model_dev import OUTPUT,LOCK
from h161.original_apps_source import QUOTAS

FIELDS={"app","language","task","image_sha","role","native_action_sent",
        "actual_original_app_success","model_output_valid","model_action_kind"}


def audit(owner):
    owner=Path(owner)
    root=owner/"private-h161"
    proof=json.loads((root/"H161_INDEPENDENT_PIXEL_QA.json").read_text("utf8"))
    native=json.loads((root/OUTPUT).read_text("utf8"))
    lock=json.loads((root/LOCK).read_text("utf8"))
    source=json.loads((root/"two-real-app-source-v1"/"manifest.json").read_text("utf8"))
    expected=[r for r in source["private_cases"] if r["split"]=="DEV"]
    if (proof.get("status")!=
        "H161_72_OF_72_CROSS_CONTEXT_EXACT_NATIVE_SCREENSHOT_MATCH" or
        native.get("status")!=
        "H161_COMPLETED_48_GENUINE_NATIVE_TRAINED_MODEL_DEV_TASKS" or
        native.get("actual_native_browser_trials")!=48 or
        native.get("model_dev_cases")!=24 or
        native.get("same_apps_as_training_not_OOD_BLIND") is not True or
        native.get("h140_blind_calls")!=0 or
        native.get("paid_GPU") is not False or
        lock.get("status")!="H161_DEV_NEURAL_48_NATIVE_ATTEMPTS_CONSUMED" or
        len(expected)!=24):
        raise ValueError("H161 trained model original task QA receipt incomplete")
    rows=native.get("private_rows")
    if not isinstance(rows,list) or len(rows)!=48:
        raise ValueError("H161 owner original app records incomplete")
    expected_keys={(s["image_sha256"],role) for s in expected
                   for role in ("before","after")}
    by={}
    for row in rows:
        if (not isinstance(row,dict) or set(row)!=FIELDS or
            (row["image_sha"],row["role"]) not in expected_keys or
            (row["image_sha"],row["role"]) in by or
            not all(type(row[k]) is bool for k in (
                "native_action_sent","actual_original_app_success",
                "model_output_valid")) or
            row["actual_original_app_success"] and not (
                row["native_action_sent"] and row["model_output_valid"])):
            raise ValueError("H161 actual native episode corrupt/duplicated")
        by[(row["image_sha"],row["role"])]=row
    if set(by)!=expected_keys:
        raise ValueError("H161 paired neural model DEV assignment incomplete")
    totals=Counter()
    groups=defaultdict(Counter)
    for c in expected:
        before=by[(c["image_sha256"],"before")]
        after=by[(c["image_sha256"],"after")]
        if any(row["app"]!=c["app"] or
               row["language"]!=c["language"] or
               row["task"]!=c["task"] for row in (before,after)):
            raise ValueError("H161 original app action role/source drift")
        b=before["actual_original_app_success"]
        a=after["actual_original_app_success"]
        totals["before"]+=int(b)
        totals["after"]+=int(a)
        totals["gains"]+=int(a and not b)
        totals["regressions"]+=int(b and not a)
        group=groups[(c["app"],c["language"])]
        group["n"]+=1
        group["before"]+=int(b)
        group["after"]+=int(a)
    if ({(app,lang):stats["n"] for (app,lang),stats in groups.items()}
        !={(app,lang):QUOTAS["DEV"][app]//2 for app in QUOTAS["DEV"]
           for lang in ("en","ar")}):
        raise ValueError("H161 native bilingual task split missing")
    d=totals["gains"]+totals["regressions"]
    m=min(totals["gains"],totals["regressions"])
    p=min(1.,2*sum(math.comb(d,k) for k in range(m+1))/(2**d)) if d else 1.
    if (totals["before"]!=native.get("before_app_success") or
        totals["after"]!=native.get("after_app_success") or
        totals["gains"]!=native.get("paired_gains") or
        totals["regressions"]!=native.get("paired_regressions") or
        p!=native.get("paired_sign_p")):
        raise ValueError("H161 private model task reward stats misrepresented")
    return {
        "status":"H161_INDEPENDENT_48_NATIVE_REWARD_AUDIT_PASS",
        "before_native_app_success":totals["before"],
        "after_native_app_success":totals["after"],
        "new_wins":totals["gains"],"regressions":totals["regressions"],
        "paired_p":p,"native_task_trials":48,"original_dev_cases":24,
        "per_app_language":{f"{a}/{l}":dict(v)
             for (a,l),v in sorted(groups.items())},
        "still_NOT_unseen_app_BLIND":True,
        "h140_blind_calls":0,"paid_GPU":False,
    }


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner",required=True)
    args=p.parse_args()
    print(json.dumps(audit(args.owner),sort_keys=True,indent=2))
