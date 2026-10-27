"""Independent owner-only H163 real new-app neural reward recomputation."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

from h163.source import GIT_SHA,SCHEMA,sha
from h163.pack_private_inference import (
    SOURCE_SHA,OLDER_MODEL_SHA,TRAINED_MODEL_SHA
)
from h163.replay_model_native import LOCK,RESULT

REQUIRED={"image_sha","role","task","language","valid",
          "click_sent","actual_original_app_success"}


def audit(owner):
    root=Path(owner)/"private-h163"
    source_data=(root/"source-once-v1"/"manifest.json").read_bytes()
    source=json.loads(source_data)
    proof=json.loads((root/"H163_SOURCE_PIXEL_QA.json").read_text("utf8"))
    lock=json.loads((root/LOCK).read_text("utf8"))
    report=json.loads((root/RESULT).read_text("utf8"))
    gpu=json.loads((root/"kaggle-output-v1"/
                    "h163-free-t4-original-app-receipt.json").read_text("utf8"))
    if (sha(source_data)!=SOURCE_SHA or source.get("schema")!=SCHEMA
        or source.get("upstream_git_sha")!=GIT_SHA
        or proof.get("status")!=
        "H163_16_OF_16_ORIGINAL_NEW_APP_EXACT_PIXEL_REPLAY_PASS"
        or lock.get("status")!=
        "H163_NEW_REAL_APP_NEURAL_MODEL_EVAL_CONSUMED"
        or lock.get("source_manifest_sha256")!=SOURCE_SHA
        or lock.get("frozen_before_sha")!=OLDER_MODEL_SHA
        or lock.get("frozen_after_sha")!=TRAINED_MODEL_SHA
        or report.get("status")!=
        "H163_NEW_ORIGINAL_GRIDSTACK_APP_REAL_NEURAL_REWARD_COMPLETE"
        or report.get("original_app_cases")!=16
        or report.get("actual_browser_trials")!=32
        or report.get("new_upstream_original_app_not_in_Hala_training") is not True
        or gpu.get("status")!=
        "H163_COMPLETED_32_REAL_FROZEN_GPU_CALLS"
        or gpu.get("gpu_calls")!=32
        or gpu.get("optimizer_steps")!=0 or gpu.get("paid_GPU") is not False):
        raise ValueError("H163 owner private model/browser provenance failed")
    cases=source["private_cases"]
    expected={(c["image_sha256"],role)
              for c in cases for role in ("before","after")}
    rows=report.get("private_rows")
    if not isinstance(rows,list) or len(rows)!=32 or len(expected)!=32:
        raise ValueError("H163 actual native trial count incomplete")
    by={}
    for row in rows:
        if (not isinstance(row,dict) or set(row)!=REQUIRED
            or (row["image_sha"],row["role"]) not in expected
            or (row["image_sha"],row["role"]) in by
            or not all(type(row[k]) is bool for k in
                    ("valid","click_sent","actual_original_app_success"))
            or (row["actual_original_app_success"] and not
                (row["valid"] and row["click_sent"]))):
            raise ValueError("H163 native source/model assignment altered")
        by[(row["image_sha"],row["role"])]=row
    if set(by)!=expected:
        raise ValueError("H163 incomplete model/case native role pairs")
    counts=Counter()
    grouped=Counter()
    for c in cases:
        old=by[(c["image_sha256"],"before")]
        new=by[(c["image_sha256"],"after")]
        if any(row["task"]!=c["task"] or
               row["language"]!=c["language"] for row in (old,new)):
            raise ValueError("H163 new app task/lang/model pair mismatch")
        a=old["actual_original_app_success"]
        b=new["actual_original_app_success"]
        counts["before"]+=int(a)
        counts["after"]+=int(b)
        counts["gains"]+=int(b and not a)
        counts["regressions"]+=int(a and not b)
        grouped[c["task"]+"/before"]+=int(a)
        grouped[c["task"]+"/after"]+=int(b)
        grouped[c["language"]+"/before"]+=int(a)
        grouped[c["language"]+"/after"]+=int(b)
    discord=counts["gains"]+counts["regressions"]
    n=min(counts["gains"],counts["regressions"])
    p=min(1.,2*sum(math.comb(discord,k) for k in range(n+1))
          /(2**discord)) if discord else 1.
    if (report.get("baseline_success")!=counts["before"]
        or report.get("trained_success")!=counts["after"]
        or report.get("paired_gains")!=counts["gains"]
        or report.get("paired_regressions")!=counts["regressions"]
        or report.get("exact_paired_sign_p")!=p
        or report.get("by_task_and_language")!=dict(grouped)):
        raise ValueError("H163 original app task reward misreported")
    return {
        "status":"H163_NEW_ORIGINAL_APP_NATIVE_REWARD_INDEPENDENT_AUDIT_PASS",
        "baseline_success":counts["before"],
        "trained_success":counts["after"],
        "paired_gains":counts["gains"],
        "paired_regressions":counts["regressions"],
        "paired_sign_p":p,
        "new_original_app_real_native_cases":16,
        "model_browser_trials":32,
        "by_task_and_language":dict(grouped),
        "one_new_app_not_universal_generalization":True,
        "h140_blind_calls":0,"paid_GPU":False,
    }


if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--owner",required=True)
    args=a.parse_args()
    print(json.dumps(audit(args.owner),indent=2,sort_keys=True))
