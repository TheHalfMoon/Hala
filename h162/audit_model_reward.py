"""H162 independently recompute 32 genuine third-app native task rewards."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

from h162.original_third_app import SCHEMA,PIN
from h162.pack_private_inference import SOURCE_SHA,BASE_SHA,TRAINED_SHA
from h162.replay_model_native import OUTPUT,LOCK

FIELDS={"role","image_sha","language","valid","native_drag_sent",
        "real_original_app_task_success"}


def audit(owner):
    root=Path(owner)/"private-h162"
    frozen=json.loads((root/"original-html5-board-source-once-v1"/
                       "manifest.json").read_text("utf8"))
    proof=json.loads((root/"H162_SOURCE_PIXEL_QA.json").read_text("utf8"))
    lock=json.loads((root/LOCK).read_text("utf8"))
    receipt=json.loads((root/OUTPUT).read_text("utf8"))
    gpu=json.loads((root/"kaggle-output-v1"/
                    "h162-free-t4-crossapp-receipt.json").read_text("utf8"))
    if (frozen.get("schema")!=SCHEMA or frozen.get("upstream_sha")!=PIN or
        proof.get("status")!="H162_16_EXACT_ORIGINAL_THIRD_APP_REPLAY_PASS" or
        lock.get("status")!="H162_MODEL_EXTERNAL_BROWSER_ONESHOT_CONSUMED" or
        lock.get("source_manifest_sha256")!=SOURCE_SHA or
        lock.get("before_model_sha")!=BASE_SHA or
        lock.get("after_model_sha")!=TRAINED_SHA or
        receipt.get("status")!=
        "H162_COMPLETED_32_REAL_ORIGINAL_THIRD_APP_MODEL_TASKS" or
        receipt.get("native_browser_trials")!=32 or
        receipt.get("original_app_cases")!=16 or
        receipt.get("historically_seen_app_NOT_sealed_BLIND") is not True or
        gpu.get("status")!="H162_COMPLETED_32_GENUINE_THIRD_APP_GPU_CALLS" or
        gpu.get("real_gpu_calls")!=32 or
        gpu.get("optimizer_steps")!=0 or
        gpu.get("paid_GPU") is not False):
        raise ValueError("H162 original third-app actual model audit provenance failed")
    original=frozen["private_cases"]
    rows=receipt.get("private_rows")
    expected={(c["image_sha256"],role) for c in original
              for role in ("before","after")}
    if not isinstance(rows,list) or len(rows)!=32 or len(expected)!=32:
        raise ValueError("H162 paired third-app model browser tasks incomplete")
    by={}
    for row in rows:
        if (not isinstance(row,dict) or set(row)!=FIELDS or
            (row["image_sha"],row["role"]) not in expected or
            (row["image_sha"],row["role"]) in by or
            type(row["valid"]) is not bool or
            type(row["native_drag_sent"]) is not bool or
            type(row["real_original_app_task_success"]) is not bool or
            row["real_original_app_task_success"] and not
            (row["valid"] and row["native_drag_sent"])):
            raise ValueError("H162 real original app native outcome tampered")
        by[(row["image_sha"],row["role"])]=row
    if set(by)!=expected:
        raise ValueError("H162 16 original scenes per model required")
    count=Counter()
    for original_case in original:
        old=by[(original_case["image_sha256"],"before")]
        new=by[(original_case["image_sha256"],"after")]
        if (old["language"]!=original_case["language"] or
            new["language"]!=original_case["language"]):
            raise ValueError("H162 model case bilingual source mismatch")
        before=old["real_original_app_task_success"]
        after=new["real_original_app_task_success"]
        count["before"]+=int(before)
        count["after"]+=int(after)
        count["gains"]+=int(after and not before)
        count["regressions"]+=int(before and not after)
        count[original_case["language"]+"/total"]+=1
        count[original_case["language"]+"/after"]+=int(after)
    d=count["gains"]+count["regressions"]
    minor=min(count["gains"],count["regressions"])
    p=min(1.,2*sum(math.comb(d,k) for k in range(minor+1))/(2**d)) if d else 1.
    if (count["before"]!=receipt["before_actual_app_success"] or
        count["after"]!=receipt["after_actual_app_success"] or
        count["gains"]!=receipt["paired_new_wins"] or
        count["regressions"]!=receipt["paired_regressions"] or
        p!=receipt["exact_paired_sign_p"]):
        raise ValueError("H162 third-app transfer success report unreproducible")
    return {
        "status":"H162_INDEPENDENT_EXTERNAL_APP_32_NATIVE_REWARD_RECOMPUTED",
        "before_real_original_app_success":count["before"],
        "after_real_original_app_success":count["after"],
        "paired_new_wins":count["gains"],
        "paired_regressions":count["regressions"],
        "paired_p":p,"original_third_app_cases":16,
        "native_trials":32,
        "third_app_not_in_H161_training":True,
        "historically_seen_app_not_sealed_BLIND":True,
        "h140_blind_calls":0,"paid_GPU":False,
    }


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner",required=True)
    args=p.parse_args()
    print(json.dumps(audit(args.owner),sort_keys=True,indent=2))
