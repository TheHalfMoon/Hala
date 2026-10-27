"""H156 independent owner-only source integrity audit, no model calls."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from h154.miniwob_source_probe import TASKS
from h156.action_contract_v2 import action_from_dict_v2
from h156.source_train_dev import SCHEMA,APP_SHA,SPLITS,excluded_images


def audit(owner):
    owner=Path(owner)
    folder=owner/"private-h156"/"miniwob-action-train-dev-v2"
    raw=(folder/"manifest.json").read_bytes()
    data=json.loads(raw)
    rows=data.get("private_cases")
    if (data.get("schema")!=SCHEMA or
        data.get("status")!="H156_NEW_ORIGINAL_EXECUTABLE_TRAIN_DEV_READY" or
        data.get("source_app_git_sha")!=APP_SHA or
        data.get("source_app_license")!="MIT" or
        data.get("train")!=48 or data.get("dev")!=12 or
        data.get("h140_sealed_blind_calls")!=0 or
        data.get("model_accuracy_measured") is not False or
        data.get("model_inference_calls")!=0 or
        data.get("optimizer_steps")!=0 or
        not isinstance(rows,list) or len(rows)!=60):
        raise ValueError("H156 frozen source manifest metadata invalid")
    historical=excluded_images(owner)
    used=set()
    seeds=set()
    counts=Counter()
    for row in rows:
        if (set(row)!={"split","task","action_family","seed","image",
                       "image_sha256","instruction","training_supervision",
                       "native_reference_actions_completed",
                       "native_controller_reads_dom","model_inference_calls"}
            or row["split"] not in SPLITS or row["task"] not in TASKS
            or row["action_family"]!=TASKS[row["task"]]
            or row["native_reference_actions_completed"] is not True
            or row["native_controller_reads_dom"] is not False
            or row["model_inference_calls"]!=0
            or not isinstance(row["seed"],int) or row["seed"] in seeds
            or not isinstance(row["instruction"],str) or not row["instruction"]):
            raise ValueError("H156 source case schema/role drift")
        name=row["image"]
        if (not isinstance(name,str) or name!=Path(name).name
            or not name.endswith(".png")):
            raise ValueError("Unsafe H156 owner media path")
        digest=hashlib.sha256((folder/"images"/name).read_bytes()).hexdigest()
        if digest!=row["image_sha256"] or digest in historical or digest in used:
            raise ValueError("H156 source media hash drift or historical leakage")
        actions=row["training_supervision"].get("actions")
        if not isinstance(actions,list):
            raise ValueError("H156 source label absent")
        parsed=[action_from_dict_v2(x) for x in actions]
        expected={
            "click-test":("click",),
            "enter-text":("click","type","submit"),
            "choose-list":("select","submit"),
        }[row["task"]]
        if tuple(a.kind for a in parsed)!=expected:
            raise ValueError("H156 action role mismatch")
        used.add(digest)
        seeds.add(row["seed"])
        counts[(row["split"],row["task"])]+=1
    if counts!={(role,task):n for role,n in SPLITS.items() for task in TASKS}:
        raise ValueError("H156 train and dev strata imbalance")
    return {
        "status":"H156_OWNER_INDEPENDENT_ACTION_SOURCE_AUDIT_PASS",
        "source_sha256":hashlib.sha256(raw).hexdigest(),
        "train":48,"dev":12,"distinct_original_png":len(used),
        "historical_excluded_images":len(historical),
        "reference_native_actions":60,
        "model_training_steps":0,"h140_blind_calls":0,
        "model_performance_claim":False,
    }


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--owner-root",required=True)
    opts=parser.parse_args()
    print(json.dumps(audit(opts.owner_root),indent=2,sort_keys=True))
