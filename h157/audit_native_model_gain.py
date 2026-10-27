"""H157 independent validation of 72 matched real neural browser DEV tasks.

Refuses duplicated tasks, unexecuted fake rewards, wrong optimizer/source,
or claiming generalization from same-app DEV. Does NOT invoke a model.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

from h157.audit_spatial_source import audit as source_audit
from h157.audit_gpu_learning import audit as gpu_audit
from h157.collect_spatial_source import COUNTS
from h157.replay_model_dev import LOCK,OUTPUT
from h157.train_gpu import START_ADAPTER_SHA

FIELDS={"role","task","image_sha","parse_valid","abstained",
        "model_actions_executed","native_success","native_reward_positive"}


def compare(episodes,cases):
    if not isinstance(episodes,list) or len(episodes)!=144:
        raise ValueError("H157 actual model comparison requires 144 native episodes")
    dev={(r["image_sha256"],r["task"]) for r in cases}
    if len(dev)!=72:
        raise ValueError("H157 72 DEV task identities must be unique")
    seen={}
    for row in episodes:
        if (not isinstance(row,dict) or set(row)!=FIELDS or
            row["role"] not in ("before","after") or
            type(row["native_success"]) is not bool or
            type(row["native_reward_positive"]) is not bool or
            row["native_success"]!=row["native_reward_positive"] or
            type(row["parse_valid"]) is not bool or
            type(row["abstained"]) is not bool or
            type(row["model_actions_executed"]) is not int or
            not 0<=row["model_actions_executed"]<=5 or
            (row["native_success"] and
                (not row["parse_valid"] or row["model_actions_executed"]==0)) or
            (row["abstained"] and
                (row["native_success"] or row["model_actions_executed"]!=0))):
            raise ValueError("H157 actual browser action/outcome contradicts receipt")
        key=(row["image_sha"],row["task"])
        if key not in dev or (key,row["role"]) in seen:
            raise ValueError("H157 unexpected or duplicate DEV model task")
        seen[(key,row["role"])]=row
    if len(seen)!=144:
        raise ValueError("H157 missing before/after paired neural task")
    counts=Counter()
    by_task={}
    for case in cases:
        key=(case["image_sha256"],case["task"])
        baseline=seen[(key,"before")]["native_success"]
        trained=seen[(key,"after")]["native_success"]
        counts["before"]+=int(baseline)
        counts["after"]+=int(trained)
        counts["new_wins"]+=int(trained and not baseline)
        counts["regressions"]+=int(baseline and not trained)
        counts["both_pass"]+=int(baseline and trained)
        counts["both_fail"]+=int(not baseline and not trained)
        item=by_task.setdefault(case["task"],Counter())
        item["n"]+=1
        item["before"]+=int(baseline)
        item["after"]+=int(trained)
    if {k:v["n"] for k,v in by_task.items()}!=COUNTS["DEV"]:
        raise ValueError("H157 paired native actions not balanced by task")
    discordant=counts["new_wins"]+counts["regressions"]
    m=min(counts["new_wins"],counts["regressions"])
    # Two-sided exact McNemar conditional sign test.
    p=(min(1,2*sum(math.comb(discordant,i) for i in range(m+1))
           /2**discordant) if discordant else 1.0)
    return {
        "paired_model_dev_tasks":72,
        "baseline_real_native_success":counts["before"],
        "trained_real_native_success":counts["after"],
        "absolute_gain_tasks":counts["after"]-counts["before"],
        "new_wins":counts["new_wins"],"regressions":counts["regressions"],
        "both_pass":counts["both_pass"],"both_fail":counts["both_fail"],
        "two_sided_exact_p":p,
        "per_original_task":{k:dict(v) for k,v in sorted(by_task.items())},
        "one_app_dev_only_not_crossapp_blind":True,
    }


def audit(owner,result_root,source_archive):
    owner=Path(owner)
    source=source_audit(owner)
    gpu=gpu_audit(result_root,source_archive)
    file=owner/"private-h157"/OUTPUT
    lock=owner/"private-h157"/LOCK
    if not file.is_file() or not lock.is_file():
        raise FileNotFoundError("No complete H157 one-shot native DEV result")
    result=json.loads(file.read_text("utf8"))
    record=json.loads(lock.read_text("utf8"))
    summary=result.get("summary")
    if (not isinstance(summary,dict) or
        summary.get("schema")!="hala.h157.private-real-neural-browser-dev-paired.v1" or
        summary.get("status")!="H157_COMPLETE_REAL_MODEL_144_NATIVE_BROWSER_DEV_EPISODES" or
        summary.get("original_chromium_model_action_episodes")!=144 or
        summary.get("unique_dev_screenshots")!=72 or
        summary.get("dev_source_manifest_sha256")!=source["manifest_sha256"] or
        summary.get("baseline_h156_adapter_sha256")!=START_ADAPTER_SHA or
        summary.get("trained_h157_adapter_sha256")!=gpu["new_adapter_sha256"] or
        summary.get("same_app_dev_not_blind") is not True or
        summary.get("multiapp_transfer_proven") is not False or
        summary.get("h140_blind_calls")!=0 or
        summary.get("paid_gpu") is not False or
        record.get("manifest_sha256")!=source["manifest_sha256"] or
        record.get("after_adapter_sha256")!=gpu["new_adapter_sha256"]):
        raise ValueError("H157 native reward provenance rejected")
    manifest=json.loads((owner/"private-h157"/
        "source-spatial-train-dev-v1"/"manifest.json").read_text("utf8"))
    dev=[r for r in manifest["private_cases"] if r["split"]=="DEV"]
    paired=compare(result.get("owner_private_cases"),dev)
    if (paired["baseline_real_native_success"]!=summary.get("before_native_success") or
        paired["trained_real_native_success"]!=summary.get("after_native_success") or
        any(paired["per_original_task"][k]["before"]!=v["before"] or
            paired["per_original_task"][k]["after"]!=v["after"]
            for k,v in summary.get("by_task",{}).items())):
        raise ValueError("H157 aggregate claimed browser reward not verified")
    return {
        "status":"H157_VERIFIED_REAL_NEURAL_NATIVE_DEV_REWARD_RESULT",
        "source_sha256":source["manifest_sha256"],
        "trained_model_sha256":gpu["new_adapter_sha256"],
        **paired,
        "h140_blind_calls":0,"paid_gpu":False,
    }


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--owner-root",required=True)
    parser.add_argument("--result-root",required=True)
    parser.add_argument("--source-archive",required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.owner_root,args.result_root,
                          args.source_archive),sort_keys=True,indent=2))
