"""Owner-only V3 DEV go/no-go; never read H140 one-shot BLIND media.

Baseline H140 V2 failed React END point diversity. Improvement on reused
DEV alone is not independent-app transfer; even passing this gate merely
qualifies the single preregistered offline BLIND evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path

from h140.train_gpu_v3 import TRAIN_MANIFEST_SHA,DEV_MANIFEST_SHA

V2_STATUS="H140_COMPLETED_128_REAL_GPU_OPTIMIZER_STEPS"
V3_STATUS="H140_V3_COMPLETED_160_REAL_GPU_OPTIMIZER_STEPS"
REQUIREMENTS={
    "total_both_min":28,
    "start_min":60,
    "end_min":28,
    "react_end_min":8,
}

def receipt_valid(doc,step_count,status):
    if (doc.get("status")!=status or doc.get("real_optimizer_steps")!=step_count or
        doc.get("frozen_train_original_screens")!=256 or
        doc.get("frozen_dev_original_screens")!=64 or
        doc.get("train_manifest_sha256")!=TRAIN_MANIFEST_SHA or
        doc.get("dev_manifest_sha256")!=DEV_MANIFEST_SHA or
        doc.get("real_blind_neural_calls")!=0 or
        doc.get("blind_gold_attached") is not False or
        doc.get("paid_compute_used") is not False or
        doc.get("flagship_Hala_35B_updated") is not False):
        raise ValueError("H140 source-pinned, zero-cost, no-BLIND GPU experiment mismatch")
    return doc["h140_adapter_dev"]["point_hits"],doc["h140_adapter_dev"]["all_apps_and_roles_diverse"]

def actual_adapter_hash(folder,doc):
    out=list(Path(folder).glob("*lora-rank8/adapter_model.safetensors"))
    if len(out)!=1:
        raise ValueError("H140 exactly one true private trained adapter must exist")
    actual=hashlib.sha256(out[0].read_bytes()).hexdigest()
    if actual!=doc.get("adapter_sha256"):
        raise ValueError("H140 GPU model checkpoint hash mismatch")
    return actual

def evaluate(v2_dir,v3_dir):
    v2_dir,v3_dir=Path(v2_dir),Path(v3_dir)
    old=json.loads((v2_dir/"private-h140-learning-evidence.json").read_text("utf8"))
    new=json.loads((v3_dir/"private-h140-v3-learning-evidence.json").read_text("utf8"))
    old_hits,old_diverse=receipt_valid(old,128,V2_STATUS)
    new_hits,new_diverse=receipt_valid(new,160,V3_STATUS)
    old_sha=actual_adapter_hash(v2_dir,old)
    new_sha=actual_adapter_hash(v3_dir,new)
    if old_sha==new_sha:
        raise ValueError("H140 V3 adapter unchanged from V2 despite retraining claim")
    v2=old_hits["__all__"]
    v3=new_hits["__all__"]
    react_old=old_hits["react-native-board"]["end"]
    react_new=new_hits["react-native-board"]["end"]
    conditions={
        "original_3_app_end_diversity":bool(new_diverse),
        "combined_start_end_hits_at_least_28_of_64":v3["both"]>=REQUIREMENTS["total_both_min"],
        "start_hits_at_least_60_of_64":v3["start"]>=REQUIREMENTS["start_min"],
        "end_hits_at_least_28_of_64":v3["end"]>=REQUIREMENTS["end_min"],
        "react_end_hits_at_least_8_of_22":react_new>=REQUIREMENTS["react_end_min"],
        "no_app_full_task_accuracy_regression":all(
            new_hits[app]["both"]>=old_hits[app]["both"]
            for app in ("vanilla-native-board","react-native-board","react-dndkit-board")),
        "new_adapter_distinct_from_v2":old_sha!=new_sha,
    }
    return {
        "schema":"hala.h140.v3.preregistered-dev-qualification.v1",
        "status":("ELIGIBLE_FOR_ONE_FINAL_OFFLINE_BLIND"
                  if all(conditions.values()) else
                  "NOT_QUALIFIED_DO_NOT_TOUCH_BLIND"),
        "source":"original_DEV64_only_no_blind_reads",
        "v2_diverse":old_diverse,
        "v3_diverse":new_diverse,
        "checks":conditions,
        "v2":{"start":v2["start"],"end":v2["end"],"both":v2["both"],
              "react_end":react_old},
        "v3":{"start":v3["start"],"end":v3["end"],"both":v3["both"],
              "react_end":react_new},
        "v2_v3_independent_blind_calls":0,
        "flagship_Hala_35B_updated":False,
    }

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--v2",required=True)
    parser.add_argument("--v3",required=True)
    args=parser.parse_args()
    print(json.dumps(evaluate(args.v2,args.v3),sort_keys=True,indent=2))
