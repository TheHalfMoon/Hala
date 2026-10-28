"""Independent H164 Kaggle training/DEV-format receipt audit. No owner gold or media."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from h164.native_action_contract import parse_action
from h164.train_gpu import TRAIN_SHA, DEV_SHA, PARENT_LORA_SHA, ACCUM, STEPS
from h164.ood_gpu_infer import TRAINED_MODEL_SHA, TRAINED_MODEL_ZIP_SHA

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def checked_rows(path, role):
    rows=json.loads(Path(path).read_text("utf8"))
    if type(rows) is not list or len(rows)!=96:
        raise ValueError("H164 exact 96 DEV inference outputs absent")
    grouped=Counter()
    valid={}
    for row in rows:
        if (type(row) is not dict or
            set(row)!={"role","app","task","language","image_sha","raw","parsed"} or
            row["role"]!=role or row["app"] not in ("calculator","simple-table") or
            row["language"] not in ("ar","en") or
            type(row["task"]) is not str or
            type(row["image_sha"]) is not str or len(row["image_sha"])!=64 or
            row["parsed"]!=parse_action(row["raw"])):
            raise ValueError("H164 genuine DEV model action/pair invalid")
        key=row["image_sha"]
        if key in valid:
            raise ValueError("H164 same DEV image repeated within model role")
        valid[key]=(row["app"],row["task"],row["language"])
        grouped[row["app"]+"/total"]+=1
        grouped[row["app"]+"/valid"]+=int(row["parsed"]["valid"])
        grouped[row["app"]+"/non_abstain"]+=int(
            row["parsed"]["kind"] not in (None,"abstain"))
    if grouped["calculator/total"]!=48 or grouped["simple-table/total"]!=48:
        raise ValueError("H164 DEV model app quotas changed")
    return rows,grouped,valid

def audit(root):
    root=Path(root)
    models=root/"hala-h164-new-original-ui-384-trained-lora"
    adapter=models/"private-h164-multi-original-app-trained-lora"/"adapter_model.safetensors"
    receipt=json.loads((models/"h164-training-receipt.json").read_text("utf8"))
    launcher=json.loads((root/"h164-free-T4-train-job-receipt.json").read_text("utf8"))
    if (sha(adapter)!=TRAINED_MODEL_SHA or
        sha(root/"eval-private-input"/"h164-trained-384-lora.zip")!=TRAINED_MODEL_ZIP_SHA or
        launcher.get("status")!="H164_COMPLETED_GENUINE_192_T4_OPTIMIZER_STEPS" or
        launcher.get("optimizer_steps")!=STEPS or
        launcher.get("trained_adapter_sha256")!=TRAINED_MODEL_SHA or
        launcher.get("paid_GPU") is not False or
        launcher.get("h140_blind_calls")!=0 or
        launcher.get("model_weights_published") is not False or
        receipt.get("schema")!="hala.h164.new-original-app-native-lang-lora-sft.v1" or
        receipt.get("status")!="H164_192_GENUINE_FREE_T4_OPTIMIZER_STEPS_COMPLETED" or
        receipt.get("optimizer_steps")!=STEPS or
        receipt.get("accumulation")!=ACCUM or
        receipt.get("unique_train_images")!=384 or
        receipt.get("train_exposures")!=384 or
        receipt.get("dev_images")!=96 or
        receipt.get("old_H161_adapter_sha256")!=PARENT_LORA_SHA or
        receipt.get("new_H164_adapter_sha256")!=TRAINED_MODEL_SHA or
        receipt.get("source_train_zip_sha256")!=TRAIN_SHA or
        receipt.get("source_dev_zip_sha256")!=DEV_SHA or
        receipt.get("dev_reference_labels_seen_by_gpu") is not False or
        receipt.get("actual_browser_task_success_measured") is not False or
        receipt.get("separate_ood_evaluation_not_performed") is not True or
        receipt.get("weights_published") is not False or
        receipt.get("paid_GPU") is not False or
        receipt.get("h140_blind_calls")!=0):
        raise ValueError("H164 genuine free T4 optimizer/adapter receipts inconsistent")
    losses=receipt.get("losses")
    if (type(losses) is not list or len(losses)!=STEPS*ACCUM or
        any(type(x) not in (float,int) or not math.isfinite(x)
            or x<0 for x in losses)):
        raise ValueError("H164 optimizer supervised loss coverage invalid")
    before,bcounts,bkeys=checked_rows(models/"h164-dev-before.json","before")
    after,acounts,akeys=checked_rows(models/"h164-dev-after.json","after")
    if (bkeys!=akeys or
        dict(bcounts)!=receipt.get("dev_model_format_before") or
        dict(acounts)!=receipt.get("dev_model_format_after")):
        raise ValueError("H164 DEV before/after matched inference counts drift")
    old={r["image_sha"]:r for r in before}
    new={r["image_sha"]:r for r in after}
    gains=sum(not old[k]["parsed"]["valid"] and new[k]["parsed"]["valid"] for k in old)
    regress=sum(old[k]["parsed"]["valid"] and not new[k]["parsed"]["valid"] for k in old)
    return {
        "status":"H164_COMPLETED_FREE_T4_TRAIN_INDEPENDENTLY_AUDITED",
        "optimizer_steps":STEPS,"unique_train_images":384,
        "dev_images":96,"old_dev_valid_json":sum(r["parsed"]["valid"] for r in before),
        "new_dev_valid_json":sum(r["parsed"]["valid"] for r in after),
        "paired_dev_json_gains":gains,
        "paired_dev_json_regressions":regress,
        "new_adapter_sha256":TRAINED_MODEL_SHA,
        "actual_browser_success_evaluated":False,
        "ood_success_evaluated":False,
        "owner_gold_on_GPU":False,"paid_GPU":False,"h140_blind_calls":0
    }

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--gpu-output",required=True)
    x=p.parse_args()
    print(json.dumps(audit(x.gpu_output),indent=2,sort_keys=True))
