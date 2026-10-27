"""Independent H157 real GPU optimizer and structured DEV audit (owner-only).

Parsing/point-format scores must NEVER be called native browser rewards.
"""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from h157.action_contract_v3 import parse_v3
from h157.train_gpu import (
    SOURCE_SHA,START_ADAPTER_SHA,STEPS,GRAD_ACCUM,read_source,
)
from h157.collect_spatial_source import COUNTS

REQUIRED_OUTPUT_FIELDS={"task","image_sha","raw","valid","sequence","text_correct"}


def checked_predictions(rows,model_sha,dev):
    expected={r["image_sha256"]:r for r in dev}
    if (not isinstance(rows,list) or len(rows)!=72 or len(expected)!=72):
        raise ValueError("H157 requires 72 exact independently frozen DEV outputs")
    seen=set()
    counts=Counter()
    for prediction in rows:
        if not isinstance(prediction,dict) or set(prediction)!=REQUIRED_OUTPUT_FIELDS:
            raise ValueError("H157 neural output schema drift")
        key=prediction["image_sha"]
        if (key not in expected or key in seen or
            prediction["task"]!=expected[key]["task"] or
            type(prediction["raw"]) is not str):
            raise ValueError("Neural output outside H157 frozen DEV")
        task=prediction["task"]
        visible_text=task in ("enter-text","enter-text-dynamic","choose-list")
        valid=False;sequence=False;correct_text=False
        try:
            actions=parse_v3(prediction["raw"],task=task,
                             model_checkpoint_sha=model_sha)
            valid=True
            sequence=actions[0].kind!="abstain"
            if sequence and visible_text:
                gold=expected[key]["training_supervision"]["actions"]
                correct_text=(
                    [a.text for a in actions if a.text is not None]
                    ==[a["text"] for a in gold if "text" in a])
        except ValueError:
            pass
        if (prediction["valid"] is not valid or
            prediction["sequence"] is not sequence or
            prediction["text_correct"] is not
                 (correct_text if visible_text else None)):
            raise ValueError("H157 DEV neural prediction counters falsified")
        counts["dev"]+=1
        counts["valid_json"]+=int(valid)
        counts["valid_action_sequences"]+=int(sequence)
        counts["requires_text"]+=int(visible_text)
        counts["correct_text_values"]+=int(correct_text)
        counts[task+"_n"]+=1
        counts[task+"_valid"]+=int(valid)
        seen.add(key)
    if (len(seen)!=72 or counts["requires_text"]!=40 or
        any(counts[t+"_n"]!=COUNTS["DEV"][t]
            for t in COUNTS["DEV"])):
        raise ValueError("H157 DEV source stratum/split contamination")
    return dict(counts)


def audit(root,archive):
    root=Path(root)
    folder=root/"hala-h157-spatial-action-lora"
    file=folder/"private-h157-learning-receipt.json"
    if not file.is_file():
        raise FileNotFoundError("No H157 genuine LoRA completion receipt")
    report=json.loads(file.read_text("utf8"))
    if (report.get("schema")!="hala.h157.private-original-spatial-action-2b-lora.v1" or
        report.get("status")!="H157_GENUINE_152_FREE_T4_OPTIMIZER_STEPS_COMPLETE" or
        report.get("optimizer_steps")!=STEPS or
        report.get("gradient_accumulation")!=GRAD_ACCUM or
        report.get("train_unique_original")!=280 or
        report.get("train_weighted_action_examples")!=304 or
        report.get("dev_unique_original")!=72 or
        report.get("source_manifest_sha256")!=SOURCE_SHA or
        report.get("initial_trained_h156_adapter_sha256")!=START_ADAPTER_SHA or
        report.get("dev_not_blind") is not True or
        report.get("true_native_task_completion_measured") is not False or
        report.get("h140_sealed_blind_calls")!=0 or
        report.get("paid_compute") is not False or
        report.get("Hala35B_unchanged") is not True or
        report.get("weights_published") is not False or
        report.get("release_qualified") is not False):
        raise ValueError("H157 genuine SFT training receipt not qualified")
    trace=report.get("loss_trace")
    if (not isinstance(trace,list) or len(trace)!=STEPS*GRAD_ACCUM or
        any(type(loss) not in (float,int) or not math.isfinite(loss)
            for loss in trace)):
        raise ValueError("H157 real training completion losses absent/nonfinite")
    checkpoint=(folder/"private-h157-spatial-research-lora"/
                "adapter_model.safetensors")
    if not checkpoint.is_file():
        raise FileNotFoundError("H157 actual trained LoRA weights missing")
    new_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if (new_sha==START_ADAPTER_SHA or
        new_sha!=report.get("new_trained_h157_adapter_sha256")):
        raise ValueError("H157 learned research adapter checksum mismatch")
    source=read_source(archive)
    dev=[row for row in source if row["split"]=="DEV"]
    summary={}
    for role,sha in (("before",START_ADAPTER_SHA),("after",new_sha)):
        result=json.loads((folder/("private-h157-dev-"+role+".json"))
                          .read_text("utf8"))
        stats=checked_predictions(result,sha,dev)
        if stats!=report.get(role+"_dev"):
            raise ValueError("H157 GPU DEV reported counts contradict original raw predictions")
        summary[role]=stats
    return {
        "status":"H157_INDEPENDENT_GPU_TRAINING_AUDIT_PASS",
        "genuine_optimizer_steps":STEPS,
        "unique_original_train":280,
        "original_dev":72,
        "source_manifest_sha256":SOURCE_SHA,
        "base_adapter_sha256":START_ADAPTER_SHA,
        "new_adapter_sha256":new_sha,
        "dev_before":summary["before"],"dev_after":summary["after"],
        "real_native_browser_score_measured":False,
        "unseen_app_transfer_proven":False,
        "paid_gpu":False,
        "h140_blind_calls":0,
    }


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root",required=True)
    parser.add_argument("--source-archive",required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.checkpoint_root,args.source_archive),
                     sort_keys=True,indent=2))
