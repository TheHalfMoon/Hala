"""H156 independent owner-only LoRA training and DEV prediction verifier.

Does not train models, touch blind data, or infer native success from JSON.
"""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from h156.action_contract_v2 import parse_v2
from h156.train_gpu import (
    SOURCE_SHA, ADAPTER_SHA, STEPS, GRAD_ACCUM,
    load_source
)

TASKS=("click-test","enter-text","choose-list")


def evaluate_predictions(rows, model_sha, dev):
    if not isinstance(rows,list) or len(rows)!=12:
        raise ValueError("Expected exactly 12 DEV model predictions")
    eligible={r["image_sha256"]:r for r in dev}
    if len(eligible)!=12:
        raise ValueError("DEV source SHA duplicates")
    result=Counter()
    observed=set()
    details=[]
    for prediction in rows:
        if not isinstance(prediction,dict) or set(prediction)!={
                "task","image_sha","raw","valid","sequence","text_correct"}:
            raise ValueError("GPU prediction schema altered")
        image=prediction["image_sha"]
        if (not isinstance(image,str) or image in observed or
            image not in eligible or prediction["task"] != eligible[image]["task"] or
            type(prediction["raw"]) is not str):
            raise ValueError("Prediction not part of DEV split")
        observed.add(image)
        task=prediction["task"]
        reference=eligible[image]["training_supervision"]["actions"]
        accepted=False
        non_abstain=False
        correct_text=False
        try:
            actions=parse_v2(prediction["raw"], task=task,
                             model_checkpoint_sha=model_sha)
            accepted=True
            non_abstain=actions[0].kind!="abstain"
            correct_text=([a.text for a in actions if a.text is not None]
                          ==[a["text"] for a in reference if "text" in a])
        except ValueError:
            pass
        if (prediction["valid"] is not accepted or
            prediction["sequence"] is not non_abstain or
            prediction["text_correct"] is not correct_text):
            raise ValueError("GPU provided scores disagree with independently parsed output")
        result["n"]+=1
        result["valid"]+=int(accepted)
        result["action_sequences"]+=int(non_abstain)
        if task!="click-test":
            result["text_required"]+=1
            result["text_correct"]+=int(accepted and correct_text)
        result[task+"_n"]+=1
        result[task+"_valid"]+=int(accepted)
        result[task+"_text_correct"]+=int(accepted and correct_text) if task!="click-test" else 0
        details.append((image,accepted,non_abstain))
    if observed!=set(eligible) or result["n"]!=12 or result["text_required"]!=8:
        raise ValueError("DEV incomplete or text role count wrong")
    for task in TASKS:
        if result[task+"_n"]!=4:
            raise ValueError("DEV source imbalance")
    return dict(result)


def audit(checkpoint_root,private_source_archive):
    root=Path(checkpoint_root)
    receipt_path=root/"hala-h156-typed-action-lora"/"private-h156-train-receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError("No completed H156 optimizer receipt")
    receipt=json.loads(receipt_path.read_text("utf8"))
    if (receipt.get("status")!="H156_COMPLETED_REAL_TYPED_ACTION_OPTIMIZER_STEPS" or
        receipt.get("schema")!="hala.h156.first-genuine-typed-action-2b-lora-pilot.v1" or
        receipt.get("actual_optimizer_steps")!=STEPS or
        receipt.get("train_cases")!=48 or receipt.get("dev_cases")!=12 or
        receipt.get("source_manifest_sha256")!=SOURCE_SHA or
        receipt.get("frozen_initial_adapter_sha256")!=ADAPTER_SHA or
        receipt.get("h140_sealed_blind_calls")!=0 or
        receipt.get("Hala_flagship_35B_updated") is not False or
        receipt.get("paid_compute") is not False or
        receipt.get("release_qualified") is not False):
        raise ValueError("No valid native-source optimizer evidence")
    losses=receipt.get("loss_trace")
    if (not isinstance(losses,list) or len(losses)!=STEPS*GRAD_ACCUM or
        any(type(v) not in (int,float) or not math.isfinite(v) for v in losses)):
        raise ValueError("Missing real per-example finite train loss evidence")
    model_file=(root/"hala-h156-typed-action-lora"/
                "private-h156-typed-action-lora"/"adapter_model.safetensors")
    if not model_file.is_file():
        raise FileNotFoundError("Actual saved research LoRA checkpoint absent")
    new_sha=hashlib.sha256(model_file.read_bytes()).hexdigest()
    if (new_sha!=receipt.get("updated_action_adapter_sha256") or
        new_sha==ADAPTER_SHA):
        raise ValueError("No new intact trained model checksum")
    source=load_source(private_source_archive)
    dev=[r for r in source if r["split"]=="DEV"]
    compare={}
    for arm,sha in (("before",ADAPTER_SHA),("after",new_sha)):
        path=(root/"hala-h156-typed-action-lora"/
              ("private-h156-dev-"+arm+".json"))
        rows=json.loads(path.read_text("utf8"))
        compare[arm]=evaluate_predictions(rows,sha,dev)
        if (receipt.get("dev_"+arm) != {
            "cases":compare[arm]["n"],
            "valid_json_actions":compare[arm]["valid"],
            "non_abstain_valid_sequences":compare[arm]["action_sequences"],
            # Existing GPU metric counts click no-text cases as correct.
            "correct_visible_text_values":sum(
                1 for row in rows if row["text_correct"])}):
            raise ValueError("GPU summary disagrees with independent parsed DEV model outputs")
    return {
        "schema":"hala.h156.independent-private-gpu-learning-audit.v1",
        "status":"PASS_NEW_2B_TYPED_ACTION_LORA_SFT",
        "optimizer_steps":STEPS,"train_examples":48,"dev_examples":12,
        "old_model_sha256":ADAPTER_SHA,"new_model_sha256":new_sha,
        "before":compare["before"],"after":compare["after"],
        "not_independent_app_evaluation":True,
        "not_native_browser_task_success":True,
        "h140_blind_calls":0,"paid_gpu":False,
        "hala35b_unchanged":True,
    }


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint-root",required=True)
    p.add_argument("--private-source-archive",required=True)
    args=p.parse_args()
    print(json.dumps(audit(args.checkpoint_root,args.private_source_archive),
                     sort_keys=True,indent=2))
