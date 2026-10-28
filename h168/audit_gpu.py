"""Independently audit actual H168 private real-browser TRAIN T4 receipts and vision adapter.

No access to old DEV/OOD or any heldout media. Independently reads saved
Safetensors header to ensure only visual merger weights were persisted.
"""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

from h132.grounder_contract import MODEL_ID,MODEL_SHA
from h166.vision_probe import (
    MAX_PEAK_GIB,EXPECTED_TORCH,EXPECTED_TRANSFORMERS,EXPECTED_PEFT)
from h167.combined_probe import (
    LANGUAGE_WEIGHTS_SHA,LANGUAGE_CONFIG_SHA,LANGUAGE_README_SHA)
from h168.train_gpu import TRAIN_ZIP_SHA,STEPS

SOURCE_COMMIT="da7d20367b7bb1db439cf23fbff0700bfb0cab89"
MODEL_DIR="hala-h168-private-real-original-train"

def sha(blob):
    return hashlib.sha256(blob).hexdigest()

def safely_read_vision_only_weights(path):
    blob=Path(path).read_bytes()
    if len(blob)<2048:
        raise ValueError("H168 private visual LoRA checkpoint too small")
    header_size=struct.unpack("<Q",blob[:8])[0]
    if header_size<10 or header_size>256000:
        raise ValueError("H168 safetensors header size invalid")
    header=json.loads(blob[8:8+header_size])
    keys=[k for k in header if k!="__metadata__"]
    if len(keys)!=4 or any(
            ".visual.merger." not in ("."+k+".") or
            not (".lora_A." in k or ".lora_B." in k) or
            "language_model" in k
            for k in keys):
        raise ValueError("H168 private checkpoint contains non-visual or missing tensors")
    return sha(blob)

def audit(root):
    root=Path(root)
    job=json.loads((root/"h168-private-training-job.json").read_text("utf8"))
    result=json.loads((root/MODEL_DIR/"h168-private-training-receipt.json").read_text("utf8"))
    if (job.get("schema")!="hala.h168.private-kaggle-train-only-job.v1" or
        job.get("status")!="H168_NEW_ORIGINAL_BROWSER_TRAIN_32_FREE_T4_PASS" or
        job.get("source_commit")!=SOURCE_COMMIT or
        job.get("paid_gpu") is not False or
        job.get("dev_ood_evaluations")!=0 or
        job.get("prior_consumed_eval_used") is not False or
        job.get("model_release") is not False or
        job.get("flagship_35b_unchanged") is not True or
        job.get("optimizer_steps")!=STEPS):
        raise ValueError("H168 real GPU job source or no-holdout policy unqualified")
    if (result.get("schema")!="hala.h168.real-original-browser-train-32-free-t4.v1"
        or result.get("status")!="H168_REAL_BROWSER_32_TRAIN_OPTIMIZER_STEPS_PASS"
        or result.get("source_train_zip_sha256")!=TRAIN_ZIP_SHA
        or result.get("language_adapter_sha256")!=LANGUAGE_WEIGHTS_SHA
        or result.get("language_adapter_config_sha256")!=LANGUAGE_CONFIG_SHA
        or result.get("language_adapter_readme_sha256")!=LANGUAGE_README_SHA
        or result.get("base_model")!=MODEL_ID
        or result.get("base_revision")!=MODEL_SHA
        or result.get("active_adapters")!=["language","vision"]
        or sorted(result.get("visual_modules",[]))!=[
          "model.visual.merger.linear_fc1","model.visual.merger.linear_fc2"]
        or result.get("trainable_visual_parameters")!=57344
        or result.get("frozen_language_parameters")!=5455872
        or result.get("language_frozen_before_after_match") is not True
        or result.get("optimizer_steps")!=STEPS
        or result.get("source_training_cases")!=STEPS
        or result.get("unique_training_exposures")!=STEPS
        or result.get("english_train_cases")!=16
        or result.get("arabic_train_cases")!=16
        or result.get("type_train_cases")!=16
        or result.get("drag_train_cases")!=16
        or result.get("installed")!={"torch":EXPECTED_TORCH,
           "transformers":EXPECTED_TRANSFORMERS,"peft":EXPECTED_PEFT,"Pillow":"11.3.0"}
        or "T4" not in result.get("gpu","")
        or result.get("memory_cap_gib")!=MAX_PEAK_GIB
        or result.get("saved_adapter_count")!=1
        or result.get("save_private_only") is not True
        or result.get("paid_gpu") is not False
        or result.get("dev_ood_evaluations")!=0
        or result.get("prior_consumed_eval_used") is not False
        or result.get("real_browser_task_reward_measured") is not False
        or result.get("model_release") is not False
        or result.get("flagship_35b_unchanged") is not True):
        raise ValueError("H168 original 32-case vision training receipt unqualified")
    expected={168000+i for i in range(16)}|{168100+i for i in range(16)}
    trace=result.get("trace")
    if (type(trace) is not list or len(trace)!=STEPS
        or {x.get("case_index") for x in trace}!=expected):
        raise ValueError("H168 missing true unique original TRAIN exposures")
    for i,row in enumerate(trace):
        ix=row["case_index"]
        expected_app="choices" if ix<168100 else "nouislider"
        parity=(ix-168000)%2 if ix<168100 else (ix-168100)%2
        if (row.get("step")!=i+1 or row.get("app")!=expected_app
            or row.get("kind")!=("type" if ix<168100 else "drag")
            or row.get("language")!=("en" if parity==0 else "ar")
            or type(row.get("loss")) not in (float,int)
            or not math.isfinite(row["loss"]) or not 0<row["loss"]<500
            or type(row.get("nonzero_visual_gradient_tensors")) is not int
            or not 1<=row["nonzero_visual_gradient_tensors"]<=4):
            raise ValueError("H168 actual train step/loss/native action provenance mismatch")
    allocated=result.get("peak_allocated_gib")
    reserved=result.get("peak_reserved_gib")
    if (type(allocated) not in (int,float) or
        type(reserved) not in (int,float) or
        not math.isfinite(allocated) or not math.isfinite(reserved)
        or not 0<allocated<=MAX_PEAK_GIB or reserved<allocated
        or job.get("peak_allocated_gib")!=allocated):
        raise ValueError("H168 paid/unsafe memory or job/receipt mismatch")
    paths=list((root/MODEL_DIR/"private-visual-adapter").rglob(
        "adapter_model.safetensors"))
    if len(paths)!=1:
        raise ValueError("H168 exactly one private vision adapter required")
    checkpoint_sha=safely_read_vision_only_weights(paths[0])
    if (checkpoint_sha!=result.get("saved_visual_only_adapter_sha256")
        or checkpoint_sha!=job.get("private_visual_sha256")):
        raise ValueError("H168 private checkpoint saved SHA mismatch")
    return {"status":"H168_INDEPENDENT_REAL_BROWSER_TRAIN_32_FREE_T4_VERIFIED",
            "optimizer_steps":STEPS,"original_unique_train_cases":32,
            "arabic_cases":16,"english_cases":16,"type_cases":16,"drag_cases":16,
            "peak_allocated_gib":allocated,"peak_reserved_gib":reserved,
            "frozen_language_unchanged":True,"private_visual_sha256":checkpoint_sha,
            "dev_ood_evaluations":0,"native_browser_reward_evaluated":False}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    a=p.parse_args()
    print(json.dumps(audit(a.output),sort_keys=True,indent=2))
