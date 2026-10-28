"""Independently recompute H166 synthetic-only free T4 projection memory gate."""
import argparse
import json
import math
from pathlib import Path

from h132.grounder_contract import MODEL_ID,MODEL_SHA
from h166.vision_probe import (
    MAX_PEAK_GIB,RANK,EXPECTED_TRANSFORMERS,EXPECTED_PEFT,EXPECTED_TORCH)

SOURCE_COMMIT="ea6b6f5cf471e2e48896a8f6cd4c74035607cc84"
RECEIPT="h166-visual-memory-receipt.json"
JOB="h166-job-receipt.json"

def audit(output):
    root=Path(output)
    job=json.loads((root/JOB).read_text(encoding="utf8"))
    model=json.loads((root/"hala-h166-synthetic-vision-probe"/RECEIPT).read_text(encoding="utf8"))
    expected_env={"torch":EXPECTED_TORCH,"transformers":EXPECTED_TRANSFORMERS,
                  "peft":EXPECTED_PEFT,"Pillow":"11.3.0"}
    selected=model.get("visual_merger_modules")
    if (job.get("schema")!="hala.h166.private-synthetic-visual-t4-job.v1" or
        job.get("status")!="H166_SYNTHETIC_FREE_T4_MEMORY_STEP_PASS" or
        job.get("source_commit")!=SOURCE_COMMIT or
        job.get("optimizer_steps")!=1 or
        job.get("real_app_training") is not False or
        job.get("owner_gold_used") is not False or
        job.get("new_ood_evaluations")!=0 or
        job.get("model_weights_published") is not False or
        job.get("paid_GPU") is not False or
        job.get("h140_blind_calls")!=0 or
        model.get("schema")!="hala.h166.synthetic-visual-merger-t4-probe.v1" or
        model.get("status")!="H166_ONE_SYNTHETIC_VISUAL_MERGER_STEP_PASSED" or
        model.get("model_id")!=MODEL_ID or model.get("model_revision")!=MODEL_SHA or
        model.get("environment")!=expected_env or
        "T4" not in model.get("gpu","") or
        model.get("optimizer_steps")!=1 or
        model.get("synthetic_only") is not True or
        model.get("source_original_app_images")!=0 or
        model.get("owner_gold_used") is not False or
        model.get("new_ood_evaluations")!=0 or
        model.get("model_weights_saved") is not False or
        model.get("paid_GPU") is not False or
        model.get("h140_blind_calls")!=0 or
        model.get("flagship_35B_unchanged") is not True or
        model.get("lora_rank")!=RANK or
        model.get("memory_cap_gib")!=MAX_PEAK_GIB or
        type(selected) is not list or not 1<=len(selected)<=4 or
        len(selected)!=len(set(selected)) or
        any(type(s) is not str or ".visual.merger." not in ("."+s+".")
            or "language_model" in s for s in selected)):
        raise ValueError("H166 research source/module/data/gpu provenance not qualified")
    for field in ("trainable_parameters","nonzero_gradient_tensors",
                  "updated_parameter_tensors","optimizer_states"):
        if type(model.get(field)) is not int or model[field]<1:
            raise ValueError("H166 measured visual gradient/optimizer states absent")
    if (model["updated_parameter_tensors"]>model["optimizer_states"] or
        model["nonzero_gradient_tensors"]>model["optimizer_states"] or
        model.get("finite_synthetic_loss") is not True):
        raise ValueError("H166 visual update counters inconsistent")
    for field in ("peak_allocated_gib","peak_reserved_gib"):
        value=model.get(field)
        if type(value) not in (float,int) or not math.isfinite(value) or value<=0:
            raise ValueError("H166 actual T4 memory measurement invalid")
    if (model["peak_allocated_gib"]>MAX_PEAK_GIB or
        model["peak_reserved_gib"]<model["peak_allocated_gib"] or
        job.get("peak_allocated_gib")!=model["peak_allocated_gib"] or
        job.get("trainable_parameters")!=model["trainable_parameters"]):
        raise ValueError("H166 T4 peak cap or kernel/model receipt mismatch")
    return {
        "status":"H166_INDEPENDENT_FREE_T4_VISUAL_MERGER_SYNTHETIC_PROBE_PASS",
        "trainable_parameters":model["trainable_parameters"],
        "selected_modules":selected,
        "nonzero_gradient_tensors":model["nonzero_gradient_tensors"],
        "updated_parameter_tensors":model["updated_parameter_tensors"],
        "peak_allocated_gib":model["peak_allocated_gib"],
        "peak_reserved_gib":model["peak_reserved_gib"],
        "optimizer_steps":1,
        "real_app_performance_measured":False,
        "new_ood_evaluations":0,"paid_GPU":False}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    args=p.parse_args()
    print(json.dumps(audit(args.output),indent=2,sort_keys=True))
