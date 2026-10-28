"""H167 independent strict audit of combined frozen-H164 + visual LoRA T4 receipts."""
import argparse
import json
import math
from pathlib import Path

from h132.grounder_contract import MODEL_ID,MODEL_SHA
from h167.combined_probe import (LANGUAGE_ZIP_SHA,LANGUAGE_WEIGHTS_SHA,
                                LANGUAGE_CONFIG_SHA,LANGUAGE_README_SHA,VISUAL_RANK)
from h166.vision_probe import (
    EXPECTED_TORCH,EXPECTED_TRANSFORMERS,EXPECTED_PEFT,MAX_PEAK_GIB)

SOURCE_COMMIT="94cb785ad6fcc61bf079d2022337efdc2a3707b1"

def audit(root):
    root=Path(root)
    job=json.loads((root/"h167-combined-job-receipt.json").read_text("utf8"))
    model=json.loads((root/"hala-h167-private-combined-t4"/
                      "h167-combined-synthetic-t4-receipt.json").read_text("utf8"))
    if (job.get("schema")!="hala.h167.private-combined-language-visual-kaggle-job.v1"
        or job.get("status")!="H167_SYNTHETIC_COMBINED_FREE_T4_PASS"
        or job.get("source_commit")!=SOURCE_COMMIT
        or job.get("optimizer_steps")!=2
        or job.get("owner_gold_used") is not False
        or job.get("previous_ood_or_dev_used") is not False
        or job.get("new_ood_evaluations")!=0
        or job.get("paid_gpu") is not False
        or job.get("model_weights_published") is not False
        or job.get("flagship_35b_unchanged") is not True):
        raise ValueError("H167 private Kaggle job provenance failed")
    if (model.get("schema")!="hala.h167.private-frozen-h164-plus-vision-synthetic-t4.v1"
        or model.get("status")!="H167_TWO_LANGUAGE_SYNTHETIC_STEPS_PASS"
        or model.get("base_model")!=MODEL_ID or model.get("base_revision")!=MODEL_SHA
        or model.get("language_adapter_zip_sha256")!=LANGUAGE_ZIP_SHA
        or model.get("language_adapter_weights_sha256")!=LANGUAGE_WEIGHTS_SHA
        or model.get("language_adapter_config_sha256")!=LANGUAGE_CONFIG_SHA
        or model.get("language_adapter_readme_sha256")!=LANGUAGE_README_SHA
        or model.get("language_adapter_frozen_before_after_match") is not True
        or model.get("active_adapters")!=["language","vision"]
        or model.get("visual_rank")!=VISUAL_RANK
        or model.get("trainable_visual_parameters")!=57344
        or not isinstance(model.get("frozen_language_parameters"),int)
        or model["frozen_language_parameters"]<=0
        or model.get("optimizer_steps")!=2
        or model.get("memory_cap_gib")!=MAX_PEAK_GIB
        or model.get("environment")!={"torch":EXPECTED_TORCH,
             "transformers":EXPECTED_TRANSFORMERS,"peft":EXPECTED_PEFT,
             "Pillow":"11.3.0"}
        or "T4" not in model.get("gpu","")
        or model.get("data")!="synthetic-only-no-original-app"
        or model.get("original_app_training_cases")!=0
        or model.get("owner_gold_used") is not False
        or model.get("previous_ood_or_dev_used") is not False
        or model.get("new_ood_evaluations")!=0
        or model.get("real_app_performance_measured") is not False
        or model.get("weights_saved") is not False
        or model.get("paid_gpu") is not False
        or model.get("flagship_35b_unchanged") is not True):
        raise ValueError("H167 frozen language + vision synthetic model receipt failed")
    modules=model.get("visual_target_modules")
    if (type(modules) is not list or len(modules)!=2
        or sorted(modules)!=["model.visual.merger.linear_fc1",
                            "model.visual.merger.linear_fc2"]):
        raise ValueError("H167 vision targets unqualified")
    rows=model.get("bilingual_synthetic")
    if (type(rows) is not list or len(rows)!=2
        or [r.get("language") for r in rows]!=["en","ar"]):
        raise ValueError("H167 two bilingual synthetic steps not proven")
    for r in rows:
        if (type(r.get("nonzero_grad_tensors")) is not int
            or not 1<=r["nonzero_grad_tensors"]<=4
            or type(r.get("updated_tensors")) is not int
            or not 1<=r["updated_tensors"]<=4
            or r.get("synthetic_finite_loss") is not True):
            raise ValueError("H167 missing true visual gradient/update")
    allocated=model.get("peak_allocated_gib")
    reserved=model.get("peak_reserved_gib")
    if (type(allocated) not in (float,int) or
        type(reserved) not in (float,int) or
        not math.isfinite(allocated) or not math.isfinite(reserved)
        or not 0<allocated<=MAX_PEAK_GIB or reserved<allocated
        or job.get("peak_allocated_gib")!=allocated
        or job.get("trainable_visual_parameters")!=model["trainable_visual_parameters"]):
        raise ValueError("H167 T4 memory cap / independent receipt mismatch")
    return dict(status="H167_INDEPENDENT_COMBINED_BILINGUAL_SYNTHETIC_T4_PASS",
                steps=2,peak_allocated_gib=allocated,peak_reserved_gib=reserved,
                frozen_language_parameters=model["frozen_language_parameters"],
                trainable_visual_parameters=model["trainable_visual_parameters"],
                languages=["en","ar"],frozen_language_unchanged=True,
                new_ood_evaluations=0,real_app_performance_measured=False)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    args=p.parse_args()
    print(json.dumps(audit(args.output),indent=2,sort_keys=True))
