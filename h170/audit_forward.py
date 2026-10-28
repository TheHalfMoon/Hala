"""Independent H170 diagnostic audit of real frozen visual forward/logit effects."""
import argparse,json,math
from pathlib import Path
from h132.grounder_contract import MODEL_ID,MODEL_SHA
from h166.vision_probe import MAX_PEAK_GIB,EXPECTED_TORCH,EXPECTED_TRANSFORMERS,EXPECTED_PEFT
from h167.combined_probe import LANGUAGE_WEIGHTS_SHA
from h168.train_gpu import TRAIN_ZIP_SHA
from h169.infer_gpu import VISION_WEIGHTS_SHA
from h170.sensitivity_gpu import SCHEMA,TRAIN_CASE

SOURCE_SHA="b61a807965963ade9305d376020087ed7d1a3fe6"

def qualify_delta(v):
    if (not isinstance(v,dict)
        or set(v)!={"max_abs_delta","l2_delta","nonzero_elements","element_count"}):
        raise ValueError("H170 numerical delta schema invalid")
    a=v["max_abs_delta"];l=v["l2_delta"]
    n=v["nonzero_elements"];count=v["element_count"]
    if (type(a) not in (float,int) or type(l) not in (float,int)
        or not math.isfinite(a) or not math.isfinite(l)
        or a<0 or l<0
        or type(n) is not int or type(count) is not int
        or not 0<=n<=count or count<1
        or (n==0)!=(a==0)
        or (n==0)!=(l==0)):
        raise ValueError("H170 numeric forward delta inconsistent")
    return n>0

def audit(directory):
    root=Path(directory)
    job=json.loads((root/"h170-private-forward-job.json").read_text("utf8"))
    result=json.loads((root/"hala-h170-private-forward-sensitivity"/
                       "h170-forward-sensitivity-receipt.json").read_text("utf8"))
    if (job.get("schema")!="hala.h170.private-frozen-visual-forward-kaggle-job.v1"
        or job.get("status")!="H170_FROZEN_TRAIN_FORWARD_DIAGNOSTIC_COMPLETE"
        or job.get("source_commit")!=SOURCE_SHA or
        job.get("optimizer_steps")!=0 or
        job.get("paired_forward_calls")!=2 or
        job.get("dev_ood_model_calls")!=0 or
        job.get("paid_gpu") is not False or
        job.get("weights_published") is not False or
        job.get("flagship_35b_unchanged") is not True):
        raise ValueError("H170 private GPU job provenance not qualified")
    if (result.get("schema")!=SCHEMA
        or result.get("status")!="H170_FROZEN_TRAIN_FORWARD_SENSITIVITY_MEASURED"
        or result.get("source_case_index")!=TRAIN_CASE
        or result.get("source_app")!="choices"
        or result.get("source_split")!="TRAIN"
        or result.get("original_train_zip_sha256")!=TRAIN_ZIP_SHA
        or result.get("frozen_language_sha256")!=LANGUAGE_WEIGHTS_SHA
        or result.get("vision_adapter_sha256")!=VISION_WEIGHTS_SHA
        or result.get("base_model")!=MODEL_ID
        or result.get("base_revision")!=MODEL_SHA
        or result.get("vision_merger_module","").endswith(
                    "visual.merger.linear_fc2") is not True
        or type(result.get("vision_adapter_parameter_l2_norm_sum")) not in (float,int)
        or result["vision_adapter_parameter_l2_norm_sum"]<=0
        or not math.isfinite(result["vision_adapter_parameter_l2_norm_sum"])
        or result.get("versions")!={"torch":EXPECTED_TORCH,
              "transformers":EXPECTED_TRANSFORMERS,"peft":EXPECTED_PEFT,
              "Pillow":"11.3.0"}
        or "T4" not in result.get("gpu","")
        or result.get("memory_cap_gib")!=MAX_PEAK_GIB
        or result.get("paired_forward_calls")!=2
        or result.get("optimizer_steps")!=0
        or result.get("dev_ood_model_calls")!=0
        or result.get("owner_gold_on_gpu_prompt") is not False
        or result.get("paid_gpu") is not False
        or result.get("weights_saved") is not False
        or result.get("real_browser_quality_measured") is not False
        or result.get("frozen_language_digest_before_after_equal") is not True
        or result.get("flagship_35b_unchanged") is not True):
        raise ValueError("H170 frozen weights, real model and TRAIN-only protocol mismatched")
    peak=result.get("peak_allocated_gib")
    if (type(peak) not in (float,int) or not math.isfinite(peak)
        or not 0<peak<=MAX_PEAK_GIB or job.get("peak_allocated_gib")!=peak):
        raise ValueError("H170 T4 memory cap or receipt mismatch")
    vis=qualify_delta(result.get("vision_merger_output_delta"))
    logits=qualify_delta(result.get("final_prompt_logit_delta"))
    if (result.get("vision_forward_effect_nonzero") is not vis
        or result.get("logit_forward_effect_nonzero") is not logits
        or job.get("vision_forward_effect_nonzero") is not vis
        or job.get("logit_forward_effect_nonzero") is not logits):
        raise ValueError("H170 declared effect vs numeric evidence mismatch")
    return dict(status="H170_INDEPENDENT_FROZEN_TRAIN_FORWARD_DIAGNOSTIC_VERIFIED",
        vision_adapter_loaded_nonzero=True,
        visual_forward_effect_nonzero=vis,
        final_token_logit_effect_nonzero=logits,
        max_visual_abs_delta=result["vision_merger_output_delta"]["max_abs_delta"],
        max_logit_abs_delta=result["final_prompt_logit_delta"]["max_abs_delta"],
        peak_allocated_gib=peak,
        new_ood_dev_calls=0,optimizer_steps=0,real_world_reward_measured=False)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    print(json.dumps(audit(parser.parse_args().output),indent=2,sort_keys=True))
