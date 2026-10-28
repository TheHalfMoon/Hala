"""Independent private real-free-T4 H171 four-native-app vision training audit.

Only TRAIN receipts and private weights. No DEV/OOD, no native reward claims.
"""
import argparse,hashlib,json,math,struct
from collections import Counter
from pathlib import Path
from h132.grounder_contract import MODEL_ID,MODEL_SHA
from h166.vision_probe import (
    MAX_PEAK_GIB,EXPECTED_TORCH,EXPECTED_TRANSFORMERS,EXPECTED_PEFT)
from h167.combined_probe import LANGUAGE_WEIGHTS_SHA
from h168.train_gpu import TRAIN_ZIP_SHA

SOURCE_SHA="d2de3c9cc3cee75e8de55fe73f533fbf5d819c4f"
NEW_TRAIN_SHA="8690b8bf40db1de82dec3620cdbb02ca1ea7fca953f253b8e4208dd559c81235"
SCHEMA="hala.h171.64-original-4-app-two-epoch-real-t4-train.v1"
JOB_SCHEMA="hala.h171.owner-private-free-t4-four-app-job.v1"

def sha(blob):return hashlib.sha256(blob).hexdigest()

def vision_only_weights(path):
    blob=Path(path).read_bytes()
    if len(blob)<1024:
        raise ValueError("H171 private adapter file too small")
    hlen=struct.unpack("<Q",blob[:8])[0]
    if not 20<=hlen<=256000:
        raise ValueError("H171 private model safetensors header invalid")
    header=json.loads(blob[8:8+hlen])
    keys=[k for k in header if k!="__metadata__"]
    if (len(keys)!=4
        or any(".visual.merger." not in ("."+k+".")
               or not (".lora_A." in k or ".lora_B." in k)
               or "language_model" in k for k in keys)):
        raise ValueError("H171 saved checkpoint contains other than visual merger")
    return sha(blob)

def audit(root):
    root=Path(root)
    job=json.loads((root/"h171-private-train-job.json").read_text("utf8"))
    stage=root/"hala-h171-private-four-original-apps"
    report=json.loads((stage/"h171-training-receipt.json").read_text("utf8"))
    if (job.get("schema")!=JOB_SCHEMA or
        job.get("status")!="H171_128_REAL_FOUR_APP_ORIGINAL_T4_TRAIN_PASS"
        or job.get("source_commit")!=SOURCE_SHA or
        job.get("optimizer_steps")!=128 or job.get("dev_ood_model_calls")!=0
        or job.get("owner_dev_gold_used") is not False
        or job.get("previous_consumed_ood_used") is not False
        or job.get("paid_gpu") is not False
        or job.get("weights_published") is not False
        or job.get("flagship_35b_unchanged") is not True):
        raise ValueError("H171 private GPU job provenance invalid")
    if (report.get("schema")!=SCHEMA
        or report.get("status")!="H171_FOUR_APP_128_REAL_ORIGINAL_OPTIMIZER_STEPS_PASSED"
        or report.get("original_h168_train_zip_sha256")!=TRAIN_ZIP_SHA
        or report.get("new_h171_click_train_zip_sha256")!=NEW_TRAIN_SHA
        or report.get("h164_frozen_language_sha256")!=LANGUAGE_WEIGHTS_SHA
        or report.get("base_model")!=MODEL_ID
        or report.get("base_revision")!=MODEL_SHA
        or report.get("active_adapters")!=["language","vision"]
        or report.get("vision_rank")!=16
        or sorted(report.get("visual_target_modules",[]))!=[
            "model.visual.merger.linear_fc1","model.visual.merger.linear_fc2"]
        or report.get("real_optimizer_steps")!=128
        or report.get("unique_original_train_cases")!=64
        or report.get("exposures_per_case")!=2
        or report.get("train_click_cases")!=32
        or report.get("train_type_cases")!=16
        or report.get("train_drag_cases")!=16
        or report.get("train_arabic_cases")!=32
        or report.get("train_english_cases")!=32
        or report.get("distinct_original_app_families")!=4
        or report.get("trainable_visual_parameters")!=229376
        or report.get("frozen_language_parameters")!=5455872
        or report.get("frozen_language_unchanged") is not True
        or report.get("memory_cap_gib")!=MAX_PEAK_GIB
        or report.get("installed")!={"torch":EXPECTED_TORCH,
            "transformers":EXPECTED_TRANSFORMERS,"peft":EXPECTED_PEFT,"Pillow":"11.3.0"}
        or "T4" not in report.get("gpu","")
        or report.get("dev_ood_model_calls")!=0
        or report.get("old_consumed_eval_used") is not False
        or report.get("private_weights_only") is not True
        or report.get("paid_gpu") is not False
        or report.get("real_browser_quality_measured") is not False
        or report.get("flagship_35b_unchanged") is not True):
        raise ValueError("H171 actual real original app optimizer evidence failed")
    trace=report.get("optimizer_trace")
    if not isinstance(trace,list) or len(trace)!=128:
        raise ValueError("H171 128 optimizer traces missing")
    counts=Counter()
    meta=Counter()
    for position,row in enumerate(trace,1):
        ix=row.get("index")
        if type(ix) is not int or ix not in (
            set(range(168000,168016))|set(range(168100,168116))|
            set(range(171000,171016))|set(range(171100,171116))):
            raise ValueError("H171 original train source index leaked")
        if row.get("step")!=position or row.get("finite_loss") is not True:
            raise ValueError("H171 optimizer step index/loss missing")
        loss=row.get("loss")
        if (type(loss) not in (float,int) or not math.isfinite(loss)
            or not 0<loss<500
            or type(row.get("nonzero_visual_gradients")) is not int
            or not 1<=row["nonzero_visual_gradients"]<=4):
            raise ValueError("H171 real training gradient or loss invalid")
        expected=("choices","type") if ix<168100 else (
            ("nouislider","drag") if ix<171000 else (
                ("flatpickr","click") if ix<171100 else ("awesomplete","click")))
        if (row.get("app"),row.get("kind"))!=expected:
            raise ValueError("H171 original app native action kind altered")
        offset=ix%100
        lang="en" if offset%2==0 else "ar"
        if row.get("language")!=lang:
            raise ValueError("H171 bilingual original train source altered")
        counts[ix]+=1
        meta[lang]+=1
    if (len(counts)!=64 or set(counts.values())!={2}
        or meta!={"en":64,"ar":64}):
        raise ValueError("H171 exactly two exposures per all 64 original TRAIN failed")
    peak=report.get("peak_allocated_gib")
    reserved=report.get("peak_reserved_gib")
    if (type(peak) not in (int,float)
        or type(reserved) not in (int,float)
        or not math.isfinite(peak) or not math.isfinite(reserved)
        or not 0<peak<=MAX_PEAK_GIB or reserved<peak
        or job.get("peak_allocated_gib")!=peak):
        raise ValueError("H171 T4 peak memory or receipts mismatch")
    weights=list((stage/"private-rank16-vision").rglob("adapter_model.safetensors"))
    if len(weights)!=1:
        raise ValueError("H171 no exactly one saved private vision checkpoint")
    digest=vision_only_weights(weights[0])
    if (digest!=report.get("private_vision_weights_sha256")
        or digest!=job.get("private_vision_sha256")):
        raise ValueError("H171 actual frozen private weights checksum drift")
    return dict(
      status="H171_INDEPENDENT_FOUR_APP_128_REAL_T4_TRAIN_VERIFIED",
      optimizer_steps=128,original_unique_train_cases=64,
      original_app_families=4,click_cases=32,type_cases=16,drag_cases=16,
      arabic_cases=32,english_cases=32,
      visual_trainable_parameters=229376,
      peak_allocated_gib=peak,peak_reserved_gib=reserved,
      private_vision_sha256=digest,
      old_dev_ood_calls=0,new_dev_ood_calls=0,
      real_native_task_reward_measured=False)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True)
    print(json.dumps(audit(p.parse_args().output),indent=2,sort_keys=True))
