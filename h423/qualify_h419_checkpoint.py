"""H423 fail-closed recovery qualifier for REAL interrupted H419 private GPU.

H419 uses the original frozen 37,808-image gradient order and saves genuine
LoRA adapter-only checkpoints at 1,500-step increments. If Kaggle expires
before 9,452 steps, a checkpoint's mere existence is NOT proof 9,452 steps
completed. Recover only a model whose original provider GPU log proves the
claimed checkpoint boundary and whose exact safe tensor bytes are verified.

This source-only qualifier NEVER trains, creates a model, claims an
uninterrupted optimizer history, performs clinical GUI actions, or releases
weights. Adapter files/logs stay owner-private outside GitHub.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import struct

H419_KERNEL="abdulazizshehri/hala-h419-private-30k-native-drag-continuation"
H419_TOTAL=9452
H419_MICRO=4
H419_INTERVAL=1500
H419_SOURCE_ROWS_SHA="e7fce082d6ccd931515beacbc6cfc2455a6683c7c92566a245fda9f2182a1605"
H419_SOURCE_RECEIPT_SHA="b7b75d6aa2587c3eba83a3934fe4cdf3cacd48a5b970b5a60a640c89ee3054cf"
H417_TRAINED_PARENT_SHA="11f06aac6c225b86d1f30272eb852bca49fd82ce6e8553113b39b4cba4216df1"
H417_PARENT_RECEIPT_SHA="227967230362eea759553ae5af892865bae4a3625947c45f30c0d9cb36f9f76a"
BASE="Qwen/Qwen3-VL-2B-Instruct"
BASE_REV="89644892e4d85e24eaac8bacfd4f463576704203"
OUT_SCHEMA="hala.h423.interrupted-h419-actual-private-checkpoint-qualification.v1"
CHECKPOINT_LOG_RE=re.compile(
    r"H419_REAL_ORIGINAL_30000_NATIVE_GRADIENT\s+(\d+)\s*/\s*9452\s+"
    r"image_conditioned_samples_seen\s+(\d+)\s+loss\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


def file_sha(path):
    d=sha256()
    with Path(path).open("rb") as src:
        for piece in iter(lambda:src.read(1024*1024),b""):
            d.update(piece)
    return d.hexdigest()


def exact_json(value):
    def no_dupe(items):
        obj={}
        for k,v in items:
            if k in obj:raise ValueError("H423 duplicate JSON keys in original model source")
            obj[k]=v
        return obj
    def finite_only(x):
        raise ValueError("H423 nonfinite original loss")
    return json.loads(value,object_pairs_hook=no_dupe,parse_constant=finite_only)


def bounded_step(step):
    if type(step) is not int or step not in range(H419_INTERVAL,H419_TOTAL,H419_INTERVAL):
        raise ValueError("H423 only original 1500-step private checkpoint boundaries accepted")
    return step


def actual_log_progress(raw,checkpoint_step):
    """Require CONTIGUOUS real original 150-step GPU progress through boundary."""
    bounded_step(checkpoint_step)
    if type(raw) is bytes:raw=raw.decode("utf8",errors="replace")
    if type(raw) is not str or len(raw)>8_000_000:
        raise ValueError("H423 original Kaggle provider run log absent or too large")
    observed={}
    for m in CHECKPOINT_LOG_RE.finditer(raw):
        step=int(m.group(1));examples=int(m.group(2))
        if (step%150 or step<=0 or step>H419_TOTAL
            or examples!=step*H419_MICRO):
            raise ValueError("H423 original provider log inconsistent GPU step/examples")
        loss=float(m.group(3))
        if not (0<=loss<1_000_000):
            raise ValueError("H423 original provider log has invalid native DRAG loss")
        if step in observed:
            raise ValueError("H423 original Kaggle progress row duplicated")
        observed[step]=loss
    wanted=set(range(150,checkpoint_step+1,150))
    if not wanted.issubset(observed):
        missing=sorted(wanted-set(observed))
        raise ValueError("H423 original provider log has not independently shown all steps: "+str(missing[:6]))
    return {"last_logged_step":max(observed),"verified_checkpoint_step":checkpoint_step,
            "log_150_step_records":len(observed),
            "loss_at_verified_boundary":observed[checkpoint_step]}


def safe_adapter_headers(path):
    """Inspect adapter's real immutable safetensors layout without loading weights."""
    path=Path(path)
    size=path.stat().st_size
    if size<100_000 or size>100_000_000:
        raise ValueError("H423 checkpoint adapter safetensors filesize implausible")
    with path.open("rb") as f:
        prefix=f.read(8)
        if len(prefix)!=8:
            raise ValueError("H423 missing original safetensors header")
        n=struct.unpack("<Q",prefix)[0]
        if not 32<=n<=2_000_000 or size<8+n:
            raise ValueError("H423 original model safetensors header bounds invalid")
        header=exact_json(f.read(n).decode("utf8"))
    if type(header) is not dict:
        raise ValueError("H423 original model adapter not a tensor dictionary")
    entries={k:v for k,v in header.items() if k!="__metadata__"}
    if len(entries)<16 or len(entries)>4096:
        raise ValueError("H423 original LoRA tensor count implausible")
    intervals=[]
    for k,v in entries.items():
        if "lora_" not in k or type(v) is not dict or set(v)!={"dtype","shape","data_offsets"}:
            raise ValueError("H423 original model has non-LoRA or malformed tensor")
        dims=v["shape"]
        offsets=v["data_offsets"]
        if (v["dtype"] not in ("F16","F32","BF16")
            or type(dims) is not list or not dims
            or any(type(z) is not int or z<=0 for z in dims)
            or type(offsets) is not list or len(offsets)!=2
            or any(type(z) is not int for z in offsets)
            or not 0<=offsets[0]<offsets[1]<=size-8-n):
            raise ValueError("H423 original model safetensors offsets/dtype invalid")
        bytes_per=2 if v["dtype"]!="F32" else 4
        cells=1
        for dim in dims:cells*=dim
        if cells*bytes_per!=offsets[1]-offsets[0]:
            raise ValueError("H423 original tensor byte length inconsistent with shape")
        intervals.append(tuple(offsets))
    if sorted(intervals)!=[(a,b) for a,b in sorted(intervals)]:
        raise ValueError("H423 unexpected safetensors interval sorting")
    intervals.sort()
    if intervals[0][0]!=0 or intervals[-1][1]!=size-8-n:
        raise ValueError("H423 safetensors not fully byte-covered by valid tensors")
    for before,after in zip(intervals,intervals[1:]):
        if before[1]!=after[0]:
            raise ValueError("H423 overlapping or missing original adapter tensor bytes")
    return {"real_lora_tensor_count":len(entries),
            "real_lora_bytes":size,"exact_safetensors_tensors_cover_all_bytes":True}


def qualify(checkpoint_dir,original_provider_log,source_version):
    checkpoint_dir=Path(checkpoint_dir)
    if type(source_version) is not str or not re.fullmatch(r"\d+",source_version) or int(source_version)<1:
        raise ValueError("H423 exact original private Kaggle source version required")
    path=checkpoint_dir/"adapter_model.safetensors"
    cfg=checkpoint_dir/"adapter_config.json"
    if path.parent.name not in ("step-1500","step-3000","step-4500","step-6000","step-7500","step-9000"):
        raise ValueError("H423 original H419 checkpoint folder is not from 1500-step evidence")
    step=bounded_step(int(path.parent.name.removeprefix("step-")))
    if not path.is_file() or not cfg.is_file():
        raise ValueError("H423 original Kaggle provider LoRA weights/config missing")
    adapter=safe_adapter_headers(path)
    c=exact_json(cfg.read_text("utf8"))
    if (type(c) is not dict or c.get("peft_type")!="LORA"
        or c.get("base_model_name_or_path")!=BASE
        or type(c.get("r")) is not int or not 1<=c["r"]<=256):
        raise ValueError("H423 original LoRA config not exact qualified Qwen base")
    logs=Path(original_provider_log)
    progress=actual_log_progress(logs.read_bytes(),step)
    return {
       "schema":OUT_SCHEMA,
       "status":"PARTIAL_H419_PRIVATE_CHECKPOINT_ONLY_NOT_FULLY_TRAINED",
       "source_notebook_slug":H419_KERNEL,
       "source_notebook_version":source_version,
       "original_log_sha256":file_sha(logs),
       "original_model_checkpoint_step":step,
       "verified_original_source_gradient_examples":step*H419_MICRO,
       "remaining_optimizer_steps_to_original_h419_plan":H419_TOTAL-step,
       "remaining_grad_examples_to_original_plan":(H419_TOTAL-step)*H419_MICRO,
       "exact_shuffled_order_resume_offset":step*H419_MICRO,
       "checkpoint_adapter_model_sha256":file_sha(path),
       "checkpoint_adapter_config_sha256":file_sha(cfg),
       "original_source_30k_rows_sha256":H419_SOURCE_ROWS_SHA,
       "original_source_30k_receipt_sha256":H419_SOURCE_RECEIPT_SHA,
       "h417_real_parent_adapter_sha256":H417_TRAINED_PARENT_SHA,
       "h417_real_parent_receipt_sha256":H417_PARENT_RECEIPT_SHA,
       "base_model":BASE,
       "base_revision":BASE_REV,
       **adapter,**progress,
       "optimizer_state_saved_in_original_h419":False,
       "optimizer_reset_required_for_resumed_training":True,
       "uninterrupted_9452_step_run":False,
       "source_only_original_heldout_evaluation_completed":False,
       "genuine_independent_external_ood":False,
       "original_source_rights_independently_signed":False,
       "model_release_qualified":False,
       "weights_public":False,
    }


def store(receipt_root,doc):
    root=Path(receipt_root);root.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(doc,sort_keys=True,indent=2,allow_nan=False).encode("utf8")
    rec=root/"receipt.json";sha_file=root/"receipt.sha256"
    if rec.exists() or sha_file.exists():
        raise FileExistsError("H423 original interrupted private checkpoint receipt CREATE ONLY")
    try:
        with rec.open("xb") as f:f.write(payload)
        with sha_file.open("x",encoding="ascii",newline="\n") as f:f.write(sha256(payload).hexdigest())
    except BaseException:
        rec.unlink(missing_ok=True);sha_file.unlink(missing_ok=True)
        raise
    return sha256(payload).hexdigest()


def run():
    p=argparse.ArgumentParser()
    p.add_argument("--actual-h419-checkpoint-directory",required=True)
    p.add_argument("--original-private-kaggle-provider-log",required=True)
    p.add_argument("--original-private-kaggle-version",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    doc=qualify(a.actual_h419_checkpoint_directory,
                a.original_private_kaggle_provider_log,
                a.original_private_kaggle_version)
    receipt_sha=store(a.out,doc)
    print("H423_REAL_GPU_CHECKPOINT_PARTIAL_ONLY",
          json.dumps({"receipt_sha256":receipt_sha,
                      "checkpoint_step":doc["original_model_checkpoint_step"],
                      "actual_adapter_sha256":doc["checkpoint_adapter_model_sha256"],
                      "optimizer_reset_required":True,
                      "full_model_training_completed":False},sort_keys=True))

if __name__=="__main__":
    run()
