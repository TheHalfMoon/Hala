"""H425 independent owner-private Kaggle T4 receipt + ORIGINAL heldout audit.

Does not run GPU and NEVER qualifies external OOD or production. All original
source media, trained adapters, raw predictions remain private to the owner.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import struct
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location(
    "h425_original_standalone",ROOT/"h425/kaggle/h425_h417_replay_paired.py")
MODEL=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)
SCHEMA=MODEL.H425_SCHEMA
COMPLETE=MODEL.H425_DONE
FAIL=MODEL.H425_FAIL
TRUE_SOURCE_NOT_RELEASE="source-only, not independent external OOD"

def sha(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()

def _int(v):
    return type(v) is int and v>=0

def _digest(v):
    return type(v) is str and len(v)==64 and all(x in "0123456789abcdef" for x in v)

def _metrics(predictions, reference=None, zips=None):
    """Check actual stored native drag predictions (and source if supplied)."""
    if type(predictions) is not list or len(predictions)!=240:
        raise ValueError("H425 240 original image-conditioned model predictions missing")
    rows={}
    for i,r in enumerate(predictions):
        if type(r) is not dict or set(r)!={
           "id","family","domain","language","image_sha256","raw_prediction",
           "correct","valid","error"}:
            raise ValueError("H425 generated prediction columns not frozen")
        if (not all(type(r[k]) is str for k in (
              "id","family","domain","language","image_sha256","raw_prediction"))
            or len(r["raw_prediction"])>8192
            or not _digest(r["image_sha256"])
            or type(r["correct"]) is not bool
            or type(r["valid"]) is not bool
            or (r["error"] is not None and type(r["error"]) is not str)):
            raise ValueError("H425 genuine model predictions malformed")
        if reference is not None:
            case=reference[i]
            if any(r[field]!=case[source] for field,source in [
                ("id","id"),("family","kind"),("domain","domain"),
                ("language","language"),("image_sha256","image_sha256")]):
                raise ValueError("H425 model prediction case or original media provenance changed")
            im,width,height=MODEL.source_image(case,zips)
            score=MODEL.score_predicted(r["raw_prediction"],MODEL.target(case,width,height))
            if any(r[key]!=score[key] for key in ("correct","valid","error")):
                raise ValueError("H425 model correctness did not replay from original source JPEG")
        for key in ("all",r["language"],r["domain"],r["family"]):
            m=rows.setdefault(key,{"n":0,"correct":0,"valid":0})
            m["n"]+=1;m["correct"]+=int(r["correct"]);m["valid"]+=int(r["valid"])
    return rows

def _eval_contract(data):
    if (type(data) is not dict or data.get("schema")!=SCHEMA
        or data.get("status")!=COMPLETE
        or data.get("official_base")!=MODEL.BASE
        or data.get("official_revision")!=MODEL.REV
        or data.get("h417_source_parent_sha256")!=MODEL.PARENT_ADAPTER_SHA
        or data.get("h417_source_parent_receipt_sha256")!=MODEL.PARENT_RECEIPT_SHA
        or data.get("h418_new_original_30k_source_manifest_sha256")!=MODEL.H418_RECEIPT_SHA
        or data.get("h422_original_pair_rows_sha256")!=MODEL.H422_ROWS_SHA
        or data.get("h422_original_paired_source_receipt_sha256")!=MODEL.H422_RECEIPT_SHA
        or data.get("h421_reserved_counterfactual_dev_sha256")!=MODEL.H421_DEV_SHA
        or data.get("original_nine_H60_source_zip_SHA256")!=MODEL.H419_ALL_SOURCE_SHAS
        or data.get("independent_external_ood") is not False
        or data.get("source_rights_review_qualified") is not False
        or data.get("release_qualified") is not False
        or data.get("weights_public") is not False
        or data.get("replaced_failed_h419_parent") is not True):
        raise ValueError("H425 model provenance or external release guard invalid")
    expected={
       "h425_unique_original_training_screens":MODEL.H425_UNIQUE_TRAIN_SCREENS,
       "h425_genuine_distinct_native_action_rows":MODEL.H425_NEW_PAIR_ACTIONS+MODEL.H425_OLD_TRAIN,
       "h425_original_same_screenshot_two_instructions":MODEL.H425_SCREENS,
       "h425_original_paired_optimizer_steps":8000,
       "h425_old_h417_english_replay_examples":MODEL.H425_OLD_EN,
       "h425_old_h417_arabic_replay_unique":MODEL.H425_OLD_AR,
       "h425_old_h417_arabic_passes":MODEL.H425_AR_TOTAL_PASSES,
       "h425_gradients_expected":MODEL.H425_SAMPLES,
       "h425_optimizer_steps_expected":MODEL.H425_STEPS,
       "untouched_old_h417_and_fresh_h419_original_dev_images":240,
       "actual_optimizer_steps":MODEL.H425_STEPS,
       "completed_optimizer_steps":MODEL.H425_STEPS,
       "processed_training_examples":MODEL.H425_SAMPLES,
       "actual_50784_gradient_examples":MODEL.H425_SAMPLES}
    for field,value in expected.items():
        if type(data.get(field)) is not int or data[field]!=value:
            raise ValueError("H425 source original training evidence incomplete: "+field)
    if (type(data.get("device_name")) is not str or data["device_name"]!="Tesla T4"
        or not _int(data.get("changed_lora_tensors"))
        or data["changed_lora_tensors"]<1
        or not _digest(data.get("model_weights_sha256"))
        or not _int(data.get("model_weights_bytes")) or data["model_weights_bytes"]<=1000
        or type(data.get("elapsed_seconds")) not in (int,float)
        or not math.isfinite(data["elapsed_seconds"]) or data["elapsed_seconds"]<=0):
        raise ValueError("H425 trained LoRA weight/optimizer/T4 evidence incomplete")
    snapshots=data.get("private_training_checkpoints_sha256")
    required_steps={str(step) for step in range(1500,MODEL.H425_STEPS+1,1500)}
    if (type(snapshots) is not dict or set(snapshots)!=required_steps
        or data.get("last_private_checkpoint_step")!=12000):
        raise ValueError("H425 all eight actually saved intermediate LoRA checkpoint receipts missing")
    for step,rec in snapshots.items():
        if (type(rec) is not dict or set(rec)!={
               "model_weights_sha256","model_weights_bytes","optimizer_step"}
            or not _digest(rec["model_weights_sha256"])
            or not _int(rec["model_weights_bytes"]) or rec["model_weights_bytes"]<=1000
            or type(rec["optimizer_step"]) is not int or rec["optimizer_step"]!=int(step)):
            raise ValueError("H425 genuine original intermediate LoRA checkpoint SHA/step invalid")
    losses=data.get("weighted_loss_history")
    if (type(losses) is not list or len(losses)!=MODEL.H425_STEPS
        or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in losses)):
        raise ValueError("H425 actual all 12696 gradient losses missing/nonfinite")
    before=_metrics(data.get("h417_parent_dev_exact"))
    after=_metrics(data.get("h425_after_dev_exact"))
    if (data.get("h417_parent_dev")!=before or data.get("h425_after_dev")!=after):
        raise ValueError("H425 claimed native DEV aggregate does not equal actual case predictions")
    old_before=sum(bool(x["correct"]) for x in data["h417_parent_dev_exact"][:144])
    new_before=sum(bool(x["correct"]) for x in data["h417_parent_dev_exact"][144:])
    old_after=sum(bool(x["correct"]) for x in data["h425_after_dev_exact"][:144])
    new_after=sum(bool(x["correct"]) for x in data["h425_after_dev_exact"][144:])
    for x,y in zip(data["h417_parent_dev_exact"],data["h425_after_dev_exact"]):
        if any(x[k]!=y[k] for k in ("id","family","domain","language","image_sha256")):
            raise ValueError("H425 H417 original parent and adapted evaluation task drift")
    if (old_before,new_before)!=(76,18):
        raise ValueError("H425 authentic H417 frozen original240 base failed reproduction")
    for field,num in (
        ("original_dev_old_parent_correct",old_before),
        ("original_dev_old_after_correct",old_after),
        ("original_dev_fresh_parent_correct",new_before),
        ("original_dev_fresh_after_correct",new_after),
        ("original_dev_parent_ar_correct",before["ar"]["correct"]),
        ("original_dev_after_ar_correct",after["ar"]["correct"])):
        if type(data.get(field)) is not int or data[field]!=num:
            raise ValueError("H425 DEV model original case count not truthful: "+field)
    valid=after["all"]["valid"]
    gate=old_after>=72 and new_after>18 and valid==240
    stronger=gate and old_after+new_after>old_before+new_before and (
        after["ar"]["correct"]>=before["ar"]["correct"]-1)
    if (data.get("source_only_H419_frozen_research_gate") is not gate
        or data.get("source_only_H425_stronger_improvement_gate") is not stronger):
        raise ValueError("H425 frozen ORIGINAL old/fresh/Arabic source-only gate fabricated")
    return old_before,new_before,old_after,new_after,gate,stronger

def _safetensors_sha(path:Path,receipt):
    if (not path.is_file() or path.stat().st_size!=receipt["model_weights_bytes"]
        or sha(path.read_bytes())!=receipt["model_weights_sha256"]):
        raise ValueError("H425 real trained private LoRA safetensors SHA or size mismatch")
    with path.open("rb") as f:
        count=struct.unpack("<Q",f.read(8))[0]
        if not 0<count<2_000_000:
            raise ValueError("H425 LoRA safetensors header length malformed")
        raw=f.read(count)
        header=json.loads(raw)
        if not any("lora_" in name for name in header if name!="__metadata__"):
            raise ValueError("H425 genuine LoRA trainable tensors missing from private weights")
    return True

def verify(receipt_root,adapter_file=None,source_root=None,paired_root=None,
           h421_root=None,checkpoints_root=None):
    root=Path(receipt_root)
    raw=(root/"receipt.json").read_bytes()
    digest=sha(raw)
    if (root/"receipt.sha256").read_text("ascii").strip()!=digest:
        raise ValueError("H425 original Kaggle receipt sidecar SHA mismatch")
    data=json.loads(raw)
    if data.get("status")==FAIL:
        if type(data.get("failure_class")) is not str:
            raise ValueError("H425 private GPU failed without authentic failure class")
        return {"receipt_sha256":digest,"status":FAIL,"failure_class":data["failure_class"],
                "source_only_qualified":False,"model_release_qualified":False}
    oldp,newp,old,fr,gate,stronger=_eval_contract(data)
    has_weights=adapter_file is not None
    if has_weights:_safetensors_sha(Path(adapter_file),data)
    snapshots_verified=False
    if checkpoints_root is not None:
        # Check EACH true owner LoRA safetensors from the provider output,
        # not just unverifiable checkpoint hashes transcribed in a receipt.
        folder=Path(checkpoints_root)
        for step,model_receipt in data["private_training_checkpoints_sha256"].items():
            _safetensors_sha(folder/("step-"+step)/"adapter_model.safetensors",
                             model_receipt)
        snapshots_verified=True
    source_verified=False
    if source_root is not None:
        if not paired_root or not h421_root:
            raise ValueError("H425 real original source replay requires private H422 pairs and H421 heldout")
        r=Path(source_root)
        paths={
          **{name:r/"h415"/name for name,_ in MODEL.SOURCE_SHAS.values()},
          **{name:r/"h418"/name for name in MODEL.H419_EXTRA_SHAS},
          MODEL.BIL_FILE:r/"h415"/MODEL.BIL_FILE,
          MODEL.CURRIC_FILE:r/"h415"/MODEL.CURRIC_FILE,
          MODEL.H418_ROWS:r/"new-30k"/MODEL.H418_ROWS,
          MODEL.H418_RECEIPT_FILE:r/"new-30k"/MODEL.H418_RECEIPT_FILE,
          MODEL.H422_ROWS_FILE:Path(paired_root)/MODEL.H422_ROWS_FILE,
          MODEL.H422_RECEIPT_FILE:Path(paired_root)/MODEL.H422_RECEIPT_FILE,
          MODEL.H421_DEV_FILE:Path(h421_root)/MODEL.H421_DEV_FILE,
        }
        if any(not p.is_file() for p in paths.values()):
            raise ValueError("H425 actual original nine private source ZIP or original DEV file absent")
        with patch.object(MODEL,"only_input_file",side_effect=lambda name:paths[name]):
            zips,extra,train,dev,order=MODEL.h425_prepare_source()
            try:
                if (len(train),len(dev),len(order))!=(43952,240,50784):
                    raise ValueError("H425 original source training/evaluation input length drift")
                for name in ("h417_parent_dev_exact","h425_after_dev_exact"):
                    stats=_metrics(data[name],reference=dev,zips=zips)
                    expected="h417_parent_dev" if name.startswith("h417") else "h425_after_dev"
                    if stats!=data[expected]:
                        raise ValueError("H425 actual model predictions do not match source JPEG")
                source_verified=True
            finally:
                for z in extra:z.close()
                for z in zips.values():
                    if hasattr(z,"close"):z.close()
    return {
        "status":COMPLETE,"receipt_sha256":digest,
        "actual_optimizer_steps":MODEL.H425_STEPS,
        "actual_original_image_conditioned_gradient_passes":MODEL.H425_SAMPLES,
        "original_training_images":MODEL.H425_UNIQUE_TRAIN_SCREENS,
        "old_parent_correct":oldp,"fresh_parent_correct":newp,
        "old_after_correct":old,"fresh_after_correct":fr,
        "original_source_only_gate":gate,
        "stronger_source_only_gate":stronger,
        "trained_adapter_sha_verified":has_weights,
        "all_eight_intermediate_checkpoint_shas_verified":snapshots_verified,
        "original_source_media_replayed":source_verified,
        "independent_external_ood":False,
        "model_release_qualified":False}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--receipt-root",type=Path,required=True)
    parser.add_argument("--adapter",type=Path)
    parser.add_argument("--sources",type=Path)
    parser.add_argument("--paired",type=Path)
    parser.add_argument("--h421",type=Path)
    parser.add_argument("--checkpoints-root",type=Path)
    p=parser.parse_args()
    print(json.dumps(verify(
        p.receipt_root,adapter_file=p.adapter,source_root=p.sources,
        paired_root=p.paired,h421_root=p.h421,
        checkpoints_root=p.checkpoints_root),sort_keys=True))
