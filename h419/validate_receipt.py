"""H419 fail-closed ORIGINAL Kaggle model receipt & 240 held-out DRAG audit.

Bilateral source-check: 144 untouched H417 DEV image-SHA + 96 fresh
source-original DEV image-SHA, all source re-derived from nine byte-SHA
pinned owner-private original TRAIN ZIPs, with H419 new 30K TRAIN images
excluded. This is only author-visible source-holdout evidence, NEVER
a third-party independent OOD or approval to execute native GUI actions.
"""
from __future__ import annotations
import argparse
import json
import math
import hashlib
from pathlib import Path
import zipfile

from h419.kaggle import h419_native_drag_continuation as h

ROOT=Path(__file__).resolve().parent/"verified_receipt"/"h419-owner-private-30k-native-drag"
PASS=h.H419_PRIVATE_STATUS
FAIL=h.H419_FAILURE_STATUS
SCHEMA="hala.h419.private-new-30000-original-native-drag.v1"


def sha(value:bytes):
    return hashlib.sha256(value).hexdigest()


def _nonneg_int(x):
    return type(x) is int and x>=0


def _source_metrics(pairs,dev,zips):
    if type(pairs) is not list or len(pairs)!=len(dev):
        raise ValueError("H419 model original generated native predictions missing")
    computed={}
    for actual,case in zip(pairs,dev):
        if type(actual) is not dict or set(actual)!={
            "id","family","domain","language","image_sha256",
            "raw_prediction","correct","valid","error"}:
            raise ValueError("H419 generated action record has unqualified properties")
        for field,source in (
            ("id",case["id"]),("family",case["kind"]),
            ("domain",case["domain"]),("language",case["language"]),
            ("image_sha256",case["image_sha256"])):
            if actual[field]!=source or type(actual[field]) is not str:
                raise ValueError("H419 prediction original source provenance/order drift")
        raw=actual["raw_prediction"]
        if type(raw) is not str or len(raw)>8192:
            raise ValueError("H419 generated native DRAG text invalid")
        im,width,height=h.source_image(case,zips)
        target=h.target(case,width,height)
        result=h.score_predicted(raw,target)
        for key in ("correct","valid","error"):
            if type(actual[key]) is not type(result[key]) or actual[key]!=result[key]:
                raise ValueError("H419 action correctness did not replay against original JPEG")
        for key in ("all",case["language"],case["domain"],case["kind"]):
            metrics=computed.setdefault(key,{"n":0,"correct":0,"valid":0})
            metrics["n"]+=1
            metrics["correct"]+=int(result["correct"])
            metrics["valid"]+=int(result["valid"])
    return computed


def _paired_receipt_source(data):
    a=data.get("parent_dev_exact")
    b=data.get("adapted_dev_exact")
    if type(a) is not list or type(b) is not list or len(a)!=h.H419_DEV_N or len(b)!=len(a):
        raise ValueError("H419 exact 240 parent/after predictions incomplete")
    for x,y in zip(a,b):
        if type(x) is not dict or type(y) is not dict:
            raise ValueError("H419 exact paired native prediction absent")
        for field in ("id","family","domain","language","image_sha256"):
            if (type(x.get(field)) is not str or
                x.get(field)!=y.get(field)):
                raise ValueError("H419 parent/trained source-image pairing forged")
        for z in (x,y):
            if (type(z.get("correct")) is not bool or
                type(z.get("valid")) is not bool or
                z.get("error") is not None and type(z.get("error")) is not str or
                type(z.get("raw_prediction")) is not str):
                raise ValueError("H419 generated model prediction schema invalid")


def _check_model_contract(data):
    if (type(data) is not dict or data.get("schema")!=SCHEMA
        or data.get("status")!=PASS
        or data.get("official_base")!=h.BASE
        or data.get("official_revision")!=h.REV
        or data.get("source_revision")!=h.SOURCE_REV
        or data.get("h417_original_parent_adapter_sha256")!=h.PARENT_ADAPTER_SHA
        or data.get("h417_original_parent_receipt_sha256")!=h.PARENT_RECEIPT_SHA
        or data.get("h418_original_source_receipt_sha256")!=h.H418_RECEIPT_SHA
        or data.get("h418_original_source_rows_sha256")!=h.H418_ROWS_SHA
        or data.get("source_only_firstparty_and_drag_train") is not True
        or data.get("h417_train_source_reused_for_en") is not False
        or data.get("h417_dev_images_used_for_gradient") is not False
        or data.get("genuinely_independent_external_ood") is not False
        or data.get("independent_source_rights_signoff") is not False
        or data.get("model_release_qualified") is not False
        or data.get("weights_public") is not False):
        raise ValueError("H419 independent source/rights/release provenance invalid")
    if (data.get("actual_original_archive_sha256")!=h.H419_ALL_SOURCE_SHAS
        or data.get("train_unique_new_original_images")!=h.H419_NEW_IMAGE_N
        or data.get("train_unique_arabic_browser_images")!=h.H419_AR_N
        or data.get("train_unique_images")!=h.H419_UNIQUE_IMAGES
        or data.get("unique_train_images")!=h.H419_UNIQUE_IMAGES
        or data.get("train_example_passes")!=h.H419_TRAINING_PASSES
        or data.get("processed_training_examples")!=h.H419_TRAINING_PASSES
        or data.get("extra_ar_repeats")!=h.H419_AR_ADDITIONAL_PASSES
        or data.get("source_disjoint_old_h417_dev_images")!=h.H419_OLD_DEV_N
        or data.get("source_disjoint_fresh_original_dev_images")!=h.H419_NEW_DEV_N
        or data.get("actual_optimizer_steps")!=h.H419_STEPS
        or data.get("completed_steps")!=h.H419_STEPS
        or data.get("device_name")!="Tesla T4"
        or data.get("train_languages")!={"en":30000,"ar":976}):
        raise ValueError("H419 true 30K ORIGINAL gradient and 240 dev evidence missing")
    if (type(data.get("adapter_sha256")) is not str or
        len(data["adapter_sha256"])!=64 or
        type(data.get("adapter_bytes")) is not int or
        data["adapter_bytes"]<=1000 or
        type(data.get("changed_lora_tensors")) is not int or
        data["changed_lora_tensors"]<1):
        raise ValueError("H419 actual changed LoRA model bytes not proved")
    losses=data.get("weighted_train_losses")
    if (type(losses) is not list or len(losses)!=h.H419_STEPS or
        any(type(v) not in (int,float) or not math.isfinite(v) or v<0
            for v in losses)):
        raise ValueError("H419 no complete actual finite native coordinate loss")
    _paired_receipt_source(data)
    for key in ("parent_dev","adapted_dev"):
        stats=data.get(key)
        if (type(stats) is not dict or stats.get("all",{}).get("n")!=h.H419_DEV_N
            or stats.get("ar",{}).get("n")!=24 or
            stats.get("en",{}).get("n")!=h.H419_DEV_N-24):
            raise ValueError("H419 actual paired source-held-out metrics incomplete")
        if not all(_nonneg_int(v) for x in stats.values() for v in x.values()):
            raise ValueError("H419 negative or noninteger native dev score")
    old_parent=sum(x["correct"] for x in data["parent_dev_exact"][:h.H419_OLD_DEV_N])
    old_post=sum(x["correct"] for x in data["adapted_dev_exact"][:h.H419_OLD_DEV_N])
    fresh_parent=sum(x["correct"] for x in data["parent_dev_exact"][h.H419_OLD_DEV_N:])
    fresh_post=sum(x["correct"] for x in data["adapted_dev_exact"][h.H419_OLD_DEV_N:])
    if any((data.get(key) is not number if type(number) is bool
            else data.get(key)!=number) for key,number in [
             ("old_h417_dev_parent_correct",old_parent),
             ("old_h417_dev_adapted_correct",old_post),
             ("fresh_new_dev_parent_correct",fresh_parent),
             ("fresh_new_dev_adapted_correct",fresh_post)]):
        raise ValueError("H419 old/new original DEV score sums inaccurate")
    strict_gate=bool(fresh_post>fresh_parent and old_post>=old_parent-4)
    if data.get("source_only_research_gate") is not strict_gate:
        raise ValueError("H419 forged source-only improvement/nonregression gate")
    return strict_gate


def verify(path=ROOT, source_root=None, adapter_file=None):
    path=Path(path)
    raw=(path/"receipt.json").read_bytes()
    digest=sha(raw)
    if (path/"receipt.sha256").read_text("ascii").strip()!=digest:
        raise ValueError("H419 original Kaggle receipt sidecar mismatch")
    data=h.strict_load(raw.decode("utf8"))
    if data.get("status")==FAIL:
        if type(data.get("failure_class")) is not str:
            raise ValueError("H419 GPU failed without actual failure class")
        return {"receipt_sha256":digest,"status":FAIL,"qualified":False,
                "failure_class":data["failure_class"]}
    gate=_check_model_contract(data)
    if adapter_file is not None:
        file=Path(adapter_file)
        if not file.is_file() or h.sha_file(file)!=data["adapter_sha256"] or file.stat().st_size!=data["adapter_bytes"]:
            raise ValueError("H419 exact original trained model safetensors SHA mismatch")
    result={
       "receipt_sha256":digest,"status":PASS,
       "qualified":False,"source_only_research_gate":gate,
       "original_30000_new_images":h.H419_NEW_IMAGE_N,
       "real_optimizer_steps":h.H419_STEPS,
       "actual_gradient_examples":h.H419_TRAINING_PASSES,
       "adapter_sha256":data["adapter_sha256"],
       "model_weights_sha_verified":adapter_file is not None,
       "source_jpeg_replayed":False,
       "parent_dev":data["parent_dev"],"adapted_dev":data["adapted_dev"],
       "external_independent_ood":False,
    }
    if source_root is not None:
        root=Path(source_root)
        sourcefiles={
            **{name:root/"h415"/name for name,_ in h.SOURCE_SHAS.values()},
            **{name:root/"h418"/name for name in h.H419_EXTRA_SHAS},
            h.BIL_FILE:root/"h415"/h.BIL_FILE,
            h.CURRIC_FILE:root/"h415"/h.CURRIC_FILE,
            h.H418_ROWS:root/"new-30k"/h.H418_ROWS,
            h.H418_RECEIPT_FILE:root/"new-30k"/h.H418_RECEIPT_FILE,
            h.PARENT_ADAPTER_FILE:root/"parent"/h.PARENT_ADAPTER_FILE,
            h.PARENT_RECEIPT_FILE:root/"parent"/h.PARENT_RECEIPT_FILE,
            "h417_original_receipt.sha256":root/"parent"/"h417_original_receipt.sha256"
        }
        if any(not p.is_file() for p in sourcefiles.values()):
            raise ValueError("H419 owner original private source bundle missing")
        old=h.only_input_file
        zips=None
        extras=[]
        try:
            h.only_input_file=lambda name:sourcefiles[name]
            if h.parent_model_source().name!=h.PARENT_ADAPTER_FILE:
                raise ValueError("H419 source parent LoRA lineage changed")
            zips,extras,train,dev,order,old_dev,new_dev=h.h419_prepare_sources()
            if (len(train),len(dev),len(order),len(old_dev),len(new_dev))!=(
                    30976,240,37808,144,96):
                raise ValueError("H419 source original partitions no longer consistent")
            if data["old_h417_dev_sha256"]!=h.sha_bytes("".join(
                    sorted(c["image_sha256"] for c in old_dev)).encode()):
                raise ValueError("H419 old144 author-visible original source DEV fingerprints shifted")
            if data["new_dev_sha256"]!=h.sha_bytes("".join(
                    sorted(c["image_sha256"] for c in new_dev)).encode()):
                raise ValueError("H419 fresh96 original unseen DEV fingerprints shifted")
            for label in ("parent","adapted"):
                scores=_source_metrics(data[label+"_dev_exact"],dev,zips)
                if scores!=data[label+"_dev"]:
                    raise ValueError("H419 original source-image DRAG metrics not reproducible")
            result["source_jpeg_replayed"]=True
        finally:
            h.only_input_file=old
            for x in extras:x.close()
            if zips:
                for x in zips.values():
                    if isinstance(x,zipfile.ZipFile):x.close()
    result["evidence_scope"]="owner author-visible TRAIN source SHA-disjoint 240 DEV; not independent third-party OOD"
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--receipt-root",default=str(ROOT))
    p.add_argument("--sources",default=None)
    p.add_argument("--adapter",default=None)
    args=p.parse_args()
    print(json.dumps(verify(args.receipt_root,args.sources,args.adapter),
                     sort_keys=True))
