"""H422 TRUE model-error analysis from the ORIGINAL H417 private T4 receipt.

Unlike training-plan speculation, this independently re-aggregates all 144
actual pretrained and fine-tuned 2-point image-conditioned DRAG predictions
from H417's SHA-pinned Kaggle GPU receipt. It exposes only aggregate
synthetic-source research scores, not model weights, source JPEGs or raw
predictions. This is owner-visible research DEV, NOT a sealed independent
external test. H419 output may change the priority ordering.
"""
from __future__ import annotations

from collections import Counter,defaultdict
from hashlib import sha256
import argparse
import json
from pathlib import Path

REAL_H417_RECEIPT_SHA="227967230362eea759553ae5af892865bae4a3625947c45f30c0d9cb36f9f76a"
REAL_H417_SCHEMA="hala.h417.private-12k-native-drag.v1"
REAL_H417_STATUS="COMPLETE_PRIVATE_NATIVE_DRAG_14880_PASSES"
REAL_SOURCE_DOMAINS=("sheet","slider","slide_resize","text_highlight")
REAL_DEV_PER_DOMAIN=24
RESEARCH_SCHEMA="hala.h422.real-h417-original-native-drag-errors.v1"


def strict_json(raw):
    def unique(items):
        obj={}
        for k,v in items:
            if k in obj:raise ValueError("H422 real H417 GPU receipt has duplicate JSON key")
            obj[k]=v
        return obj
    def no_nonfinite(x):
        raise ValueError("H422 actual original GPU metric non-finite")
    return json.loads(raw,object_pairs_hook=unique,parse_constant=no_nonfinite)


def verify_real_h417(receipt):
    data=Path(receipt).read_bytes()
    if sha256(data).hexdigest()!=REAL_H417_RECEIPT_SHA:
        raise ValueError("H422 cannot attribute model errors to any unverified H417 GPU receipt")
    r=strict_json(data.decode("utf8"))
    if (r.get("schema")!=REAL_H417_SCHEMA
        or r.get("status")!=REAL_H417_STATUS
        or r.get("official_base")!="Qwen/Qwen3-VL-2B-Instruct"
        or r.get("official_revision")!="89644892e4d85e24eaac8bacfd4f463576704203"
        or r.get("actual_optimizer_steps")!=3720
        or r.get("processed_training_examples")!=14880
        or r.get("source_only_dev_improved") is not True
        or r.get("model_release_qualified") is not False
        or r.get("weights_public") is not False):
        raise ValueError("H422 underlying H417 actual GPU source receipt contract altered")
    groups={}
    for name in ("base_dev_exact","adapted_dev_exact"):
        rows=r.get(name)
        if type(rows) is not list or len(rows)!=144:
            raise ValueError("H422 missing complete actual H417 144 generated model predictions")
        ids=set()
        image_shas=set()
        domains=defaultdict(Counter)
        families=defaultdict(Counter)
        languages=defaultdict(Counter)
        errors=Counter()
        for row in rows:
            if (type(row) is not dict or
                type(row.get("id")) is not str or
                type(row.get("image_sha256")) is not str
                or len(row["image_sha256"])!=64
                or type(row.get("raw_prediction")) is not str
                or type(row.get("valid")) is not bool
                or type(row.get("correct")) is not bool
                or row.get("domain") not in REAL_SOURCE_DOMAINS
                or row.get("family") not in ("original","firstparty")
                or row.get("language") not in ("en","ar")
                or row["id"] in ids or row["image_sha256"] in image_shas):
                raise ValueError("H422 original H417 real GPU case or unique source image invalid")
            ids.add(row["id"])
            image_shas.add(row["image_sha256"])
            if row["correct"] and not row["valid"]:
                raise ValueError("H422 cannot score exact source native action with invalid JSON")
            if row["family"]=="original" and row["language"]!="en":
                raise ValueError("H422 original frozen source DragOn DEV language wrong")
            error=row.get("error")
            if (row["correct"] and error is not None
                or not row["correct"] and error not in ("schema","coordinates","ordered_mismatch")):
                raise ValueError("H422 H417 original source-metric failure reason corrupted")
            errors["correct" if row["correct"] else error]+=1
            for key,table in ((row["domain"],domains),(row["family"],families),(row["language"],languages)):
                table[key]["n"]+=1
                table[key]["valid"]+=int(row["valid"])
                table[key]["correct"]+=int(row["correct"])
        if (len(ids)!=144 or len(image_shas)!=144 or
            {key:val["n"] for key,val in domains.items()}!={"sheet":34,"slider":36,"slide_resize":34,"text_highlight":40}
            or {key:val["n"] for key,val in families.items()}!={"original":96,"firstparty":48}
            or {key:val["n"] for key,val in languages.items()}!={"ar":24,"en":120}):
            raise ValueError("H422 actual H417 source holdout partition or unique images drift")
        if r.get(name.removesuffix("_exact"),{}).get("all")!={
            "n":144,"valid":sum(val["valid"] for val in domains.values()),
            "correct":sum(val["correct"] for val in domains.values())
        }:
            raise ValueError("H422 H417 original GPU receipt aggregate disagrees with per-image predictions")
        groups[name]={
            "all":{k:sum(v[k] for v in domains.values()) for k in ("n","valid","correct")},
            "by_source_domain":{k:dict(v) for k,v in sorted(domains.items())},
            "by_source_family":{k:dict(v) for k,v in sorted(families.items())},
            "by_language":{k:dict(v) for k,v in sorted(languages.items())},
            "errors":dict(sorted(errors.items())),
        }
    true_original=defaultdict(Counter)
    for row in r["adapted_dev_exact"]:
        if row["family"]!="original":continue
        v=true_original[row["domain"]]
        v["n"]+=1;v["valid"]+=int(row["valid"]);v["correct"]+=int(row["correct"])
        if not row["correct"]:v["incorrect"]+=1
        if row["error"]=="coordinates":v["coordinate_errors"]+=1
        if row["error"]=="ordered_mismatch":v["ordered_errors"]+=1
    if any(true_original[d]["n"]!=REAL_DEV_PER_DOMAIN for d in REAL_SOURCE_DOMAINS):
        raise ValueError("H422 original 24 per family source image counts changed")
    priority=sorted(REAL_SOURCE_DOMAINS,
                    key=lambda k:(true_original[k]["correct"],
                                  -true_original[k]["coordinate_errors"],k))
    return {
        "schema":RESEARCH_SCHEMA,
        "status":"REAL_T4_H417_ORIGINAL_SOURCE_DEV_ERRORS_NOT_NEW_MODEL_TRAINING",
        "source_h417_genuine_gpu_original_receipt_sha256":REAL_H417_RECEIPT_SHA,
        "original_H417_completed_optimizer_steps":3720,
        "original_H417_true_image_gradient_passes":14880,
        "original_H417_trained_native_model":True,
        "source_holdout_scope":"author-visible original source family + firstparty synthetic",
        "source_total_original_drag_dev_images":96,
        "source_total_firstparty_dev_images":48,
        "base":groups["base_dev_exact"],
        "genuinely_finetuned_H417":groups["adapted_dev_exact"],
        "original_drag_only_domain_errors":{k:dict(v) for k,v in sorted(true_original.items())},
        "evidence_only_priority_for_future_curriculum":priority,
        "H419_trained_model_evaluated_by_this_report":False,
        "H421_unseen_source_counterfactual_dev_evaluated":False,
        "external_independent_ood":False,
        "model_release_qualified":False,
    }


def save(out,report):
    path=Path(out)
    path.mkdir(parents=True,exist_ok=True)
    original=json.dumps(report,sort_keys=True,indent=2,
                        ensure_ascii=False,allow_nan=False).encode("utf8")
    receipt=path/"receipt.json";sidecar=path/"receipt.sha256"
    if receipt.exists() or sidecar.exists():
        raise FileExistsError("H422 actual H417 model error analysis create-only")
    try:
        with receipt.open("xb") as f:f.write(original)
        with sidecar.open("x",encoding="ascii",newline="\n") as f:f.write(sha256(original).hexdigest())
    except BaseException:
        receipt.unlink(missing_ok=True);sidecar.unlink(missing_ok=True)
        raise
    return sha256(original).hexdigest()


def run():
    p=argparse.ArgumentParser()
    p.add_argument("--actual-h417-gpu-receipt",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    report=verify_real_h417(a.actual_h417_gpu_receipt)
    digest=save(a.out,report)
    print("H422_GENUINE_H417_SOURCE_MODEL_ERRORS",
          json.dumps({
            "real_original_T4_receipt_sha":REAL_H417_RECEIPT_SHA,
            "error_audit_sha":digest,
            "real_native_dev144":report["genuinely_finetuned_H417"]["all"],
            "original_domain_correct":{k:v["correct"] for k,v in
                report["original_drag_only_domain_errors"].items()},
            "original_domain_coordinate_errors":{k:v["coordinate_errors"] for k,v in
                report["original_drag_only_domain_errors"].items()},
            "research_hard_domain_priority":report["evidence_only_priority_for_future_curriculum"],
            "H419_eval_performed":False,
            "independent_external_OOD":False
          },sort_keys=True),flush=True)


if __name__=="__main__":
    run()
