"""Strict forensic receipt replay for ORIGINAL H417 private native drag model.

Model weights are NEVER copied into GitHub; passing any author-visible DEV
cannot itself establish independent OOD or approve autonomous GUI actions.
To replay exact gold geometry, supply owner-private --sources with the six
unchanged H415 Kaggle source files. Read-only source media are SHA checked.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

from h417.kaggle import h417_native_drag_lora as h

ROOT=Path(__file__).resolve().parent/"verified_receipt"/"h417-owner-private-native-drag"
PASS="COMPLETE_PRIVATE_NATIVE_DRAG_14880_PASSES"
FAIL="FAILED_PRIVATE_NATIVE_DRAG_NO_RELEASE"
SCHEMA="hala.h417.private-12k-native-drag.v1"


def sha(blob:bytes)->str:
    return hashlib.sha256(blob).hexdigest()


def _check_record_pair(a,b):
    if type(a) is not dict or type(b) is not dict:
        raise ValueError("H417 missing actual native model pair")
    frozen=("id","family","domain","language","image_sha256")
    for field in frozen:
        if a.get(field)!=b.get(field) or type(a.get(field)) is not str:
            raise ValueError("H417 before/after paired original image mismatch")
    for x in (a,b):
        if set(x)!={"id","family","domain","language","image_sha256",
                   "raw_prediction","correct","valid","error"}:
            raise ValueError("H417 unexpected native model output schema")
        if type(x["raw_prediction"]) is not str or len(x["raw_prediction"])>8192:
            raise ValueError("H417 original native model output text missing")
        if type(x["correct"]) is not bool or type(x["valid"]) is not bool:
            raise ValueError("H417 model results must carry strict Booleans")
        if x["error"] is not None and type(x["error"]) is not str:
            raise ValueError("H417 original generated result error malformed")


def _rescore(details,dev,zips):
    if type(details) is not list or len(details)!=len(dev):
        raise ValueError("H417 generated case count incomplete")
    totals={}
    for record,case in zip(details,dev):
        if (record["id"]!=case["id"] or record["family"]!=case["kind"]
            or record["domain"]!=case["domain"]
            or record["language"]!=case["language"]
            or record["image_sha256"]!=case["image_sha256"]):
            raise ValueError("H417 output provenance/order mismatch")
        im,w,height=h.source_image(case,zips)   # SHA-check real JPEG/PNG
        original_gold=h.target(case,w,height)
        recomputed=h.score_predicted(record["raw_prediction"],original_gold)
        for field in ("correct","valid","error"):
            if (type(record[field]) is not type(recomputed[field])
                or record[field]!=recomputed[field]):
                raise ValueError("H417 fabricated native endpoint score")
        for k in ("all",case["language"],case["domain"],case["kind"]):
            s=totals.setdefault(k,{"n":0,"correct":0,"valid":0})
            s["n"]+=1;s["correct"]+=int(recomputed["correct"])
            s["valid"]+=int(recomputed["valid"])
    return totals


def verify(path=ROOT,source_root=None):
    path=Path(path)
    raw=(path/"receipt.json").read_bytes()
    digest=sha(raw)
    if digest!=(path/"receipt.sha256").read_text("ascii").strip():
        raise ValueError("H417 original Kaggle receipt sidecar mismatch")
    state=h.strict_load(raw.decode("utf8"))
    if (type(state) is not dict or state.get("schema")!=SCHEMA
        or state.get("official_base")!=h.BASE
        or state.get("official_revision")!=h.REV
        or state.get("source_revision")!=h.SOURCE_REV
        or state.get("source_curriculum_sha256")!=h.DATASET_SHA
        or state.get("bilingual_labels_sha256")!=h.FIRSTPARTY_LABEL_SHA
        or state.get("private_sources_only") is not True
        or state.get("train_only_sources") is not True
        or state.get("benchmark_answers_used") is not False
        or state.get("full_upstream_tar_sha_verified") is not False
        or state.get("independent_external_ood") is not False
        or state.get("weights_public") is not False
        or state.get("model_release_qualified") is not False):
        raise ValueError("H417 real owner/source/release contract invalid")
    if state.get("status")==FAIL:
        if type(state.get("failure_class")) is not str:
            raise ValueError("H417 failed GPU receipt not qualified")
        return {"receipt_sha256":digest,"status":FAIL,"qualified":False,
                "failure_class":state["failure_class"]}
    if state.get("status")!=PASS:
        raise ValueError("H417 not a finished real native-drag fit")
    if (state.get("train_unique_images")!=h.TRAIN_UNIQUE
        or state.get("unique_train_images")!=h.TRAIN_UNIQUE
        or state.get("train_example_passes")!=h.EXPECTED_PASSES
        or state.get("processed_training_examples")!=h.EXPECTED_PASSES
        or state.get("dev_source_disjoint_images")!=144
        or state.get("actual_optimizer_steps")!=h.STEPS
        or state.get("completed_steps")!=h.STEPS
        or state.get("device_name")!="Tesla T4"
        or state.get("train_languages")!={"en":10976,"ar":976}
        or state.get("dev_languages")!={"en":120,"ar":24}):
        raise ValueError("H417 insufficient genuine data/optimizer/GPU evidence")
    repairs=state.get("original_image_collision_repairs")
    if type(repairs) is not list or len(repairs)!=28 or len(set(repairs))!=28:
        raise ValueError("H417 missing 28 source collision and sheet holdout repairs")
    if (type(state.get("adapter_sha256")) is not str
        or len(state["adapter_sha256"])!=64
        or type(state.get("adapter_bytes")) is not int
        or state["adapter_bytes"]<1000
        or type(state.get("changed_lora_tensors")) is not int
        or state["changed_lora_tensors"]<=0):
        raise ValueError("H417 actual saved model weights not proven")
    losses=state.get("weighted_train_losses")
    if (type(losses) is not list or len(losses)!=h.STEPS
        or any(type(v) not in (int,float) or not math.isfinite(v) or v<0
               for v in losses)):
        raise ValueError("H417 no complete finite weighted optimizer loss")
    for k in ("base_dev","adapted_dev"):
        details=state.get(k+"_exact")
        if type(details) is not list or len(details)!=144:
            raise ValueError("H417 source-disjoint native-drag DEV missing")
    for a,b in zip(state["base_dev_exact"],state["adapted_dev_exact"]):
        _check_record_pair(a,b)

    result={"receipt_sha256":digest,"status":PASS,"qualified":False,
            "train_unique_images":h.TRAIN_UNIQUE,
            "real_optimizer_steps":h.STEPS,
            "adapter_sha256":state["adapter_sha256"],
            "source_replayed":False,
            "base":state["base_dev"],"adapted":state["adapted_dev"]}
    if source_root is not None:
        source_root=Path(source_root)
        want={name for name,_ in h.SOURCE_SHAS.values()}
        want.update([h.CURRIC_FILE,h.BIL_FILE])
        if not all((source_root/name).is_file() for name in want):
            raise ValueError("H417 private original source bundle incomplete")
        original=h.only_input_file
        zips=None
        try:
            h.only_input_file=lambda name:source_root/name
            zips,originals,curriculum,firstparty=h.get_research_sources()
            train,dev,order=h.partition(originals,curriculum,firstparty)
            if (len(train),len(dev),len(order))!=(11952,144,14880):
                raise ValueError("H417 original source partition changed")
            if [r["id"] for r in train if r.get("source_repair")]!=repairs:
                raise ValueError("H417 28 replacement training image IDs altered")
            for label in ("base","adapted"):
                got=_rescore(state[label+"_dev_exact"],dev,zips)
                if got!=state[label+"_dev"]:
                    raise ValueError("H417 native model metrics not replayable")
            result["source_replayed"]=True
        finally:
            h.only_input_file=original
            if zips:
                for archive in zips.values():archive.close()

    source_improved=(state["adapted_dev"]["all"]["correct"] >
                     state["base_dev"]["all"]["correct"])
    if state.get("source_only_dev_improved") is not source_improved:
        raise ValueError("H417 false model improvement status")
    result["source_only_dev_improved"]=source_improved
    result["evidence_scope"]="author-visible image-disjoint TRAIN holdout, NOT external sealed OOD"
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--receipt-root",default=str(ROOT))
    parser.add_argument("--sources",default=None)
    options=parser.parse_args()
    print(json.dumps(verify(options.receipt_root,options.sources),sort_keys=True))
