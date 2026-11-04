"""Fail-closed author-visible H421 original same-JPEG counterfactual scoring.

The two user instructions share the exact original JPEG but demand distinct
drag start/end actions. Judge model instruction sensitivity: BOTH correct
rather than separately counting single-task accuracy. All examples remain
owner-author-visible original source TRAIN-withheld research DEV, not
independent third-party OOD or production authorization.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path

from h419.kaggle.h419_native_drag_continuation import score_predicted

SOURCE_JSONL_SHA="42d93219ec4083920462d4194b175154f1f12d9b3e7b36ffd1068d8f8a4d96b9"
SOURCE_RECEIPT_SHA="02a4d17ead71fc06019a83669a04c9779430de5a7a48f800429ad2241d2ebaf4"
SAMPLE_N=384
IMAGE_PAIRS=192
DOMAINS={"slider":64,"slide_resize":64,"text_highlight":64}
SCHEMA="hala.h421.same-original-image-two-instructions.v1"


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for buf in iter(lambda:f.read(1024*1024),b""):
            h.update(buf)
    return h.hexdigest()


def load_sources(rows_path,receipt_path,sidecar_path):
    if digest(rows_path)!=SOURCE_JSONL_SHA or digest(receipt_path)!=SOURCE_RECEIPT_SHA:
        raise ValueError("H421 original DEV source JPEG/receipt SHA changed")
    if Path(sidecar_path).read_text("ascii").strip()!=SOURCE_RECEIPT_SHA:
        raise ValueError("H421 source receipt sidecar is not original Kaggle SHA")
    doc=json.loads(Path(receipt_path).read_text("utf8"))
    if (doc.get("schema")!="hala.h421.source-original-counterfactual-dev.v1" or
        doc.get("frozen_original_dev_rows_sha256")!=SOURCE_JSONL_SHA or
        doc.get("owner_heldout_source_image_pairs")!=IMAGE_PAIRS or
        doc.get("native_two_instruction_eval_cases")!=SAMPLE_N or
        doc.get("native_pairs_by_source_family")!=DOMAINS or
        doc.get("source_target_different_instructions_and_coordinates") is not True or
        doc.get("excluded_h417_and_h419_train_and_dev_source_images") is not True or
        doc.get("excluded_source_task_sha_of_h417_h419_models_and_dev") is not True or
        doc.get("original_jpeg_media_sha_replayed") is not True or
        doc.get("model_evaluation_executed") is not False or
        doc.get("model_training_executed") is not False or
        doc.get("independent_external_ood") is not False or
        doc.get("model_release_qualified") is not False):
        raise ValueError("H421 original counterfactual DEV source contract altered")
    rows=[]
    for line in Path(rows_path).read_text("utf8").splitlines():
        if not line.strip():continue
        row=json.loads(line)
        if (row.get("schema")!=SCHEMA or row.get("split")!="dev" or
            row.get("source_original_split")!="train" or
            type(row.get("pair_id")) is not int or
            not 0<=row["pair_id"]<IMAGE_PAIRS or
            type(row.get("alternative_index")) is not int or
            row["alternative_index"] not in (0,1) or
            row.get("source_domain") not in DOMAINS or
            row.get("source_original_jpeg_bytes_sha_verified") is not True or
            row.get("external_independent_ood") is not False):
            raise ValueError("H421 original source paired DEV row changed")
        rows.append(row)
    return validate_pairs(rows)


def validate_pairs(rows):
    if type(rows) is not list or len(rows)!=SAMPLE_N:
        raise ValueError("H421 missing 384 original paired source DEV cases")
    pairs={}
    counts=Counter()
    screen_sha=set()
    task_ids=set()
    for row in rows:
        p=row["pair_id"]
        q=row["alternative_index"]
        group=pairs.setdefault(p,{})
        if q in group:
            raise ValueError("H421 same original image repeated with same instruction slot")
        group[q]=row
        if row["id"] in task_ids:
            raise ValueError("H421 duplicate underlying evaluation action ID")
        task_ids.add(row["id"])
    if set(pairs)!=set(range(IMAGE_PAIRS)):
        raise ValueError("H421 entire frozen original source counterfactual DEV missing")
    for pair_id in range(IMAGE_PAIRS):
        group=pairs[pair_id]
        if set(group)!={0,1}:
            raise ValueError("H421 each heldout original image needs two tasks")
        a,b=group[0],group[1]
        if (a["image_sha256"]!=b["image_sha256"] or
            a["source_domain"]!=b["source_domain"] or
            a["intent"]==b["intent"] or
            a["native_drag_target"]==b["native_drag_target"] or
            float(a["same_image_pair_native_target_ratio"])<0.15 or
            float(b["same_image_pair_native_target_ratio"])<0.15):
            raise ValueError("H421 same original image target not instruction-contrastive")
        if a["image_sha256"] in screen_sha:
            raise ValueError("H421 repeated original screenshot across different pairs")
        screen_sha.add(a["image_sha256"])
        counts[a["source_domain"]]+=1
    if counts!=DOMAINS:
        raise ValueError("H421 frozen 64 per original action source family changed")
    return pairs


def calculate(rows,predictions):
    pairs=validate_pairs(rows)
    if type(predictions) is not list or len(predictions)!=SAMPLE_N:
        raise ValueError("H421 actual 384 generated native DRAG predictions required")
    mapping={}
    for p in predictions:
        if (type(p) is not dict or
            set(p)!={"id","image_sha256","raw_prediction"} or
            type(p["raw_prediction"]) is not str or
            len(p["raw_prediction"])>8192 or
            p["id"] in mapping):
            raise ValueError("H421 malformed or repeated real-model native output")
        mapping[p["id"]]=p
    stats={"image_pairs":IMAGE_PAIRS,"native_instructions":SAMPLE_N,
           "single_action_correct":0,"single_action_schema_valid":0,
           "both_actions_correct":0,"one_action_correct":0,
           "neither_action_correct":0,"collapsed_same_raw_action":0,
           "wrong_instruction_source_target":0,
           "by_domain":{},"independent_external_ood":False,
           "model_release_qualified":False}
    for i in range(IMAGE_PAIRS):
        a,b=pairs[i][0],pairs[i][1]
        ra=mapping.get(a["id"]);rb=mapping.get(b["id"])
        if (ra is None or rb is None or
            ra["image_sha256"]!=a["image_sha256"] or
            rb["image_sha256"]!=b["image_sha256"]):
            raise ValueError("H421 original model output source-image pairing drift")
        sa=score_predicted(ra["raw_prediction"],a["native_drag_target"])
        sb=score_predicted(rb["raw_prediction"],b["native_drag_target"])
        wrong_a=score_predicted(ra["raw_prediction"],b["native_drag_target"])
        wrong_b=score_predicted(rb["raw_prediction"],a["native_drag_target"])
        both=int(sa["correct"])+int(sb["correct"])
        stats["single_action_correct"]+=both
        stats["single_action_schema_valid"]+=int(sa["valid"])+int(sb["valid"])
        stats["both_actions_correct"]+=int(both==2)
        stats["one_action_correct"]+=int(both==1)
        stats["neither_action_correct"]+=int(both==0)
        stats["collapsed_same_raw_action"]+=int(
            ra["raw_prediction"].strip()==rb["raw_prediction"].strip())
        stats["wrong_instruction_source_target"]+=int(wrong_a["correct"])+int(wrong_b["correct"])
        by=stats["by_domain"].setdefault(a["source_domain"],{
            "pairs":0,"both":0,"single_correct":0})
        by["pairs"]+=1;by["both"]+=int(both==2);by["single_correct"]+=both
    if (stats["both_actions_correct"]+
        stats["one_action_correct"]+
        stats["neither_action_correct"])!=IMAGE_PAIRS:
        raise ValueError("H421 counterfactual original source pair arithmetic invalid")
    return stats
