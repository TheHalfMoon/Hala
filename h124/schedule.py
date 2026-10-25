"""H124 disjoint-new-original-TRAIN source for joint model-produced START/END.

Select from 10k H71 original TRAIN only; exclude ALL 512 prior H105/H113
and ALL 1024 H123 TRAIN+heldout images. Includes NO previous heldout
endpoints, benchmark EVAL or hidden TEST in training or cohort choice.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from dataclasses import dataclass
import hashlib,json
from pathlib import Path

from h105.curriculum import propose,source_geometry
from h106.training_plan import H105_QUALIFIED_PLAN_SHA
from h113.dual_point import QuantizationQuarantine
from h122.source_bound import admit_original_source_map,original_for_sft
from h123.schedule import (derive as previous_h123,FAMILIES,
                            _balanced_choose,_holdout_indices,PER_FAMILY)
from h123.training_plan import SCHEDULE_SHA as H123_SCHEDULE_SHA
from h124.objective import from_original,SCHEMA

ORIGINAL=1024
TRAIN=896
HOLDOUT=128
PREVIOUS_H105=512
PREVIOUS_H123=1024
SEED="hala.h124.original-unseen-complex-joint-start-end-20261025.v1"


@dataclass(frozen=True)
class Schedule:
    train:tuple[dict,...]
    heldout:tuple[dict,...]
    train_indices:tuple[int,...]
    heldout_indices:tuple[int,...]
    summary:dict


def _hash(s):
    return hashlib.sha256(s.encode("utf8")).hexdigest()


def _jsonsha(rows):
    return hashlib.sha256(b"".join(
        json.dumps(x,sort_keys=True,ensure_ascii=False,
                   separators=(",",":")).encode("utf8")+b"\n"
        for x in rows)).hexdigest()


def derive(rows,root:Path):
    if type(rows) is not list or len(rows)!=10000:
        raise ValueError("H124 needs exact original H71 10k TRAIN SFT")
    sources=admit_original_source_map(root)
    older=propose(rows,sources,count=PREVIOUS_H105)
    previous=previous_h123(rows,root)
    if (older.sha256!=H105_QUALIFIED_PLAN_SHA or
            previous.summary.get("schedule_sha256")!=H123_SCHEDULE_SHA
            or len(previous.train)+len(previous.heldout)!=PREVIOUS_H123):
        raise ValueError("H124 all previous original source exclusions unverified")
    excluded_ids={rows[i]["id"] for i in older.indices}
    excluded_images={rows[i]["image_sha256"] for i in older.indices}
    for role in previous.train+previous.heldout:
        excluded_ids.add(role["original_source_id"])
        excluded_images.add(role["image_sha256"])
    if len(excluded_ids)!=PREVIOUS_H105+PREVIOUS_H123 or len(excluded_images)!=len(excluded_ids):
        raise ValueError("H124 prior original TRAIN source lineage overlaps/drifts")
    pool=defaultdict(list)
    seen_raw=set()
    for index,sft in enumerate(rows):
        if (sft.get("split")!="train" or sft.get("id") in seen_raw
                or type(sft.get("id")) is not str):
            raise ValueError("H124 original TRAIN ID/dataset invariant changed")
        seen_raw.add(sft["id"])
        if (sft["id"] in excluded_ids or
                sft["image_sha256"] in excluded_images or
                sft.get("family") not in FAMILIES):
            continue
        try:
            original=original_for_sft(sft,sources)
        except ValueError as error:
            if str(error)!="H122 original source geometry was excluded by H105":
                raise
            continue
        features=source_geometry(sft,original)
        if features is None:
            continue
        try:
            role=from_original(sft,original)
        except QuantizationQuarantine:
            continue
        pool[sft["family"]].append({
            "index":index,"source_id":sft["id"],"sha":sft["image_sha256"],
            "direction":features["direction"],"scale":features["scale"],
            "end_cell":features["end_cell"],"role":role})
    if set(pool)!=set(FAMILIES):
        raise ValueError("H124 source original complex domains incomplete")
    train=[];heldout=[];train_index=[];heldout_index=[]
    used=set();per_family={}
    for family in FAMILIES:
        selected=_balanced_choose(pool[family],PER_FAMILY)
        withheld=_holdout_indices(selected)
        for i,record in enumerate(selected):
            if record["sha"] in used or record["sha"] in excluded_images:
                raise ValueError("H124 new source reused previous H113/H123/heldout image")
            used.add(record["sha"])
            if i in withheld:
                heldout.append(record["role"]);heldout_index.append(record["index"])
            else:
                train.append(record["role"]);train_index.append(record["index"])
        per_family[family]={
            "train":PER_FAMILY-len(withheld),
            "heldout":len(withheld),
            "supported_drag_directions":len({x["direction"] for x in pool[family]}),
            "supported_original_END_regions":len({tuple(x["end_cell"]) for x in pool[family]})}
    if (len(train)!=TRAIN or len(heldout)!=HOLDOUT
            or len(used)!=ORIGINAL
            or len(set(train_index)&set(heldout_index))!=0
            or any(role["language"]!="en" for role in train+heldout)):
        raise ValueError("H124 exactly 896/128 new English original TRAIN cohort failed")
    train_sha=_jsonsha(train)
    heldout_sha=_jsonsha(heldout)
    source_sha=_hash(SEED+H105_QUALIFIED_PLAN_SHA+
                     H123_SCHEDULE_SHA+train_sha+heldout_sha)
    summary={
      "schema":SCHEMA,"source_split":"TRAIN_ONLY",
      "new_original_TRAIN_images":ORIGINAL,
      "source_train_images":TRAIN,"source_heldout_images":HOLDOUT,
      "all_previous_H105_H123_images_excluded":PREVIOUS_H105+PREVIOUS_H123,
      "previous_original_H105_sha":H105_QUALIFIED_PLAN_SHA,
      "previous_H123_schedule_sha":H123_SCHEDULE_SHA,
      "per_complex_family":per_family,
      "train_private_sha256":train_sha,
      "heldout_private_sha256":heldout_sha,
      "schedule_sha256":source_sha,
      "objective":"one-shot model-generated START+END+signed_delta",
      "english_complex_only":True,"no_original_EVAL_TEST":True,
      "heldout_never_optimizer":True,
      "no_original_source_or_model_published":True,
      "trained_model_or_improvement_claimed":False}
    return Schedule(tuple(train),tuple(heldout),
                    tuple(train_index),tuple(heldout_index),summary)
