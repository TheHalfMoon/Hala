"""H125 source-only NEW 64 original GUI grounding images, 16/domain.

Exclude ALL previous 512 H105/H113 +1024 H123 +1024 H124
TRAIN AND heldout original screenshots by ID AND genuine media SHA.
No prior heldout labels, original EVAL or hidden TEST in baseline.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import hashlib,json
from pathlib import Path
from h105.curriculum import propose,source_geometry
from h106.training_plan import H105_QUALIFIED_PLAN_SHA
from h113.dual_point import derive_two_independent_points,QuantizationQuarantine
from h122.source_bound import admit_original_source_map,original_for_sft
from h123.schedule import FAMILIES,_balanced_choose,derive as derive_h123
from h123.training_plan import SCHEDULE_SHA as H123_SHA
from h124.schedule import derive as derive_h124
from h124.training_plan import SCHEDULE_SHA as H124_SHA

PER_FAMILY=16
TOTAL=PER_FAMILY*len(FAMILIES)
OLD=512+1024+1024
NONCE="hala.h125.new-original64-gui-pretrained-2b-nofinetune-20261026.v1"

@dataclass(frozen=True)
class Schedule:
    cases:tuple[dict,...]
    indices:tuple[int,...]
    summary:dict


def derive(rows,root:Path):
    if type(rows) is not list or len(rows)!=10000:
        raise ValueError("H125 require original H71 exactly 10000 TRAIN labels")
    originals=admit_original_source_map(root)
    h105=propose(rows,originals,count=512)
    h123=derive_h123(rows,root)
    h124=derive_h124(rows,root)
    if (h105.sha256!=H105_QUALIFIED_PLAN_SHA or
            h123.summary.get("schedule_sha256")!=H123_SHA or
            h124.summary.get("schedule_sha256")!=H124_SHA):
        raise ValueError("H125 all past original TRAIN and holdout exclusions must qualify")
    previous_ids={rows[idx]["id"] for idx in h105.indices}
    previous_shas={rows[idx]["image_sha256"] for idx in h105.indices}
    for role in (h123.train+h123.heldout+h124.train+h124.heldout):
        previous_ids.add(role["original_source_id"])
        previous_shas.add(role["image_sha256"])
    if len(previous_ids)!=OLD or len(previous_shas)!=OLD:
        raise ValueError("H125 old original TRAIN image SHA source overlap")
    pool=defaultdict(list);seen=set()
    for idx,row in enumerate(rows):
        if (type(row.get("id")) is not str or row["id"] in seen
                or row.get("split")!="train"):
            raise ValueError("H125 genuine original TRAIN source identities invalid")
        seen.add(row["id"])
        if (row["id"] in previous_ids or row["image_sha256"] in previous_shas
                or row.get("family") not in FAMILIES):
            continue
        try:
            src=original_for_sft(row,originals)
        except ValueError as exc:
            if str(exc)!="H122 original source geometry was excluded by H105":
                raise
            continue
        features=source_geometry(row,src)
        if features is None:continue
        try:
            pair=derive_two_independent_points(row,src)
        except QuantizationQuarantine:
            continue
        pool[row["family"]].append({
          "index":idx,"source_id":row["id"],"sha":row["image_sha256"],
          "direction":features["direction"],"scale":features["scale"],
          "end_cell":features["end_cell"],
          "case":{
             "id":row["id"],"split":"train","family":row["family"],
             "language":row["language"],"image_path":row["image_path"],
             "image_sha256":row["image_sha256"],
             "source_image_width":src["image_width"],
             "source_image_height":src["image_height"],
             "intent":src["intent"],
             "start_prompt":pair[0]["prompt"],"end_prompt":pair[1]["prompt"]
          }})
    if set(pool)!=set(FAMILIES):
        raise ValueError("H125 exact four untouched GUI families unavailable")
    selected=[];indices=[];shas=set();families={}
    for family in FAMILIES:
        cohort=_balanced_choose(pool[family],PER_FAMILY)
        if len(cohort)!=PER_FAMILY:
            raise ValueError("H125 exact original 16 per GUI domain required")
        for item in cohort:
            if item["sha"] in shas or item["sha"] in previous_shas:
                raise ValueError("H125 previous model TRAIN or heldout SHA reused")
            shas.add(item["sha"])
            selected.append(item["case"]);indices.append(item["index"])
        families[family]=PER_FAMILY
    if (len(selected)!=TOTAL or len(set(indices))!=TOTAL
            or len(shas)!=TOTAL or any(x["language"]!="en" for x in selected)):
        raise ValueError("H125 original heldout English complex source count/identity invalid")
    digest=hashlib.sha256(b"".join(
        json.dumps(case,sort_keys=True,separators=(",",":"),ensure_ascii=False)
        .encode("utf8")+b"\n" for case in selected)).hexdigest()
    schedule_sha=hashlib.sha256(
        (NONCE+H105_QUALIFIED_PLAN_SHA+H123_SHA+H124_SHA+digest)
        .encode("utf8")).hexdigest()
    return Schedule(tuple(selected),tuple(indices),{
       "schema":"hala.h125.pretrained-gui-2b-original-train64-source-only.v1",
       "source_split":"TRAIN_ONLY_FRESH_NEVER_TRAINED",
       "new_original_images":TOTAL,
       "per_family":families,
       "prior_original_screenshots_excluded":OLD,
       "prior_h123_schedule_sha":H123_SHA,
       "prior_h124_schedule_sha":H124_SHA,
       "private_selected_source_sha256":digest,
       "schedule_sha256":schedule_sha,
       "models":["osunlp/UGround-V1-2B","showlab/ShowUI-2B"],
       "model_pretraining_only":True,
       "Hala_H125_optimizer_steps":0,
       "all_models_same_paired_images":True,
       "original_EVAL_or_TEST_used":False,
       "no_prior_heldout_reuse":True,
       "no_heldout_optimizer_input":True,
       "no_model_or_source_published":True})
