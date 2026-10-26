"""H129 immutable NEW original TRAIN16 marker-feedback exploratory pilot.

Exploratory GPU inference only: preregistered new cases, never earlier H125/H126
or H123/H124 heldout, and no model tuning on any selected cases.
"""
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from h105.curriculum import propose as propose_h105
from h106.training_plan import H105_QUALIFIED_PLAN_SHA
from h123.schedule import derive as derive_h123
from h123.training_plan import SCHEDULE_SHA as H123_SHA
from h124.schedule import derive as derive_h124
from h124.training_plan import SCHEDULE_SHA as H124_SHA
from h125.source import derive as derive_h125
from h125.eval_plan import SCHEDULE_SHA as H125_SHA
from h126.source_audit import propose as derive_h126, FAMILIES
from h128.source import propose as derive_h128
from h122.source_bound import admit_original_source_map, original_for_sft

# Source schedule is pinned to independent H126 owner CPU evidence.
H128_SHA = "bd0f0099cc42a15d1fb11691964adbb2bd01ae13fbe2f039d2c8d714d3f30c2b"
TOTAL = 16
PER_FAMILY = 4
PREVIOUS = 512 + 1024 + 1024 + 64 + 64 + 16
NONCE = "hala.h129.original-new-train16-visual-marker-feedback-pilot-v1"
@dataclass(frozen=True)
class Schedule:
    cases: tuple
    summary: dict

def sha_cases(cases):
    return hashlib.sha256(b"".join(
        json.dumps(c,sort_keys=True,separators=(",",":"),ensure_ascii=False)
        .encode("utf8")+b"\n" for c in cases)).hexdigest()

def propose(rows,root:Path):
    if type(rows) is not list or len(rows)!=10000:
        raise ValueError("H129 exact genuine original 10000 TRAIN manifest needed")
    original=admit_original_source_map(root)
    h105=propose_h105(rows,original,count=512)
    h123=derive_h123(rows,root)
    h124=derive_h124(rows,root)
    h125=derive_h125(rows,root)
    h126=derive_h126(rows,root)
    h128=derive_h128(rows,root)
    if (h105.sha256!=H105_QUALIFIED_PLAN_SHA or
        h123.summary.get("schedule_sha256")!=H123_SHA or
        h124.summary.get("schedule_sha256")!=H124_SHA or
        h125.summary.get("schedule_sha256")!=H125_SHA or
        h126.summary.get("schedule_sha256")!="8a54ff0af4cfdeb9572ef741b205cbe377738eee0bbd630373ee95739b20ab2d" or
        h128.summary.get("schedule_sha256")!=H128_SHA):
        raise ValueError("H129 source exclusions must exactly match ALL earlier 2704 sources")
    old_ids={rows[i]["id"] for i in h105.indices}
    old_shas={rows[i]["image_sha256"] for i in h105.indices}
    for c in h123.train+h123.heldout+h124.train+h124.heldout:
        old_ids.add(c["original_source_id"])
        old_shas.add(c["image_sha256"])
    for c in h125.cases+h126.cases+h128.cases:
        old_ids.add(c["id"])
        old_shas.add(c["image_sha256"])
    if len(old_ids)!=PREVIOUS or len(old_shas)!=PREVIOUS:
        raise ValueError("H129 prior source ID/SHA disjoint exclusion incomplete")
    pool=defaultdict(list)
    for row in rows:
        if (row.get("split")!="train" or row.get("family") not in FAMILIES or
            row.get("id") in old_ids or row.get("image_sha256") in old_shas):
            continue
        try:
            src=original_for_sft(row,original)
        except ValueError as exc:
            if str(exc)!="H122 original source geometry was excluded by H105":
                raise
            continue
        if (src["id"]!=row["id"] or src["image_sha256"]!=row["image_sha256"]):
            raise ValueError("H129 mismatched original source identity")
        case={"id":row["id"],"split":"train","family":row["family"],
              "image_path":row["image_path"],"image_sha256":row["image_sha256"],
              "image_width":src["image_width"],"image_height":src["image_height"],
              "intent":src["intent"],"start_bbox":list(src["start_bbox"]),
              "end_bbox":list(src["end_bbox"])}
        key=hashlib.sha256((NONCE+src["image_sha256"]).encode("utf8")).hexdigest()
        pool[row["family"]].append((key,case))
    cases=[]; seen=set()
    for family in FAMILIES:
        if len(pool[family])<PER_FAMILY:
            raise ValueError("H129 insufficient new true TRAIN original GUI screenshots")
        for _,case in sorted(pool[family],key=lambda pair:pair[0])[:PER_FAMILY]:
            if case["image_sha256"] in seen:
                raise ValueError("H129 new GUI pilot repeated media SHA")
            seen.add(case["image_sha256"])
            cases.append(case)
    if len(cases)!=TOTAL or len(seen)!=TOTAL:
        raise ValueError("H129 exact fresh unique TRAIN16 required")
    digest=sha_cases(cases)
    schedule_sha=hashlib.sha256((NONCE+H128_SHA+digest).encode("utf8")).hexdigest()
    return Schedule(tuple(cases),{
        "schema":"hala.h129.original-disjoint-TRAIN16-cursor-marker-pilot-source.v1",
        "status":"SOURCE_ONLY_NOT_YET_GPU_EVALUATED","per_family":PER_FAMILY,
        "original_new_train_images":TOTAL,
        "all_previous_ID_and_SHAs_excluded":PREVIOUS,
        "h128_source_schedule_sha256":H128_SHA,
        "cases_sha256":digest,"schedule_sha256":schedule_sha,
        "prior_EVAL_TEST_or_heldout_reused":False,
        "optimizer_steps":0,
        "model_inference_calls":0,
    })
