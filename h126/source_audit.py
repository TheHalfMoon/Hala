"""H126 NEW original TRAIN64 screenshot+instruction sufficiency diagnostics.

This is strictly a *source and task-definition audit*; heuristic intent
features cannot prove human or model visual identifiability.
Excludes all 2560 prior H105/H113/H123/H124 plus 64 H125 sources.
"""
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import hashlib,json,re,statistics

from h125.source import derive as derive_h125
from h125.eval_plan import get_plan as require_h125
from h125.source import FAMILIES,_balanced_choose
from h105.curriculum import source_geometry,propose as propose_h105
from h106.training_plan import H105_QUALIFIED_PLAN_SHA
from h123.schedule import derive as derive_h123
from h123.training_plan import SCHEDULE_SHA as H123_SOURCE_SHA
from h124.schedule import derive as derive_h124
from h124.training_plan import SCHEDULE_SHA as H124_SOURCE_SHA
from h113.dual_point import derive_two_independent_points,QuantizationQuarantine
from h122.source_bound import admit_original_source_map,original_for_sft

TOTAL=64
PER_FAMILY=16
PREVIOUS=2560+64
NONCE="hala.h126.never-before-used-original-TRAIN64-instruction-sufficiency-v1"

@dataclass(frozen=True)
class Source:
    cases:tuple[dict,...]
    indices:tuple[int,...]
    summary:dict


def sha_rows(rows):
    return hashlib.sha256(b"".join(
        json.dumps(row,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf8")+b"\n"
        for row in rows)).hexdigest()


def propose(rows,root:Path):
    if type(rows) is not list or len(rows)!=10000:
        raise ValueError("H126 genuine original H71 TRAIN source of exact size needed")
    h125=require_h125(derive_h125(rows,root))
    upstream=admit_original_source_map(root)
    h105=propose_h105(rows,upstream,count=512)
    h123=derive_h123(rows,root)
    h124=derive_h124(rows,root)
    if (h105.sha256!=H105_QUALIFIED_PLAN_SHA
            or h123.summary.get("schedule_sha256")!=H123_SOURCE_SHA
            or h124.summary.get("schedule_sha256")!=H124_SOURCE_SHA):
        raise ValueError("H126 all prior 2624 original source exclusion hashes required")
    old_ids={rows[i]["id"] for i in h105.indices}
    old_shas={rows[i]["image_sha256"] for i in h105.indices}
    for prior in (h123.train+h123.heldout+h124.train+h124.heldout):
        old_ids.add(prior["original_source_id"])
        old_shas.add(prior["image_sha256"])
    for prior in h125.cases:
        old_ids.add(prior["id"])
        old_shas.add(prior["image_sha256"])
    if len(old_ids)!=PREVIOUS or len(old_shas)!=PREVIOUS:
        raise ValueError("H126 must exclude all 2624 previous TRAIN and heldout screenshots")
    pool=defaultdict(list)
    for i,row in enumerate(rows):
        if row.get("split")!="train" or row["id"] in old_ids or row["image_sha256"] in old_shas:
            continue
        if row.get("family") not in FAMILIES:
            continue
        try:
            genuine=original_for_sft(row,upstream)
        except ValueError as exc:
            if str(exc)!="H122 original source geometry was excluded by H105":
                raise
            continue
        info=source_geometry(row,genuine)
        if info is None:
            continue
        try:
            pair=derive_two_independent_points(row,genuine)
        except QuantizationQuarantine:
            continue
        if any(x["original_source_id"]!=row["id"] for x in pair):
            raise ValueError("H126 source START or END identity drift")
        pool[row["family"]].append({
            "index":i,"sha":row["image_sha256"],
            "source_id":row["id"],
            "direction":info["direction"],"scale":info["scale"],
            "end_cell":info["end_cell"],
            "role":{
               "id":row["id"],"split":"train","family":row["family"],
               "image_path":row["image_path"],
               "image_sha256":row["image_sha256"],
               "image_width":genuine["image_width"],
               "image_height":genuine["image_height"],
               "start_bbox":list(genuine["start_bbox"]),
               "end_bbox":list(genuine["end_bbox"]),
               "intent":genuine["intent"],
               "ordered":genuine["ordered"]
            }})
    if set(pool)!=set(FAMILIES):
        raise ValueError("H126 four original complex TRAIN families unavailable")
    selected=[];indices=[];image_shas=set()
    for family in FAMILIES:
        cohort=_balanced_choose(pool[family],PER_FAMILY)
        if len(cohort)!=PER_FAMILY:
            raise ValueError("H126 genuine 16 NEW scenes per family absent")
        for sample in cohort:
            if sample["sha"] in image_shas or sample["sha"] in old_shas:
                raise ValueError("H126 duplicate original screenshot in new source audit")
            image_shas.add(sample["sha"])
            selected.append(sample["role"]);indices.append(sample["index"])
    if len(selected)!=TOTAL or len(set(indices))!=TOTAL or len(image_shas)!=TOTAL:
        raise ValueError("H126 new unique original TRAIN64 cohort incomplete")
    sha=sha_rows(selected)
    return Source(tuple(selected),tuple(indices),{
      "schema":"hala.h126.new-original-TRAIN64-task-definition-audit-source.v1",
      "original_source_scenes":TOTAL,"families":list(FAMILIES),
      "excluded_previous_source_screenshots":PREVIOUS,
      "h125_previous_source_sha":h125.summary["schedule_sha256"],
      "source_cases_SHA":sha,
      "schedule_sha256":hashlib.sha256(
          (NONCE+h125.summary["schedule_sha256"]+sha).encode("utf8")).hexdigest(),
      "TRAIN_only":True,"original_EVAL_or_TEST_used":False,
      "heldout_or_previous_source_trained_on":False,
      "model_inference_calls":0,"optimizer_steps":0
    })


NUMBER=re.compile(r"\b\d+(?:\.\d+)?\b")
DIRECTION=re.compile(r"\b(?:left|right|up|down|top|bottom|next|previous|forward|backward|to|towards|from|into|out of|over)\b",re.I)
SOURCE_REFERENT=re.compile(r"\b(?:slider|handle|cell|row|column|text|word|selection|image|picture|object|shape|edge|corner|window|element|item)\b",re.I)
TARGET_DEICTIC=re.compile(r"\b(?:there|here|that|this|selected|highlighted|marked|indicated|destination|target|requested|specific)\b",re.I)


def metadata_summary(cohort:Source):
    if len(cohort.cases)!=TOTAL or cohort.summary.get("original_source_scenes")!=TOTAL:
        raise ValueError("H126 exact original new source64 proof needed")
    families=defaultdict(list)
    for case in cohort.cases:
        text=case["intent"]
        if (type(text) is not str or not 4<=len(text)<=1200
                or type(case["image_width"]) is not int
                or type(case["image_height"]) is not int
                or not 64<=case["image_width"]<=1024
                or not 64<=case["image_height"]<=640):
            raise ValueError("H126 original instruction/geometry corrupted")
        src=case["start_bbox"];dst=case["end_bbox"]
        if any(type(v) is not int for v in src+dst):
            raise ValueError("H126 original target boxes require genuine integer pixel values")
        width=case["image_width"];height=case["image_height"]
        if any(not 0<=box[0]<=box[2]<width or not 0<=box[1]<=box[3]<height
               for box in (src,dst)):
            raise ValueError("H126 genuine original START/END source boxes out of image")
        families[case["family"]].append({
            "tokens":len(text.split()),
            "explicit_numeral":bool(NUMBER.search(text)),
            "direction_word":bool(DIRECTION.search(text)),
            "source_widget_word":bool(SOURCE_REFERENT.search(text)),
            "deictic_or_target_word":bool(TARGET_DEICTIC.search(text)),
            "start_box_width_px":src[2]-src[0]+1,
            "start_box_height_px":src[3]-src[1]+1,
            "end_box_width_px":dst[2]-dst[0]+1,
            "end_box_height_px":dst[3]-dst[1]+1,
            "start_box_fraction":(src[2]-src[0]+1)*(src[3]-src[1]+1)/(width*height),
            "end_box_fraction":(dst[2]-dst[0]+1)*(dst[3]-dst[1]+1)/(width*height)})
    if set(families)!=set(FAMILIES) or any(len(families[f])!=PER_FAMILY for f in FAMILIES):
        raise ValueError("H126 exact source domain balance missing")
    aggregate={}
    for family,items in sorted(families.items()):
        rec={"original_source_images":len(items)}
        for field in ("tokens","start_box_width_px","start_box_height_px",
                      "end_box_width_px","end_box_height_px",
                      "start_box_fraction","end_box_fraction"):
            rec["median_"+field]=round(statistics.median(x[field] for x in items),8)
        for field in ("explicit_numeral","direction_word","source_widget_word",
                      "deictic_or_target_word"):
            rec["text_with_"+field]=sum(x[field] for x in items)
        aggregate[family]=rec
    return {
      "schema":"hala.h126.original-new-TRAIN64-language-and-geometry-diagnostics.v1",
      "status":"SOURCE_ONLY_DESCRIPTIVE_AUDIT_NOT_VISUAL_IDENTIFIABILITY_PROOF",
      "original_untouched_screenshots":TOTAL,
      "excluded_all_prior_2624_source_images":PREVIOUS,
      "schedule_sha256":cohort.summary["schedule_sha256"],
      "source_cases_sha256":cohort.summary["source_cases_SHA"],
      "per_domain_aggregates":aggregate,
      "no_original_case_text_boxes_or_source_media_published":True,
      "semantic_identifiability_inferred":False,
      "new_optimizer_steps":0,"new_model_inference_calls":0
    }
