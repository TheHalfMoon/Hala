"""H105 original H71 TRAIN-only, image-unique, geometry-aware sampler.

Deterministic, source-label-bound, immutable. This builds a proposal for
training rather than training or modifying the frozen H71 curriculum.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math

from h102.oracle import in_bounds, inside

STRATEGY="hala.h105.source-bound-bilingual-family-direction-spatial-curriculum.v1"
SEED="hala.h105.first-actual-512-source-20261024"
COUNT=512


def keyhash(text):
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def source_geometry(row:dict,original:dict):
    """Require ORIGINAL TRAIN label and immutable completion to match boxes."""
    if (row.get("id")!=original.get("id") or row.get("split")!="train"
            or original.get("split")!="train" or original.get("action_type")!="drag"
            or row.get("image_sha256")!=original.get("image_sha256")
            or row.get("image_path")!=original.get("image_path")
            or type(original.get("ordered")) is not bool):
        raise ValueError("original TRAIN source identity mismatch")
    a=json.loads(row["completion"])
    w,h=original["image_width"],original["image_height"]
    if a!=original["completion"]:
        raise ValueError("TRAIN completion has been modified")
    if not in_bounds(original["start_bbox"],w,h) or not in_bounds(original["end_bbox"],w,h):
        return None
    start,end=a["start_point"],a["end_point"]
    if not inside(start,original["start_bbox"]) or not inside(end,original["end_bbox"]):
        raise ValueError("original source target outside true box")
    dx,dy=end[0]-start[0],end[1]-start[1]
    distance=math.hypot(dx,dy)
    if distance<=0:
        return None
    direction=int(((math.atan2(dy,dx)+math.pi/8)%(2*math.pi))/(math.pi/4))
    scale=("short" if distance<80 else "medium" if distance<240 else "long")
    start_cell=(min(3,start[0]*4//w),min(3,start[1]*4//h))
    end_cell=(min(3,end[0]*4//w),min(3,end[1]*4//h))
    if row["language"] not in ("ar","en") or not row["family"]:
        raise ValueError("original bilingual and family identity needed")
    return {"id":row["id"],"image_sha256":row["image_sha256"],
            "language":row["language"],"family":row["family"],
            "direction":direction,"scale":scale,
            "start_cell":start_cell,"end_cell":end_cell}


def aggregate(selected):
    result={"count":len(selected),"unique_images":len(set(x["image_sha256"] for x in selected)),
            "language":dict(sorted(Counter(x["language"] for x in selected).items())),
            "family":dict(sorted(Counter(x["family"] for x in selected).items())),
            "direction":dict(sorted(Counter(str(x["direction"]) for x in selected).items())),
            "scale":dict(sorted(Counter(x["scale"] for x in selected).items())),
            "start_cells_occupied":len(set(x["start_cell"] for x in selected)),
            "end_cells_occupied":len(set(x["end_cell"] for x in selected))}
    result["largest_direction_fraction"]=(max(Counter(x["direction"] for x in selected).values())
                                           /len(selected) if selected else 0)
    return result


@dataclass(frozen=True)
class Proposal:
    indices:tuple[int,...]
    sha256:str
    summary:dict


def propose(records:list[dict],sources:dict[str,dict],count:int=COUNT,seed:str=SEED):
    if (type(count) is not int or not 32<=count<=2048
            or len(records)!=10000 or len(sources)<10000 or not seed):
        raise ValueError("H105 original immutable 10k bounded proposal only")
    eligible=[]
    quarantined=Counter()
    seen_ids=set()
    for index,row in enumerate(records):
        if row["id"] in seen_ids:raise ValueError("original source duplicate TRAIN ID")
        seen_ids.add(row["id"])
        src=sources.get(row["id"])
        if src is None:raise ValueError("H105 missing original upstream task")
        features=source_geometry(row,src)
        if features is None:
            quarantined["source_boundary_or_degenerate"]+=1
            continue
        features["index"]=index
        eligible.append(features)
    # All original data remain unchanged, including quarantined source rows.
    if len(eligible)<count:raise ValueError("insufficient verified original TRAIN examples")
    eligible.sort(key=lambda x:(keyhash(seed+":"+x["id"]),x["id"]))
    selected=[]
    used_images=set()
    ctr={field:Counter() for field in
         ("language","family","direction","scale","start_cell","end_cell")}
    n_families=len(set(x["family"] for x in eligible))
    # Cost is concentration relative to a target, not global source popularity.
    # The direction term intentionally dominates H99's collapsed dir0 prefix.
    quotas={"direction":count/8,"language_ar":count*.3,
            "language_en":count*.7,"family":count/n_families,
            "scale":count/3,"cell":count/16}
    for _ in range(count):
        best=None
        best_score=float("inf")
        for x in eligible:
            if x["image_sha256"] in used_images:
                continue
            score=(3.6*ctr["direction"][x["direction"]]/quotas["direction"]+
                   2.0*ctr["language"][x["language"]]/quotas["language_"+x["language"]]+
                   1.6*ctr["family"][x["family"]]/quotas["family"]+
                   .65*ctr["scale"][x["scale"]]/quotas["scale"]+
                   .85*ctr["start_cell"][x["start_cell"]]/quotas["cell"]+
                   .85*ctr["end_cell"][x["end_cell"]]/quotas["cell"])
            if score<best_score:
                best_score,best=score,x
        if best is None:
            raise ValueError("not enough source-independent TRAIN screenshots")
        selected.append(best)
        used_images.add(best["image_sha256"])
        for field in ctr:
            ctr[field][best[field]]+=1
    summary=aggregate(selected)
    h99_equivalent=False
    gates={
        "unique_original_train_images":summary["unique_images"]==count,
        "at_least_six_directions":len(summary["direction"])>=6,
        "dominant_direction_below_40pct":summary["largest_direction_fraction"]<.4,
        "arabic_at_least_20pct":summary["language"].get("ar",0)>=math.ceil(count*.2),
        "all_five_original_source_families":len(summary["family"])>=5,
        "at_least_six_start_regions":summary["start_cells_occupied"]>=6,
        "at_least_six_end_regions":summary["end_cells_occupied"]>=6,
    }
    digest=keyhash(json.dumps({"strategy":STRATEGY,"seed":seed,
                    "chosen_IDS":[r["id"] for r in selected]},
                    sort_keys=True,separators=(",",":")))
    # Keep only distribution-level aggregates; chosen IDs and image hashes
    # stay only in owner-private runtime, never exported to PR metadata.
    summary.update(strategy=STRATEGY,requested_count=count,
                   eligible_original_train_rows=len(eligible),
                   original_source_quarantined_counts=dict(quarantined),
                   private_source_identifiers_in_receipt=False,
                   source_split="TRAIN_ONLY",
                   all_gates_passed=all(gates.values()),
                   gates=gates,
                   plan_sha256=digest)
    return Proposal(tuple(x["index"] for x in selected),digest,summary)
