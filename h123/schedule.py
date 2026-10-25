"""H123 source-bound, image-disjoint complex GUI spatial curriculum.

Select four original complex DragOn TRAIN domains, 256 unique screenshots
each; hold 32/domain aside, train with 224/domain. Exclude ALL original
H105/H113/H122 512 screenshots, including their previous heldout.
No original EVAL/TEST, hidden labels, model outputs or observed H122
heldout answers influence this deterministic TRAIN-only selector.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from h105.curriculum import propose,source_geometry
from h106.training_plan import H105_QUALIFIED_PLAN_SHA
from h113.dual_point import QuantizationQuarantine
from h122.source_bound import admit_original_source_map,original_for_sft
from h123.objective import from_original,SCHEMA

SOURCE=10000
OLD=512
ORIGINAL=1024
TRAIN=896
HOLDOUT=128
PER_FAMILY=256
HELDOUT_PER_FAMILY=32
SEED="hala.h123.original-train-complex-domain-unseen-20261025.v1"
FAMILIES=tuple("hcompany_dragon_h61_"+domain+"_train" for domain in
               ("sheet","slide_resize","slider","text_highlight"))


@dataclass(frozen=True)
class Schedule:
    train:tuple[dict,...]
    heldout:tuple[dict,...]
    train_indices:tuple[int,...]
    heldout_indices:tuple[int,...]
    summary:dict


def _hashed(s:str)->str:
    return hashlib.sha256(s.encode("utf8")).hexdigest()


def _jsonhash(rows):
    return hashlib.sha256(b"".join(
        json.dumps(row,sort_keys=True,ensure_ascii=False,
                   separators=(",",":")).encode("utf8")+b"\n"
        for row in rows)).hexdigest()


def _balanced_choose(candidates,count):
    """Greedy original TRAIN direction+target-region+length balancing."""
    if len(candidates)<count:
        raise ValueError("H123 insufficient independent original complex TRAIN")
    chosen=[]
    used_images=set()
    direction=Counter();cells=Counter();scales=Counter()
    remaining=sorted(candidates,key=lambda x:(_hashed(SEED+x["source_id"]),
                                              x["source_id"]))
    for _ in range(count):
        best=None;score=None
        for x in remaining:
            if x["sha"] in used_images:
                continue
            rank=(4*direction[x["direction"]]+
                  2*cells[tuple(x["end_cell"])]+
                  scales[x["scale"]])
            if score is None or rank<score:
                score=rank;best=x
        if best is None:
            raise ValueError("H123 original source image uniqueness unavailable")
        chosen.append(best)
        remaining.remove(best)
        used_images.add(best["sha"])
        direction[best["direction"]]+=1
        cells[tuple(best["end_cell"])]+=1
        scales[best["scale"]]+=1
    available_dirs={x["direction"] for x in candidates}
    available_cells={tuple(x["end_cell"]) for x in candidates}
    if (len(chosen)!=count or len(used_images)!=count
            or len(available_dirs)<1 or len(available_cells)<2
            or len(direction)<min(6,len(available_dirs))
            or len(cells)<min(8,len(available_cells))):
        raise ValueError(
            "H123 original TRAIN support-aware spatial/direction diversity "
            f"failed: chosen_dirs={len(direction)}, available_dirs={len(available_dirs)}, "
            f"chosen_cells={len(cells)}, available_cells={len(available_cells)}")
    return chosen


def _holdout_indices(selected):
    """Reserve 32 per domain across its authentic TRAIN-supported directions."""
    if len(selected)!=PER_FAMILY or len({x["sha"] for x in selected})!=PER_FAMILY:
        raise ValueError("H123 original 256 unique screenshots required")
    by_dir=defaultdict(list)
    for i,x in enumerate(selected):
        if type(x["direction"]) is not int or not 0<=x["direction"]<8:
            raise ValueError("H123 original eight-way drag direction required")
        by_dir[x["direction"]].append(
            (_hashed("heldout:"+SEED+x["sha"]),i))
    if not 1<=len(by_dir)<=8 or any(len(items)<4 for items in by_dir.values()):
        raise ValueError("H123 original source has insufficient supported directions")
    # Stratify over actual TRAIN directions, rather than demanding GUI
    # motions that do not exist in a particular original source domain.
    ordering={k:[i for _,i in sorted(v)] for k,v in by_dir.items()}
    used={k:1 for k in ordering}
    indices={indices[0] for indices in ordering.values()}
    while len(indices)<HELDOUT_PER_FAMILY:
        choices=[k for k in ordering if used[k]<len(ordering[k])-1]
        if not choices:
            raise ValueError("H123 32 independent TRAIN holdouts unavailable")
        direction=min(choices,key=lambda k:(used[k]/len(ordering[k]),k))
        indices.add(ordering[direction][used[direction]])
        used[direction]+=1
    if len(indices)!=HELDOUT_PER_FAMILY:
        raise ValueError("H123 exact source 32/domain holdout count differs")
    return indices


def derive(rows,root:Path):
    if type(rows) is not list or len(rows)!=SOURCE:
        raise ValueError("H123 original H71 10k TRAIN-only SFT required")
    sources=admit_original_source_map(root)
    # The original frozen parent H105 first-512 SHA must remain unchanged
    # and NONE of its 512 images may be in the new TRAIN OR heldout.
    original=propose(rows,sources,count=OLD)
    if original.sha256!=H105_QUALIFIED_PLAN_SHA or not original.summary["all_gates_passed"]:
        raise ValueError("H123 prior H113/H122 parent 512 SHA qualification failed")
    forbidden_ids={rows[i]["id"] for i in original.indices}
    forbidden_shas={rows[i]["image_sha256"] for i in original.indices}
    if len(forbidden_ids)!=OLD or len(forbidden_shas)!=OLD:
        raise ValueError("H123 prior H113 source duplicate")
    pool=defaultdict(list)
    visited=set()
    for index,sft in enumerate(rows):
        ident=sft.get("id")
        if (type(ident) is not str or ident in visited
                or sft.get("split")!="train"):
            raise ValueError("H123 original 10k TRAIN source IDs/split drift")
        visited.add(ident)
        if ident in forbidden_ids or sft["image_sha256"] in forbidden_shas:
            continue
        if sft.get("family") not in FAMILIES:
            continue
        try:
            original_source=original_for_sft(sft,sources)
        except ValueError as error:
            # H105 already quarantines genuinely degenerate or boundary-
            # violating original TRAIN geometry. All identity, task,
            # image-SHA and EVAL guards still fail closed.
            if str(error)!="H122 original source geometry was excluded by H105":
                raise
            continue
        features=source_geometry(sft,original_source)
        if features is None:
            continue
        try:
            output=from_original(sft,original_source)
        except QuantizationQuarantine:
            continue
        pool[sft["family"]].append({
            "index":index,"source_id":ident,"sha":sft["image_sha256"],
            "direction":features["direction"],"scale":features["scale"],
            "end_cell":features["end_cell"],"role":output})
    if set(pool)!=set(FAMILIES):
        raise ValueError("H123 four original DragOn complex domains absent")
    train=[];heldout=[];train_indices=[];heldout_indices=[]
    seen=set();domain_sizes={};languages=Counter()
    for family in FAMILIES:
        selected=_balanced_choose(pool[family],PER_FAMILY)
        heldout_pos=_holdout_indices(selected)
        family_t=family_h=0
        for i,item in enumerate(selected):
            if item["sha"] in seen or item["sha"] in forbidden_shas:
                raise ValueError("H123 prior H113/H122 or new source reused")
            seen.add(item["sha"])
            if item["role"].get("language")!="en":
                raise ValueError("H123 complex-domain English pilot only; no bilingual claim")
            languages[item["role"]["language"]]+=1
            if i in heldout_pos:
                heldout.append(item["role"]);heldout_indices.append(item["index"])
                family_h+=1
            else:
                train.append(item["role"]);train_indices.append(item["index"])
                family_t+=1
        domain_sizes[family]={
            "train":family_t,"heldout":family_h,
            "available_original_TRAIN_directions":
                len({x["direction"] for x in pool[family]}),
            "selected_TRAIN_directions":
                dict(sorted((str(k),v) for k,v in
                     Counter(x["direction"] for x in selected).items())),
            "available_original_TRAIN_target_regions":
                len({tuple(x["end_cell"]) for x in pool[family]})}
        if family_t!=224 or family_h!=32:
            raise ValueError("H123 exact original TRAIN domain allocation drift")
    if (len(train)!=TRAIN or len(heldout)!=HOLDOUT
            or len(seen)!=ORIGINAL
            or len(set(train_indices)&set(heldout_indices))!=0
            or set(languages)!={"en"}):
        raise ValueError("H123 original TRAIN cohort disjointness failed")
    train_sha=_jsonhash(train);holdout_sha=_jsonhash(heldout)
    digest=_hashed(SEED+original.sha256+train_sha+holdout_sha)
    summary={
        "schema":SCHEMA,"source_split":"TRAIN_ONLY",
        "original_H105_parent_SHA":original.sha256,
        "excluded_all_previous_H113_H122_images":OLD,
        "source_original_TRAIN_only":ORIGINAL,
        "train_source_images":TRAIN,"heldout_source_images":HOLDOUT,
        "per_complex_family":domain_sizes,
        "train_private_sha256":train_sha,
        "heldout_private_sha256":holdout_sha,
        "schedule_sha256":digest,
        "objective":"END quadrant4+grid10+absolute1000",
        "language_scope":"en_only_complex_pilot",
        "no_original_PUBLIC_EVAL_or_TEST":True,
        "no_prev_heldout_in_optimizer_or_new_holdout":True,
        "heldout_never_optimizer":True,
        "raw_sources_or_target_boxes_published":False,
        "trained_model_or_improvement_claimed":False,
    }
    return Schedule(tuple(train),tuple(heldout),
                    tuple(train_indices),tuple(heldout_indices),summary)
