"""H122 fixed H113-verified 448 TRAIN / 64 TRAIN-heldout original screenshots.

The heldout original TRAIN screenshots are NEVER optimizer inputs.
Nothing from original public EVAL/TEST may enter this scheduler.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from h113.dual_point import ROLES,COORDINATE_SPACE
from h114.schedule import (ORIGINAL_H113_DERIVED_SHA,
                           ORIGINAL_H105_TRAIN_PLAN_SHA,
                           _read_verified_rows,qualify,QualifiedSchedule)
from h61.official_eval import strict_json
from h122.grid import end_prompt,end_completion,grid_cell,SCHEMA

ORIGINAL=512
TRAIN=448
HOLDOUT=64
NONCE=b"hala.h122.source-sha-split-original-H113-TRAIN.v1"

@dataclass(frozen=True)
class Schedule:
    train:tuple[dict,...]
    heldout:tuple[dict,...]
    train_original_indices:tuple[int,...]
    heldout_original_indices:tuple[int,...]
    summary:dict

def _select_heldout(pairs):
    groups=defaultdict(list)
    for i,(start,end) in enumerate(pairs):
        groups[(start["family"],start["language"])].append(
            (hashlib.sha256(NONCE+bytes.fromhex(start["image_sha256"])).hexdigest(),i))
    counts={group:len(items) for group,items in groups.items()}
    quota={group:(HOLDOUT*size)//ORIGINAL for group,size in counts.items()}
    remain=HOLDOUT-sum(quota.values())
    remainders=sorted(groups,key=lambda group:
                     (-(HOLDOUT*counts[group])%ORIGINAL,group))
    # Largest remainder: sorting explicit positive remainder descending.
    remainders=sorted(groups,key=lambda group:
                     (-(HOLDOUT*counts[group] % ORIGINAL),group))
    for group in remainders[:remain]:
        quota[group]+=1
    holdout=set()
    for group,items in groups.items():
        holdout.update(i for _,i in sorted(items)[:quota[group]])
    if len(holdout)!=HOLDOUT:
        raise ValueError("H122 64 proportional original TRAIN heldout sources required")
    return holdout

def derive(parent:QualifiedSchedule)->Schedule:
    if (not isinstance(parent,QualifiedSchedule)
            or parent.summary.get("source_split")!="TRAIN_ONLY"
            or parent.summary.get("original_h113_point_jsonl_sha256")!=ORIGINAL_H113_DERIVED_SHA
            or parent.summary.get("original_h105_frozen_plan_sha256")!=ORIGINAL_H105_TRAIN_PLAN_SHA
            or parent.summary.get("pilot_selected_original_screenshots")!=ORIGINAL
            or len(parent.selected)!=ORIGINAL*2):
        raise ValueError("H122 only all 512 owner-SHA-qualified original TRAIN pairs")
    pairs=list(zip(parent.selected[::2],parent.selected[1::2]))
    holdout=_select_heldout(pairs)
    train=[];validation=[];train_idx=[];valid_idx=[]
    langs=Counter();families=Counter();hold_lang=Counter();hold_family=Counter()
    seen=set()
    for i,(start,end) in enumerate(pairs):
        if (start.get("endpoint_role")!="start" or end.get("endpoint_role")!="end"
                or start.get("original_source_id")!=end.get("original_source_id")
                or start.get("image_sha256")!=end.get("image_sha256")
                or start.get("split")!="train" or end.get("split")!="train"
                or end.get("original_source_split")!="train"
                or not end.get("source_bound_validation_passed")
                or not end.get("no_eval_or_test_source")):
            raise ValueError("H122 exact original paired TRAIN-only provenance required")
        if start["image_sha256"] in seen:
            raise ValueError("H122 original source repeated between optimizer and holdout")
        seen.add(start["image_sha256"])
        answer=strict_json(end["completion"])
        if (type(answer) is not dict or set(answer)!={"role","point","coordinate_space"}
                or answer["role"]!="end" or answer["coordinate_space"]!=COORDINATE_SPACE):
            raise ValueError("H122 original H113 END label schema invalid")
        marker="\nLocate ONLY the END endpoint"
        if end["prompt"].count(marker)!=1:
            raise ValueError("H122 source END task intent missing")
        prompt=end_prompt(end["prompt"].split(marker)[0],True)
        derived=dict(end)
        derived.update(
            schema=SCHEMA,id=start["original_source_id"]+"::h122_end",
            prompt=prompt,completion=end_completion(answer["point"]),
            endpoint_role="end")
        if i in holdout:
            validation.append(derived);valid_idx.append(i)
            hold_lang[end["language"]]+=1;hold_family[end["family"]]+=1
        else:
            train.append(derived);train_idx.append(i)
            langs[end["language"]]+=1;families[end["family"]]+=1
    if (len(train)!=TRAIN or len(validation)!=HOLDOUT
            or len(seen)!=ORIGINAL or len(set(train_idx)&set(valid_idx))!=0
            or set(langs)!={"ar","en"} or set(hold_lang)!={"ar","en"}
            or len(families)!=5 or len(hold_family)!=5):
        raise ValueError("H122 full original TRAIN split language/domain coverage missing")
    canonical=lambda rows:b"".join(
        json.dumps(row,sort_keys=True,ensure_ascii=False,
                   separators=(",",":")).encode("utf8")+b"\n" for row in rows)
    train_sha=hashlib.sha256(canonical(train)).hexdigest()
    heldout_sha=hashlib.sha256(canonical(validation)).hexdigest()
    signature=hashlib.sha256((train_sha+heldout_sha+
                              ORIGINAL_H113_DERIVED_SHA).encode()).hexdigest()
    return Schedule(tuple(train),tuple(validation),tuple(train_idx),tuple(valid_idx),
        {"schema":SCHEMA,"source_split":"TRAIN_ONLY",
         "original_h113_sha256":ORIGINAL_H113_DERIVED_SHA,
         "original_h105_plan_sha256":ORIGINAL_H105_TRAIN_PLAN_SHA,
         "original_source_images":ORIGINAL,
         "train_source_images":TRAIN,"heldout_train_source_images":HOLDOUT,
         "train_languages":dict(sorted(langs.items())),
         "train_families":dict(sorted(families.items())),
         "heldout_languages":dict(sorted(hold_lang.items())),
         "heldout_families":dict(sorted(hold_family.items())),
         "train_private_sha256":train_sha,
         "heldout_private_sha256":heldout_sha,
         "schedule_sha256":signature,
         "objective":"END absolute 10x10 grid cell plus exact normalized point",
         "source_original_images_unchanged":True,
         "no_EVAL_or_TEST_source":True,
         "no_holdout_optimizer_input":True,
         "raw_train_ids_or_coordinates_exposed":False,
         "model_trained_or_improvement_claimed":False})

def qualify_owner_private(path:Path)->Schedule:
    return derive(qualify(_read_verified_rows(path),512,512))
