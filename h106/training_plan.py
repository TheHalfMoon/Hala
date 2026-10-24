"""Opt-in H106 source-bound 256-step prefix of QUALIFIED H105 512 plan.

Training results are model candidates, not model improvements. All EVAL/TEST
are forbidden. The original H71 source and H105 code are immutable.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from dataclasses import dataclass

from h61.curriculum import read_shard,shard_roots
from h105.curriculum import propose,source_geometry,aggregate

STRATEGY="hala.h106.joint-vision-language-h105-source-prefix256.v1"
STEPS=256
H105_QUALIFIED_PLAN_SHA="ed69e0ea119c4e208ba6aa0a486cb29fb0c857e61fb278d6543152d022c40d0d"


@dataclass(frozen=True)
class FrozenTrainPrefix:
    indices:tuple[int,...]
    sha256:str
    summary:dict


def get_plan(rows,training_root:Path,steps:int=STEPS):
    if (type(steps) is not int or steps!=STEPS or len(rows)!=10000):
        raise ValueError("H106 strictly bounded 256 optimizer steps on original H71")
    source={}
    root=Path(training_root)
    for shard in shard_roots(root):
        prefix=shard.relative_to(root).as_posix() if shard!=root else ""
        originals,_=read_shard(shard,prefix)
        for item in originals:
            if item["id"] in source:raise ValueError("duplicate original source ID")
            source[item["id"]]=item
    if len(source)<10000:
        raise ValueError("original H71 upstream labels insufficient")
    proposal=propose(rows,source,512)
    if proposal.sha256!=H105_QUALIFIED_PLAN_SHA or not proposal.summary["all_gates_passed"]:
        raise ValueError("H105 prior owner-private accepted TRAIN plan identity changed")
    indices=proposal.indices[:STEPS]
    selected=[source_geometry(rows[i],source[rows[i]["id"]]) for i in indices]
    if any(x is None for x in selected):
        raise ValueError("invalid source label in H106 TRAIN optimizer prefix")
    report=aggregate(selected)
    if (len(indices)!=STEPS or report["unique_images"]!=STEPS
            or report["language"].get("ar",0)<48
            or len(report["family"])!=5
            or report["largest_direction_fraction"]>=.45):
        raise ValueError("H106 selected TRAIN prefix failed source diversity gate")
    digest=hashlib.sha256((STRATEGY+":"+proposal.sha256+":"+str(STEPS)).encode()).hexdigest()
    report.update(strategy=STRATEGY,steps=STEPS,
                  parent_original_h105_512_plan_sha256=proposal.sha256,
                  source_split="TRAIN_ONLY",
                  no_eval_or_test=True,source_ids_in_receipt=False,
                  plan_sha256=digest)
    return FrozenTrainPrefix(indices,digest,report)
