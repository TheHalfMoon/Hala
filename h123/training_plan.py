"""Freeze source-qualified H123 true TRAIN 896 / original TRAIN holdout 128."""
from dataclasses import dataclass
import hashlib

from h123.schedule import Schedule,TRAIN,HOLDOUT,ORIGINAL,_jsonhash
from h106.training_plan import H105_QUALIFIED_PLAN_SHA

STEPS=TRAIN
TRAIN_SHA="921705d11c4e4d9edfa5662f706aaccc3bd3d5c74eeb5c88353f31abface2040"
HELDOUT_SHA="5b6cf06392ca028b8c96f8ee21425541d2f18f69ea3298b5e698b471ad15d7ba"
SCHEDULE_SHA="246e393eae0106926de333491da988266f15c0fbf364f842a05478229ff10837"
NONCE="hala.h123.source-original-complex896-128-dualgrid-fresh-v1"


@dataclass(frozen=True)
class Plan:
    indices:tuple[int,...]
    sha256:str
    summary:dict


def get_plan(schedule:Schedule,steps=STEPS):
    if (not isinstance(schedule,Schedule) or type(steps) is not int
            or steps!=STEPS or len(schedule.train)!=TRAIN
            or len(schedule.heldout)!=HOLDOUT
            or schedule.summary.get("source_split")!="TRAIN_ONLY"
            or schedule.summary.get("train_private_sha256")!=TRAIN_SHA
            or schedule.summary.get("heldout_private_sha256")!=HELDOUT_SHA
            or schedule.summary.get("schedule_sha256")!=SCHEDULE_SHA
            or schedule.summary.get("original_H105_parent_SHA")!=H105_QUALIFIED_PLAN_SHA
            or schedule.summary.get("no_prev_heldout_in_optimizer_or_new_holdout") is not True
            or schedule.summary.get("no_original_PUBLIC_EVAL_or_TEST") is not True
            or len(set(schedule.train_indices))!=TRAIN
            or len(set(schedule.heldout_indices))!=HOLDOUT
            or set(schedule.train_indices)&set(schedule.heldout_indices)):
        raise ValueError("H123 exact source SHA-qualified 896/128 unseen original TRAIN only")
    if (_jsonhash(schedule.train)!=TRAIN_SHA or
            _jsonhash(schedule.heldout)!=HELDOUT_SHA):
        raise ValueError("H123 actual source row bytes differ from frozen original TRAIN digests")
    ids={r["original_source_id"] for r in schedule.train}
    heldout_ids={r["original_source_id"] for r in schedule.heldout}
    media={r["image_sha256"] for r in schedule.train+schedule.heldout}
    if (len(ids)!=TRAIN or len(heldout_ids)!=HOLDOUT or
            ids&heldout_ids or len(media)!=ORIGINAL):
        raise ValueError("H123 unique original source images/IDs reused")
    digest=hashlib.sha256((NONCE+TRAIN_SHA+HELDOUT_SHA+SCHEDULE_SHA).encode()).hexdigest()
    return Plan(tuple(schedule.train_indices),digest,{
        "source_split":"TRAIN_ONLY",
        "source_original_screenshots":ORIGINAL,
        "train_source_images":TRAIN,
        "heldout_original_TRAIN_images":HOLDOUT,
        "original_h105_parent_SHA":H105_QUALIFIED_PLAN_SHA,
        "train_sha256":TRAIN_SHA,"heldout_sha256":HELDOUT_SHA,
        "schedule_sha256":SCHEDULE_SHA,"plan_sha256":digest,
        "no_previous_H113_H122_source_used":True,
        "no_original_EVAL_or_TEST":True,
        "no_heldout_optimizer_input":True,
    })


def checkpoint_due(step,max_steps=STEPS):
    if (type(step) is not int or type(max_steps) is not int
            or max_steps!=STEPS or not 1<=step<=max_steps):
        raise ValueError("H123 exactly 896 source-original optimizer steps required")
    return step%64==0 or step==max_steps
