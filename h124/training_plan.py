"""H124 frozen original TRAIN896+heldout128 joint START END source plan."""
import hashlib
from dataclasses import dataclass

from h124.schedule import Schedule,TRAIN,HOLDOUT,ORIGINAL,_jsonsha

STEPS=TRAIN
TRAIN_SHA="5cdb08b3abae660c14268aceef58f5f10b9216e8d070abad316e41ddf58063dc"
HELDOUT_SHA="922099432b818530486bec9ff01aa7fab452b0d0fd1e47d930c39c5c5659f6c1"
SCHEDULE_SHA="37d55eb4a1aa2429e4d13a00e8a15838fe87911748f13fa97b6bb4107f9003d6"
NONCE="hala.h124.fresh-original-complex896-joint-start-end-delta.v1"

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
            or schedule.summary.get("all_previous_H105_H123_images_excluded")!=1536
            or schedule.summary.get("no_original_EVAL_TEST") is not True
            or len(set(schedule.train_indices))!=TRAIN
            or len(set(schedule.heldout_indices))!=HOLDOUT
            or set(schedule.train_indices)&set(schedule.heldout_indices)
            or _jsonsha(schedule.train)!=TRAIN_SHA
            or _jsonsha(schedule.heldout)!=HELDOUT_SHA):
        raise ValueError("H124 exact new original TRAIN896/heldout128 source SHA missing")
    ids={row["original_source_id"] for row in schedule.train}
    heldout_ids={row["original_source_id"] for row in schedule.heldout}
    images={row["image_sha256"] for row in schedule.train+schedule.heldout}
    if (len(ids)!=TRAIN or len(heldout_ids)!=HOLDOUT
            or ids&heldout_ids or len(images)!=ORIGINAL):
        raise ValueError("H124 duplicate previously trained or heldout source")
    plan_sha=hashlib.sha256((NONCE+TRAIN_SHA+HELDOUT_SHA+SCHEDULE_SHA).encode()).hexdigest()
    return Plan(tuple(schedule.train_indices),plan_sha,{
        "source_split":"TRAIN_ONLY",
        "train_source_images":TRAIN,"heldout_source_images":HOLDOUT,
        "original_source_images":ORIGINAL,
        "train_private_sha256":TRAIN_SHA,
        "heldout_private_sha256":HELDOUT_SHA,
        "schedule_sha256":SCHEDULE_SHA,
        "plan_sha256":plan_sha,
        "all_previous_H105_H123_images_excluded":1536,
        "no_EVAL_or_TEST":True,
        "no_heldout_optimizer_input":True})

def checkpoint_due(step,max_steps=STEPS):
    if type(step) is not int or type(max_steps) is not int or max_steps!=STEPS or not 1<=step<=STEPS:
        raise ValueError("H124 exact 896 optimizer steps required")
    return step%64==0 or step==max_steps
