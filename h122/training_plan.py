"""H122 exact image-unique 448 original TRAIN images with heldout 64 excluded."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
from h119.training_plan import get_plan as original_full512_plan
from h122.schedule import TRAIN,HOLDOUT,ORIGINAL,Schedule

STEPS=TRAIN

@dataclass(frozen=True)
class Plan:
    indices:tuple[int,...]
    sha256:str
    summary:dict

def get_plan(rows,root:Path,schedule:Schedule,steps:int=STEPS):
    if (type(steps) is not int or steps!=STEPS
            or not isinstance(schedule,Schedule)
            or schedule.summary.get("source_split")!="TRAIN_ONLY"
            or len(schedule.train)!=TRAIN or len(schedule.heldout)!=HOLDOUT):
        raise ValueError("H122 exact SHA-qualified TRAIN 448 / heldout 64 schedule required")
    original=original_full512_plan(rows,root,ORIGINAL*2)
    train_indices=tuple(original.indices[i] for i in schedule.train_original_indices)
    heldout_indices={original.indices[i] for i in schedule.heldout_original_indices}
    if (len(train_indices)!=STEPS or len(set(train_indices))!=STEPS
            or set(train_indices)&heldout_indices or len(heldout_indices)!=HOLDOUT):
        raise ValueError("original 64 TRAIN heldout images included in optimizer source")
    digest=hashlib.sha256((
        "hala.h122.end-absolute-grid10-448train-64holdout.v1"+
        original.sha256+schedule.summary["schedule_sha256"]).encode()).hexdigest()
    return Plan(train_indices,digest,{
        "source_split":"TRAIN_ONLY",
        "original_full512_train_sha256":original.sha256,
        "h122_schedule_sha256":schedule.summary["schedule_sha256"],
        "actual_optimizer_image_steps":STEPS,
        "heldout_train_original_images":HOLDOUT,
        "source_original_train_images":ORIGINAL,
        "no_overlap_with_heldout":True,
        "no_EVAL_or_TEST_data":True,
        "plan_sha256":digest
    })

def checkpoint_due(step:int,max_steps:int=STEPS):
    if (type(step) is not int or type(max_steps) is not int
            or max_steps!=STEPS or not 1<=step<=max_steps):
        raise ValueError("H122 only 448 actual real optimizer steps")
    return step%64==0 or step==max_steps
