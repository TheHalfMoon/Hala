"""H122 training checkpoint owner attestation: fail closed until real bytes pinned.

The three constants remain None until independent Windows SHA replay of
an actually finished 448-step noncommercial original TRAIN-only experiment.
An unverified Kaggle COMPLETE status alone is NEVER sufficient.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

from h73.kaggle_h73_cpu import TRAIN_SHA as H71_TRAIN_SHA
from h106.training_plan import STRATEGY as H106_STRATEGY
from h122.research_train import (config as h122_config,TRAIN_SHA_H122,
                                 HELDOUT_SHA_H122,SCHEDULE_SHA_H122)
from h122.training_plan import get_plan,STEPS
from h122.schedule import TRAIN,HOLDOUT,Schedule
from training.scripts.train_adapter import (JOINT_LORA_SCOPE_NONCE,
                                            H122_GRID_SCOPE_NONCE)
from training.checkpoints import load_checkpoint
from training.artifacts import sha256_file

SOURCE_KERNEL="hala-h122-private-end-grid448-t4"
SOURCE_DIR="hala-h122-private-end-grid448-checkpoint"
SOURCE_RECEIPT="hala-h122-private-train-receipt.json"

# These must be pinned ONLY from a true independently replayed private
# Windows adapter/checkpoint/optimizer readback, not copied from a
# hypothetical experiment, text plan, or unverified Kaggle status.
H122_VERIFIED_ADAPTER_SHA=None
H122_VERIFIED_CHECKPOINT_SHA=None
H122_VERIFIED_OPTIMIZER_SHA=None


def expected_training_identity(repo:Path, original_rows, source_root, schedule:Schedule):
    """Exactly reproduce opt-in H122 frozen training signature, no EVAL."""
    if (not isinstance(schedule,Schedule)
            or schedule.summary.get("train_private_sha256")!=TRAIN_SHA_H122
            or schedule.summary.get("heldout_private_sha256")!=HELDOUT_SHA_H122
            or schedule.summary.get("schedule_sha256")!=SCHEDULE_SHA_H122):
        raise ValueError("H122 original source trained/heldout digest differs")
    plan=get_plan(original_rows,source_root,schedule,STEPS)
    config=h122_config(repo)
    signature=hashlib.sha256(
        json.dumps(config,sort_keys=True).encode()).hexdigest()
    signature=hashlib.sha256((signature+JOINT_LORA_SCOPE_NONCE).encode()).hexdigest()
    signature=hashlib.sha256((signature+H106_STRATEGY+plan.sha256).encode()).hexdigest()
    signature=hashlib.sha256((
        signature+H122_GRID_SCOPE_NONCE+
        TRAIN_SHA_H122+HELDOUT_SHA_H122+plan.sha256).encode()).hexdigest()
    return signature,plan


def admit_checkpoint(inputs:Path,repo:Path,original_rows,root,schedule:Schedule):
    expected=(H122_VERIFIED_ADAPTER_SHA,H122_VERIFIED_CHECKPOINT_SHA,
              H122_VERIFIED_OPTIMIZER_SHA)
    if not all(type(value) is str and len(value)==64 and all(
           letter in "0123456789abcdef" for letter in value) for value in expected):
        raise ValueError("H122 448-step independent owner model binary SHA not yet admitted")
    attached=Path(inputs)
    if attached.is_symlink() or not attached.is_dir():
        raise ValueError("H122 private attached Kaggle model kernel source required")
    candidates=[p.parent for p in attached.rglob("latest.json")
                if (not p.is_symlink() and p.parent.name==SOURCE_DIR
                    and SOURCE_KERNEL in p.as_posix())]
    if len(candidates)!=1:
        raise ValueError("one exact original PRIVATE H122 Kaggle checkpoint required")
    checkpoint_root=candidates[0]
    receipt_file=checkpoint_root.parent/SOURCE_RECEIPT
    if not receipt_file.is_file() or receipt_file.is_symlink():
        raise ValueError("H122 true original GPU optimizer receipt required")
    info=json.loads(receipt_file.read_text("utf8"))
    version,state=load_checkpoint(checkpoint_root,None)
    paths={"adapter":version/"adapter/adapter_model.safetensors",
           "checkpoint":version/"checkpoint.json",
           "optimizer":version/"resume.pt"}
    if any(not x.is_file() or x.is_symlink() for x in paths.values()):
        raise ValueError("H122 original model/optimizer binary files unavailable")
    actual={k:sha256_file(v) for k,v in paths.items()}
    signature,plan=expected_training_identity(repo,original_rows,root,schedule)
    updates=state.get("runtime",{}).get("qualification_steps",[])
    observed=[record.get("record_id") for step in updates
              for record in step.get("microbatches",[])]
    verified=(
       info.get("status")=="H122_REAL448_TRAIN_ONLY_END_GRID_OPTIMIZER_VERIFIED"
       and info.get("actual_optimizer_steps")==STEPS
       and info.get("positive_gradient_steps")==STEPS
       and info.get("train_private_sha256")==TRAIN_SHA_H122
       and info.get("heldout_private_sha256")==HELDOUT_SHA_H122
       and info.get("schedule_sha256")==SCHEDULE_SHA_H122
       and info.get("no_EVAL_or_TEST_used") is True
       and info.get("actual_validation_scored") is False
       and info.get("model_weights_published") is False
       and info.get("paid_compute") is False
       and state.get("step")==STEPS and state.get("max_steps")==STEPS
       and state.get("dataset_sha256")==H71_TRAIN_SHA
       and state.get("config_sha256")==signature
       and state.get("changed_parameter_tensors",0)>0
       and len(state.get("metrics",[]))==STEPS
       and len(updates)==STEPS
       and observed==[r["id"] for r in schedule.train]
       and len(set(observed))==TRAIN
       and not set(observed)&{r["id"] for r in schedule.heldout}
       and state.get("runtime",{}).get("h122_end_processor_audits")==8
       and state.get("runtime",{}).get("h122_end_grid_train_sampler",{}).get(
           "schedule_sha256")==SCHEDULE_SHA_H122
       and state.get("runtime",{}).get("h106_joint_lora_verified") is True
       and all(step.get("step")==i
          and step.get("changed_parameter_tensors_this_step",0)>0
          and type(step.get("gradient_norm_before_clip")) in (int,float)
          and math.isfinite(step["gradient_norm_before_clip"])
          and step["gradient_norm_before_clip"]>0
          for i,step in enumerate(updates,1))
       and actual["adapter"]==expected[0]==info.get("final_adapter_sha256")
       and actual["checkpoint"]==expected[1]==info.get("final_checkpoint_sha256")
       and actual["optimizer"]==expected[2]==info.get("final_optimizer_sha256"))
    if not verified:
        raise ValueError("H122 real optimizer/source/heldout/binary owner gate failed")
    return version/"adapter",{"verified_steps":STEPS,
       "adapter_sha256":actual["adapter"],
       "checkpoint_sha256":actual["checkpoint"],
       "optimizer_sha256":actual["optimizer"],
       "train_source_sha256":TRAIN_SHA_H122,
       "heldout_source_sha256":HELDOUT_SHA_H122,
       "model_GUI_improvement_qualified":False}
