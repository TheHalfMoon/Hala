"""H123 real training model SHA gate; unarmed before Windows byte readback."""
from pathlib import Path
import hashlib,json,math
from h73.kaggle_h73_cpu import TRAIN_SHA as H71_SHA
from h123.schedule import Schedule,TRAIN,HOLDOUT
from h123.training_plan import (get_plan,STEPS,TRAIN_SHA,HELDOUT_SHA,SCHEDULE_SHA)
from h123.research_train import config
from h106.training_plan import STRATEGY as H106_SAMPLER
from training.scripts.train_adapter import JOINT_LORA_SCOPE_NONCE,H123_COMPLEX_SCOPE_NONCE
from training.checkpoints import load_checkpoint
from training.artifacts import sha256_file

MODEL_KERNEL="hala-h123-private-complex896-dualgrid-t4"
MODEL_DIR="hala-h123-private-complex896-checkpoint"
RECEIPT="hala-h123-private-training-receipt.json"

# Fill only after actual owner independent Windows replay of all three real binaries.
ADAPTER_SHA=None
CHECKPOINT_SHA=None
OPTIMIZER_SHA=None

def expected_identity(repo,schedule:Schedule):
    plan=get_plan(schedule)
    sha=lambda x:hashlib.sha256(x.encode("utf8")).hexdigest()
    signature=sha(json.dumps(config(repo),sort_keys=True))
    signature=sha(signature+JOINT_LORA_SCOPE_NONCE)
    signature=sha(signature+H106_SAMPLER+plan.sha256)
    signature=sha(signature+H123_COMPLEX_SCOPE_NONCE+TRAIN_SHA+
                  HELDOUT_SHA+plan.sha256)
    return signature,plan

def admit(inputs,repo,schedule):
    expected={"adapter":ADAPTER_SHA,"checkpoint":CHECKPOINT_SHA,
              "optimizer":OPTIMIZER_SHA}
    if any(type(x) is not str or len(x)!=64 or any(
            c not in "0123456789abcdef" for c in x) for x in expected.values()):
        raise ValueError("H123 independently verified training binary SHA not yet admitted")
    attached=Path(inputs)
    if not attached.is_dir() or attached.is_symlink():
        raise ValueError("H123 private Kaggle checkpoint attachment absent")
    results=[p.parent for p in attached.rglob("latest.json") if
             p.parent.name==MODEL_DIR and MODEL_KERNEL in p.as_posix()
             and p.is_file() and not p.is_symlink()]
    if len(results)!=1:
        raise ValueError("H123 exact one original private trained checkpoint required")
    root=results[0]
    receipt=root.parent/RECEIPT
    if not receipt.is_file() or receipt.is_symlink():
        raise ValueError("H123 actual model training receipt missing")
    data=json.loads(receipt.read_text("utf8"))
    version,state=load_checkpoint(root,None)
    paths={"adapter":version/"adapter"/"adapter_model.safetensors",
           "checkpoint":version/"checkpoint.json","optimizer":version/"resume.pt"}
    if any(not p.is_file() or p.is_symlink() for p in paths.values()):
        raise ValueError("H123 exact trained checkpoint bytes unavailable")
    actual={k:sha256_file(p) for k,p in paths.items()}
    signature,plan=expected_identity(repo,schedule)
    runtime=state.get("runtime",{})
    history=runtime.get("qualification_steps",[])
    observed=[x.get("record_id") for step in history
              for x in step.get("microbatches",[])]
    if not (
        data.get("status")=="H123_REAL896_NEW_COMPLEX_TRAIN_ONLY_OPTIMIZER_VERIFIED"
        and data.get("actual_train_optimizer_steps")==STEPS
        and data.get("positive_gradient_steps")==STEPS
        and data.get("train_private_sha256")==TRAIN_SHA
        and data.get("heldout_private_sha256")==HELDOUT_SHA
        and data.get("schedule_sha256")==SCHEDULE_SHA
        and data.get("original_EVAL_or_TEST_in_training") is False
        and data.get("previous_H113_H122_source_reused") is False
        and data.get("heldout_scored_or_used_for_optimizer") is False
        and state.get("step")==state.get("max_steps")==STEPS
        and state.get("dataset_sha256")==H71_SHA
        and state.get("config_sha256")==signature
        and len(state.get("metrics",[]))==len(history)==STEPS
        and observed==[row["id"] for row in schedule.train]
        and len(set(observed))==STEPS
        and runtime.get("h123_original_complex_processor_audits")==12
        and runtime.get("h106_joint_lora_verified") is True
        and all(x.get("step")==n and x.get("changed_parameter_tensors_this_step",0)>0
            and type(x.get("gradient_norm_before_clip")) in (int,float)
            and math.isfinite(x["gradient_norm_before_clip"])
            and x["gradient_norm_before_clip"]>0
            for n,x in enumerate(history,1))
        and all(actual[k]==expected[k]==data.get("final_"+k+"_sha256") for k in actual)):
        raise ValueError("H123 independent model/source optimizer gates failed")
    return version/"adapter",{"actual_steps":STEPS,"SHA":actual,
                               "heldout_untrained":True,
                               "external_GUI_skill_qualified":False}
