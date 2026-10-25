"""H124 private real training original source/optimizer adapter SHA admission.

Only owner independently verified real TRAIN896 model binaries may be
attached to genuine new original H71 TRAIN128 heldout inference.
"""
from pathlib import Path
import hashlib,json,math
from h73.kaggle_h73_cpu import TRAIN_SHA as H71_SHA
from h124.schedule import Schedule,TRAIN,HOLDOUT
from h124.training_plan import get_plan,STEPS,TRAIN_SHA,HELDOUT_SHA,SCHEDULE_SHA
from h124.research_train import config
from h106.training_plan import STRATEGY as H106_SAMPLER
from training.scripts.train_adapter import JOINT_LORA_SCOPE_NONCE,H124_JOINT_SCOPE_NONCE
from training.checkpoints import load_checkpoint
from training.artifacts import sha256_file

MODEL_KERNEL="hala-h124-private-new-joint896-t4"
MODEL_DIR="hala-h124-private-new-joint896-checkpoint"
RECEIPT="hala-h124-private-training-receipt.json"
ADAPTER_SHA="509655e618defe8b5677e9235e4a281da63fbb0914c68738231b0a64956f122d"
CHECKPOINT_SHA="94397d8869bb49439ac08ca40cde919190d3c58563fd8542c96a0849cd1b9e84"
OPTIMIZER_SHA="a9080a58831ad7b7e1759fb54be5118d18b0b57b59e0e78401772ce935e03ac9"

def expected_identity(repo,schedule:Schedule):
    plan=get_plan(schedule)
    sha=lambda x:hashlib.sha256(x.encode("utf8")).hexdigest()
    identity=sha(json.dumps(config(repo),sort_keys=True))
    identity=sha(identity+JOINT_LORA_SCOPE_NONCE)
    identity=sha(identity+H106_SAMPLER+plan.sha256)
    identity=sha(identity+H124_JOINT_SCOPE_NONCE+TRAIN_SHA+
                 HELDOUT_SHA+plan.sha256)
    return identity,plan

def admit(inputs,repo,schedule:Schedule):
    expected={"adapter":ADAPTER_SHA,"checkpoint":CHECKPOINT_SHA,
              "optimizer":OPTIMIZER_SHA}
    if any(type(x) is not str or len(x)!=64 or any(
            letter not in "0123456789abcdef" for letter in x)
            for x in expected.values()):
        raise ValueError("H124 owner exact real model binaries not admitted")
    inputs=Path(inputs)
    if not inputs.is_dir() or inputs.is_symlink():
        raise ValueError("H124 owner private model kernel attachment absent")
    roots=[f.parent for f in inputs.rglob("latest.json")
           if f.is_file() and not f.is_symlink()
           and f.parent.name==MODEL_DIR and MODEL_KERNEL in f.as_posix()]
    if len(roots)!=1:
        raise ValueError("H124 exact one private source pinned model root required")
    root=roots[0];receipt=root.parent/RECEIPT
    if not receipt.is_file() or receipt.is_symlink():
        raise ValueError("H124 original actual 896 optimizer receipt missing")
    data=json.loads(receipt.read_text("utf8"))
    version,state=load_checkpoint(root,None)
    files={"adapter":version/"adapter"/"adapter_model.safetensors",
           "checkpoint":version/"checkpoint.json",
           "optimizer":version/"resume.pt"}
    if any(not file.is_file() or file.is_symlink() for file in files.values()):
        raise ValueError("H124 exact three private trained model binaries missing")
    real={name:sha256_file(file) for name,file in files.items()}
    signature,plan=expected_identity(repo,schedule)
    runtime=state.get("runtime",{})
    updates=runtime.get("qualification_steps",[])
    rows=[item.get("record_id") for step in updates
          for item in step.get("microbatches",[])]
    if not (
        data.get("status")=="H124_REAL896_FRESH_JOINT_TRAIN_ONLY_OPTIMIZER_VERIFIED"
        and data.get("actual_optimizer_steps")==STEPS
        and data.get("positive_gradient_steps")==STEPS
        and data.get("train_private_sha256")==TRAIN_SHA
        and data.get("heldout_private_sha256")==HELDOUT_SHA
        and data.get("schedule_sha256")==SCHEDULE_SHA
        and data.get("old_H105_H123_images_excluded")==1536
        and data.get("original_EVAL_or_TEST_in_training") is False
        and data.get("prior_H113_H123_images_reused") is False
        and data.get("heldout_used_for_optimizer") is False
        and state.get("step")==state.get("max_steps")==STEPS
        and state.get("dataset_sha256")==H71_SHA
        and state.get("config_sha256")==signature
        and len(state.get("metrics",[]))==len(updates)==STEPS
        and rows==[row["id"] for row in schedule.train]
        and len(set(rows))==STEPS
        and runtime.get("h124_joint_original_processor_audits")==12
        and runtime.get("h106_joint_lora_verified") is True
        and all(u.get("step")==i and
                u.get("changed_parameter_tensors_this_step",0)>0
                and type(u.get("gradient_norm_before_clip")) in (int,float)
                and math.isfinite(u["gradient_norm_before_clip"])
                and u["gradient_norm_before_clip"]>0
                for i,u in enumerate(updates,1))
        and all(real[key]==expected[key]==data.get("final_"+key+"_sha256")
                for key in real)):
        raise ValueError("H124 exact owner real-model/source 896-step SHA proof failed")
    return version/"adapter",{
        "owner_verified_model_bytes_SHA":real,"actual_optimizer_steps":STEPS,
        "new_original_train_holdout_unoptimized":True,
        "externally_qualified_GUI_skill":False}
