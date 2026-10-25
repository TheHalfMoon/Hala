"""H120 private real next64 H61 GUI evaluation for H114/H117/H119.

Hard fail until independent OWNER H119 final checkpoint binary SHA pins
are committed; never silently accept unverified GPU artifacts. No training,
no hidden TEST, no oracle prompt labels, no edited H61 scoring.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

from h61.official_eval import (
    DOMAINS,REPO,REV,canonical,fetch_domain,load_scenes,apply_frame,strict_json)
from h85.model_cache import HF_HOME,pinned_cache_kwargs
from h93.eval import _infer
from h100.eval import require_owner_private_runtime
from h115.eval import admit_h114_checkpoint
from h118.eval import admit_h117_checkpoint,strict_start_anchor,input_proof
from h115.compose import role_input,compose_strict_original_drag
from h116.delta import prompt_end_delta,compose_model_drag_from_delta
from h120.protocol import (START,END,ARMS,PER_DOMAIN,freeze_domain,freeze_all,
                           score_paired,SCHEMA)
from h119.research_train import config as h119_config, EXPECTED_TRAIN_SHA
from h119.training_plan import get_plan,STEPS,H105_QUALIFIED_PLAN_SHA
from h73.kaggle_h73_cpu import TRAIN_SHA
from h106.training_plan import STRATEGY as H106_SAMPLER
from training.scripts.train_adapter import (
    JOINT_LORA_SCOPE_NONCE,H119_FULL512_SCOPE_NONCE)
from training.artifacts import sha256_file
from training.checkpoints import load_checkpoint
from training.scripts.baseline import MODEL_ID,MODEL_REVISION

# Independently owner-verified 2026-10-25: 18/18 real Windows binary,
# optimizer, gradient, exact original TRAIN/source identity checks passed.
H119_ADAPTER_SHA = "6b5c01eff8097321926373e251bc9a3e7a6cc17e524bdf860fc1543e551b2ed7"
H119_CHECKPOINT_SHA = "a9c3b7d66450b910cea78a3876529c3408b67d3ad1080e629e72b2d239a93f19"
H119_OPTIMIZER_SHA = "0f73481865b4f2ead8726362db495224eb4cd02a14857593f3df79719c33f4f7"
H119_KERNEL="hala-h119-private-robust-full512-1024-t4"
H119_ROOT="hala-h119-private-robust512-1024-checkpoint"
SUMMARY_SCHEMA="hala.h120.owner-private-real-next64-three-arm-GUI.v1"


def expected_h119_training_identity(repo):
    cfg=h119_config(Path(repo))
    config_sha=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    joint=hashlib.sha256((config_sha+JOINT_LORA_SCOPE_NONCE).encode()).hexdigest()
    plan_sha=hashlib.sha256(("hala.h119.full512:"+H105_QUALIFIED_PLAN_SHA).encode()).hexdigest()
    shared=hashlib.sha256((joint+H106_SAMPLER+plan_sha).encode()).hexdigest()
    final=hashlib.sha256((shared+H119_FULL512_SCOPE_NONCE+
                          EXPECTED_TRAIN_SHA+plan_sha).encode()).hexdigest()
    return final


def admit_h119_checkpoint(attached=Path("/kaggle/input"),repo=None):
    if (repo is None or not all(
            isinstance(x,str) and len(x)==64 for x in
            (H119_ADAPTER_SHA,H119_CHECKPOINT_SHA,H119_OPTIMIZER_SHA))):
        raise ValueError("H119 owner 1024-step adapter, checkpoint and optimizer SHA not pinned")
    root=Path(attached)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("owner-private Kaggle source attachments required")
    choices=[p.parent for p in root.rglob("latest.json")
             if (not p.is_symlink() and p.parent.name==H119_ROOT and
                 H119_KERNEL in p.as_posix())]
    if len(choices)!=1:
        raise ValueError("exactly one private H119 1024-step source checkpoint required")
    checkpoint_root=choices[0]
    receipt=checkpoint_root.parent/"hala-h119-private-train-receipt.json"
    if not receipt.is_file() or receipt.is_symlink():
        raise ValueError("H119 real private optimizer receipt missing")
    info=strict_json(receipt.read_text("utf8"))
    version,state=load_checkpoint(checkpoint_root,None)
    runtime=state.get("runtime",{})
    updates=runtime.get("qualification_steps",[])
    outputs={
       "adapter":version/"adapter"/"adapter_model.safetensors",
       "checkpoint":version/"checkpoint.json",
       "optimizer":version/"resume.pt"}
    if any(not p.is_file() or p.is_symlink() for p in outputs.values()):
        raise ValueError("H119 original private final binaries missing")
    actual={key:sha256_file(p) for key,p in outputs.items()}
    expected={"adapter":H119_ADAPTER_SHA,"checkpoint":H119_CHECKPOINT_SHA,
              "optimizer":H119_OPTIMIZER_SHA}
    if actual!=expected:
        raise ValueError("H119 independent owner binary hashes differ")
    if (info.get("status")!="H119_REAL_1024_ORIGINAL_FULL512_ROBUST_TRAIN_STEPS_VERIFIED"
            or state.get("step")!=STEPS or state.get("max_steps")!=STEPS
            or len(updates)!=STEPS
            or info.get("actual_optimizer_steps")!=STEPS
            or info.get("actual_positive_gradient_steps")!=STEPS
            or info.get("original_source_images")!=512
            or info.get("original_h119_derived_sha256")!=EXPECTED_TRAIN_SHA
            or runtime.get("h119_full512_robust_anchor_sampler",{}).get(
                "derived_private_train_sha256")!=EXPECTED_TRAIN_SHA
            or state.get("dataset_sha256")!=TRAIN_SHA
            or state.get("config_sha256")!=expected_h119_training_identity(repo)
            or runtime.get("h106_joint_lora_verified") is not True
            or len(runtime.get("h106_trainable_visual_modules",[]))!=8
            or runtime.get("h119_original_role_processor_audits")!=12
            or any(step.get("step")!=i or
                   step.get("changed_parameter_tensors_this_step",0)<=0 or
                   not isinstance(step.get("gradient_norm_before_clip"),(int,float)) or
                   not math.isfinite(step["gradient_norm_before_clip"]) or
                   step["gradient_norm_before_clip"]<=0
                   for i,step in enumerate(updates,1))
            or info.get("no_EVAL_or_TEST_training") is not True
            or info.get("paid_resources_used") is not False
            or info.get("model_weights_published") is not False
            or actual["adapter"]!=info.get("adapter_sha256")
            or actual["checkpoint"]!=info.get("checkpoint_sha256")
            or actual["optimizer"]!=info.get("optimizer_sha256")):
        raise ValueError("H119 exact original 1024-step TRAIN receipt or optimizer invalid")
    return version/"adapter",{"adapter_sha256":actual["adapter"],
                               "checkpoint_sha256":actual["checkpoint"],
                               "optimizer_sha256":actual["optimizer"],
                               "steps":STEPS}


def _conditioned_inputs(scenes,raw_starts):
    selected=[]
    for scene in scenes:
        anchor=strict_start_anchor(raw_starts.get(scene["id"]),
                                   scene["image_width"],scene["image_height"])
        if anchor is not None:
            item=dict(scene)
            # This request contains original TASK INTENT plus MODEL-ONLY START.
            # Original task start/end boxes never enter a prompt.
            item["prompt"]=prompt_end_delta(scene["task"]["intent"],anchor)
            selected.append(item)
    return selected


def run(*,repo,work,output,attached_inputs=Path("/kaggle/input")):
    require_owner_private_runtime(work=work,output=output)
    repo,work,output=map(Path,(repo,work,output))
    # Model allocation and any EVAL scene download are below binary admission.
    h114,_=admit_h114_checkpoint(attached_inputs)
    h117,_=admit_h117_checkpoint(attached_inputs)
    h119,owner=admit_h119_checkpoint(attached_inputs,repo)
    work.mkdir(parents=True,exist_ok=False)
    frozen_prefixes={}
    for domain in DOMAINS:
        folder=work/domain
        fetch_domain(domain,END,folder)
        framed=apply_frame(load_scenes(folder/"scenes.json"),folder)
        if len(framed)!=END:
            raise ValueError("H120 exact original EVAL 108 prefix unavailable")
        frozen_prefixes[domain]=framed
    all_scenes=freeze_all(frozen_prefixes)
    originals={domain:freeze_domain(frozen_prefixes[domain]) for domain in DOMAINS}
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    if not torch.cuda.is_available() or transformers.__version__!="5.16.1":
        raise ValueError("H120 private free T4 and frozen Qwen3 VL runtime required")
    os.environ.update(HF_HOME=HF_HOME,HF_HUB_DISABLE_TELEMETRY="1",
                      DO_NOT_TRACK="1")
    torch.manual_seed(42)
    processor=AutoProcessor.from_pretrained(
        MODEL_ID,**pinned_cache_kwargs(MODEL_ID,MODEL_REVISION))
    actions={arm:{} for arm in ARMS}
    evidence={arm:{} for arm in ARMS}
    inference_calls={arm:0 for arm in ARMS}
    started=time.monotonic()
    for arm in ARMS:
        base=Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,**pinned_cache_kwargs(MODEL_ID,MODEL_REVISION),
            torch_dtype=torch.float16,attn_implementation="sdpa").to("cuda:0").eval()
        adapter={ARMS[0]:h114,ARMS[1]:h117,ARMS[2]:h119}[arm]
        model=PeftModel.from_pretrained(
            base,adapter,is_trainable=False,trust_remote_code=False).eval()
        for domain in DOMAINS:
            source=originals[domain]
            start_inputs=[role_input(s,"start") for s in source]
            starts,start_evidence=_infer(model,processor,start_inputs,torch)
            input_proof(start_inputs,start_evidence)
            inference_calls[arm]+=len(start_evidence)
            if arm==ARMS[0]:
                end_inputs=[role_input(s,"end") for s in source]
                ends,end_evidence=_infer(model,processor,end_inputs,torch)
                input_proof(end_inputs,end_evidence)
                inference_calls[arm]+=len(end_evidence)
                actions[arm].update({s["id"]:compose_strict_original_drag(
                    s,starts[s["id"]],ends[s["id"]]) for s in source})
                evidence[arm][domain]={"start":start_evidence,"end":end_evidence}
            else:
                conditioned=_conditioned_inputs(source,starts)
                if conditioned:
                    delta_outputs,delta_evidence=_infer(
                        model,processor,conditioned,torch)
                    input_proof(conditioned,delta_evidence)
                else:
                    delta_outputs,delta_evidence={},[]
                inference_calls[arm]+=len(delta_evidence)
                actions[arm].update({s["id"]:compose_model_drag_from_delta(
                    s,starts[s["id"]],delta_outputs[s["id"]])
                    if s["id"] in delta_outputs else None for s in source})
                evidence[arm][domain]={"start":start_evidence,
                                       "delta":delta_evidence}
            print("H120_REAL_NEXT64_STRICT",arm,domain,len(source),flush=True)
        del model,base
        gc.collect()
        torch.cuda.empty_cache()
    if any(set(actions[arm])!={s["id"] for s in all_scenes} for arm in ARMS):
        raise ValueError("H120 paired scene/model missing actions")
    summary=score_paired(all_scenes,actions)
    if not summary["unchanged_original_H61_strict_scorer"]:
        raise ValueError("H120 original scorer changed")
    output.mkdir(parents=True,exist_ok=False)
    private_path=output/"private-raw-neural-attempts.json"
    private_path.write_bytes(canonical(evidence))
    result={**summary,"schema":SUMMARY_SCHEMA,
            "status":"H120_REAL_NEXT64_H114_H117_H119_STRICT_GUI_COMPLETE",
            "source_repo":REPO,"source_commit":REV,
            "source_split":"eval",
            "model_revision":MODEL_REVISION,"model_id":MODEL_ID,
            "model_arms":list(ARMS),
            "model_inference_calls":inference_calls,
            "original_h119_checkpoint":owner,
            "private_raw_attempts_sha256":sha256_file(private_path),
            "training_during_eval":False,
            "model_quality_improvement_verified":False,
            "paid_resources_used":False,
            "hidden_TEST_used":False,
            "elapsed_seconds":round(time.monotonic()-started,2)}
    (output/"h120-private-next64-strict-summary.json").write_bytes(canonical(result))
    return result
