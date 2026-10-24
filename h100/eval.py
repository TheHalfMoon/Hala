"""H100: disjoint official public DragOn cases 12-15: Qwen/H94/H97/H99.

Uses pinned original H61 geometry scorer unchanged. Indices 8..11/domain
are disjoint from H93 (0..3), H95 (4..7) and H98 (8..11). Raw outputs PRIVATE.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import gc
from pathlib import Path

from h61.official_eval import (
    REPO, REV, DOMAINS, apply_frame, canonical, fetch_domain, load_scenes, score)
from h73.kaggle_h73_cpu import TRAIN_SHA
from h85.model_cache import HF_HOME,pinned_cache_kwargs
from h93.eval import _infer
from h95.eval import admit_h94_adapter, normalize_fenced_json
from h97.research_train import STEPS, training_config
from h99.research_train import STEPS as H99_STEPS, training_config as h99_training_config
from h99.sampler import STRATEGY as H99_STRATEGY
from training.scripts.train_adapter import VISION_LORA_SCOPE_NONCE
from training.artifacts import sha256_file
from training.checkpoints import load_checkpoint
from training.scripts.baseline import MODEL_ID,MODEL_REVISION

H97_KERNEL="hala-h97-private-vision-thirtytwo-step"
H97_ADAPTER_SHA="5dbc350c56dcdbbcec5feda97a300d925e48e7e6b11d5bdb51e3bdcd8632fedb"
H97_CHECKPOINT_SHA="16eb940c91e7e4fc8599019a722819e8867ad9eabc60571336969fa5ebf101b1"
H99_KERNEL="hala-h99-private-balanced-visual-128"
H99_ADAPTER_SHA="701109c5fbf61df8a101ae6c23093e6b1e35224b701a4983c5f4f0784d915eb1"
H99_CHECKPOINT_SHA="d15c8bf63b1ebb48897c597c5502b30457e962671d5c5251e8bce5c2cf51a613"
H99_PLAN_SHA="76ea4d7105fd2e3cce02d9e009e3df6c8fce20e69e877495a2320b6ae86c7c63"
START_INDEX=12
PER_DOMAIN=4
FETCH_COUNT=START_INDEX+PER_DOMAIN
ARMS=("base","h94_language","h97_vision","h99_balanced_vision")
SCHEMA="hala.h100.private-fresh-original-drag-eval-fourway.v1"


def admit_h97_vision(root,repo):
    root=Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("authorized private Kaggle kernel input root required")
    choices=[pointer.parent for pointer in root.rglob("latest.json")
             if (not pointer.is_symlink()
                 and pointer.parent.name=="hala-h97-vision-checkpoint"
                 and H97_KERNEL in pointer.as_posix())]
    if len(choices)!=1:
        raise ValueError("exactly one owner-private H97 optimizer checkpoint required")
    version,state=load_checkpoint(choices[0],None)
    adapter=version/"adapter"
    weights=adapter/"adapter_model.safetensors"
    cfg=training_config(Path(repo))
    cfg_sha=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    expected_sha=hashlib.sha256((cfg_sha+VISION_LORA_SCOPE_NONCE).encode()).hexdigest()
    expected={f"model.visual.blocks.{i}.attn.{part}"
              for i in range(20,24) for part in ("qkv","proj")}
    runtime=state.get("runtime",{})
    if (weights.is_symlink() or not weights.is_file()
            or not (adapter/"adapter_config.json").is_file()
            or sha256_file(weights)!=H97_ADAPTER_SHA
            or sha256_file(version/"checkpoint.json")!=H97_CHECKPOINT_SHA
            or state.get("config_sha256")!=expected_sha
            or state.get("dataset_sha256")!=TRAIN_SHA
            or state.get("step")!=STEPS or state.get("max_steps")!=STEPS
            or runtime.get("h96_vision_lora_only") is not True
            or set(runtime.get("h96_vision_target_modules",[]))!=expected
            or runtime.get("trainable_parameter_tensors")!=16):
        raise ValueError("H97 immutable visual adapter identity was not qualified")
    return adapter


def admit_h99_balanced_vision(root,repo):
    """Exact private H99 real final step-128 optimizer, adapter and plan SHA."""
    root=Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("actual private H99 Kaggle kernel source required")
    choices=[pointer.parent for pointer in root.rglob("latest.json")
             if (not pointer.is_symlink()
                 and pointer.parent.name=="hala-h99-balanced-visual-checkpoint"
                 and H99_KERNEL in pointer.as_posix())]
    if len(choices)!=1:
        raise ValueError("exactly one attached private H99 checkpoint allowed")
    version,state=load_checkpoint(choices[0],None)
    adapter=version/"adapter"
    weights=adapter/"adapter_model.safetensors"
    cfg=h99_training_config(Path(repo))
    cfg_digest=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    vision_digest=hashlib.sha256((cfg_digest+VISION_LORA_SCOPE_NONCE).encode()).hexdigest()
    model_digest=hashlib.sha256((vision_digest+H99_STRATEGY+H99_PLAN_SHA).encode()).hexdigest()
    runtime=state.get("runtime",{})
    sampler=runtime.get("h99_training_sampler",{})
    targets={f"model.visual.blocks.{i}.attn.{part}"
             for i in range(20,24) for part in ("qkv","proj")}
    if (weights.is_symlink() or not weights.is_file()
            or not (adapter/"adapter_config.json").is_file()
            or sha256_file(weights)!=H99_ADAPTER_SHA
            or sha256_file(version/"checkpoint.json")!=H99_CHECKPOINT_SHA
            or state.get("config_sha256")!=model_digest
            or state.get("dataset_sha256")!=TRAIN_SHA
            or state.get("step")!=H99_STEPS
            or state.get("max_steps")!=H99_STEPS
            or runtime.get("h96_vision_lora_only") is not True
            or runtime.get("trainable_parameter_tensors")!=16
            or set(runtime.get("h96_vision_target_modules",[]))!=targets
            or sampler.get("plan_sha256")!=H99_PLAN_SHA
            or sampler.get("unique_source_images_selected")!=H99_STEPS
            or set(sampler.get("selected_languages",{}))!={"ar","en"}
            or len(sampler.get("selected_families",{}))<4):
        raise ValueError("H99 source+optimizer+vision+balanced plan SHA failed")
    return adapter


def require_owner_private_runtime(*,env=None,system=None,work=None,output=None):
    import platform
    env=os.environ if env is None else env
    system=platform.system() if system is None else system
    if (system!="Linux" or not env.get("KAGGLE_KERNEL_RUN_TYPE")
            or env.get("HALA_NONCOMMERCIAL_RESEARCH")!="1"
            or work is None or output is None
            or not str(work).startswith("/kaggle/tmp/")
            or not str(output).startswith("/kaggle/working/")
            or Path(output).exists()):
        raise ValueError("approved read-only private owner Kaggle evaluation required")


def run(*,repo,work,output,attached_inputs=Path("/kaggle/input")):
    require_owner_private_runtime(work=work,output=output)
    repo,work,output=Path(repo),Path(work),Path(output)
    # SHA-bind full private checkpoint and optimizer BEFORE model allocation.
    language=admit_h94_adapter(attached_inputs,repo)
    vision=admit_h97_vision(attached_inputs,repo)
    balanced=admit_h99_balanced_vision(attached_inputs,repo)
    work.mkdir(parents=True,exist_ok=False)
    selected={}
    for domain in DOMAINS:
        folder=work/domain
        fetch_domain(domain,FETCH_COUNT,folder)
        scenes=apply_frame(load_scenes(folder/"scenes.json"),folder)
        if len(scenes)!=FETCH_COUNT:
            raise ValueError("original official pinned EVAL scenes missing")
        selected[domain]=scenes[START_INDEX:FETCH_COUNT]
        if len(selected[domain])!=PER_DOMAIN:
            raise ValueError("fixed NEW four per domain missing")
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    if not torch.cuda.is_available() or transformers.__version__!="5.16.1":
        raise ValueError("genuine pinned CUDA provider experiment required")
    os.environ.update(HF_HOME=HF_HOME,HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
    torch.manual_seed(42)
    processor=AutoProcessor.from_pretrained(
        MODEL_ID,**pinned_cache_kwargs(MODEL_ID,MODEL_REVISION))
    scores={}
    proofs={}
    started=time.monotonic()
    for label in ARMS:
        # Reload pretrained base for every arm. DO NOT mix adapters with
        # different target structures or leak the previous arm into next.
        core=Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,**pinned_cache_kwargs(MODEL_ID,MODEL_REVISION),
            torch_dtype=torch.float16,attn_implementation="sdpa").to("cuda:0").eval()
        if label=="base":
            model=core
        else:
            adapter=(language if label=="h94_language" else
                     vision if label=="h97_vision" else balanced)
            model=PeftModel.from_pretrained(
                core,adapter,is_trainable=False,
                trust_remote_code=False).eval()
        scores[label]={}
        proofs[label]={}
        for domain in DOMAINS:
            scenes=selected[domain]
            outputs,attempts=_infer(model,processor,scenes,torch)
            strict=score(scenes,outputs)
            diagnostic=score(scenes,{
                key:normalize_fenced_json(value) for key,value in outputs.items()})
            fields=("scenes","action_valid","start_hit","end_hit",
                    "task_success","task_success_rate",
                    "degenerate_start_equals_end")
            scores[label][domain]={
                "strict":{k:strict[k] for k in fields},
                "format_only_diagnostic":{k:diagnostic[k] for k in fields}}
            proofs[label][domain]=attempts
            print("H100_PRIVATE_NEW_SOURCE_SCENE_EVAL",label,domain,len(scenes),flush=True)
        del model,core
        gc.collect()
        torch.cuda.empty_cache()
    output.mkdir(parents=True,exist_ok=False)
    path=output/"private-raw-neural-attempts.json"
    path.write_bytes(canonical(proofs))
    counts={label:sum(scores[label][d]["strict"]["scenes"] for d in DOMAINS)
            for label in ARMS}
    if counts!={name:16 for name in ARMS}:
        raise ValueError("failed matched four-arm 16-scene benchmark")
    strict_successes={
        label:sum(scores[label][d]["strict"]["task_success"] for d in DOMAINS)
        for label in ARMS}
    format_successes={
        label:sum(scores[label][d]["format_only_diagnostic"]["task_success"] for d in DOMAINS)
        for label in ARMS}
    report={
        "schema":SCHEMA,
        "status":"REAL_NEW_FOURWAY_BALANCED_VISUAL_VS_OTHERS_STRICT_EVAL",
        "source_repo":REPO,"source_commit":REV,"source_split":"eval",
        "sample_selection":"fixed original upstream EVAL ordered indices 12..15 per domain",
        "samples_per_domain":PER_DOMAIN,
        "sample_counts":counts,
        "strict_successes":strict_successes,
        "format_only_successes_diagnostic":format_successes,
        "per_domain_scores":scores,
        "h97_vision_adapter_sha256":H97_ADAPTER_SHA,
        "h97_vision_checkpoint_sha256":H97_CHECKPOINT_SHA,
        "h99_balanced_adapter_sha256":H99_ADAPTER_SHA,
        "h99_balanced_checkpoint_sha256":H99_CHECKPOINT_SHA,
        "h99_ordered_training_plan_sha256":H99_PLAN_SHA,
        "model_id":MODEL_ID,"model_revision":MODEL_REVISION,
        "raw_evidence_sha256":sha256_file(path),
        "official_strict_scorer_modified":False,
        "training_performed_during_evaluation":False,
        "small_public_eval_not_hidden_test":True,
        "bilingual_model_qualified":False,
        "commercial_use":False,"public_model_released":False,
        "elapsed_seconds":round(time.monotonic()-started,1)}
    (output/"h100-private-eval-summary.json").write_bytes(canonical(report))
    return report
