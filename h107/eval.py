"""H107 preregistered fresh DragOn strict paired model evaluation.

PUBLIC original EVAL indices 20..35 per each of four domains. NOT used in
H93/H95/H98/H100/H101, NOT a hidden TEST or release benchmark. Base/H99/H106
receive *identical* original source images/prompts/scorer/inference conditions.
All original task boxes are used only AFTER model output, for strict scoring.
"""
from __future__ import annotations
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

from h61.official_eval import REPO,REV,DOMAINS,canonical,apply_frame,fetch_domain,load_scenes,score
from h73.kaggle_h73_cpu import TRAIN_SHA
from h85.model_cache import HF_HOME,pinned_cache_kwargs
from h93.eval import _infer
from h95.eval import normalize_fenced_json
from h100.eval import admit_h99_balanced_vision,H99_ADAPTER_SHA,H99_CHECKPOINT_SHA
from h100.eval import require_owner_private_runtime
from h106.research_train import config as h106_config
from h106.training_plan import STRATEGY as H106_STRATEGY,STEPS as H106_STEPS,H105_QUALIFIED_PLAN_SHA
from training.scripts.train_adapter import JOINT_LORA_SCOPE_NONCE
from training.checkpoints import load_checkpoint
from training.artifacts import sha256_file
from training.scripts.baseline import MODEL_ID,MODEL_REVISION

START_INDEX=20
PER_DOMAIN=16
FETCH_COUNT=36
ARMS=("base","h99_vision","h106_joint")
H106_PRIVATE_KERNEL="hala-h106-private-joint-grounding-t4"
H106_OUTPUT_ROOT="hala-h106-private-joint-checkpoint"
SCHEMA="hala.h107.new-source-64-paired-base-h99-h106-native-drag-eval.v1"


def select_fresh(scenes):
    if len(scenes)!=FETCH_COUNT:
        raise ValueError("H107 first untouched EVAL 36-scene source prefix required")
    selected=scenes[START_INDEX:FETCH_COUNT]
    if (len(selected)!=PER_DOMAIN
            or len({row["id"] for row in selected})!=PER_DOMAIN):
        raise ValueError("H107 16 exact new source scenes required")
    return selected


def plan_sha():
    return hashlib.sha256((
        "hala.h106.joint-vision-language-h105-source-prefix256.v1:"+H105_QUALIFIED_PLAN_SHA+
        ":"+str(H106_STEPS)).encode()).hexdigest()


def expected_training_identity(repo):
    cfg=h106_config(Path(repo))
    if cfg.get("max_steps")!=256:
        raise ValueError("H106 original exact training configuration changed")
    signature=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    signature=hashlib.sha256((signature+JOINT_LORA_SCOPE_NONCE).encode()).hexdigest()
    signature=hashlib.sha256((signature+H106_STRATEGY+plan_sha()).encode()).hexdigest()
    return signature


def admit_h106_checkpoint(input_root,repo):
    """Refuse missing, incomplete, stale or tampered actual H106 GPU training."""
    input_root=Path(input_root)
    if input_root.is_symlink() or not input_root.is_dir():
        raise ValueError("attached owner-private H106 Kaggle kernel required")
    pointers=[p.parent for p in input_root.rglob("latest.json")
              if (not p.is_symlink() and
                  p.parent.name==H106_OUTPUT_ROOT and
                  H106_PRIVATE_KERNEL in p.as_posix())]
    if len(pointers)!=1:
        raise ValueError("exactly one real final private H106 optimizer output")
    root=pointers[0]
    receipt=root.parent/"hala-h106-private-joint-training-receipt.json"
    if not receipt.is_file() or receipt.is_symlink():
        raise ValueError("owner-private H106 final optimizer receipt missing")
    report=json.loads(receipt.read_text("utf8"))
    version,state=load_checkpoint(root,None)
    runtime=state.get("runtime",{})
    selected=runtime.get("h106_joint_training_sampler",{})
    visual=runtime.get("h106_trainable_visual_modules",[])
    language=runtime.get("h106_trainable_language_modules",[])
    steps=runtime.get("qualification_steps",[])
    adapter=version/"adapter"
    actual_ckpt=sha256_file(version/"checkpoint.json")
    actual_adapter=sha256_file(adapter/"adapter_model.safetensors")
    if (report.get("status")!="ORIGINAL_H71_REAL_256_JOINT_OPTIMIZER_STEPS_VERIFIED"
            or report.get("new_optimizer_steps")!=H106_STEPS
            or report.get("gradient_bearing_steps")!=H106_STEPS
            or report.get("original_h71_train_sha256")!=TRAIN_SHA
            or report.get("checkpoint_sha256")!=actual_ckpt
            or report.get("adapter_sha256")!=actual_adapter
            or state.get("step")!=H106_STEPS or state.get("max_steps")!=H106_STEPS
            or state.get("dataset_sha256")!=TRAIN_SHA
            or state.get("config_sha256")!=expected_training_identity(repo)
            or selected.get("plan_sha256")!=plan_sha()
            or selected.get("source_split")!="TRAIN_ONLY"
            or selected.get("parent_original_h105_512_plan_sha256")!=H105_QUALIFIED_PLAN_SHA
            or runtime.get("h106_joint_lora_verified") is not True
            or len(visual)!=8 or not 16<=len(language)<=256
            or runtime.get("trainable_parameter_tensors")!=2*(len(visual)+len(language))
            or len(steps)!=H106_STEPS or any(
                s.get("step")!=i or s.get("changed_parameter_tensors_this_step",0)<=0
                or not math.isfinite(s.get("gradient_norm_before_clip",float("nan")))
                or s["gradient_norm_before_clip"]<=0
                for i,s in enumerate(steps,1))
            or not (adapter/"adapter_config.json").is_file()):
        raise ValueError("H106 original PRIVATE 256-step joint checkpoint not qualified")
    return adapter,{"checkpoint_sha256":actual_ckpt,
                    "adapter_sha256":actual_adapter,
                    "source_train_sha256":state["dataset_sha256"],
                    "trained_steps":state["step"],
                    "plan_sha256":selected["plan_sha256"]}


def run(*,repo,work,output,attached_inputs=Path("/kaggle/input")):
    require_owner_private_runtime(work=work,output=output)
    repo,work,output=map(Path,(repo,work,output))
    # Verify EXACT owner-private weights, optimizer and source *before* model inference.
    h99=admit_h99_balanced_vision(attached_inputs,repo)
    h106,training=admit_h106_checkpoint(attached_inputs,repo)
    work.mkdir(parents=True,exist_ok=False)
    scenes={}
    for domain in DOMAINS:
        folder=work/domain
        fetch_domain(domain,FETCH_COUNT,folder)
        candidates=select_fresh(apply_frame(load_scenes(folder/"scenes.json"),folder))
        if any(c["domain"]!=domain for c in candidates):
            raise ValueError("upstream source domain mismatch")
        scenes[domain]=candidates

    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    if not torch.cuda.is_available() or transformers.__version__!="5.16.1":
        raise ValueError("free Kaggle T4 and pinned processor needed")
    os.environ.update(HF_HOME=HF_HOME,HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
    torch.manual_seed(42)
    processor=AutoProcessor.from_pretrained(
        MODEL_ID,**pinned_cache_kwargs(MODEL_ID,MODEL_REVISION))
    scores={}
    proof={}
    started=time.monotonic()
    fields=("scenes","action_valid","start_hit","end_hit",
            "task_success","task_success_rate","degenerate_start_equals_end")
    for arm in ARMS:
        core=Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,**pinned_cache_kwargs(MODEL_ID,MODEL_REVISION),
            torch_dtype=torch.float16,attn_implementation="sdpa").to("cuda:0").eval()
        model=(core if arm=="base" else PeftModel.from_pretrained(
            core,h99 if arm=="h99_vision" else h106,
            is_trainable=False,trust_remote_code=False).eval())
        scores[arm],proof[arm]={},{}
        for domain in DOMAINS:
            original=scenes[domain]
            outputs,attempts=_infer(model,processor,original,torch)
            strict=score(original,outputs)
            format_only=score(original,{
                key:normalize_fenced_json(value) for key,value in outputs.items()})
            scores[arm][domain]={
                "strict":{k:strict[k] for k in fields},
                "format_only_diagnostic":{k:format_only[k] for k in fields}}
            proof[arm][domain]=attempts
            print("H107_REAL_NEW_STRICT_GUI",arm,domain,len(original),flush=True)
        del model,core
        gc.collect()
        torch.cuda.empty_cache()
    output.mkdir(parents=True,exist_ok=False)
    raw=output/"private-raw-neural-attempts.json"
    raw.write_bytes(canonical(proof))
    sample_counts={arm:sum(scores[arm][d]["strict"]["scenes"] for d in DOMAINS) for arm in ARMS}
    expected={arm:len(DOMAINS)*PER_DOMAIN for arm in ARMS}
    if sample_counts!=expected:
        raise ValueError("real H107 comparison not 64 paired scenes for every arm")
    # Every arm must have identical source-media+prompt hashes per scene.
    for dom in DOMAINS:
        source_signatures=[
            [(x["id"],x["image_sha256"],x["prompt_sha256"])
             for x in proof[arm][dom]] for arm in ARMS]
        if source_signatures[0]!=source_signatures[1] or source_signatures[0]!=source_signatures[2]:
            raise ValueError("H107 arms received different source scenes or prompts")
    strict_success={arm:sum(scores[arm][d]["strict"]["task_success"]
                            for d in DOMAINS) for arm in ARMS}
    fmt_success={arm:sum(scores[arm][d]["format_only_diagnostic"]["task_success"]
                         for d in DOMAINS) for arm in ARMS}
    report={
        "schema":SCHEMA,
        "status":"REAL_H107_FRESH_PAIRED_64_SOURCE_STRICT_EVAL_COMPLETE",
        "source_repo":REPO,"source_commit":REV,"source_split":"eval",
        "original_scene_selection":"original public upstream DragOn EVAL indices 20..35 per four domains",
        "samples_per_domain":PER_DOMAIN,"total_distinct_source_cases":64,
        "sample_counts":sample_counts,
        "strict_successes":strict_success,
        "format_only_successes_diagnostic":fmt_success,
        "per_domain_scores":scores,
        "h99_adapter_sha256":H99_ADAPTER_SHA,
        "h99_checkpoint_sha256":H99_CHECKPOINT_SHA,
        "h106_verified_original_train":training,
        "model_id":MODEL_ID,"model_revision":MODEL_REVISION,
        "source_image_prompt_actions_paired":True,
        "original_H61_scorer_modified":False,
        "inference_settings_identical":True,
        "raw_neural_attempts_sha256":sha256_file(raw),
        "training_executed_in_evaluation":False,
        "not_hidden_test":True,"not_bilingual_skill_qualification":True,
        "production_or_release_qualified":False,
        "model_improvement_claimed":False,
        "private_weights_and_raw_data_preserved":True,
        "commercial_use":False,
        "elapsed_seconds":round(time.monotonic()-started,1)}
    (output/"h107-private-fresh-strict-eval-summary.json").write_bytes(canonical(report))
    return report
