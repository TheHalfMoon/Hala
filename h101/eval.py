"""H101 controlled geometry-instruction intervention on disjoint DragOn EVAL.

No training, no model adaptation, no box or label exposure to inference.
Four matched arms: base/H99 x untouched-H61/scaffold. H61 score unchanged.
"""
from __future__ import annotations
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from h61.official_eval import REPO, REV, DOMAINS, apply_frame, canonical, fetch_domain, load_scenes, score
from h73.kaggle_h73_cpu import TRAIN_SHA
from h85.model_cache import HF_HOME, pinned_cache_kwargs
from h93.eval import _infer
from h95.eval import normalize_fenced_json
from h100.eval import admit_h99_balanced_vision, require_owner_private_runtime, H99_ADAPTER_SHA, H99_CHECKPOINT_SHA, H99_PLAN_SHA
from training.artifacts import sha256_file
from training.scripts.baseline import MODEL_ID, MODEL_REVISION

START_INDEX = 16
PER_DOMAIN = 4
FETCH_COUNT = START_INDEX + PER_DOMAIN
SCHEMA = "hala.h101.private-new-geometry-prompt-controlled-eval.v1"
ARMS = ("base_original", "base_geometry_prompt", "h99_original", "h99_geometry_prompt")
INTERVENTION_REV = "hala.h101.geometry-instruction-only-v1"
INSTRUCTION = (
    "\nVisual coordinate check (perform silently, do not print your reasoning): "
    "Locate the draggable START element within the screenshot, then locate the "
    "intended END destination independently. Estimate center pixel coordinates for "
    "both visual targets in the full original-image coordinate system, not relative "
    "to a crop. Use a distinct end position when the two visual targets differ; "
    "never default to start=end. Check left/right and up/down direction and "
    "ensure both points lie inside the image bounds. "
    "Output exactly ONE raw JSON object, without Markdown fences, commentary "
    "or additional keys. Retain the five required drag fields and the specified "
    "ordered flag."
)

def coordinate_prompt_from_scene(scene):
    """Text-only change: absolutely never reads task boxes or labels."""
    original = scene["prompt"]
    width, height = scene["image_width"], scene["image_height"]
    if (not isinstance(original, str) or not original
            or type(width) is not int or type(height) is not int
            or width < 1 or height < 1 or len(original) > 4096):
        raise ValueError("source-bound original scene prompt required")
    return original + "\nFull image bounds: x=0.." + str(width-1) + ", y=0.." + str(height-1) + "." + INSTRUCTION

def select_unseen(scenes):
    if len(scenes) != FETCH_COUNT:
        raise ValueError("fixed original DragOn EVAL source selection requires 20 scenes")
    selected=scenes[START_INDEX:FETCH_COUNT]
    if len(selected)!=PER_DOMAIN or len({s["id"] for s in selected})!=PER_DOMAIN:
        raise ValueError("duplicate/missing fresh EVAL scenes")
    return selected

def run(*, repo, work, output, attached_inputs=Path("/kaggle/input")):
    require_owner_private_runtime(work=work,output=output)
    repo,work,output=Path(repo),Path(work),Path(output)
    adapter=admit_h99_balanced_vision(attached_inputs,repo)
    work.mkdir(parents=True,exist_ok=False)
    original={}
    for domain in DOMAINS:
        folder=work/domain
        fetch_domain(domain,FETCH_COUNT,folder)
        original[domain]=select_unseen(apply_frame(load_scenes(folder/"scenes.json"),folder))
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    if not torch.cuda.is_available() or transformers.__version__!="5.16.1":
        raise ValueError("real pinned CUDA evaluation required")
    os.environ.update(HF_HOME=HF_HOME,HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
    torch.manual_seed(42)
    processor=AutoProcessor.from_pretrained(MODEL_ID,**pinned_cache_kwargs(MODEL_ID,MODEL_REVISION))
    scores, proofs={},{}
    started=time.monotonic()
    for arm in ARMS:
        core=Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,**pinned_cache_kwargs(MODEL_ID,MODEL_REVISION),
            torch_dtype=torch.float16,attn_implementation="sdpa").to("cuda:0").eval()
        model=core if arm.startswith("base_") else PeftModel.from_pretrained(
            core,adapter,is_trainable=False,trust_remote_code=False).eval()
        scores[arm],proofs[arm]={},{}
        for domain in DOMAINS:
            canonical_scenes=original[domain]
            # Source scene task/score remain identical across both prompt conditions.
            inference_scenes=([dict(s,prompt=coordinate_prompt_from_scene(s)) for s in canonical_scenes]
                              if arm.endswith("_geometry_prompt") else canonical_scenes)
            outputs,attempts=_infer(model,processor,inference_scenes,torch)
            strict=score(canonical_scenes,outputs)
            fmt=score(canonical_scenes,{k:normalize_fenced_json(v) for k,v in outputs.items()})
            fields=("scenes","action_valid","start_hit","end_hit",
                    "task_success","task_success_rate","degenerate_start_equals_end")
            scores[arm][domain]={
                "strict":{k:strict[k] for k in fields},
                "format_only_diagnostic":{k:fmt[k] for k in fields}}
            proofs[arm][domain]=attempts
            print("H101_PRIVATE_NEW_EVAL",arm,domain,len(canonical_scenes),flush=True)
        del model,core
        gc.collect()
        torch.cuda.empty_cache()
    output.mkdir(parents=True,exist_ok=False)
    raw=output/"private-raw-neural-attempts.json"
    raw.write_bytes(canonical(proofs))
    counts={arm:sum(scores[arm][dom]["strict"]["scenes"] for dom in DOMAINS) for arm in ARMS}
    if counts!={arm:16 for arm in ARMS}:
        raise ValueError("paired original source sample incomplete")
    strict_successes={arm:sum(scores[arm][dom]["strict"]["task_success"] for dom in DOMAINS) for arm in ARMS}
    fmt_successes={arm:sum(scores[arm][dom]["format_only_diagnostic"]["task_success"] for dom in DOMAINS) for arm in ARMS}
    report={"schema":SCHEMA,"status":"REAL_H101_CONTROLLED_NEW_UNSEEN_PROMPT_GEOMETRY_EVAL",
            "source_repo":REPO,"source_commit":REV,"source_split":"eval",
            "sample_selection":"fixed NEW original upstream DragOn EVAL indices 16..19 per domain",
            "samples_per_domain":PER_DOMAIN,"sample_counts":counts,
            "strict_successes":strict_successes,
            "format_only_successes_diagnostic":fmt_successes,
            "per_domain_scores":scores,
            "h99_adapter_sha256":H99_ADAPTER_SHA,
            "h99_checkpoint_sha256":H99_CHECKPOINT_SHA,
            "h99_sampler_plan_sha256":H99_PLAN_SHA,
            "intervention_revision":INTERVENTION_REV,
            "intervention_sha256":hashlib.sha256(INSTRUCTION.encode("utf8")).hexdigest(),
            "original_h61_prompt_preserved_for_control":True,
            "model_id":MODEL_ID,"model_revision":MODEL_REVISION,
            "raw_evidence_sha256":sha256_file(raw),
            "official_strict_scorer_modified":False,
            "training_performed_during_evaluation":False,
            "small_public_eval_not_hidden_test":True,
            "bilingual_model_qualified":False,
            "commercial_use":False,"public_model_released":False,
            "elapsed_seconds":round(time.monotonic()-started,1)}
    (output/"h101-private-eval-summary.json").write_bytes(canonical(report))
    return report
