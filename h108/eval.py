"""H108 four-arm original vs visual-ruler action generation, fresh DragOn EVAL.

Pre-registered original upstream 36..43/dom, four domains, 32 source scenes.
Base and exact byte-qualified H106 models, each shown unchanged and ruler
annotated images. Same original H61 strict scorer, ZERO training/new weights.
"""
from __future__ import annotations
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from PIL import Image

from h61.official_eval import (REPO,REV,DOMAINS,canonical,fetch_domain,
                                load_scenes,apply_frame,score)
from h85.model_cache import HF_HOME,pinned_cache_kwargs
from h93.eval import _infer
from h95.eval import normalize_fenced_json
from h100.eval import require_owner_private_runtime
from h107.eval import admit_h106_checkpoint
from h108.ruler import position_ruler,annotated_prompt,GRID_STEP,SCHEMA as RULER_SCHEMA
from training.artifacts import sha256_file
from training.scripts.baseline import MODEL_ID,MODEL_REVISION

START_INDEX=36
PER_DOMAIN=8
FETCH_COUNT=START_INDEX+PER_DOMAIN
ARMS=("base_original","base_ruler","h106_original","h106_ruler")
SCHEMA="hala.h108.private-fresh-source-position-ruler-ab-v1"

def frozen_source(scenes):
    if len(scenes)!=FETCH_COUNT:
        raise ValueError("exact original source EVAL 44-prefix required")
    selected=scenes[START_INDEX:FETCH_COUNT]
    if len(selected)!=PER_DOMAIN or len({s["id"] for s in selected})!=PER_DOMAIN:
        raise ValueError("eight untouched original EVAL scenes required")
    return selected


def variants(scenes,folder):
    """Return unchanged original and altered pixels+prompt with original IDs/boxes."""
    originals=list(scenes)
    result=[]
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=False)
    for index,scene in enumerate(originals):
        actual=Path(scene["image_path"])
        with Image.open(actual) as picture:
            rendered=position_ruler(picture)
            if picture.size!=rendered.size:
                raise ValueError("position grid changed original source dimensions")
        target=folder/f"source-scene-{index:02d}-ruler.png"
        rendered.save(target,format="PNG")
        augmented=dict(scene)
        augmented["image_path"]=str(target)
        augmented["image_sha256"]=sha256_file(target)
        augmented["prompt"]=annotated_prompt(scene["prompt"],scene["image_width"],scene["image_height"])
        if augmented["task"]!=scene["task"]:
            raise ValueError("source original benchmark labels mutated")
        if (augmented["image_width"],augmented["image_height"])!=rendered.size:
            raise ValueError("original H61 source-pixel frame was altered")
        result.append(augmented)
    return originals,result


def run(*,repo,work,output,attached_inputs=Path("/kaggle/input")):
    require_owner_private_runtime(work=work,output=output)
    repo,work,output=map(Path,(repo,work,output))
    adapter,owner_proof=admit_h106_checkpoint(attached_inputs,repo)
    work.mkdir(parents=True,exist_ok=False)
    sources={}
    for domain in DOMAINS:
        folder=work/domain
        fetch_domain(domain,FETCH_COUNT,folder)
        scenes=frozen_source(apply_frame(load_scenes(folder/"scenes.json"),folder))
        originals,ruler=variants(scenes,work/(domain+"-ruler"))
        sources[domain]={"original":originals,"ruler":ruler}

    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    if not torch.cuda.is_available() or transformers.__version__!="5.16.1":
        raise ValueError("real free T4 and pinned Qwen3-VL processor required")
    os.environ.update(HF_HOME=HF_HOME,HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
    torch.manual_seed(42)
    processor=AutoProcessor.from_pretrained(
        MODEL_ID,**pinned_cache_kwargs(MODEL_ID,MODEL_REVISION))
    scores={}
    proofs={}
    began=time.monotonic()
    fields=("scenes","action_valid","start_hit","end_hit",
            "task_success","task_success_rate","degenerate_start_equals_end")
    for arm in ARMS:
        core=Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,**pinned_cache_kwargs(MODEL_ID,MODEL_REVISION),
            torch_dtype=torch.float16,attn_implementation="sdpa").to("cuda:0").eval()
        model=(core if arm.startswith("base_") else PeftModel.from_pretrained(
            core,adapter,is_trainable=False,trust_remote_code=False).eval())
        variant=("ruler" if arm.endswith("_ruler") else "original")
        scores[arm]={};proofs[arm]={}
        for domain in DOMAINS:
            source_original=sources[domain]["original"]
            seen=sources[domain][variant]
            outputs,raw=_infer(model,processor,seen,torch)
            original=score(source_original,outputs)
            repair=score(source_original,{
                k:normalize_fenced_json(v) for k,v in outputs.items()})
            scores[arm][domain]={
              "strict":{k:original[k] for k in fields},
              "format_only_diagnostic":{k:repair[k] for k in fields}}
            proofs[arm][domain]=raw
            expected=[(s["id"],s["image_sha256"],
                       hashlib.sha256(s["prompt"].encode("utf8")).hexdigest())
                      for s in seen]
            observed=[(r["id"],r["image_sha256"],r["prompt_sha256"])
                      for r in raw]
            if expected!=observed:
                raise ValueError("model did not see the expected original/ruler paired screenshot")
            print("H108_REAL_NEW_SOURCE_GRID",arm,domain,len(seen),flush=True)
        del model,core
        gc.collect();torch.cuda.empty_cache()
    output.mkdir(parents=True,exist_ok=False)
    rawpath=output/"private-raw-neural-attempts.json"
    rawpath.write_bytes(canonical(proofs))
    count={arm:sum(scores[arm][domain]["strict"]["scenes"]
                   for domain in DOMAINS) for arm in ARMS}
    if count!={a:32 for a in ARMS}:
        raise ValueError("new benchmark arms not exact 32 source scenes")
    strict={a:sum(scores[a][domain]["strict"]["task_success"]
                  for domain in DOMAINS) for a in ARMS}
    format_only={a:sum(scores[a][domain]["format_only_diagnostic"]["task_success"]
                       for domain in DOMAINS) for a in ARMS}
    report={"schema":SCHEMA,"status":"REAL_H108_ORIGINAL_VS_EXPLICIT_RULER_32_SOURCE_AB",
            "source_repo":REPO,"source_commit":REV,"source_split":"eval",
            "source_selection":"original upstream EVAL indices 36..43 per domain",
            "ruler_schema":RULER_SCHEMA,
            "grid_spacing_source_pixels":GRID_STEP,
            "unaltered_original_source_bbox_scorer":True,
            "max_original_frame_pixels":[1024,640],
            "original_aspect_ratio_preserved":True,
            "ruler_exposes_no_original_bbox_or_answer":True,
            "sample_counts":count,
            "strict_successes":strict,
            "format_only_success_diagnostic":format_only,
            "per_domain_scores":scores,
            "raw_neural_attempts_sha256":sha256_file(rawpath),
            "h106_source_checkpoint":owner_proof,
            "model_id":MODEL_ID,"model_revision":MODEL_REVISION,
            "no_model_weights_modified":True,
            "model_training_executed":False,
            "hidden_test_used":False,
            "production_quality_established":False,
            "private_checkpoint_published":False,
            "paid_resources_used":False,
            "commercial_use":False,
            "elapsed_seconds":round(time.monotonic()-began,2)}
    (output/"h108-private-ruler-ab-summary.json").write_bytes(canonical(report))
    return report
