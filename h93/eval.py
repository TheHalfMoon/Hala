"""H93: REAL paired unchanged-Qwen vs H92-adapter on official DragOn eval.

Private noncommercial Kaggle ONLY, no training. Reuse H61 upstream pinned
source, frame, prompts, and independent geometry scorer unchanged. This
bounded sample is diagnostic, not release or bilingual qualification.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import time

from h61.official_eval import DOMAINS, REV, REPO, apply_frame, canonical, fetch_domain, load_scenes, score
from h85.model_cache import HF_HOME, pinned_cache_kwargs
from h92.research_train import H87_ADAPTER_SHA, H92_STEPS, TRAIN_SHA, research_configuration
from training.artifacts import sha256_file
from training.checkpoints import load_checkpoint
from training.scripts.baseline import MODEL_ID, MODEL_REVISION

H92_KERNEL_SLUG="hala-h92-private-qwen-h87-warmstart-eight-steps"
H92_ADAPTER_SHA="85fa3f6da1dd1b2ddd86d3db795ea19858b3e58bb12305abe2094c2a22e52497"
H92_CHECKPOINT_SHA="6e3a2c678da49001e7309096ff33dcae8df70a0fa9a41c0cfd00b46ec0273d88"
MAX_SCENES_PER_DOMAIN=4
MAX_GENERATED_TOKENS=128
SCHEMA="hala.h93.private-official-eval-paired-real-qwen-v1"


def admit_h92_adapter(attached_inputs:Path, repo:Path)->Path:
    """Source/optimizer/checkpoint identity must match verified H92, or fail."""
    root=Path(attached_inputs)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("attached private Kaggle kernel inputs required")
    choices=[]
    for p in root.rglob("latest.json"):
        if (not p.is_symlink() and p.parent.name=="hala-h92-private-checkpoint"
                and H92_KERNEL_SLUG in p.as_posix()):
            choices.append(p.parent)
    if len(choices)!=1:
        raise ValueError("exactly one owner-private H92 checkpoint source is required")
    version,state=load_checkpoint(choices[0],None)
    path=version/"adapter"
    weights=path/"adapter_model.safetensors"
    config=research_configuration(repo)
    cfg_digest=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    signed_digest=hashlib.sha256((cfg_digest+H87_ADAPTER_SHA).encode()).hexdigest()
    if (not weights.is_file() or weights.is_symlink()
            or not (path/"adapter_config.json").is_file()
            or sha256_file(weights)!=H92_ADAPTER_SHA
            or sha256_file(version/"checkpoint.json")!=H92_CHECKPOINT_SHA
            or state["step"]!=H92_STEPS or state["max_steps"]!=H92_STEPS
            or state["dataset_sha256"]!=TRAIN_SHA
            or state["config_sha256"]!=signed_digest
            or state["changed_parameter_tensors"]<=0):
        raise ValueError("H92 adapter/model training source identity mismatch")
    return path


def require_research_runtime():
    if (platform.system()!="Linux" or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
            or os.environ.get("HALA_NONCOMMERCIAL_RESEARCH")!="1"
            or not Path("/kaggle/working").is_dir()):
        raise ValueError("approved personal noncommercial Kaggle evaluation only")


def _infer(model,processor,scenes,torch):
    from PIL import Image
    outputs={}
    proofs=[]
    for scene in scenes:
        with Image.open(scene["image_path"]) as image:
            picture=image.convert("RGB")
        messages=[{"role":"user","content":[{"type":"image"},
                    {"type":"text","text":scene["prompt"]}]}]
        rendered=processor.apply_chat_template(messages,tokenize=False,
                                               add_generation_prompt=True)
        inputs=processor(text=[rendered],images=[picture],
                         return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            predicted=model.generate(**inputs,max_new_tokens=MAX_GENERATED_TOKENS,
                                     do_sample=False,max_time=45)
        torch.cuda.synchronize(0)
        raw=processor.batch_decode(predicted[:,inputs.input_ids.shape[1]:],
                                   skip_special_tokens=True)[0]
        if not isinstance(raw,str) or len(raw)>8192:
            raise ValueError("unbounded neural action output")
        outputs[scene["id"]]=raw
        proofs.append({"id":scene["id"],"image_sha256":scene["image_sha256"],
                       "prompt_sha256":hashlib.sha256(scene["prompt"].encode()).hexdigest(),
                       "output_sha256":hashlib.sha256(raw.encode()).hexdigest(),
                       "raw_output":raw})
        del inputs,predicted
    return outputs,proofs


def run(*,repo:Path,work:Path,output:Path,attached_inputs:Path=Path("/kaggle/input")):
    require_research_runtime()
    if output.exists() or not str(output).startswith("/kaggle/working/"):
        raise ValueError("create-only Kaggle private evaluation output required")
    adapter=admit_h92_adapter(attached_inputs,repo)
    if not str(work).startswith("/kaggle/tmp/"):
        raise ValueError("official evaluation scratch must remain private temporary storage")
    work.mkdir(parents=True,exist_ok=False)
    original={}
    for domain in DOMAINS:
        folder=work/domain
        source=fetch_domain(domain,MAX_SCENES_PER_DOMAIN,folder)
        scenes=apply_frame(load_scenes(folder/"scenes.json"),folder)
        if not scenes or len(scenes)>MAX_SCENES_PER_DOMAIN:
            raise ValueError("empty or unbounded original official eval domain")
        original[domain]=scenes
    import torch
    import transformers
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from peft import PeftModel
    if (not torch.cuda.is_available() or transformers.__version__!="5.16.1"):
        raise ValueError("qualified real GPU / pinned transformers required")
    os.environ.update(HF_HOME=HF_HOME, HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
    torch.manual_seed(42)
    processor=AutoProcessor.from_pretrained(
        MODEL_ID,**pinned_cache_kwargs(MODEL_ID,MODEL_REVISION))
    model=Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,**pinned_cache_kwargs(MODEL_ID,MODEL_REVISION),
        torch_dtype=torch.float16,attn_implementation="sdpa").to("cuda:0").eval()
    raw_evidence={}
    scores={}
    started=time.monotonic()
    for label in ("base","h92"):
        if label=="h92":
            model=PeftModel.from_pretrained(
                model,adapter,is_trainable=False,trust_remote_code=False).eval()
        group={}
        raw_evidence[label]={}
        for domain,scenes in original.items():
            outputs,proofs=_infer(model,processor,scenes,torch)
            actual=score(scenes,outputs)
            group[domain]={k:actual[k] for k in
                ("scenes","action_valid","start_hit","end_hit","task_success",
                 "task_success_rate","degenerate_start_equals_end")}
            raw_evidence[label][domain]=proofs
            print("HALA_H93_REAL_EVAL_CASES",label,domain,len(scenes),flush=True)
        scores[label]=group
    output.mkdir(parents=True,exist_ok=False)
    (output/"private-raw-neural-attempts.json").write_bytes(canonical(raw_evidence))
    total={key:sum(group[d]["scenes"] for d in DOMAINS) for key,group in scores.items()}
    successes={key:sum(group[d]["task_success"] for d in DOMAINS) for key,group in scores.items()}
    if total["base"]!=total["h92"] or total["base"]<=0:
        raise ValueError("unpaired or empty benchmark evaluation sample")
    report={
        "schema":SCHEMA,"status":"REAL_PAIRED_OFFICIAL_PUBLIC_EVAL_DIAGNOSTIC",
        "model_id":MODEL_ID,"model_revision":MODEL_REVISION,
        "source_repo":REPO,"source_commit":REV,"source_split":"eval",
        "sample_per_domain_requested":MAX_SCENES_PER_DOMAIN,
        "sample_counts":total,
        "base_successes":successes["base"],"h92_successes":successes["h92"],
        "base_success_rate":successes["base"]/total["base"],
        "h92_success_rate":successes["h92"]/total["h92"],
        "paired_delta_successes":successes["h92"]-successes["base"],
        "domain_scores":scores,
        "h92_adapter_sha256":H92_ADAPTER_SHA,
        "h92_checkpoint_sha256":H92_CHECKPOINT_SHA,
        "raw_attempts_sha256":sha256_file(output/"private-raw-neural-attempts.json"),
        "elapsed_seconds":round(time.monotonic()-started,1),
        "training_executed_in_evaluation":False,
        "english_arabic_bilingual_qualified":False,
        "statistically_reliable_benchmark":False,
        "full_official_evaluation_completed":False,
        "public_model_released":False,
        "commercial_use":False
    }
    (output/"h93-private-eval-summary.json").write_bytes(canonical(report))
    return report
