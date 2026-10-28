"""H167: actual synthetic bilingual T4 feasibility of frozen H164 language LoRA plus vision LoRA.

No original screens, no consumed DEV/OOD, no native success claim, no weights saved.
"""
import hashlib
import importlib.metadata
import json
import math
import os
import zipfile
from pathlib import Path

from h132.grounder_contract import MODEL_ID,MODEL_SHA
from h166.vision_probe import (
    synth,require_free_t4,visual_merger_targets,MAX_PEAK_GIB,
    EXPECTED_TORCH,EXPECTED_TRANSFORMERS,EXPECTED_PEFT)

LANGUAGE_ZIP_SHA="4f03a7520c69f2e4e5f2546ff1c05d975e163089c215bfd935eda674927cf8da"
LANGUAGE_WEIGHTS_SHA="43e0f6a260a40bda0fb5604aabde897d8aa3fdf1ced2877a32babddabbcf6b88"
EXPECTED_ZIP_FILES={"README.md","adapter_config.json","adapter_model.safetensors"}
EXPECTED_RANK=8
VISUAL_RANK=4

def file_sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def admit_frozen_language_zip(package,destination):
    """Validate exact previously trained H164 weights, never TRAIN/DEV/consumed OOD media."""
    package=Path(package)
    destination=Path(destination)
    if destination.exists() or file_sha(package)!=LANGUAGE_ZIP_SHA:
        raise ValueError("H167 H164-language frozen ZIP source mismatch")
    with zipfile.ZipFile(package) as z:
        if set(z.namelist())!=EXPECTED_ZIP_FILES:
            raise ValueError("H167 H164-language archive contains unexpected files")
        config=json.loads(z.read("adapter_config.json"))
        if (config.get("r")!=EXPECTED_RANK or
            config.get("base_model_name_or_path")!=MODEL_ID or
            config.get("peft_type")!="LORA" or
            config.get("inference_mode") is not True or
            config.get("target_modules") is None or
            "visual" in json.dumps(config.get("target_modules"))):
            raise ValueError("H167 parent research adapter is not frozen rank8 language")
        weights=z.read("adapter_model.safetensors")
        if hashlib.sha256(weights).hexdigest()!=LANGUAGE_WEIGHTS_SHA:
            raise ValueError("H167 frozen H164 trained adapter weights mismatch")
        destination.mkdir(parents=True,exist_ok=False)
        for name in sorted(EXPECTED_ZIP_FILES):
            (destination/name).write_bytes(z.read(name))
    return destination

def is_visual_trainable(name):
    return (".visual.merger." in ("."+name+".")
            and (".lora_A.vision." in name or ".lora_B.vision." in name))

def is_language_adapter(name):
    return ".lora_A.language." in name or ".lora_B.language." in name

def check_adapter_partition(named_parameters):
    language=[]
    vision=[]
    wrong=[]
    for name,param in named_parameters:
        if is_language_adapter(name):
            language.append((name,param))
            if param.requires_grad: wrong.append(name)
        elif is_visual_trainable(name):
            vision.append((name,param))
            if not param.requires_grad: wrong.append(name)
        elif param.requires_grad:
            wrong.append(name)
    if wrong or not language or len(vision)!=4 or len(set(n for n,_ in vision))!=4:
        raise ValueError("H167 visual-only trainable partition or frozen H164 language violated")
    return language,vision

def tensor_digest(params):
    h=hashlib.sha256()
    for name,param in sorted(params):
        h.update(name.encode())
        h.update(param.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def language_rows():
    pic,row=synth()
    en=dict(row,language="en")
    ar=dict(row,language="ar",
        instruction="على اللوحة البيضاء الاصطناعية، انقر في مركز المربع الأحمر.")
    if en["training_supervision"]!=ar["training_supervision"]:
        raise ValueError("H167 synthetic bilingual labels inconsistent")
    return [(pic,en),(pic,ar)]

def run(package,output):
    import torch
    from transformers import AutoProcessor,AutoModelForImageTextToText
    from peft import PeftModel,LoraConfig
    from h164.train_gpu import assemble

    output=Path(output)
    if (output.exists() or not str(output).startswith("/kaggle/working/")
        or os.environ.get("HALA_NONCOMMERCIAL_RESEARCH")!="1"
        or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")):
        raise ValueError("H167 private Kaggle free research output required")
    versions={name:importlib.metadata.version(name).split("+")[0]
              for name in ("torch","transformers","peft","Pillow")}
    gpu=require_free_t4(torch,versions)
    torch.manual_seed(167)
    frozen=admit_frozen_language_zip(package,
                                    Path("/kaggle/working/h167-private-frozen-language"))
    processor=AutoProcessor.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,trust_remote_code=False)
    base=AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,torch_dtype=torch.float16,
        trust_remote_code=False,attn_implementation="sdpa").to("cuda:0")
    targets=visual_merger_targets(base.named_modules())
    model=PeftModel.from_pretrained(base,str(frozen),
                                    adapter_name="language",is_trainable=False)
    model.add_adapter("vision",LoraConfig(
        r=VISUAL_RANK,lora_alpha=8,lora_dropout=0.0,
        target_modules=targets,bias="none",task_type="CAUSAL_LM"))
    model.set_adapter(["language","vision"])
    active=list(model.active_adapters)
    if active!=["language","vision"]:
        raise ValueError("H167 both language and vision adapters must be active")
    for name,p in model.named_parameters():
        p.requires_grad_(is_visual_trainable(name))
    language,visual=check_adapter_partition(model.named_parameters())
    before_language=tensor_digest(language)
    opt=torch.optim.AdamW([p for _,p in visual],lr=1e-5,weight_decay=0)
    results=[]
    torch.cuda.reset_peak_memory_stats(0)
    if hasattr(model,"enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model,"gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config,"use_cache"):
        model.config.use_cache=False
    for image,row in language_rows():
        model.train()
        opt.zero_grad(set_to_none=True)
        batch=assemble(processor,image,row,"cuda:0")
        result=model(**batch)
        if result.loss is None or not torch.isfinite(result.loss):
            raise ValueError("H167 nonfinite synthetic bilingual loss")
        result.loss.backward()
        positive=sum(int(p.grad is not None
                         and torch.isfinite(p.grad).all().item()
                         and torch.count_nonzero(p.grad).item()>0)
                     for _,p in visual)
        if positive<1:
            raise ValueError("H167 visual merger received zero bilingual gradients")
        before=[p.detach().clone() for _,p in visual]
        opt.step()
        changed=sum(int(not torch.equal(old,p.detach()))
                    for old,(_,p) in zip(before,visual))
        if changed<1:
            raise ValueError("H167 no visual optimizer update")
        torch.cuda.synchronize(0)
        results.append({"language":row["language"],
            "nonzero_grad_tensors":positive,"updated_tensors":changed,
            "synthetic_finite_loss":True})
    language_after=tensor_digest(language)
    peak=torch.cuda.max_memory_allocated(0)/2**30
    reserved=torch.cuda.max_memory_reserved(0)/2**30
    if (not math.isfinite(peak) or not 0<peak<=MAX_PEAK_GIB
        or before_language!=language_after
        or not all(opt.state.get(p) for _,p in visual)):
        raise ValueError("H167 frozen language changed, no states, or T4 cap exceeded")
    doc={
        "schema":"hala.h167.private-frozen-h164-plus-vision-synthetic-t4.v1",
        "status":"H167_TWO_LANGUAGE_SYNTHETIC_STEPS_PASS",
        "language_adapter_zip_sha256":LANGUAGE_ZIP_SHA,
        "language_adapter_weights_sha256":LANGUAGE_WEIGHTS_SHA,
        "language_adapter_frozen_before_after_match":True,
        "base_model":MODEL_ID,"base_revision":MODEL_SHA,
        "environment":versions,"gpu":gpu,"active_adapters":active,
        "visual_target_modules":targets,"visual_rank":VISUAL_RANK,
        "trainable_visual_parameters":sum(p.numel() for _,p in visual),
        "frozen_language_parameters":sum(p.numel() for _,p in language),
        "bilingual_synthetic":results,"optimizer_steps":len(results),
        "peak_allocated_gib":round(peak,3),
        "peak_reserved_gib":round(reserved,3),
        "memory_cap_gib":MAX_PEAK_GIB,
        "data":"synthetic-only-no-original-app",
        "original_app_training_cases":0,"owner_gold_used":False,
        "previous_ood_or_dev_used":False,"new_ood_evaluations":0,
        "real_app_performance_measured":False,"weights_saved":False,
        "paid_gpu":False,"flagship_35b_unchanged":True}
    output.mkdir(parents=True,exist_ok=False)
    (output/"h167-combined-synthetic-t4-receipt.json").write_text(
        json.dumps(doc,sort_keys=True,indent=2)+"\n",encoding="utf8")
    return doc
