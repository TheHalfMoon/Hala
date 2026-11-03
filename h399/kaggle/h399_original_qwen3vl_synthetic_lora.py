"""H399 original synthetic-only Qwen3-VL-2B LoRA research; never Hala release.

Download ONLY independently Apache-2.0 upstream model weights from HF.
All training/eval pixels are invented in memory by this file, not screenshots.
No H199 original-GUI data admission, no external app or patient media.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

MODEL="Qwen/Qwen3-VL-2B-Instruct"
MODEL_REV="89644892e4d85e24eaac8bacfd4f463576704203"
SEED=399
STEPS=16
DIR=Path("/kaggle/working/h399-private-original-synthetic-2b-lora")
TEXTS=(
    ("Click the red square.","انقر على المربع الأحمر."),
    ("Click the green square.","انقر على المربع الأخضر."),
    ("Click the blue square.","انقر على المربع الأزرق."),
)
COLORS=((230,48,49),(42,205,66),(55,92,230))

def sample(i,seed,style="normal"):
    from PIL import Image,ImageDraw
    rng=random.Random(seed*10000+i)
    color=i%3
    language=(i//3)%2
    positions=rng.sample(range(9),9)
    target=positions[color]
    bg=(26,26,34) if style=="normal" else (82,74,82)
    im=Image.new("RGB",(224,224),bg)
    draw=ImageDraw.Draw(im)
    for which,p in enumerate(positions):
        row,col=divmod(p,3)
        center=(32+col*64,32+row*64)
        x,y=center
        fill=COLORS[which] if which<3 else (101+which*9,)*3
        if style=="unseen":
            fill=tuple(min(255,round(c*0.78+32)) for c in fill)
        draw.rectangle((x-15,y-15,x+15,y+15),fill=fill)
    x,y=32+(target%3)*64,32+(target//3)*64
    point=[round(x/224*1000),round(y/224*1000)]
    prompt=(TEXTS[color][language]+" Reply ONLY JSON point_2d in normalized"
            " [0,1000] coordinates, for example {\"point_2d\":[500,500]}.")
    return im,prompt,json.dumps({"point_2d":point},separators=(",",":")),point,language

def build_batch(processor,datum):
    image,prompt,gold,_,_=datum
    messages=[{"role":"user","content":[{"type":"image"},
               {"type":"text","text":prompt}]}]
    prefix=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    full=processor.apply_chat_template(
        messages+[{"role":"assistant","content":[{"type":"text","text":gold}]}],
        tokenize=False,add_generation_prompt=False)
    context=processor(text=[prefix],images=[image],return_tensors="pt")
    encoded=processor(text=[full],images=[image],return_tensors="pt")
    k=int(context["input_ids"].shape[1])
    if not (0<k<int(encoded["input_ids"].shape[1])):
        raise ValueError("H399 assistant completion must follow real image+prompt")
    labels=encoded["input_ids"].clone()
    labels[:,:k]=-100
    if "attention_mask" in encoded:
        labels[encoded["attention_mask"]==0]=-100
    if (labels!=-100).sum().item()<5:
        raise ValueError("H399 no learnable assistant point target")
    encoded["labels"]=labels
    return {name:v.to("cuda:0") if hasattr(v,"to") else v
            for name,v in encoded.items()}

def evaluate_loss(model,processor,items):
    import torch
    model.eval()
    losses=[]
    with torch.inference_mode():
        for datum in items:
            loss=model(**build_batch(processor,datum)).loss
            if not torch.isfinite(loss):
                raise ValueError("H399 nonfinite dev loss")
            losses.append(float(loss.detach().cpu()))
    return round(sum(losses)/len(losses),6)

def save_report(data):
    DIR.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(data,indent=2,sort_keys=True,allow_nan=False).encode()
    (DIR/"receipt.json").write_bytes(encoded)
    (DIR/"receipt.sha256").write_text(hashlib.sha256(encoded).hexdigest())

def run():
    import torch
    state={"schema":"hala.h399.private-invented-gui-qwen3vl-2b-lora.v1",
           "status":"NOT_STARTED",
           "base_model":MODEL,"base_revision":MODEL_REV,
           "license":"apache-2.0 (upstream model card)",
           "dataset":"only first-party invented in-memory colored squares",
           "training_source_rights":"entirely programmatic; no third-party UI screenshots",
           "official_h199_train_manifest_admitted":False,
           "official_hala_model_trained":False,
           "independent_ood_gain_proven":False,
           "private_kaggle_research_only":True,
           "optimizer_steps":0,"model_weights_published":False,
           "real_blind_calls":0}
    if (not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or
        not torch.cuda.is_available() or
        not any("T4" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count()))):
        raise ValueError("H399 PRIVATE noncommercial Kaggle T4 only")
    if DIR.exists():
        raise ValueError("H399 fresh output required")
    save_report(state)
    started=time.monotonic()
    try:
        subprocess.run([sys.executable,"-m","pip","install","-q",
            "transformers==4.57.3","peft==0.18.0",
            "accelerate==1.12.0","Pillow==11.3.0"],
            timeout=330,check=True)
        import transformers
        import peft
        from transformers import AutoProcessor,AutoModelForImageTextToText
        from peft import LoraConfig,get_peft_model
        if transformers.__version__!="4.57.3" or peft.__version__!="0.18.0":
            raise ValueError("H399 dependency drift")
        torch.manual_seed(SEED)
        train=[sample(i,3981) for i in range(24)]
        dev=[sample(i,3982) for i in range(6)]
        alternate=[sample(i,3983,"unseen") for i in range(6)]
        # Verify independently seeded image+prompt identity before model download.
        def identity(rows):
            return {hashlib.sha256(image.tobytes()+prompt.encode("utf8")).hexdigest()
                    for image,prompt,_,_,_ in rows}
        ts,ds,us=map(identity,(train,dev,alternate))
        if (len(ts)!=len(train) or len(ds)!=len(dev) or
                len(us)!=len(alternate) or ts&ds or ts&us or ds&us):
            raise ValueError("H399 invented TRAIN/DEV/alternate data leakage")
        # Images are never written into datasets nor passed to remote inference APIs.
        processor=AutoProcessor.from_pretrained(MODEL,revision=MODEL_REV,
                                                  trust_remote_code=False)
        model=AutoModelForImageTextToText.from_pretrained(
            MODEL,revision=MODEL_REV,dtype=torch.float16,
            trust_remote_code=False,attn_implementation="sdpa").to("cuda:0")
        if "qwen3_vl" not in model.config.model_type.lower():
            raise ValueError("H399 upstream model architecture mismatch")
        for p in model.parameters():
            p.requires_grad_(False)
        before=evaluate_loss(model,processor,dev)
        before_alt=evaluate_loss(model,processor,alternate)
        targets=[name for name,module in model.named_modules()
            if ("language_model.layers." in name and
                name.rsplit(".",1)[-1] in ("q_proj","v_proj") and
                isinstance(module,torch.nn.Linear))]
        if len(targets)<20 or any("visual" in x or "vision" in x for x in targets):
            raise ValueError("H399 pinned language-only LoRA module inventory absent")
        model=get_peft_model(model,LoraConfig(r=4,lora_alpha=8,lora_dropout=0.0,
                bias="none",task_type="CAUSAL_LM",target_modules=targets))
        if hasattr(model,"enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model,"gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model.config,"use_cache"):
            model.config.use_cache=False
        params=[p for n,p in model.named_parameters() if p.requires_grad]
        if (not params or any("lora_" not in n for n,p in model.named_parameters()
                              if p.requires_grad)):
            raise ValueError("H399 only tiny language adapter can update")
        opt=torch.optim.AdamW(params,lr=0.00015)
        history=[]
        for step in range(STEPS):
            model.train()
            opt.zero_grad(set_to_none=True)
            batch=build_batch(processor,train[step])
            loss=model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("H399 nonfinite original synthetic LoRA loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,1.0)
            opt.step()
            history.append(round(float(loss.detach().cpu()),6))
            state["optimizer_steps"]=step+1
            if step%4==0 or step==STEPS-1:
                print("H399 ORIGINAL_SYNTHETIC_REAL_2B_LORA_STEP",step+1,
                      STEPS,"loss",history[-1],flush=True)
            del loss,batch
            torch.cuda.empty_cache()
        model.eval()
        if hasattr(model.config,"use_cache"):model.config.use_cache=True
        after=evaluate_loss(model,processor,dev)
        after_alt=evaluate_loss(model,processor,alternate)
        adapter=DIR/"adapter-private-research"
        model.save_pretrained(adapter,safe_serialization=True)
        if not (adapter/"adapter_model.safetensors").is_file():
            raise ValueError("H399 trained LoRA adapter weights not saved")
        state.update(status="COMPLETED_REAL_2B_SYNTHETIC_RESEARCH_LORA",
          base_dev_loss=before,after_dev_loss=after,
          base_author_exposed_alt_loss=before_alt,
          after_author_exposed_alt_loss=after_alt,
          steps=history,language_only_trainable_parameters=sum(p.numel() for p in params),
          train_n=24,dev_n=6,alternate_style_n=6,
          train_seconds=round(time.monotonic()-started,3),
          device_name=torch.cuda.get_device_name(0),
          torch_version=torch.__version__,transformers_version=transformers.__version__,
          peft_version=peft.__version__)
        save_report(state)
        print("H399_RESULT",json.dumps({k:state[k] for k in (
            "status","optimizer_steps","base_dev_loss","after_dev_loss",
            "base_author_exposed_alt_loss","after_author_exposed_alt_loss",
            "train_seconds")},sort_keys=True),flush=True)
    except Exception as exc:
        state["status"]="FAILED_NO_HALA_MODEL_QUALIFICATION"
        state["failure_class"]=type(exc).__name__
        save_report(state)
        raise

if __name__=="__main__":
    run()
