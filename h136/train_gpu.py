"""H136 OWNER-PRIVATE LoRA SFT experimental 2B GUI START/END grounder.

Free Kaggle T4 only. Separate source fixture TRAIN & DEV datasets; sealed
holdout NOT attached. No H71 data or third-party training screenshots.
Baseline and post-training DEV evaluation on same independently sampled
DEV16. NEVER claim Hala flagship model weights have changed.
"""
import json
import math
import os
import random
import time
from pathlib import Path

from h136.source import check,sha
from h132.grounder_contract import MODEL_ID,MODEL_SHA,instruction_for,native_point
from h130.browser_env import WIDTH,HEIGHT

DATASET_ROOT=Path("/kaggle/input")
BASE_MODEL=MODEL_ID
BASE_REV=MODEL_SHA
STEPS=64
GRAD_ACCUM=2
LORA_RANK=8
RANDOM_SEED=136

def admit_private_split(split):
    matched=[]
    for path in DATASET_ROOT.rglob("manifest.json"):
        try:
            receipt=check(path.parent,split)
        except (ValueError,KeyError,TypeError,FileNotFoundError,json.JSONDecodeError):
            continue
        matched.append((path.parent,receipt))
    if len(matched)!=1:
        raise ValueError(f"H136 requires exactly one owner-private {split} source; saw {len(matched)}")
    return matched[0]

def make_prompt_and_gold(row,role):
    if role not in ("start","end"):
        raise ValueError("H136 SFT target START or END required")
    instruction=instruction_for(row["instruction"],role)
    pixels=row[role+"_px"]
    if type(pixels) is not list or len(pixels)!=2 or any(type(x) is not int for x in pixels):
        raise ValueError("H136 actual owner browser pixel target missing")
    grid=[round(pixels[0]/WIDTH*1000),round(pixels[1]/HEIGHT*1000)]
    if (type(grid) is not list or len(grid)!=2 or
        any(type(v) is not int or not 0<=v<=1000 for v in grid)):
        raise ValueError("H136 gold original 1000-grid target malformed")
    gold=json.dumps({"point_2d":grid},separators=(",",":"))
    return instruction,gold

def assemble_example(processor,image,row,role,device):
    import torch
    prompt,gold=make_prompt_and_gold(row,role)
    content=[{"type":"image"},{"type":"text","text":prompt}]
    messages=[{"role":"user","content":content}]
    template=processor.apply_chat_template(
        messages,tokenize=False,add_generation_prompt=True,
        enable_thinking=False)
    full=processor.apply_chat_template(messages+[
        {"role":"assistant","content":[{"type":"text","text":gold}]}],
        tokenize=False,add_generation_prompt=False,enable_thinking=False)
    if not full.startswith(template):
        raise ValueError("H136 frozen Qwen3.5 chat template prompt prefix mismatch")
    # Full original 960x720 image: model processor applies its native smart-resize.
    if image.size!=(WIDTH,HEIGHT):
        raise ValueError("H136 only original native-size owner Chromium screenshots")
    context=processor(text=[template],images=[image],return_tensors="pt")
    encoded=processor(text=[full],images=[image],return_tensors="pt")
    if "input_ids" not in context or "input_ids" not in encoded:
        raise ValueError("H136 actual multimodal processor omitted tokens")
    prefix=int(context["input_ids"].shape[1])
    if not 0<prefix<int(encoded["input_ids"].shape[1]):
        raise ValueError("H136 completion-only loss cannot be isolated")
    labels=encoded["input_ids"].clone()
    labels[:,:prefix]=-100
    if "attention_mask" in encoded:
        labels[encoded["attention_mask"]==0]=-100
    if (labels!=-100).sum().item()<5:
        raise ValueError("H136 training gold JSON unlearnable / tokenizer misaligned")
    encoded["labels"]=labels
    return {k:v.to(device) if hasattr(v,"to") else v for k,v in encoded.items()}

def model_predict(processor,model,torch,image,instruction):
    messages=[{"role":"user","content":[{"type":"image"},
               {"type":"text","text":instruction}]}]
    prompt=processor.apply_chat_template(messages,tokenize=False,
                         add_generation_prompt=True,enable_thinking=False)
    if image.size!=(WIDTH,HEIGHT):
        raise ValueError("H136 model inference expects original native screenshot")
    batch=processor(text=[prompt],images=[image],return_tensors="pt").to("cuda:0")
    model.eval()
    with torch.inference_mode():
        generated=model.generate(**batch,max_new_tokens=80,do_sample=False,max_time=45)
    torch.cuda.synchronize(0)
    raw=processor.batch_decode(generated[:,batch["input_ids"].shape[1]:],
                               skip_special_tokens=True)[0].strip()
    return raw,native_point(raw,WIDTH,HEIGHT)

def score_dev(processor,model,torch,dir,manifest):
    from PIL import Image
    stats={"start":{"hit":0,"valid":0},"end":{"hit":0,"valid":0}}
    examples=[]
    for row in manifest["cases"]:
        with Image.open(dir/row["image"]) as im:
            image=im.convert("RGB")
        for role in ("start","end"):
            instruction,_=make_prompt_and_gold(row,role)
            raw,point=model_predict(processor,model,torch,image,instruction)
            bbox=row[role+"_bbox"]
            hit=(point is not None and
                 bbox["x"]<=point[0]<=bbox["x"]+bbox["width"] and
                 bbox["y"]<=point[1]<=bbox["y"]+bbox["height"])
            stats[role]["hit"]+=int(hit)
            stats[role]["valid"]+=int(point is not None)
            examples.append({"case_id":row["id"],"role":role,
                "raw":raw,"prediction":point,"hit":bool(hit)})
    return stats,examples

def find_lora_targets(model,torch):
    allowed={"q_proj","k_proj","v_proj","o_proj",
             "gate_proj","up_proj","down_proj"}
    names=[name for name,module in model.named_modules()
           if ("language_model.layers." in name and
               name.rsplit(".",1)[-1] in allowed and
               isinstance(module,torch.nn.Linear))]
    if len(names)<30:
        raise ValueError("H136 pinned language-only Qwen3.5 LoRA layer names absent")
    if any("vision" in n or "visual" in n for n in names):
        raise ValueError("H136 vision tower must remain frozen")
    return names

def run(private_output):
    import torch,transformers
    from PIL import Image
    from transformers import AutoProcessor,AutoModelForImageTextToText
    from peft import LoraConfig,get_peft_model
    private_output=Path(private_output)
    if (private_output.exists() or
        not str(private_output).startswith("/kaggle/working/") or
        transformers.__version__!="5.17.0" or
        not torch.cuda.is_available() or
        not any("T4" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count()))):
        raise ValueError("H136 only fresh owner PRIVATE FREE T4 and pinned runtime")
    train_root,train_manifest=admit_private_split("train")
    dev_root,dev_manifest=admit_private_split("dev")
    if set(row["image_sha256"] for row in train_manifest["cases"]) & set(
            row["image_sha256"] for row in dev_manifest["cases"]):
        raise ValueError("H136 train/dev screenshot leakage")
    if len(train_manifest["cases"])!=288 or len(dev_manifest["cases"])!=24:
        raise ValueError("H136 exact independent 288 TRAIN and 24 DEV source prerequisite")
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    os.environ.update(HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
    processor=AutoProcessor.from_pretrained(BASE_MODEL,revision=BASE_REV,
                                              trust_remote_code=False)
    model=AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,revision=BASE_REV,torch_dtype=torch.float16,
        trust_remote_code=False,attn_implementation="sdpa").to("cuda:0")
    if "qwen3_5" not in model.config.model_type.lower():
        raise ValueError("H136 LoRA expected frozen pretrained Qwen3.5-2B")
    start=time.monotonic()
    before,before_examples=score_dev(processor,model,torch,dev_root,dev_manifest)
    for p in model.parameters():p.requires_grad_(False)
    targets=find_lora_targets(model,torch)
    lora=LoraConfig(r=LORA_RANK,lora_alpha=16,lora_dropout=0.05,bias="none",
                    task_type="CAUSAL_LM",target_modules=targets)
    model=get_peft_model(model,lora)
    if hasattr(model,"enable_input_require_grads"):
        model.enable_input_require_grads()
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if (not trainable or any("lora_" not in n for n,p in trainable) or
        any("vision" in n or "visual" in n for n,p in trainable)):
        raise ValueError("H136 only language LoRA adapters may be optimized")
    if hasattr(model,"gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config,"use_cache"):
        model.config.use_cache=False
    # Freeze the base and only optimize Qwen3.5 language projections.
    optimizer=torch.optim.AdamW([p for n,p in trainable],
                                 lr=1e-4,weight_decay=0.01)
    examples=[(i,role) for i in range(288) for role in ("start","start","start","end")]
    random.shuffle(examples)
    steps_taken=0
    losses=[]
    for step in range(STEPS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for k in range(GRAD_ACCUM):
            row_idx,role=examples[(step*GRAD_ACCUM+k)%len(examples)]
            row=train_manifest["cases"][row_idx]
            with Image.open(train_root/row["image"]) as im:
                image=im.convert("RGB")
            batch=assemble_example(processor,image,row,role,"cuda:0")
            output=model(**batch)
            loss=output.loss/GRAD_ACCUM
            if not torch.isfinite(loss):
                raise ValueError("H136 LoRA training loss NaN/Inf")
            loss.backward()
            losses.append(round(float(loss.detach().cpu())*GRAD_ACCUM,6))
            del batch,output,loss
        torch.nn.utils.clip_grad_norm_([p for n,p in trainable],1.0)
        optimizer.step()
        steps_taken+=1
        if step%4==0:
            print(f"H136 LORA REAL OPTIMIZER STEPS {steps_taken}/{STEPS}",flush=True)
        torch.cuda.empty_cache()
    if steps_taken!=STEPS:raise ValueError("H136 optimizer steps missing")
    model.eval()
    if hasattr(model.config,"use_cache"):model.config.use_cache=True
    after,after_examples=score_dev(processor,model,torch,dev_root,dev_manifest)
    private_output.mkdir(parents=True,exist_ok=False)
    adapter_dir=private_output/"private-hala-h136-small-handle-adapter"
    model.save_pretrained(adapter_dir,safe_serialization=True)
    processor.save_pretrained(adapter_dir)
    if not (adapter_dir/"adapter_model.safetensors").exists():
        raise ValueError("H136 actual adapter weights not saved")
    file=adapter_dir/"adapter_model.safetensors"
    results={
      "schema":"hala.h136.private-original-start-end-LoRA-pilot.v1",
      "status":"H136_REAL_64_OPTIMIZER_STEP_NATIVE_GUI_LORA_EXPERIMENT",
      "base_model":BASE_MODEL,"base_revision":BASE_REV,
      "train_original_screens":288,"dev_original_screens":24,
      "sealed_holdout_attached":False,
      "train_dev_id_shas_disjoint":True,
      "actual_optimizer_steps":steps_taken,
      "lora_rank":LORA_RANK,"trainable_param_count":sum(p.numel() for n,p in trainable),
      "train_losses":losses,"dev_before":before,"dev_after":after,
      "train_source_manifest_sha":sha((train_root/"manifest.json").read_bytes()),
      "dev_source_manifest_sha":sha((dev_root/"manifest.json").read_bytes()),
      "adapter_sha256":sha(file.read_bytes()),
      "elapsed_seconds":round(time.monotonic()-start,2),
      "paid_compute_used":False,"published_weights":False,
      "main_Hala_flagship_weights_updated":False,
      "independent_holdout_accuracy_proven":False,
    }
    (private_output/"private-h136-learning-evidence.json").write_text(
        json.dumps(results,sort_keys=True,indent=2)+"\n","utf8")
    (private_output/"private-dev-before.json").write_text(
        json.dumps(before_examples,sort_keys=True,indent=2)+"\n","utf8")
    (private_output/"private-dev-after.json").write_text(
        json.dumps(after_examples,sort_keys=True,indent=2)+"\n","utf8")
    return results
