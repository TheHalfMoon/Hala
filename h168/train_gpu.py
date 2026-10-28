"""H168 private FREE-T4 32 original browser TRAIN-only vision LoRA SFT.

No DEV, OOD, historical heldout or benchmark inference. This experiment only
establishes that 32 fresh source-native examples can update a visual LoRA.
"""
import hashlib
import importlib.metadata
import io
import json
import math
import os
import random
import zipfile
from collections import Counter
from pathlib import Path

from h132.grounder_contract import MODEL_ID,MODEL_SHA
from h164.native_action_contract import supervision
from h166.vision_probe import (
    require_free_t4,visual_merger_targets,MAX_PEAK_GIB)
from h167.combined_probe import (
    LANGUAGE_WEIGHTS_SHA,LANGUAGE_CONFIG_SHA,LANGUAGE_README_SHA,
    activate_frozen_language_and_trainable_vision,
    admit_frozen_language_directory,check_adapter_partition,
    is_visual_trainable,tensor_digest)

TRAIN_ZIP_SHA="d6d23a0b4c0cebd87d5ebd9830ceeca18108037f82bcbbc2c0fb7749aae86fc6"
TRAIN_SCHEMA="hala.h168.owner-private-new-original-train-32.v1"
SEED=168
STEPS=32
LR=1e-5
MAX_LOSS=500.0

def sha(blob):
    return hashlib.sha256(blob).hexdigest()

def admit_train(archive):
    archive=Path(archive)
    if sha(archive.read_bytes())!=TRAIN_ZIP_SHA:
        raise ValueError("H168 immutable real original TRAIN ZIP hash changed")
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist())!=33 or set(z.namelist())!=(
            {"manifest.json"}|{"images/"+str(x)+".png" for x in
                list(range(168000,168016))+list(range(168100,168116))}):
            raise ValueError("H168 TRAIN-only archive has unknown files")
        doc=json.loads(z.read("manifest.json"))
        rows=doc.get("cases")
        if (doc.get("schema")!=TRAIN_SCHEMA or doc.get("split")!="TRAIN"
            or doc.get("count")!=STEPS or doc.get("dev_ood_count")!=0
            or doc.get("reference_gold_uploaded_for_train_only") is not True
            or doc.get("historical_consumed_images_excluded") is not True
            or doc.get("data_role")!="H168_NEW_SOURCE_TRAIN_PILOT_ONLY"
            or not isinstance(rows,list) or len(rows)!=STEPS):
            raise ValueError("H168 original TRAIN manifest missing source safeguards")
        ids=set()
        images=set()
        for row in rows:
            if (row.get("split")!="TRAIN"
                or row.get("app") not in ("choices","nouislider")
                or row.get("language") not in ("ar","en")
                or row.get("action_type") not in ("type","drag")
                or row["index"] in ids or row["image_sha256"] in images
                or row["image"]!="images/"+str(row["index"])+".png"
                or sha(z.read(row["image"]))!=row["image_sha256"]):
                raise ValueError("H168 original TRAIN case/source image mismatch")
            supervision(row)
            ids.add(row["index"]);images.add(row["image_sha256"])
        if (Counter(x["language"] for x in rows)!={"en":16,"ar":16}
            or Counter(x["app"] for x in rows)!={"choices":16,"nouislider":16}
            or Counter(x["action_type"] for x in rows)!={"type":16,"drag":16}):
            raise ValueError("H168 bilingual real GUI native-action TRAIN not balanced")
    return rows

def run(train_zip,language_source,output):
    import torch
    from PIL import Image
    from transformers import AutoProcessor,AutoModelForImageTextToText
    from peft import PeftModel,LoraConfig
    from h164.train_gpu import assemble

    output=Path(output)
    if (output.exists() or not str(output).startswith("/kaggle/working/")
        or os.environ.get("HALA_NONCOMMERCIAL_RESEARCH")!="1"
        or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")):
        raise ValueError("H168 private owner FREE Kaggle output required")
    installed={name:importlib.metadata.version(name).split("+")[0]
               for name in ("torch","transformers","peft","Pillow")}
    gpu=require_free_t4(torch,installed)
    rows=admit_train(train_zip)
    torch.manual_seed(SEED)
    random.seed(SEED)
    frozen=admit_frozen_language_directory(
        language_source,Path("/kaggle/tmp/h168-verified-frozen-language"))
    processor=AutoProcessor.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,trust_remote_code=False)
    base=AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,torch_dtype=torch.float16,
        trust_remote_code=False,attn_implementation="sdpa").to("cuda:0")
    targets=visual_merger_targets(base.named_modules())
    model=PeftModel.from_pretrained(base,str(frozen),
                                   adapter_name="language",is_trainable=False)
    model.add_adapter("vision",LoraConfig(r=4,lora_alpha=8,lora_dropout=0.0,
                          target_modules=targets,bias="none",task_type="CAUSAL_LM"))
    active=activate_frozen_language_and_trainable_vision(model)
    for name,p in model.named_parameters():
        p.requires_grad_(is_visual_trainable(name))
    language,visual=check_adapter_partition(model.named_parameters())
    old_language_digest=tensor_digest(language)
    if hasattr(model,"enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model,"gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config,"use_cache"):
        model.config.use_cache=False
    optim=torch.optim.AdamW([p for _,p in visual],lr=LR,weight_decay=0)
    schedule=list(range(STEPS))
    random.Random(SEED).shuffle(schedule)
    trace=[]
    torch.cuda.reset_peak_memory_stats(0)
    with zipfile.ZipFile(train_zip) as z:
        for step,index in enumerate(schedule):
            row=rows[index]
            with Image.open(io.BytesIO(z.read(row["image"]))) as im:
                picture=im.convert("RGB")
            batch=assemble(processor,picture,row,"cuda:0")
            optim.zero_grad(set_to_none=True)
            model.train()
            loss=model(**batch).loss
            if loss is None or not torch.isfinite(loss):
                raise ValueError("H168 original real GUI train loss not finite")
            value=float(loss.item())
            if not 0<value<MAX_LOSS:
                raise ValueError("H168 training loss unsafe")
            loss.backward()
            grads=sum(int(p.grad is not None
                          and torch.isfinite(p.grad).all().item()
                          and torch.count_nonzero(p.grad).item()>0)
                      for _,p in visual)
            if grads<1:
                raise ValueError("H168 visual merger gradients missing")
            optim.step()
            torch.cuda.synchronize(0)
            allocated=torch.cuda.max_memory_allocated(0)/2**30
            if allocated>MAX_PEAK_GIB:
                raise ValueError("H168 T4 memory cap hit: stop immediately")
            trace.append({"step":step+1,"app":row["app"],"language":row["language"],
                "kind":row["action_type"],"loss":round(value,6),
                "nonzero_visual_gradient_tensors":grads,
                "case_index":row["index"]})
            if step%4==0 or step==STEPS-1:
                print("H168_REAL_BROWSER_T4_TRAIN",step+1,STEPS,
                      round(value,4),round(allocated,3),flush=True)
    if (len(trace)!=STEPS or tensor_digest(language)!=old_language_digest
        or not all(optim.state.get(p) for _,p in visual)):
        raise ValueError("H168 frozen-language drift or incomplete 32 true GUI steps")
    peak=torch.cuda.max_memory_allocated(0)/2**30
    reserved=torch.cuda.max_memory_reserved(0)/2**30
    if not math.isfinite(peak) or peak>MAX_PEAK_GIB:
        raise ValueError("H168 T4 GPU memory cap exceeded")
    output.mkdir(exist_ok=False,parents=True)
    private=output/"private-visual-adapter"
    model.save_pretrained(str(private),selected_adapters=["vision"],
                          safe_serialization=True)
    saved=list(private.rglob("adapter_model.safetensors"))
    if len(saved)!=1 or saved[0].stat().st_size<1000:
        raise ValueError("H168 private visual LoRA weights not saved exactly once")
    receipt={"schema":"hala.h168.real-original-browser-train-32-free-t4.v1",
        "status":"H168_REAL_BROWSER_32_TRAIN_OPTIMIZER_STEPS_PASS",
        "source_train_zip_sha256":TRAIN_ZIP_SHA,
        "language_adapter_sha256":LANGUAGE_WEIGHTS_SHA,
        "language_adapter_config_sha256":LANGUAGE_CONFIG_SHA,
        "language_adapter_readme_sha256":LANGUAGE_README_SHA,
        "base_model":MODEL_ID,"base_revision":MODEL_SHA,
        "optimizer_steps":STEPS,"source_training_cases":STEPS,
        "unique_training_exposures":len(set(x["case_index"] for x in trace)),
        "english_train_cases":16,"arabic_train_cases":16,
        "type_train_cases":16,"drag_train_cases":16,
        "trainable_visual_parameters":sum(p.numel() for _,p in visual),
        "frozen_language_parameters":sum(p.numel() for _,p in language),
        "language_frozen_before_after_match":True,
        "active_adapters":active,"visual_modules":targets,
        "gpu":gpu,"installed":installed,
        "peak_allocated_gib":round(peak,3),
        "peak_reserved_gib":round(reserved,3),
        "memory_cap_gib":MAX_PEAK_GIB,
        "saved_visual_only_adapter_sha256":sha(saved[0].read_bytes()),
        "saved_adapter_count":len(saved),
        "save_private_only":True,"paid_gpu":False,
        "dev_ood_evaluations":0,"prior_consumed_eval_used":False,
        "real_browser_task_reward_measured":False,
        "model_release":False,"flagship_35b_unchanged":True,
        "trace":trace}
    (output/"h168-private-training-receipt.json").write_text(
        json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf8")
    return receipt
