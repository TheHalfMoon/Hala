"""H170: REAL frozen visual-adapter forward sensitivity on H168 TRAIN only.

We do not use any DEV/OOD or historical heldout data. This distinguishes
PEFT-loaded-but-forward-ineffective adapters from small logit effects which
are insufficient to change greedy action output. No optimizer or weights saved.
"""
import hashlib
import importlib.metadata
import io
import json
import math
import os
import zipfile
from pathlib import Path

from h132.grounder_contract import MODEL_ID,MODEL_SHA
from h164.native_action_contract import visible_prompt
from h166.vision_probe import require_free_t4,MAX_PEAK_GIB
from h167.combined_probe import (
    admit_frozen_language_directory,activate_frozen_language_and_trainable_vision,
    LANGUAGE_WEIGHTS_SHA,tensor_digest)
from h168.train_gpu import TRAIN_ZIP_SHA
from h169.infer_gpu import admit_vision,VISION_WEIGHTS_SHA,sha_file

SCHEMA="hala.h170.frozen-visual-adapter-forward-sensitivity-on-train.v1"
TRAIN_CASE=168000

def one_train_image(archive):
    if sha_file(archive)!=TRAIN_ZIP_SHA:
        raise ValueError("H170 only pinned H168 TRAIN source, never old DEV/OOD")
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist())!={"manifest.json"}|{
            "images/"+str(i)+".png" for i in
            list(range(168000,168016))+list(range(168100,168116))}:
            raise ValueError("H170 original TRAIN archive contains extra media")
        doc=json.loads(z.read("manifest.json"))
        rows=doc.get("cases",[])
        if (doc.get("schema")!="hala.h168.owner-private-new-original-train-32.v1"
            or doc.get("split")!="TRAIN" or len(rows)!=32
            or doc.get("dev_ood_count")!=0):
            raise ValueError("H170 data source not qualified TRAIN-only")
        target=next((x for x in rows if x["index"]==TRAIN_CASE),None)
        if (not target or target["split"]!="TRAIN"
            or target["app"]!="choices" or target["language"]!="en"
            or target["image"]!="images/168000.png"):
            raise ValueError("H170 fixed source case changed")
        png=z.read(target["image"])
        if hashlib.sha256(png).hexdigest()!=target["image_sha256"]:
            raise ValueError("H170 original browser training image digest drift")
    clean={k:v for k,v in target.items() if k not in (
        "training_supervision","owner_gold","gold","reference_point")}
    clean["split"]="TRAIN_DIAGNOSTIC_INPUT_ONLY"
    return png,clean

def absolute_norm_metrics(a,b):
    """Independent finite-value comparison; zero delta is a real null finding."""
    import torch
    if a.shape!=b.shape or a.numel()==0:
        raise ValueError("H170 equal-dimension tensors required")
    a=a.detach().to(dtype=torch.float32)
    b=b.detach().to(dtype=torch.float32)
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("H170 nonfinite visual or logit evidence")
    delta=b-a
    return {"max_abs_delta":float(delta.abs().max().item()),
            "l2_delta":float(torch.linalg.vector_norm(delta).item()),
            "nonzero_elements":int(torch.count_nonzero(delta).item()),
            "element_count":int(delta.numel())}

def run(train_source,language_source,vision_source,output):
    import torch
    from PIL import Image
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText,AutoProcessor

    output=Path(output)
    if (output.exists() or not str(output).startswith("/kaggle/working/")
        or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
        or os.environ.get("HALA_NONCOMMERCIAL_RESEARCH")!="1"):
        raise ValueError("H170 strict owner-private free Kaggle T4 only")
    versions={key:importlib.metadata.version(key).split("+")[0]
              for key in ("torch","transformers","peft","Pillow")}
    gpu=require_free_t4(torch,versions)
    original_bytes,row=one_train_image(train_source)
    language=admit_frozen_language_directory(
        language_source,Path("/kaggle/tmp/h170-verified-h164-language"))
    vision=admit_vision(vision_source)
    processor=AutoProcessor.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,trust_remote_code=False)
    base=AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,torch_dtype=torch.float16,
        trust_remote_code=False,attn_implementation="sdpa").to("cuda:0")
    model=PeftModel.from_pretrained(
        base,str(language),adapter_name="language",is_trainable=False)
    model.load_adapter(str(vision),adapter_name="vision",is_trainable=False)
    model.eval()
    frozen_params=[(name,p) for name,p in model.named_parameters()
                   if ".lora_A.language." in name or ".lora_B.language." in name]
    visual_params=[(name,p) for name,p in model.named_parameters()
                   if ".lora_A.vision." in name or ".lora_B.vision." in name]
    if len(visual_params)!=4 or not frozen_params:
        raise ValueError("H170 visual/language loaded LoRA tensors incomplete")
    norm=sum(float(torch.linalg.vector_norm(p.detach().float()).item())
             for _,p in visual_params)
    if not math.isfinite(norm) or norm<=0:
        raise ValueError("H170 H168 vision checkpoint weights all zero")
    old_language=tensor_digest(frozen_params)
    modules=[(name,m) for name,m in model.named_modules()
             if name.endswith("visual.merger.linear_fc2")]
    if len(modules)!=1:
        raise ValueError("H170 visual merger output hook target ambiguous")
    current=[None]
    visual_out={}
    def hook(module,inputs,result):
        if not isinstance(result,torch.Tensor):
            raise ValueError("H170 real vision merger hook was not a tensor")
        visual_out[current[0]]=result.detach().float().cpu().contiguous()
    handle=modules[0][1].register_forward_hook(hook)
    from h164.train_gpu import predict
    with Image.open(io.BytesIO(original_bytes)) as image:
        pic=image.convert("RGB")
    if pic.size!=(960,720):
        raise ValueError("H170 original TRAIN image shape invalid")
    messages=[{"role":"user","content":[
        {"type":"image"},{"type":"text","text":visible_prompt(row)}]}]
    prompt=processor.apply_chat_template(
        messages,tokenize=False,add_generation_prompt=True,
        enable_thinking=False)
    batch=processor(text=[prompt],images=[pic],return_tensors="pt").to("cuda:0")
    torch.cuda.reset_peak_memory_stats(0)
    logits={}
    try:
        for role in ("before","after"):
            if role=="before":
                model.set_adapter("language")
                if list(model.active_adapters)!=["language"]:
                    raise ValueError("H170 frozen baseline not truly language-only")
            else:
                if activate_frozen_language_and_trainable_vision(model)!=[
                    "language","vision"]:
                    raise ValueError("H170 double LoRA not truly active")
            current[0]=role
            with torch.inference_mode():
                out=model(**batch)
                final=out.logits[:,-1,:].detach().float().cpu().contiguous()
                del out
            logits[role]=final
            torch.cuda.synchronize(0)
            print("H170_FROZEN_TRAIN_FORWARD",role,flush=True)
    finally:
        handle.remove()
    if set(logits)!={"before","after"} or set(visual_out)!={
        "before","after"}:
        raise ValueError("H170 paired frozen visual forward hook incomplete")
    logit_delta=absolute_norm_metrics(logits["before"],logits["after"])
    vision_delta=absolute_norm_metrics(visual_out["before"],visual_out["after"])
    if tensor_digest(frozen_params)!=old_language:
        raise ValueError("H170 frozen language changed during inference")
    peak=torch.cuda.max_memory_allocated(0)/2**30
    if peak>MAX_PEAK_GIB:
        raise ValueError("H170 free T4 memory cap exceeded")
    receipt={
        "schema":SCHEMA,
        "status":"H170_FROZEN_TRAIN_FORWARD_SENSITIVITY_MEASURED",
        "source_case_index":TRAIN_CASE,"source_app":"choices",
        "source_split":"TRAIN","original_train_zip_sha256":TRAIN_ZIP_SHA,
        "frozen_language_sha256":LANGUAGE_WEIGHTS_SHA,
        "vision_adapter_sha256":VISION_WEIGHTS_SHA,
        "base_model":MODEL_ID,"base_revision":MODEL_SHA,
        "vision_adapter_parameter_l2_norm_sum":norm,
        "vision_merger_module":modules[0][0],
        "vision_merger_output_delta":vision_delta,
        "final_prompt_logit_delta":logit_delta,
        "vision_forward_effect_nonzero":vision_delta["nonzero_elements"]>0,
        "logit_forward_effect_nonzero":logit_delta["nonzero_elements"]>0,
        "frozen_language_digest_before_after_equal":True,
        "gpu":gpu,"versions":versions,
        "peak_allocated_gib":round(peak,3),"memory_cap_gib":MAX_PEAK_GIB,
        "paired_forward_calls":2,"optimizer_steps":0,
        "dev_ood_model_calls":0,"owner_gold_on_gpu_prompt":False,
        "paid_gpu":False,"weights_saved":False,
        "real_browser_quality_measured":False,
        "flagship_35b_unchanged":True}
    output.mkdir(parents=True,exist_ok=False)
    (output/"h170-forward-sensitivity-receipt.json").write_text(
        json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf8")
    return receipt
