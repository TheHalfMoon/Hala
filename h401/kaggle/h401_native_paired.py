"""H401 PRIVATE source-only paired native click test on Qwen3-VL 2B.

Invented colored squares. Base vs H399 private LoRA adapter on author-exposed
new source seed. This does not evaluate real UI or independent sealed OOD.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path

BASE="Qwen/Qwen3-VL-2B-Instruct"
REV="89644892e4d85e24eaac8bacfd4f463576704203"
ADAPTER_SHA="8b6d9bdf5df8fd4ecba939a5f454e3deffd5704df3ff41622697e63ef767a537"
PARENT_RECEIPT_SHA="17ef16c9091ca411b03df892c12cb843fbbb14544e445b94a19d29660dd53333"
SOURCE="abdulazizshehri/hala-h399-original-synthetic-qwen3vl-lora"
OUT=Path("/kaggle/working/h401-private-native-paired-research")
TEMPLATE=(
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
    p=positions[color]
    bg=(26,26,34) if style=="normal" else (82,74,82)
    im=Image.new("RGB",(224,224),bg)
    draw=ImageDraw.Draw(im)
    for which,cell in enumerate(positions):
        row,col=divmod(cell,3)
        x,y=32+col*64,32+row*64
        fill=COLORS[which] if which<3 else (101+which*9,)*3
        if style=="unseen":
            fill=tuple(min(255,round(v*.78+32)) for v in fill)
        draw.rectangle((x-15,y-15,x+15,y+15),fill=fill)
    x,y=32+(p%3)*64,32+(p//3)*64
    target=[round(x/224*1000),round(y/224*1000)]
    prompt=(TEMPLATE[color][language]+" Reply ONLY JSON point_2d in normalized"
            " [0,1000] coordinates, for example {\"point_2d\":[500,500]}.")
    return im,prompt,target,language

def checksum(data):
    return hashlib.sha256(data).hexdigest()

def fingerprint(datum):
    image,prompt,_,_=datum
    return checksum(image.tobytes()+prompt.encode("utf8"))

def evidence(adapter_path):
    path=Path(adapter_path)
    if checksum(path.read_bytes())!=ADAPTER_SHA:
        raise ValueError("H401 external private adapter SHA mismatch")
    parent=path.parent.parent
    raw=(parent/"receipt.json").read_bytes()
    if checksum(raw)!=PARENT_RECEIPT_SHA:
        raise ValueError("H401 parent Kaggle verified train receipt changed")
    if (parent/"receipt.sha256").read_text().strip()!=PARENT_RECEIPT_SHA:
        raise ValueError("H401 parent receipt hash sidecar mismatched")
    old=json.loads(raw)
    if (old.get("status")!="COMPLETED_REAL_2B_SYNTHETIC_RESEARCH_LORA"
        or old.get("base_model")!=BASE or old.get("base_revision")!=REV
        or old.get("optimizer_steps")!=16 or old.get("official_hala_model_trained") is not False):
        raise ValueError("H401 parent synthetic-only LoRA provenance wrong")

def parse(raw):
    try:
        doc=json.loads(raw.strip())
    except (ValueError,TypeError):
        return None
    if type(doc) is not dict or set(doc)!={"point_2d"}:
        return None
    value=doc["point_2d"]
    if type(value) is not list or len(value)!=2 or any(
        type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1000
        for v in value
    ):
        return None
    return [float(value[0]),float(value[1])]

def hit(point,target):
    # Original author-rendered colored square bounds are 30x30 px.
    return bool(point is not None and
                abs(point[0]-target[0])<=15/224*1000 and
                abs(point[1]-target[1])<=15/224*1000)

def report(model,processor,items,tag):
    import torch
    model.eval()
    counts={k:{"n":0,"valid_json":0,"hit":0,"cells":set()} for k in ("all","en","ar")}
    with torch.inference_mode():
        for i,(image,prompt,target,language) in enumerate(items):
            messages=[{"role":"user","content":[{"type":"image"},
                      {"type":"text","text":prompt}]}]
            formatted=processor.apply_chat_template(
                messages,tokenize=False,add_generation_prompt=True)
            values=processor(text=[formatted],images=[image],
                             return_tensors="pt").to("cuda:0")
            generated=model.generate(
                **values,max_new_tokens=70,do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id)
            completion=generated[:,values["input_ids"].shape[1]:]
            raw=processor.batch_decode(
                completion,skip_special_tokens=True)[0].strip()
            point=parse(raw)
            success=hit(point,target)
            for key in ("all","en" if language==0 else "ar"):
                box=counts[key]
                box["n"]+=1
                box["valid_json"]+=int(point is not None)
                box["hit"]+=int(success)
                if point:
                    cell=(min(2,int(point[0]*3/1001)),
                          min(2,int(point[1]*3/1001)))
                    box["cells"].add(cell)
            print("H401_EVAL",tag,i+1,"valid",point is not None,
                  "target_hit",success,flush=True)
    return {name:{"n":v["n"],"valid_json":v["valid_json"],
                  "hits":v["hit"],"unique_cells":len(v["cells"])}
            for name,v in counts.items()}

def run():
    import torch
    state={"schema":"hala.h401.private-paired-native-invented-qwen3vl.v1",
           "status":"NOT_STARTED","base_model":BASE,"revision":REV,
           "adapter_sha256":ADAPTER_SHA,
           "input_kernel_source":SOURCE,
           "evaluation_is_independent":False,
           "official_hala_model_qualified":False,
           "source_rights_original_screenshots_approved":False,
           "external_ood":False,
           "all_images_are_first_party_generated":True,
           "model_weights_published":False}
    if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not torch.cuda.is_available():
        raise ValueError("H401 private owner Kaggle GPU only")
    if OUT.exists():
        raise ValueError("H401 require fresh private Kaggle output")
    OUT.mkdir(parents=True)
    try:
        subprocess.run([sys.executable,"-m","pip","install","-q",
                        "transformers==4.57.3","peft==0.18.0",
                        "accelerate==1.12.0","Pillow==11.3.0"],
                       check=True,timeout=330)
        from transformers import AutoProcessor,AutoModelForImageTextToText
        from peft import PeftModel
        paths=list(Path("/kaggle/input").rglob("adapter_model.safetensors"))
        if len(paths)!=1:
            raise ValueError("H401 expected exactly one linked PRIVATE H399 LoRA adapter")
        evidence(paths[0])
        evals=[sample(i,4901,"unseen") for i in range(18)]
        baseline_sources=[
            *[sample(i,3981) for i in range(24)],
            *[sample(i,3982) for i in range(6)],
            *[sample(i,3983,"unseen") for i in range(6)]]
        if (len({fingerprint(x) for x in evals})!=18 or
                set(map(fingerprint,evals)) & set(map(fingerprint,baseline_sources))):
            raise ValueError("H401 author source split reused")
        processor=AutoProcessor.from_pretrained(BASE,revision=REV,trust_remote_code=False)
        model=AutoModelForImageTextToText.from_pretrained(
            BASE,revision=REV,dtype=torch.float16,
            trust_remote_code=False,attn_implementation="sdpa").to("cuda:0")
        state["base"]=report(model,processor,evals,"base")
        lora=PeftModel.from_pretrained(model,paths[0].parent,is_trainable=False)
        state["adapted"]=report(lora,processor,evals,"adapted")
        state["status"]="COMPLETE_AUTHOR_SYNTHETIC_NATIVE_PAIRED_DIAGNOSTIC"
        state["sample_count"]=len(evals)
        state["device"]=torch.cuda.get_device_name(0)
    except Exception as exc:
        state["status"]="FAILED_NO_OOD_QUALIFICATION"
        state["failure_class"]=type(exc).__name__
        raise
    finally:
        encoded=json.dumps(state,sort_keys=True,indent=2,allow_nan=False).encode()
        (OUT/"receipt.json").write_bytes(encoded)
        (OUT/"receipt.sha256").write_text(checksum(encoded))
        print("H401_RECEIPT",json.dumps(state,sort_keys=True),flush=True)

if __name__=="__main__":
    run()
