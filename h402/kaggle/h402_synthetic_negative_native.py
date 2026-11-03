"""H402 private genuine generation stress test; invented GUIs and negative targets.

No H199 approved original screenshots. Developer-authored task family ONLY.
No third-party model training or model weight publication.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

BASE="Qwen/Qwen3-VL-2B-Instruct"
REV="89644892e4d85e24eaac8bacfd4f463576704203"
ADAPTER_SHA="8b6d9bdf5df8fd4ecba939a5f454e3deffd5704df3ff41622697e63ef767a537"
H399_RECEIPT="17ef16c9091ca411b03df892c12cb843fbbb14544e445b94a19d29660dd53333"
OUT=Path("/kaggle/working/h402-private-synthetic-negative-eval")
COLORS=((230,48,49),(42,205,66),(55,92,230))
NAMES=(("red","الأحمر"),("green","الأخضر"),("blue","الأزرق"))
CENTERS=((37,51),(96,40),(174,61),(43,151),(124,131),(183,174),(82,201),(208,112))

def checksum(binary):
    return hashlib.sha256(binary).hexdigest()

def sample(i,seed=40202):
    """Alter layout and prompts; half items intentionally lack target."""
    from PIL import Image,ImageDraw
    rng=random.Random(seed*10000+i)
    color=i%3
    language=(i//3)%2
    absent=i>=12
    image=Image.new("RGB",(224,224),(52,59,74) if i%2 else (114,104,119))
    draw=ImageDraw.Draw(image)
    positions=rng.sample(CENTERS,6)
    target=None
    for j,(x,y) in enumerate(positions):
        kind=j%3 if not absent else [k for k in range(3) if k!=color][j%2]
        fill=COLORS[kind] if j<3 else (120+j*8,)*3
        half=11+(j%3)*3
        if not absent and j==color:
            target=[round(x/224*1000),round(y/224*1000),half]
        draw.rectangle((x-half,y-half,x+half,y+half),fill=fill)
    if absent:
        target=None
    if language==0:
        prompt=(f"Find the {NAMES[color][0]} square, if present, and click its center."
                ' Reply with exact JSON {"point_2d":[x,y]} in 0-1000 coordinates.'
                ' If absent, reply exactly {"abstain":true}.')
    else:
        prompt=(f"حدد المربع {NAMES[color][1]} إن وُجد، وانقر على مركزه."
                ' أجب بإحداثيات في JSON {"point_2d":[x,y]} ضمن 0-1000.'
                ' إذا لم يوجد المربع المطلوب فأجب {"abstain":true}.')
    return image,prompt,target,language,absent

def parse(raw):
    try:
        d=json.loads(raw.strip())
    except (ValueError,TypeError):
        return "invalid",None
    if type(d) is not dict:
        return "invalid",None
    if set(d)=={"abstain"} and d["abstain"] is True:
        return "abstain",None
    if set(d)!={"point_2d"}:
        return "invalid",None
    x=d["point_2d"]
    if (type(x) is not list or len(x)!=2 or
        any(type(v) not in (int,float) or
            not math.isfinite(v) or not 0<=v<=1000 for v in x)):
        return "invalid",None
    return "point",[float(x[0]),float(x[1])]

def scoring(kind,point,target):
    if target is None:
        return {"correct":kind=="abstain","false_click":kind=="point"}
    if kind!="point":
        return {"correct":False,"false_click":False}
    half=target[2]/224*1000
    return {"correct":abs(point[0]-target[0])<=half and
                     abs(point[1]-target[1])<=half,
            "false_click":False}

def verify_private_adapter(f):
    if checksum(f.read_bytes())!=ADAPTER_SHA:
        raise ValueError("H402 parent LoRA file SHA changed")
    parent=f.parent.parent
    if checksum((parent/"receipt.json").read_bytes())!=H399_RECEIPT:
        raise ValueError("H402 upstream H399 training receipt changed")
    if (parent/"receipt.sha256").read_text().strip()!=H399_RECEIPT:
        raise ValueError("H402 upstream parent hash sidecar changed")

def evaluate(model,processor,items,name):
    import torch
    model.eval()
    groups={k:{"n":0,"correct":0,"invalid":0,"false_clicks":0,"abstentions":0}
            for k in ("all","en","ar","positive","absent")}
    with torch.inference_mode():
        for i,(image,prompt,target,language,absent) in enumerate(items):
            messages=[{"role":"user","content":[{"type":"image"},
                {"type":"text","text":prompt}]}]
            rendered=processor.apply_chat_template(messages,
                tokenize=False,add_generation_prompt=True)
            values=processor(text=[rendered],images=[image],
                             return_tensors="pt").to("cuda:0")
            gen=model.generate(**values,max_new_tokens=80,do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id)
            raw=processor.batch_decode(
                gen[:,values["input_ids"].shape[1]:],
                skip_special_tokens=True)[0].strip()
            kind,point=parse(raw)
            result=scoring(kind,point,target)
            for key in ("all","en" if language==0 else "ar",
                        "absent" if absent else "positive"):
                cell=groups[key]
                cell["n"]+=1
                cell["correct"]+=int(result["correct"])
                cell["invalid"]+=int(kind=="invalid")
                cell["false_clicks"]+=int(result["false_click"])
                cell["abstentions"]+=int(kind=="abstain")
            print("H402_SAMPLE",name,i+1,"absent",absent,"kind",kind,
                  "correct",result["correct"],flush=True)
    return groups

def run():
    import torch
    state={"schema":"hala.h402.private-author-negative-native-v1",
           "status":"NOT_STARTED","base_model":BASE,"base_revision":REV,
           "adapter_sha256":ADAPTER_SHA,
           "dataset":"first-party-only invented rectangles and paraphrases",
           "evaluator_is_independent":False,"real_app_gui":False,
           "source_rights_approved":False,"externally_sealed_ood":False,
           "model_weights_published":False,
           "training_performed_in_this_run":False}
    if (not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or
        not torch.cuda.is_available() or OUT.exists()):
        raise ValueError("H402 private fresh Kaggle GPU-only evaluation")
    OUT.mkdir(parents=True)
    try:
        subprocess.run([sys.executable,"-m","pip","install","-q",
            "transformers==4.57.3","peft==0.18.0",
            "accelerate==1.12.0","Pillow==11.3.0"],
            check=True,timeout=330)
        from transformers import AutoProcessor,AutoModelForImageTextToText
        from peft import PeftModel
        files=list(Path("/kaggle/input").rglob("adapter_model.safetensors"))
        if len(files)!=1:
            raise ValueError("H402 exactly one owner linked private H399 adapter required")
        verify_private_adapter(files[0])
        data=[sample(i) for i in range(24)]
        keys={checksum(im.tobytes()+prompt.encode("utf8")) for im,prompt,_,_,_ in data}
        if len(keys)!=24:
            raise ValueError("H402 repeated original invented evaluation item")
        processor=AutoProcessor.from_pretrained(BASE,revision=REV,trust_remote_code=False)
        model=AutoModelForImageTextToText.from_pretrained(
            BASE,revision=REV,dtype=torch.float16,trust_remote_code=False,
            attn_implementation="sdpa").to("cuda:0")
        state["base"]=evaluate(model,processor,data,"base")
        adapted=PeftModel.from_pretrained(model,files[0].parent,is_trainable=False)
        state["adapted"]=evaluate(adapted,processor,data,"adapted")
        state["status"]="COMPLETE_AUTHOR_SYNTHETIC_STRESS_ONLY"
        state["n"]=24
        state["device"]=torch.cuda.get_device_name(0)
    except Exception as e:
        state.update(status="FAILED_NO_GENUINE_OOD_QUALIFICATION",
                     failure_class=type(e).__name__)
        raise
    finally:
        raw=json.dumps(state,sort_keys=True,indent=2,allow_nan=False).encode()
        (OUT/"receipt.json").write_bytes(raw)
        (OUT/"receipt.sha256").write_text(checksum(raw))
        print("H402_RECEIPT",json.dumps(state,sort_keys=True),flush=True)

if __name__=="__main__":
    run()
