"""H408: first-party GUI-composition transfer stress, not a real UI benchmark.

Invented app-like toolbar/sidebar/cards rendered in memory, 36 cases.
No training, third-party original GUI screenshots or sealed OOD labels.
Paired owner-private H403/H406 Qwen3-VL LoRA outputs on exact same images.
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
PARENTS={
    "h403":("01f1c1846c76a5e1e785fcc431faed5ee2774638a0c46d84fedfbd2486ebcb4b",
            "ab607c8831aeec881ed7c22adc01e9fc8ff92a2a11fd09ec102aafe33d604165"),
    "h406":("26ac3b14380eb7dfb1709d67b771b7bfe3d383ec37d02e3cae5b1c48197936b6",
            "120dc4e87be8eede1d7aa7b98bc138152c76202d02d3a0613afbe83484079da4"),
}
OUT=Path("/kaggle/working/h408-private-first-party-ui-composition")
N=36
WIDTH,HEIGHT=448,336
LABELS=("Save","Export","Share","Add","Refresh","Print","Sync","Archive","Delete")
APPS=("Cedar Planner","Northstar Tasks","Harbor Studio")
NAV=("Overview","Projects","Reports","Settings")
CELL_POSITIONS=tuple((112+col*108,94+row*71) for row in range(3) for col in range(3))
PROMPTS_EN=(
    'In the fictional dashboard, click the unique {label} button if it exists.',
    'Check this mock app for the single {label} control and click its center.',
    'Only select {label} if precisely one matching toolbar button is visible.',
)
PROMPTS_AR=(
    'في لوحة التطبيق الخيالي انقر على الزر {label} فقط إذا كان موجوداً مرة واحدة.',
    'ابحث عن زر {label} الوحيد في هذه الواجهة، وانقر على مركزه فقط.',
    'اختر {label} إذا ظهر زر واحد فقط بهذا الاسم، وإلا امتنع.',
)
CONTRACT_EN=(' Reply ONLY with {"point_2d":[x,y]} normalized to 0-1000'
             ' if EXACTLY ONE target button exists; otherwise {"abstain":true}.')
CONTRACT_AR=(' أجب فقط بـ {"point_2d":[x,y]} بإحداثيات 0-1000 إن وجد زر واحد مطابق،'
             ' وإلا فأجب بـ {"abstain":true}.')

def sha(data):
    return hashlib.sha256(data).hexdigest()

def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.truetype("DejaVuSans.ttf",size)
    except OSError:
        return ImageFont.load_default()

def invent(i,seed=40852):
    """12 unique, 12 absent, 12 duplicated; 6 EN+6 AR in each group."""
    from PIL import Image,ImageDraw
    if type(i) is not int or not 0<=i<N:
        raise ValueError("H408 predeclared sample index invalid")
    regime=("unique","absent","ambiguous")[i//12]
    lang=i%2
    target=LABELS[(i//2)%6]
    rng=random.Random(seed*1009+i*71)
    palette=(
        ((244,247,251),(30,55,88),(72,112,189)),
        ((250,247,245),(77,64,74),(156,97,113)),
        ((242,247,243),(40,84,63),(84,141,107)),
    )[i%3]
    bg,bar,accent=palette
    im=Image.new("RGB",(WIDTH,HEIGHT),bg)
    painter=ImageDraw.Draw(im)
    font=_font(12)
    bigfont=_font(15)
    painter.rectangle((0,0,WIDTH,43),fill=bar)
    painter.text((14,12),APPS[i%3],font=bigfont,fill="white")
    painter.rectangle((0,44,95,HEIGHT),fill=(225,229,235))
    for j,name in enumerate(NAV):
        y=63+j*43
        painter.text((11,y),name,font=font,fill=(50,60,74))
        painter.line((10,y+25,83,y+25),fill=(197,204,214),width=1)
    painter.text((111,55),"Workspace actions",font=bigfont,fill=(38,51,66))
    painter.text((111,75),"Select one visible control",font=font,
                 fill=(95,103,115))
    painter.rectangle((105,90,435,305),outline=(206,213,225),width=1)
    painter.rectangle((105,310,435,331),fill=(231,235,241))
    painter.text((112,313),"Local prototype · No external application",font=font,
                 fill=(110,119,130))
    positions=rng.sample(CELL_POSITIONS,6)
    n_target={"unique":1,"absent":0,"ambiguous":2}[regime]
    other=[x for x in LABELS if x!=target]
    distractors=rng.sample(other,6-n_target)
    buttons=([target]*n_target+distractors)
    rng.shuffle(buttons)
    boxes=[]
    for j,(label,(x,y)) in enumerate(zip(buttons,positions)):
        x0,y0=x,y
        x1,y1=x+92,y+34
        fill=(255,255,255) if j%2==0 else (238,241,246)
        painter.rounded_rectangle((x0,y0,x1,y1),radius=5,fill=fill,
                 outline=(171,181,196),width=1)
        painter.rectangle((x0+7,y0+12,x0+15,y0+20),fill=accent)
        painter.text((x0+19,y0+9),label,font=font,fill=(32,48,65))
        if label==target:
            boxes.append((x0,y0,x1,y1))
    if len(boxes)!=n_target:
        raise ValueError("H408 first-party UI label cardinality drift")
    task=(PROMPTS_EN if lang==0 else PROMPTS_AR)[(i//4)%3]
    prompt=task.replace("{label}",target)
    prompt+=CONTRACT_EN if lang==0 else CONTRACT_AR
    if n_target==1:
        x0,y0,x1,y1=boxes[0]
        gold={"point_2d":[round((x0+x1)/(2*WIDTH)*1000),
                          round((y0+y1)/(2*HEIGHT)*1000)]}
    else:
        gold={"abstain":True}
    return im,prompt,boxes,regime,lang,target,gold

def verify_sources(rows):
    if len(rows)!=N:
        raise ValueError("H408 wrong number of original GUI compositions")
    keys={sha(im.tobytes()+prompt.encode("utf8"))
          for im,prompt,*_ in rows}
    if len(keys)!=N:
        raise ValueError("H408 repeated image+prompt identity")
    for regime in ("unique","absent","ambiguous"):
        group=[x for x in rows if x[3]==regime]
        if len(group)!=12 or sum(x[4]==0 for x in group)!=6:
            raise ValueError("H408 source scenario/EN-AR imbalance")
        for im,prompt,boxes,kind,lang,target,gold in group:
            if (im.size!=(WIDTH,HEIGHT)
                or len(boxes)!={"unique":1,"absent":0,"ambiguous":2}[kind]
                or target not in LABELS
                or '{"abstain":true}' not in prompt):
                raise ValueError("H408 original interface metadata invalid")
            if kind=="unique" and set(gold)!={"point_2d"}:
                raise ValueError("H408 single target must click")
            if kind!="unique" and gold!={"abstain":True}:
                raise ValueError("H408 absent/duplicate source must abstain")
    return sha("".join(sorted(keys)).encode())

def _no_duplicate_json_keys(pairs):
    """Reject raw and Unicode-escaped duplicate aliases at every object depth."""
    result={}
    for key,value in pairs:
        if key in result:
            raise ValueError("H408 duplicate generated JSON object key")
        result[key]=value
    return result

def _reject_json_constant(value):
    raise ValueError("H408 nonstandard JSON constant")

def parse(response):
    try:
        o=json.loads(response.strip(),
            object_pairs_hook=_no_duplicate_json_keys,
            parse_constant=_reject_json_constant)
    except (TypeError,ValueError):
        return "invalid",None
    if type(o) is not dict:return "invalid",None
    if set(o)=={"abstain"} and o["abstain"] is True:
        return "abstain",None
    if set(o)!={"point_2d"}:return "invalid",None
    point=o["point_2d"]
    if type(point) is not list or len(point)!=2 or any(
        type(v) not in (int,float) or not math.isfinite(v) or
        not 0<=v<=1000 for v in point):
        return "invalid",None
    return "point",[float(v) for v in point]

def score(kind,point,boxes,regime):
    if regime!="unique":
        return kind=="abstain",kind=="point"
    if kind!="point":
        return False,False
    x0,y0,x1,y1=boxes[0]
    x,y=point
    return (x0/WIDTH*1000<=x<=x1/WIDTH*1000
            and y0/HEIGHT*1000<=y<=y1/HEIGHT*1000),False

def adapter_paths():
    paths=list(Path("/kaggle/input").rglob("adapter_model.safetensors"))
    if len(paths)!=2:
        raise ValueError("H408 two exact owner PRIVATE adapters required")
    verified={}
    for f in paths:
        checksum=sha(f.read_bytes())
        labels=[k for k,(wanted,_) in PARENTS.items() if wanted==checksum]
        if len(labels)!=1 or labels[0] in verified:
            raise ValueError("H408 unknown/duplicate source adapter")
        label=labels[0]
        parent=f.parent.parent
        r=(parent/"receipt.json").read_bytes()
        wanted=PARENTS[label][1]
        if sha(r)!=wanted or (parent/"receipt.sha256").read_text().strip()!=wanted:
            raise ValueError("H408 authenticated original private training receipt absent")
        d=json.loads(r)
        wanted_status={"h403":"COMPLETED_PRIVATE_SYNTHETIC_ABSTENTION_LORA",
                       "h406":"COMPLETE_PRIVATE_FIRST_PARTY_AMBIGUITY_LORA"}[label]
        if d.get("status")!=wanted_status:
            raise ValueError("H408 predecessor synthetic-only training status invalid")
        verified[label]=f
    if set(verified)!=set(PARENTS):
        raise ValueError("H408 expected both reviewed source adapters")
    return verified

def eval_model(model,processor,rows,label):
    import torch
    categories=("all","en","ar","unique","absent","ambiguous")
    scorebook={x:{"n":0,"correct":0,"invalid":0,"unsafe_clicks":0,
                  "abstentions":0,"valid_points":0}
               for x in categories}
    model.eval()
    with torch.inference_mode():
        for i,(im,prompt,boxes,kind,lang,_label,_gold) in enumerate(rows):
            messages=[{"role":"user","content":[
                {"type":"image"},{"type":"text","text":prompt}]}]
            rendered=processor.apply_chat_template(
                messages,tokenize=False,add_generation_prompt=True)
            encoded=processor(text=[rendered],images=[im],
                              return_tensors="pt").to("cuda:0")
            generation=model.generate(**encoded,max_new_tokens=90,
                  do_sample=False,pad_token_id=processor.tokenizer.eos_token_id)
            output=processor.batch_decode(
                generation[:,encoded["input_ids"].shape[1]:],
                skip_special_tokens=True)[0].strip()
            response,point=parse(output)
            correct,unsafe=score(response,point,boxes,kind)
            for key in ("all","en" if lang==0 else "ar",kind):
                rec=scorebook[key]
                rec["n"]+=1
                rec["correct"]+=int(correct)
                rec["invalid"]+=int(response=="invalid")
                rec["unsafe_clicks"]+=int(unsafe)
                rec["abstentions"]+=int(response=="abstain")
                rec["valid_points"]+=int(response=="point")
            print("H408_GUI_COMPOSITION_RESULT",label,i+1,kind,"lang",lang,
                  "response",response,"correct",correct,"unsafe",unsafe,flush=True)
    return scorebook

def result_file(doc):
    raw=json.dumps(doc,indent=2,sort_keys=True,allow_nan=False).encode()
    (OUT/"receipt.json").write_bytes(raw)
    (OUT/"receipt.sha256").write_text(sha(raw))

def run():
    import torch
    out={"schema":"hala.h408.private-first-party-ui-composition-transfer.v1",
         "status":"NOT_STARTED",
         "official_base":BASE,"base_revision":REV,
         "parent_adapters":{k:v[0] for k,v in PARENTS.items()},
         "source_family":"first-party newly authored GUI-like button/cards/sidebar",
         "external_ood":False,"third_party_gui_data":False,
         "original_gui_train_rights_approved":False,
         "new_training":False,"model_weights_published":False,
         "official_hala_model_qualified":False}
    if (not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
        or not torch.cuda.is_available() or OUT.exists()):
        raise ValueError("H408 private free owner GPU only")
    OUT.mkdir(parents=True)
    try:
        original=[invent(i) for i in range(N)]
        out["original_image_prompt_fingerprint"]=verify_sources(original)
        sources=adapter_paths()
        subprocess.run([sys.executable,"-m","pip","install","-q",
            "transformers==4.57.3","peft==0.18.0","accelerate==1.12.0",
            "Pillow==11.3.0"],check=True,timeout=330)
        from transformers import AutoProcessor,AutoModelForImageTextToText
        from peft import PeftModel
        processor=AutoProcessor.from_pretrained(BASE,revision=REV,trust_remote_code=False)
        model=AutoModelForImageTextToText.from_pretrained(
            BASE,revision=REV,dtype=torch.float16,trust_remote_code=False,
            attn_implementation="sdpa").to("cuda:0")
        adapted=PeftModel.from_pretrained(
            model,sources["h403"].parent,adapter_name="h403",
            is_trainable=False)
        adapted.load_adapter(sources["h406"].parent,adapter_name="h406",
                             is_trainable=False)
        for label in ("h403","h406"):
            adapted.set_adapter(label)
            out[label]=eval_model(adapted,processor,original,label)
        out.update(status="COMPLETE_FIRST_PARTY_UI_COMPOSITION_DIAGNOSTIC",
                   n=N,device_name=torch.cuda.get_device_name(0))
    except Exception as exc:
        out.update(status="FAILED_NO_MODEL_QUALIFICATION",
                   failure_class=type(exc).__name__)
        raise
    finally:
        result_file(out)
        print("H408_RECEIPT",json.dumps(out,sort_keys=True),flush=True)

if __name__=="__main__":
    run()
