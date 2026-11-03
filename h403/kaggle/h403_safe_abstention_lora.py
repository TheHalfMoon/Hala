"""H403: private, first-party synthetic no-target LoRA continuation.

All images and bilingual prompts invented in source. No H402 evaluation records
or images used as training inputs. This is NOT official Hala model release.
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
REV="89644892e4d85e24eaac8bacfd4f463576704203"
PARENT_ADAPTER_SHA="8b6d9bdf5df8fd4ecba939a5f454e3deffd5704df3ff41622697e63ef767a537"
PARENT_RECEIPT_SHA="17ef16c9091ca411b03df892c12cb843fbbb14544e445b94a19d29660dd53333"
OUT=Path("/kaggle/working/h403-private-no-target-lora")
STEPS=48
TRAIN_N=48
DEV_N=24
AUDIT_N=24
PALETTE=((230,48,49),(42,205,66),(55,92,230))
TEXT=(("red","الأحمر"),("green","الأخضر"),("blue","الأزرق"))
PROMPTS=(
    "Check this image for the {color} square. When present, select its exact center."
    ' Output ONLY JSON {"point_2d":[x,y]} normalized to 0-1000.'
    ' If it does not exist output ONLY {"abstain":true}.',
    "Select the center of the {color} square only if it actually exists."
    ' Use JSON {"point_2d":[x,y]} for a visible target.'
    ' Otherwise respond with {"abstain":true}.',
    "ابحث عن المربع {color} في الصورة. إذا وجدته، حدد مركزه."
    ' أجب فقط ب JSON {"point_2d":[x,y]} ضمن 0-1000.'
    ' إن لم يكن موجوداً أجب فقط {"abstain":true}.',
    "اختر مركز المربع {color} إذا كان موجوداً بالفعل."
    ' أجب بصيغة {"point_2d":[x,y]} وإلا امتنع بصيغة {"abstain":true}.',
)

def sha(value):
    return hashlib.sha256(value).hexdigest()

def make_case(i,seed,style="normal"):
    from PIL import Image,ImageDraw
    rng=random.Random(seed*1000003+i*97)
    color=i%3
    lang=(i//3)%2
    missing=(i//6)%2==1
    variant=(i//12)%2
    bg=((22,35,52),(89,91,102))[(i//2)%2]
    if style=="shifted":
        bg=(75,54,62) if i%2 else (118,116,126)
    image=Image.new("RGB",(224,224),bg)
    painter=ImageDraw.Draw(image)
    positions=rng.sample([(30+q*54,30+r*54) for r in range(4) for q in range(4)],7)
    target=None
    for j,(x0,y0) in enumerate(positions):
        dx,dy=rng.randint(-6,6),rng.randint(-6,6)
        x,y=x0+dx,y0+dy
        if j<3:
            if missing:
                other=tuple(k for k in range(3) if k!=color)
                which=other[j%2]
            else:
                which=j
            fill=PALETTE[which]
            if style=="shifted":
                fill=tuple(min(255,round(0.85*v+24)) for v in fill)
        else:
            value=110+j*12
            fill=(value,value,value)
        half=12+(j%3)
        painter.rectangle((x-half,y-half,x+half,y+half),fill=fill)
        if not missing and j==color:
            target=[round(x/224*1000),round(y/224*1000)]
    if missing:
        gold='{"abstain":true}'
    else:
        gold=json.dumps({"point_2d":target},separators=(",",":"))
    # Replace only the color placeholder: JSON braces are literal output text.
    prompt=PROMPTS[2*lang+variant].replace("{color}",TEXT[color][lang])
    return image,prompt,gold,lang,missing,target

def fingerprint(rows):
    return {sha(image.tobytes()+prompt.encode("utf8"))
            for image,prompt,_,_,_,_ in rows}

def validate_groups(train,dev,audit):
    if (len(train)!=TRAIN_N or len(dev)!=DEV_N or len(audit)!=AUDIT_N):
        raise ValueError("H403 frozen independent source sample count mismatch")
    a,b,c=fingerprint(train),fingerprint(dev),fingerprint(audit)
    if (len(a),len(b),len(c))!=(TRAIN_N,DEV_N,AUDIT_N) or a&b or a&c or b&c:
        raise ValueError("H403 original dataset split collision")
    for rows in (train,dev,audit):
        n=len(rows)
        if (sum(item[4] for item in rows)!=n//2 or
                sum(item[3]==0 for item in rows)!=n//2):
            raise ValueError("H403 missing-target and EN/AR balance violated")
        for img,prompt,gold,lang,missing,target in rows:
            if img.size!=(224,224) or (target is None)!=missing:
                raise ValueError("H403 invalid synthetic input/label")
            decoded=json.loads(gold)
            if decoded!=({"abstain":True} if missing else {"point_2d":target}):
                raise ValueError("H403 invalid generated target evidence")
    return {"train":sha("".join(sorted(a)).encode()),
            "dev":sha("".join(sorted(b)).encode()),
            "audit_author_visible":sha("".join(sorted(c)).encode())}

def verify_parent(path):
    if sha(path.read_bytes())!=PARENT_ADAPTER_SHA:
        raise ValueError("H403 private H399 parent adapter SHA mismatch")
    parent=path.parent.parent
    original=(parent/"receipt.json").read_bytes()
    if sha(original)!=PARENT_RECEIPT_SHA:
        raise ValueError("H403 H399 training receipt mismatch")
    if (parent/"receipt.sha256").read_text().strip()!=PARENT_RECEIPT_SHA:
        raise ValueError("H403 H399 receipt sidecar mismatch")
    doc=json.loads(original)
    if (doc.get("status")!="COMPLETED_REAL_2B_SYNTHETIC_RESEARCH_LORA" or
        doc.get("optimizer_steps")!=16 or
        doc.get("official_hala_model_trained") is not False):
        raise ValueError("H403 original parent is not source-only synthetic")

def batch(processor,case):
    image,prompt,gold,*_=case
    user=[{"role":"user","content":[{"type":"image"},{"type":"text","text":prompt}]}]
    prefix=processor.apply_chat_template(user,tokenize=False,add_generation_prompt=True)
    whole=processor.apply_chat_template(user+[
        {"role":"assistant","content":[{"type":"text","text":gold}]}],
        tokenize=False,add_generation_prompt=False)
    context=processor(text=[prefix],images=[image],return_tensors="pt")
    encoded=processor(text=[whole],images=[image],return_tensors="pt")
    k=context["input_ids"].shape[1]
    if k<1 or k>=encoded["input_ids"].shape[1]:
        raise ValueError("H403 supervision assistant-only prefix wrong")
    labels=encoded["input_ids"].clone()
    labels[:,:k]=-100
    if "attention_mask" in encoded:
        labels[encoded["attention_mask"]==0]=-100
    if (labels!=-100).sum().item()<5:
        raise ValueError("H403 assistant label missing")
    encoded["labels"]=labels
    return {name:value.to("cuda:0") for name,value in encoded.items()}

def loss_eval(model,processor,cases):
    import torch
    model.eval()
    values=[]
    with torch.inference_mode():
        for case in cases:
            loss=model(**batch(processor,case)).loss
            if not torch.isfinite(loss):
                raise ValueError("H403 held out loss not finite")
            values.append(float(loss.detach().cpu()))
    return round(sum(values)/len(values),6)

def parse(raw):
    try:
        parsed=json.loads(raw.strip())
    except (ValueError,TypeError):
        return "invalid",None
    if type(parsed) is not dict:
        return "invalid",None
    if set(parsed)=={"abstain"} and parsed["abstain"] is True:
        return "abstain",None
    if set(parsed)!={"point_2d"}:
        return "invalid",None
    point=parsed["point_2d"]
    if (type(point) is not list or len(point)!=2 or
        any(type(x) not in (int,float) or not math.isfinite(x) or
            not 0<=x<=1000 for x in point)):
        return "invalid",None
    return "point",[float(v) for v in point]

def generated(model,processor,rows,tag):
    import torch
    model.eval()
    grouped={key:{"n":0,"correct":0,"false_clicks":0,"invalid":0,"abstain":0}
             for key in ("all","en","ar","positive","absent")}
    with torch.inference_mode():
        for i,(im,prompt,gold,lang,missing,target) in enumerate(rows):
            messages=[{"role":"user","content":[{"type":"image"},
                      {"type":"text","text":prompt}]}]
            prefix=processor.apply_chat_template(
                messages,tokenize=False,add_generation_prompt=True)
            enc=processor(text=[prefix],images=[im],return_tensors="pt").to("cuda:0")
            ids=model.generate(**enc,max_new_tokens=80,do_sample=False,
                               pad_token_id=processor.tokenizer.eos_token_id)
            output=processor.batch_decode(
                ids[:,enc["input_ids"].shape[1]:],skip_special_tokens=True)[0]
            kind,point=parse(output)
            ok=(kind=="abstain" if missing else
                kind=="point" and point is not None and
                abs(point[0]-target[0])<=13/224*1000 and
                abs(point[1]-target[1])<=13/224*1000)
            for key in ("all","en" if lang==0 else "ar",
                        "absent" if missing else "positive"):
                group=grouped[key]
                group["n"]+=1
                group["correct"]+=int(ok)
                group["false_clicks"]+=int(missing and kind=="point")
                group["invalid"]+=int(kind=="invalid")
                group["abstain"]+=int(kind=="abstain")
            print("H403_REAL_GENERATION",tag,i+1,"missing",missing,
                  "kind",kind,"correct",ok,flush=True)
    return grouped

def run():
    import torch
    state={"schema":"hala.h403.private-invented-absent-target-lora.v1",
           "status":"NOT_STARTED","model":MODEL,"revision":REV,
           "parent_adapter_sha256":PARENT_ADAPTER_SHA,
           "public_base_license":"apache-2.0",
           "train_source":"originally authored algorithmic rectangles only",
           "train_uses_prior_h402_eval":False,
           "authorized_original_gui_train":False,
           "independent_external_ood":False,
           "official_hala_model_trained":False,
           "weights_published":False,
           "optimizer_steps":0}
    if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not torch.cuda.is_available():
        raise ValueError("H403 private owner Kaggle GPU only")
    if OUT.exists(): raise ValueError("H403 original fresh notebook output required")
    OUT.mkdir(parents=True)
    started=time.monotonic()
    try:
        subprocess.run([sys.executable,"-m","pip","install","-q",
             "transformers==4.57.3","peft==0.18.0",
             "accelerate==1.12.0","Pillow==11.3.0"],check=True,timeout=330)
        from transformers import AutoProcessor,AutoModelForImageTextToText
        from peft import PeftModel
        import transformers,peft
        source=list(Path("/kaggle/input").rglob("adapter_model.safetensors"))
        if len(source)!=1:
            raise ValueError("H403 requires exactly one private H399 LoRA adapter")
        verify_parent(source[0])
        train=[make_case(i,40311) for i in range(TRAIN_N)]
        dev=[make_case(i,40321) for i in range(DEV_N)]
        audit=[make_case(i,40331,style="shifted") for i in range(AUDIT_N)]
        state["split_fingerprints"]=validate_groups(train,dev,audit)
        torch.manual_seed(403)
        processor=AutoProcessor.from_pretrained(MODEL,revision=REV,
                                                  trust_remote_code=False)
        base=AutoModelForImageTextToText.from_pretrained(
            MODEL,revision=REV,dtype=torch.float16,
            trust_remote_code=False,attn_implementation="sdpa").to("cuda:0")
        model=PeftModel.from_pretrained(base,source[0].parent,is_trainable=True)
        params=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
        if (not params or any("lora_" not in name for name,_ in params)):
            raise ValueError("H403 GPU only LoRA matrices can update")
        before_snapshot={name:p.detach().float().cpu().clone() for name,p in params}
        state["parent_dev_loss"]=loss_eval(model,processor,dev[:12])
        state["parent_audit_generated"]=generated(model,processor,audit[:12],"parent")
        if hasattr(model,"enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model,"gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model.config,"use_cache"):
            model.config.use_cache=False
        optim=torch.optim.AdamW([p for _,p in params],lr=0.00008)
        perm=list(range(TRAIN_N))
        random.Random(403).shuffle(perm)
        history=[]
        for step in range(STEPS):
            model.train()
            optim.zero_grad(set_to_none=True)
            b=batch(processor,train[perm[step]])
            l=model(**b).loss
            if not torch.isfinite(l):
                raise RuntimeError("H403 nonfinite native refusal LoRA gradient")
            l.backward()
            torch.nn.utils.clip_grad_norm_([p for _,p in params],1.)
            optim.step()
            history.append(round(l.item(),6))
            state["optimizer_steps"]=step+1
            if step%12==0 or step==STEPS-1:
                print("H403_REAL_PRIVATE_SAFETY_LORA_STEP",step+1,
                      STEPS,"loss",history[-1],flush=True)
            del b,l
            torch.cuda.empty_cache()
        moved=sum(1 for name,p in params if bool(
            (p.detach().float().cpu()!=before_snapshot[name]).any()))
        if moved==0:
            raise ValueError("H403 no trained weights actually changed")
        state["changed_trainable_tensors"]=moved
        if hasattr(model,"gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        if hasattr(model.config,"use_cache"):
            model.config.use_cache=True
        state["adapted_dev_loss"]=loss_eval(model,processor,dev[:12])
        state["adapted_audit_generated"]=generated(model,processor,audit[:12],"adapted")
        folder=OUT/"private-invented-abstention-adapter"
        model.save_pretrained(folder,safe_serialization=True)
        adapter=folder/"adapter_model.safetensors"
        if not adapter.is_file() or adapter.stat().st_size<1000:
            raise RuntimeError("H403 private LoRA safetensors missing")
        state.update(status="COMPLETED_PRIVATE_SYNTHETIC_ABSTENTION_LORA",
                     train_n=TRAIN_N,dev_n=DEV_N,audit_n=AUDIT_N,
                     training_losses=history,
                     adapter_sha256=sha(adapter.read_bytes()),
                     adapter_size=adapter.stat().st_size,
                     train_seconds=round(time.monotonic()-started,3),
                     device=torch.cuda.get_device_name(0),
                     torch_version=torch.__version__,
                     transformers_version=transformers.__version__,
                     peft_version=peft.__version__)
    except Exception as err:
        state["status"]="FAILED_PRIVATE_SYNTHETIC_TRAIN_NO_RELEASE"
        state["failure_class"]=type(err).__name__
        raise
    finally:
        report=json.dumps(state,indent=2,sort_keys=True,allow_nan=False).encode()
        (OUT/"receipt.json").write_bytes(report)
        (OUT/"receipt.sha256").write_text(sha(report))
        print("H403_RECEIPT",json.dumps(state,sort_keys=True),flush=True)

if __name__=="__main__":
    run()
