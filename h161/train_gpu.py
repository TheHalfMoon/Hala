"""H161 mixed native click/drag original-app 2B LoRA SFT on free Kaggle T4.

TRAIN only supplies gold labels; DEV labels never leave owner Windows.
No optimizer update on H158/H159/H160 consumed scenes or H140 blind.
"""
import hashlib
import io
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

from h156.train_gpu import open_media

TRAIN_SHA="c39bd3259e07210c004773de6305b8b816f9a13509e2c180b40a3eb42fb54c5d"
DEV_SHA="a6310404c89d50ea0f4f854b829a9ad67f6b6f68bf6e890bd680f8349d82afce"
OLD_MODEL_ZIP_SHA="14e3a9a61123a92fb23d53297531831fa19ff7d2435d6704717af7a90c08e808"
OLD_LORA_SHA="181ade5652daf797c27945cbeafca2967cf465d36af58870648400890414a5a8"
PARENT_SHA="0710797074d8407640fa8c0e65bbfe8672175a644483701ce156aaab02621997"
STEPS=48
ACCUM=2
SEED=161
MODEL_FILES={"README.md","adapter_config.json","adapter_model.safetensors"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def actions(raw):
    if not isinstance(raw,str) or len(raw)>1200:
        return {"valid":False,"kind":None}
    try:
        doc=json.loads(raw)
        if type(doc) is not dict or set(doc)!={"actions"}:
            raise ValueError("Expected one actions array")
        value=doc["actions"]
        if not isinstance(value,list) or len(value)!=1 or not isinstance(value[0],dict):
            raise ValueError("H161 exactly one native action")
        a=value[0]
        kind=a.get("kind")
        if kind=="abstain" and set(a)=={"kind"}:
            return {"valid":True,"kind":"abstain"}
        if kind=="click" and set(a)=={"kind","point"}:
            points=[a["point"]]
        elif kind=="drag" and set(a)=={"kind","start","end"}:
            points=[a["start"],a["end"]]
        else:
            raise ValueError("Unexpected click/drag shape")
        for xy in points:
            if (type(xy) is not list or len(xy)!=2 or
                any(type(x) not in (int,float) or not 0<=x<bound
                    for x,bound in zip(xy,(960,720)))):
                raise ValueError("Out-of-range native pixel point")
        return {"valid":True,"kind":kind,"points":points}
    except (ValueError,TypeError,KeyError):
        return {"valid":False,"kind":None}


def prompt(row):
    return (
        "You are Hala, a visual computer-use agent. Read the 960x720 "
        "original-application screenshot and the user's task. Reply ONLY "
        "with JSON {'actions':[...]} with ONE action. For a click use "
        "{\"kind\":\"click\",\"point\":[x,y]}. For a native drag use "
        "{\"kind\":\"drag\",\"start\":[x,y],\"end\":[x,y]}. Coordinates "
        "are absolute screenshot pixels. If unclear, use "
        "{\"kind\":\"abstain\"}. Do not use DOM or hidden HTML. "
        "Task: " + row["instruction"]
    )


def label(row):
    return json.dumps(row["training_supervision"],
                      ensure_ascii=False,separators=(",",":"),sort_keys=True)


def admit(archive,split):
    archive=Path(archive)
    wanted=TRAIN_SHA if split=="TRAIN" else DEV_SHA
    if archive.is_file() and sha(archive.read_bytes())!=wanted:
        raise ValueError("H161 frozen original-app archive checksum changed")
    with open_media(archive) as z:
        raw=z.read("manifest.json")
        doc=json.loads(raw)
        records=doc["private_cases"]
        n=48 if split=="TRAIN" else 24
        if (doc.get("schema")!=
            "hala.h161.freeze-private-"+split.lower()+"-source.v1"
            or doc.get("parent_manifest_sha256")!=PARENT_SHA
            or doc.get("split")!=split or doc.get("case_count")!=n
            or doc.get("h140_blind_calls")!=0 or doc.get("paid_gpu") is not False
            or len(records)!=n
            or set(z.namelist())!={"manifest.json"}|
            {"images/"+r["image"] for r in records}):
            raise ValueError("H161 frozen original app split/coverage changed")
        counted=Counter()
        seen=set()
        for r in records:
            if (r["split"]!=split or
                r["app"] not in ("todomvc-es6","react-dnd-kit-board") or
                r["image_sha256"] in seen or
                sha(z.read("images/"+r["image"]))!=r["image_sha256"]):
                raise ValueError("H161 original task/PNG split changed")
            seen.add(r["image_sha256"])
            counted[r["app"]]+=1
            if split=="TRAIN":
                if ("training_supervision" not in r or
                    not actions(label(r))["valid"]):
                    raise ValueError("H161 original native SFT label invalid")
                expected="click" if r["app"]=="todomvc-es6" else "drag"
                if actions(label(r))["kind"]!=expected:
                    raise ValueError("H161 native task mismatch")
            elif "training_supervision" in r:
                raise ValueError("H161 DEV GOLD LEAK TO GPU")
        if counted!={name:n//2 for name in
                    ("todomvc-es6","react-dnd-kit-board")}:
            raise ValueError("H161 training/DEV mixed-app quotas mismatch")
    return records


def assemble(processor,image,row,device):
    if image.size!=(960,720):
        raise ValueError("H161 original image resolution changed")
    user=[{"role":"user","content":[
        {"type":"image"},{"type":"text","text":prompt(row)}]}]
    a=processor.apply_chat_template(user,tokenize=False,
        add_generation_prompt=True,enable_thinking=False)
    b=processor.apply_chat_template(user+[{"role":"assistant","content":[
        {"type":"text","text":label(row)}]}],tokenize=False,
        add_generation_prompt=False,enable_thinking=False)
    if not b.startswith(a):
        raise ValueError("H161 model SFT completion prompt misaligned")
    before=processor(text=[a],images=[image],return_tensors="pt")
    full=processor(text=[b],images=[image],return_tensors="pt")
    prefix=int(before["input_ids"].shape[1])
    if not 0<prefix<int(full["input_ids"].shape[1]):
        raise ValueError("H161 cannot mask image/input safely")
    labels=full["input_ids"].clone()
    labels[:,:prefix]=-100
    if "attention_mask" in full:
        labels[full["attention_mask"]==0]=-100
    if (labels!=-100).sum().item()<5:
        raise ValueError("H161 native action loss tokens missing")
    full["labels"]=labels
    return {k:v.to(device) if hasattr(v,"to") else v for k,v in full.items()}


def predict(processor,model,torch,image,row):
    model.eval()
    messages=[{"role":"user","content":[{"type":"image"},
               {"type":"text","text":prompt(row)}]}]
    prepared=processor.apply_chat_template(messages,tokenize=False,
        add_generation_prompt=True,enable_thinking=False)
    batch=processor(text=[prepared],images=[image],
                    return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        result=model.generate(**batch,max_new_tokens=160,do_sample=False,
                              max_time=50)
    torch.cuda.synchronize(0)
    return processor.batch_decode(
        result[:,batch["input_ids"].shape[1]:],
        skip_special_tokens=True)[0].strip()


def import_old(source,dest):
    if dest.exists() or (Path(source).is_file() and
                          sha(Path(source).read_bytes())!=OLD_MODEL_ZIP_SHA):
        raise ValueError("H161 starting model ZIP changed or already extracted")
    with open_media(source) as z:
        if set(z.namelist())!=MODEL_FILES or sha(
            z.read("adapter_model.safetensors"))!=OLD_LORA_SHA:
            raise ValueError("H161 requires exact frozen H157 research model")
        conf=json.loads(z.read("adapter_config.json"))
        if conf.get("r")!=8:
            raise ValueError("H161 language adapter rank changed")
        dest.mkdir(parents=True,exist_ok=False)
        for name in sorted(MODEL_FILES):
            (dest/name).write_bytes(z.read(name))
    return dest


def gpu_predictions(processor,model,torch,archive,rows,label_name):
    from PIL import Image
    entries=[]
    counts=Counter()
    with open_media(archive) as z:
        for n,row in enumerate(rows,1):
            with Image.open(io.BytesIO(z.read("images/"+row["image"]))) as im:
                picture=im.convert("RGB")
            raw=predict(processor,model,torch,picture,row)
            parsed=actions(raw)
            counts[row["app"]+"/total"]+=1
            counts[row["app"]+"/valid"]+=int(parsed["valid"])
            counts[row["app"]+"/non_abstain"]+=int(
                parsed["kind"] not in (None,"abstain"))
            entries.append({
                "app":row["app"],"task":row["task"],"language":row["language"],
                "image_sha":row["image_sha256"],"raw":raw,
                "parsed":parsed,"role":label_name})
            if n%8==0:
                print("H161_REAL_FREE_T4_DEV_INFERENCES",label_name,n,len(rows),
                      flush=True)
    return dict(counts),entries


def run(train_archive,dev_archive,model_archive,output):
    import torch
    import transformers
    from PIL import Image
    from peft import PeftModel
    from transformers import AutoProcessor,AutoModelForImageTextToText
    from h132.grounder_contract import MODEL_ID,MODEL_SHA

    output=Path(output)
    if (output.exists() or
        not str(output).startswith("/kaggle/working/") or
        transformers.__version__!="5.17.0" or
        not torch.cuda.is_available() or
        not any("T4" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count()))):
        raise ValueError("H161 owner-private free noncommercial T4 only")
    train=admit(train_archive,"TRAIN")
    dev=admit(dev_archive,"DEV")
    if {r["image_sha256"] for r in train} & {r["image_sha256"] for r in dev}:
        raise ValueError("H161 DEV source leaked into training")
    random.seed(SEED)
    torch.manual_seed(SEED)
    os.environ.update(HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
    old=import_old(model_archive,Path("/kaggle/working/h161-starting-h157-lora"))
    processor=AutoProcessor.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,trust_remote_code=False)
    base=AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,torch_dtype=torch.float16,
        trust_remote_code=False,attn_implementation="sdpa").to("cuda:0")
    model=PeftModel.from_pretrained(base,str(old),is_trainable=True)
    model.eval()
    before,before_rows=gpu_predictions(processor,model,torch,dev_archive,dev,
                                       "before")
    for name,p in model.named_parameters():
        p.requires_grad_("lora_" in name)
    trainable=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
    if (not trainable or any(
        "lora_" not in name or "visual" in name or "vision" in name
        for name,p in trainable)):
        raise ValueError("H161 language-only research adapter trainable guard")
    if hasattr(model,"enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model,"gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config,"use_cache"):
        model.config.use_cache=False
    optimizer=torch.optim.AdamW([p for _,p in trainable],lr=3e-5,
                                weight_decay=.01)
    # Exactly two epoch passes over 48 unique TRAIN images; 0 DEV in optimizer.
    schedule=list(range(48))*2
    random.Random(SEED).shuffle(schedule)
    if len(schedule)!=STEPS*ACCUM:
        raise ValueError("H161 fixed two-app train-only optimizer schedule")
    losses=[]
    started=time.monotonic()
    with open_media(train_archive) as originals:
        for step in range(STEPS):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for j in range(ACCUM):
                row=train[schedule[step*ACCUM+j]]
                with Image.open(io.BytesIO(originals.read(
                    "images/"+row["image"]))) as im:
                    picture=im.convert("RGB")
                batch=assemble(processor,picture,row,"cuda:0")
                output_model=model(**batch)
                loss=output_model.loss/ACCUM
                if not torch.isfinite(loss):
                    raise ValueError("H161 nonfinite real original native action loss")
                loss.backward()
                losses.append(round(float(loss.detach().cpu())*ACCUM,6))
                del batch,output_model,loss
            torch.nn.utils.clip_grad_norm_([p for _,p in trainable],1.0)
            optimizer.step()
            if (step+1)%8==0:
                print("H161_GENUINE_MIXED_TWO_APP_OPTIMIZER_STEPS",
                      step+1,STEPS,flush=True)
            torch.cuda.empty_cache()
    model.eval()
    if hasattr(model.config,"use_cache"):
        model.config.use_cache=True
    output.mkdir(parents=True,exist_ok=False)
    target=output/"private-h161-mixed-original-app-research-lora"
    model.save_pretrained(target,safe_serialization=True)
    newfile=target/"adapter_model.safetensors"
    if not newfile.is_file():
        raise ValueError("H161 no updated trained neural weights saved")
    learned_sha=sha(newfile.read_bytes())
    if learned_sha==OLD_LORA_SHA:
        raise ValueError("H161 weights unchanged from frozen H157")
    after,after_rows=gpu_predictions(processor,model,torch,dev_archive,dev,
                                     "after")
    receipt={
        "status":"H161_GENUINE_48_T4_TWO_REAL_APP_TRAIN_STEPS_COMPLETE",
        "schema":"hala.h161.real-two-app-mixed-native-free-t4.v1",
        "optimizer_steps":STEPS,"gradient_accumulation":ACCUM,
        "unique_train":48,"weighted_train_samples":96,
        "dev_image_count":24,"dev_labels_seen_by_GPU":False,
        "source_train_zip_sha256":TRAIN_SHA,
        "source_dev_zip_sha256":DEV_SHA,
        "starting_H157_adapter_sha256":OLD_LORA_SHA,
        "new_mixed_app_adapter_sha256":learned_sha,
        "before_dev_model_format":before,"after_dev_model_format":after,
        "loss_trace":losses,
        "native_browser_task_success_measured":False,
        "cross_app_blind_proven":False,
        "h140_blind_calls":0,"paid_GPU":False,
        "Hala35B_unchanged":True,"model_weights_published":False,
        "elapsed_seconds":round(time.monotonic()-started,1),
    }
    (output/"h161-genuine-learning-receipt.json").write_text(
        json.dumps(receipt,sort_keys=True,indent=2)+"\n","utf8")
    for name,rows in (("before",before_rows),("after",after_rows)):
        (output/("h161-private-dev-"+name+".json")).write_text(
            json.dumps(rows,ensure_ascii=False,sort_keys=True,indent=2)+"\n","utf8")
    return {k:v for k,v in receipt.items() if k!="loss_trace"}
