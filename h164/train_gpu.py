"""H164 FREE-T4 language-LoRA SFT on new 384 TRAIN; independent 96 DEV.

Requires pinning the one-shot fully-qualified input ZIP SHAs before GPU launch.
No old consumed model/DEV/OOD images in optimizer. NEVER use H140 blind.
"""
import hashlib,io,json,os,random,time
from collections import Counter
from pathlib import Path
from h156.train_gpu import open_media
from h164.native_action_contract import parse_action,visible_prompt,supervision

# These MUST be pinned to new private owner source receipts before launch.
TRAIN_SHA="ed08ef5a7e8b38d6fe8c38c235d93d8fa0625437ca3c67a5578f4fc941db4495"
DEV_SHA="17adcc78fae62d7de3ae73c2ae7d3863abd0b2f1105edf4f45db7f4c78b503b0"
PARENT_LORA_SHA="72227e6ca39b5be1052812640e868a17f583367e4a82d03ff6f5ba569f02007c"
PARENT_ZIP_SHA="8436ab1a9adbdd90b4fe21e6bb389a185994eb0ab631b6408877c2b8c796bb35"
ACCUM=2
STEPS=192
SEED=164
EXPECTED_KAGGLE_FILES={"README.md","adapter_config.json","adapter_model.safetensors"}

def sha(blob):
    return hashlib.sha256(blob).hexdigest()

def admit(path,split):
    fixed=TRAIN_SHA if split=="TRAIN" else DEV_SHA
    path=Path(path)
    if (len(fixed)!=64 or path.is_file() and sha(path.read_bytes())!=fixed):
        raise ValueError("H164 frozen supervised research source SHA unqualified")
    with open_media(path) as z:
        raw=z.read("manifest.json")
        doc=json.loads(raw)
        examples=doc.get("private_cases")
        n=384 if split=="TRAIN" else 96
        if (doc.get("schema")!=f"hala.h164.private-{split.lower()}-source.v1"
            or doc.get("split")!=split or doc.get("count")!=n
            or doc.get("reference_gold_uploaded") is not (split=="TRAIN")
            or not isinstance(examples,list) or len(examples)!=n
            or set(z.namelist())!={"manifest.json"}|
            {"images/"+row["image"] for row in examples}):
            raise ValueError("H164 native source frozen manifest/ZIP changed")
        identities=set()
        counts=Counter()
        for row in examples:
            if (row.get("split")!=split or
                row.get("language") not in ("en","ar") or
                row.get("action_type") not in ("click","drag","type") or
                row["image_sha256"] in identities or
                sha(z.read("images/"+row["image"]))!=row["image_sha256"]):
                raise ValueError("H164 native image/case mismatch")
            identities.add(row["image_sha256"])
            counts[row["app"]]+=1
            if split=="TRAIN":
                if (parse_action(supervision(row))["valid"] is not True or
                    parse_action(supervision(row))["kind"]!=row["action_type"]):
                    raise ValueError("H164 original supervised point/drag/type invalid")
            elif any(x in row for x in (
                "training_supervision","owner_gold_click","reference_point",
                "owner_supervision")):
                raise ValueError("H164 DEV reference leaked to model")
        if split=="TRAIN" and (len(counts)<4 or not {"click","drag","type"}.issubset(
            {r["action_type"] for r in examples})):
            raise ValueError("H164 missing independent native TRAIN UI coverage")
    return examples

def assemble(processor,image,row,device):
    if image.size!=(960,720):
        raise ValueError("H164 original GUI screenshot not 960x720")
    user=[{"role":"user","content":[
        {"type":"image"},{"type":"text","text":visible_prompt(row)}]}]
    first=processor.apply_chat_template(
        user,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    labeled=processor.apply_chat_template(
        user+[{"role":"assistant","content":[
            {"type":"text","text":supervision(row)}]}],
        tokenize=False,add_generation_prompt=False,enable_thinking=False)
    if not labeled.startswith(first):
        raise ValueError("H164 supervisor prefix/prompt mismatch")
    before=processor(text=[first],images=[image],return_tensors="pt")
    full=processor(text=[labeled],images=[image],return_tensors="pt")
    begin=int(before["input_ids"].shape[1])
    if not 0<begin<int(full["input_ids"].shape[1]):
        raise ValueError("H164 tokenized original native gold alignment missing")
    labels=full["input_ids"].clone()
    labels[:,:begin]=-100
    if "attention_mask" in full:
        labels[full["attention_mask"]==0]=-100
    if int((labels!=-100).sum().item())<5:
        raise ValueError("H164 native label-bearing tokens absent")
    full["labels"]=labels
    return {key:val.to(device) if hasattr(val,"to") else val
            for key,val in full.items()}

def predict(processor,model,torch,image,row):
    model.eval()
    messages=[{"role":"user","content":[
        {"type":"image"},{"type":"text","text":visible_prompt(row)}]}]
    txt=processor.apply_chat_template(
        messages,tokenize=False,add_generation_prompt=True,
        enable_thinking=False)
    batch=processor(text=[txt],images=[image],return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        seq=model.generate(**batch,max_new_tokens=190,
                           do_sample=False,max_time=55)
    torch.cuda.synchronize(0)
    return processor.batch_decode(
        seq[:,batch["input_ids"].shape[1]:],
        skip_special_tokens=True)[0].strip()

def compare_dev(processor,model,torch,dev_zip,examples,label):
    from PIL import Image
    rec=[];counts=Counter()
    with open_media(dev_zip) as z:
        for i,row in enumerate(examples,1):
            with Image.open(io.BytesIO(z.read("images/"+row["image"]))) as raw:
                picture=raw.convert("RGB")
            proposal=predict(processor,model,torch,picture,row)
            parsed=parse_action(proposal)
            counts[row["app"]+"/total"]+=1
            counts[row["app"]+"/valid"]+=int(parsed["valid"])
            counts[row["app"]+"/non_abstain"]+=int(
                parsed["kind"] not in (None,"abstain"))
            rec.append({
                "role":label,"app":row["app"],"task":row["task"],
                "language":row["language"],"image_sha":row["image_sha256"],
                "raw":proposal,"parsed":parsed})
            if i%16==0:
                print("H164_REAL_FREE_GPU_DEV",label,i,len(examples),flush=True)
    return dict(counts),rec

def import_parent(source,dest):
    source=Path(source)
    if dest.exists() or len(PARENT_ZIP_SHA)!=64 or (
        source.is_file() and sha(source.read_bytes())!=PARENT_ZIP_SHA):
        raise ValueError("H164 original trained research LoRA ZIP identity changed")
    with open_media(source) as z:
        if (set(z.namelist())!=EXPECTED_KAGGLE_FILES or
            sha(z.read("adapter_model.safetensors"))!=PARENT_LORA_SHA):
            raise ValueError("H164 pretrained specialist LoRA weights changed")
        config=json.loads(z.read("adapter_config.json"))
        if config.get("r")!=8:
            raise ValueError("H164 must start with frozen rank8 specialist")
        dest.mkdir(parents=True,exist_ok=False)
        for name in sorted(EXPECTED_KAGGLE_FILES):
            (dest/name).write_bytes(z.read(name))
    return dest

def run(train_zip,dev_zip,parent_zip,output):
    import torch,transformers
    from PIL import Image
    from peft import PeftModel
    from transformers import AutoProcessor,AutoModelForImageTextToText
    from h132.grounder_contract import MODEL_ID,MODEL_SHA

    output=Path(output)
    if (output.exists() or not str(output).startswith("/kaggle/working/") or
        transformers.__version__!="5.17.0" or not torch.cuda.is_available() or
        not any("T4" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count()))):
        raise ValueError("H164 paid GPU forbidden; require free private T4")
    train=admit(train_zip,"TRAIN")
    dev=admit(dev_zip,"DEV")
    if len(train)!=384 or len(dev)!=96 or (
        {r["image_sha256"] for r in train}&
        {r["image_sha256"] for r in dev}):
        raise ValueError("H164 384/96 independent source not qualified")
    random.seed(SEED);torch.manual_seed(SEED)
    os.environ.update(HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
    old=import_parent(parent_zip,Path("/kaggle/working/h164-frozen-h161-parent"))
    processor=AutoProcessor.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,trust_remote_code=False)
    base=AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,torch_dtype=torch.float16,
        trust_remote_code=False,attn_implementation="sdpa").to("cuda:0")
    model=PeftModel.from_pretrained(base,str(old),is_trainable=True)
    before,before_rows=compare_dev(
        processor,model,torch,dev_zip,dev,"before")
    for name,param in model.named_parameters():
        param.requires_grad_("lora_" in name)
    trainable=[(name,p) for name,p in model.named_parameters()
               if p.requires_grad]
    if (not trainable or any("lora_" not in name or "visual" in name or
        "vision" in name for name,p in trainable)):
        raise ValueError("H164 no visual encoder training in safe T4 control")
    if hasattr(model,"enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model,"gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config,"use_cache"):model.config.use_cache=False
    optim=torch.optim.AdamW([p for _,p in trainable],lr=2e-5,weight_decay=.01)
    schedule=list(range(384))
    random.Random(SEED).shuffle(schedule)
    if len(schedule)!=STEPS*ACCUM:
        raise ValueError("H164 one exposure per fresh TRAIN original screenshot")
    losses=[];started=time.monotonic()
    with open_media(train_zip) as originals:
        for step in range(STEPS):
            model.train()
            optim.zero_grad(set_to_none=True)
            for j in range(ACCUM):
                row=train[schedule[step*ACCUM+j]]
                with Image.open(io.BytesIO(
                    originals.read("images/"+row["image"]))) as im:
                    picture=im.convert("RGB")
                batch=assemble(processor,picture,row,"cuda:0")
                results=model(**batch)
                loss=results.loss/ACCUM
                if not torch.isfinite(loss):
                    raise ValueError("H164 actual original-task supervised loss nonfinite")
                loss.backward()
                losses.append(round(float(loss.detach().cpu())*ACCUM,6))
                del batch,results,loss
            norm=torch.nn.utils.clip_grad_norm_(
                [p for _,p in trainable],1.0)
            if not torch.isfinite(norm):
                raise ValueError("H164 T4 training gradient nonfinite")
            optim.step()
            if (step+1)%16==0:
                print("H164_GENUINE_384_FRESH_TRAIN_IMAGES",step+1,STEPS,flush=True)
            torch.cuda.empty_cache()
    model.eval()
    if hasattr(model.config,"use_cache"):model.config.use_cache=True
    output.mkdir(parents=True,exist_ok=False)
    adapter_dir=output/"private-h164-multi-original-app-trained-lora"
    model.save_pretrained(adapter_dir,safe_serialization=True)
    weight=adapter_dir/"adapter_model.safetensors"
    if not weight.is_file() or sha(weight.read_bytes())==PARENT_LORA_SHA:
        raise ValueError("H164 genuine model weight update missing")
    after,after_rows=compare_dev(
        processor,model,torch,dev_zip,dev,"after")
    receipt={
        "schema":"hala.h164.new-original-app-native-lang-lora-sft.v1",
        "status":"H164_192_GENUINE_FREE_T4_OPTIMIZER_STEPS_COMPLETED",
        "optimizer_steps":STEPS,"accumulation":ACCUM,
        "unique_train_images":384,"train_exposures":384,
        "dev_images":96,"dev_reference_labels_seen_by_gpu":False,
        "source_train_zip_sha256":TRAIN_SHA,
        "source_dev_zip_sha256":DEV_SHA,
        "old_H161_adapter_sha256":PARENT_LORA_SHA,
        "new_H164_adapter_sha256":sha(weight.read_bytes()),
        "dev_model_format_before":before,
        "dev_model_format_after":after,
        "actual_browser_task_success_measured":False,
        "separate_ood_evaluation_not_performed":True,
        "losses":losses,
        "elapsed_seconds":round(time.monotonic()-started,1),
        "h140_blind_calls":0,"paid_GPU":False,
        "flagship_Hala35B_unchanged":True,"weights_published":False}
    (output/"h164-training-receipt.json").write_text(
        json.dumps(receipt,indent=2,sort_keys=True)+"\n","utf8")
    for name,rows in (("before",before_rows),("after",after_rows)):
        (output/("h164-dev-"+name+".json")).write_text(
            json.dumps(rows,ensure_ascii=False,sort_keys=True,indent=2)+"\n","utf8")
    return {k:v for k,v in receipt.items() if k!="losses"}
