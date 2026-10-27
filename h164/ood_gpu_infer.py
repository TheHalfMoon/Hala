"""H164 frozen TWO truly new original OOD apps paired LoRA free T4 inference.

64 previously unseen original Pell/Pikaday browser screenshots × 2 frozen
models = 128 GPU inference calls. No optimizer; no owner gold point uploads.
Do not run until exact newly trained LoRA SHA and ZIP are frozen/pinned.
"""
import hashlib,io,json
from collections import Counter
from pathlib import Path
from h156.train_gpu import open_media
from h158.gpu_infer import import_adapter
from h164.train_gpu import predict
from h164.native_action_contract import parse_action

SOURCE_ZIP_SHA="6056b16ef9f6ef86ec2a514e7d71a6d2840d0930dcf770ffa198e1367f444a85"
PARENT_ZIP_SHA="8436ab1a9adbdd90b4fe21e6bb389a185994eb0ab631b6408877c2b8c796bb35"
PARENT_LORA_SHA="72227e6ca39b5be1052812640e868a17f583367e4a82d03ff6f5ba569f02007c"
TRAINED_MODEL_SHA="UNQUALIFIED"
TRAINED_MODEL_ZIP_SHA="UNQUALIFIED"

def sha(data):
    return hashlib.sha256(data).hexdigest()

def admit(archive):
    archive=Path(archive)
    if archive.is_file() and sha(archive.read_bytes())!=SOURCE_ZIP_SHA:
        raise ValueError("H164 original two-new-app OOD source SHA changed")
    with open_media(archive) as z:
        doc=json.loads(z.read("manifest.json"))
        records=doc.get("private_cases")
        if (doc.get("schema")!="hala.h164.private-ood-source.v1" or
            doc.get("split")!="OOD" or doc.get("count")!=64 or
            doc.get("reference_gold_uploaded") is not False or
            not isinstance(records,list) or len(records)!=64 or
            set(z.namelist())!={"manifest.json"}|
            {"images/"+row["image"] for row in records}):
            raise ValueError("H164 new-application source schema or count drift")
        seen=set();stats=Counter()
        for row in records:
            if (row.get("split")!="OOD" or
                row.get("app") not in ("pell","pikaday") or
                row.get("task") not in ("format_text","select_calendar_day") or
                row.get("action_type")!="click" or
                row.get("language") not in ("ar","en") or
                any(key in row for key in (
                    "training_supervision","owner_supervision",
                    "owner_gold_click","reference_point","owner_reference_action")) or
                row["image_sha256"] in seen or
                sha(z.read("images/"+row["image"]))!=row["image_sha256"]):
                raise ValueError("H164 OOD original media/reference leakage")
            seen.add(row["image_sha256"])
            stats[row["app"]]+=1
        if stats!={"pell":32,"pikaday":32}:
            raise ValueError("H164 two-independent-application OOD quotas changed")
    return records

def run(source,before_model,new_model,output):
    import torch,transformers
    from PIL import Image
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText,AutoProcessor
    from h132.grounder_contract import MODEL_ID,MODEL_SHA

    output=Path(output)
    if (output.exists() or
        not str(output).startswith("/kaggle/working/") or
        transformers.__version__!="5.17.0" or
        not torch.cuda.is_available() or
        not any("T4" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count()))):
        raise ValueError("H164 only private FREE Kaggle T4 research inference")
    if (len(TRAINED_MODEL_SHA)!=64 or
        len(TRAINED_MODEL_ZIP_SHA)!=64 or
        Path(new_model).is_file() and
        sha(Path(new_model).read_bytes())!=TRAINED_MODEL_ZIP_SHA):
        raise ValueError("H164 new research LoRA not frozen; no OOD model calls")
    records=admit(source)
    previous=import_adapter(
        before_model,Path("/kaggle/working/h164-ood-old-h161-adapter"),
        PARENT_LORA_SHA)
    current=import_adapter(
        new_model,Path("/kaggle/working/h164-ood-new-h164-adapter"),
        TRAINED_MODEL_SHA)
    processor=AutoProcessor.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,trust_remote_code=False)
    base=AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,torch_dtype=torch.float16,
        trust_remote_code=False,attn_implementation="sdpa").to("cuda:0")
    model=PeftModel.from_pretrained(
        base,str(previous),adapter_name="baseline",is_trainable=False)
    model.load_adapter(str(current),adapter_name="trained",is_trainable=False)
    model.eval()
    saved=[];summary=Counter()
    with open_media(source) as z:
        for role,name in (("before","baseline"),("after","trained")):
            model.set_adapter(name)
            for row in records:
                with Image.open(io.BytesIO(z.read(
                    "images/"+row["image"]))) as f:
                    photo=f.convert("RGB")
                if photo.size!=(960,720):
                    raise ValueError("H164 original OOD viewport mutated")
                raw=predict(processor,model,torch,photo,row)
                parsed=parse_action(raw)
                summary[role+"/calls"]+=1
                summary[role+"/valid"]+=int(parsed["valid"])
                summary[role+"/click"]+=int(parsed["kind"]=="click")
                saved.append({
                    "role":role,"image_sha":row["image_sha256"],
                    "app":row["app"],"task":row["task"],
                    "language":row["language"],"raw":raw,"parsed":parsed})
                if len(saved)%16==0:
                    print("H164_REAL_NEW_APP_GPU_OOD",len(saved),128,flush=True)
    if summary["before/calls"]!=64 or summary["after/calls"]!=64:
        raise ValueError("H164 two-original-app GPU proposals incomplete")
    receipt={
        "schema":"hala.h164.two-new-original-app-free-t4-paired-eval.v1",
        "status":"H164_COMPLETED_128_REAL_GPU_OOD_CALLS",
        "source_zip_sha256":SOURCE_ZIP_SHA,
        "frozen_H161_adapter_sha256":PARENT_LORA_SHA,
        "new_H164_adapter_sha256":TRAINED_MODEL_SHA,
        "real_gpu_calls":128,"optimizer_steps":0,
        "private_owner_gold_on_gpu":False,
        "actual_browser_native_success_measured":False,
        "h140_blind_calls":0,"paid_GPU":False,
        "flagship_Hala35B_unchanged":True,
        "counts":dict(summary),"private_predictions":saved
    }
    output.mkdir(parents=True,exist_ok=False)
    (output/"h164-private-two-new-ood-predictions.json").write_text(
        json.dumps(receipt,indent=2,sort_keys=True,ensure_ascii=False)+"\n",
        "utf8")
    return {k:v for k,v in receipt.items() if k!="private_predictions"}
