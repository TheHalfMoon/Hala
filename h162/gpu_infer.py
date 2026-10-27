"""H162 paired H157-vs-H161 frozen 2B model 3rd-app OOD free T4 GPU calls.

Independent original MIT HTML5 Kanban app at pinned SHA, new 16 scenes.
No optimizer steps, no owner-reference coordinates and no H140 blind.
"""
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

from h156.train_gpu import open_media
from h158.gpu_infer import import_adapter
from h161.train_gpu import actions,predict
from h162.pack_private_inference import (
    APP_SHA, SOURCE_SHA,BASE_SHA,TRAINED_SHA,
    SOURCE_ZIP,OLD_ZIP,NEW_ZIP,
)

SOURCE_ZIP_SHA="8bd15a9129160bc7908fbe388a58c61a564e35ad2984e440487ec07c83a989e3"
NEW_ZIP_SHA="8436ab1a9adbdd90b4fe21e6bb389a185994eb0ab631b6408877c2b8c796bb35"


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def source_cases(source):
    source=Path(source)
    if source.is_file() and sha(source.read_bytes())!=SOURCE_ZIP_SHA:
        raise ValueError("H162 source frozen original HTML5 zip changed")
    with open_media(source) as z:
        raw=z.read("manifest.json")
        doc=json.loads(raw)
        records=doc["private_cases"]
        if (doc.get("schema")!=
            "hala.h162.original-html5-kanban-external-app-model-input.v1" or
            doc.get("source_app_sha")!=APP_SHA or
            doc.get("source_manifest_sha256")!=SOURCE_SHA or
            doc.get("source_cases")!=16 or
            len(records)!=16 or
            set(z.namelist())!={"manifest.json"}|
            {"images/"+r["image"] for r in records}):
            raise ValueError("H162 original third-app independent model source invalid")
        seen=set()
        for item in records:
            if (set(item)!={"index","task","language","app",
                            "image","image_sha256","instruction"} or
                item["app"]!="react-html5-board" or
                item["task"]!="drag_backlog_to_todo" or
                item["language"] not in ("en","ar") or
                sha(z.read("images/"+item["image"]))!=item["image_sha256"] or
                item["image_sha256"] in seen):
                raise ValueError("H162 third app model image leaked gold or duplicated")
            seen.add(item["image_sha256"])
    return records


def run(source,prior_zip,new_zip,output):
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
        raise ValueError("H162 FREE T4 private research inference ONLY")
    rows=source_cases(source)
    if Path(new_zip).is_file() and sha(Path(new_zip).read_bytes())!=NEW_ZIP_SHA:
        raise ValueError("H162 new trained H161 adapter zip mutated")
    previous=import_adapter(
        prior_zip,Path("/kaggle/working/h162-older-h157-lora"),BASE_SHA)
    newest=import_adapter(
        new_zip,Path("/kaggle/working/h162-trained-h161-lora"),TRAINED_SHA)
    processor=AutoProcessor.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,trust_remote_code=False)
    base=AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,torch_dtype=torch.float16,
        trust_remote_code=False,attn_implementation="sdpa").to("cuda:0")
    model=PeftModel.from_pretrained(
        base,str(previous),adapter_name="prior",is_trainable=False)
    model.load_adapter(str(newest),adapter_name="trained",is_trainable=False)
    model.eval()
    records=[]
    counters=Counter()
    with open_media(source) as z:
        for role,name in (("before","prior"),("after","trained")):
            model.set_adapter(name)
            for item in rows:
                with Image.open(io.BytesIO(
                    z.read("images/"+item["image"]))) as im:
                    image=im.convert("RGB")
                if image.size!=(960,720):
                    raise ValueError("H162 original third-app PNG resolution altered")
                raw=predict(processor,model,torch,image,item)
                parsed=actions(raw)
                counters[role+"_calls"]+=1
                counters[role+"_valid"]+=int(parsed["valid"])
                counters[role+"_drag"]+=int(parsed["kind"]=="drag")
                records.append({
                    "role":role,"image_sha":item["image_sha256"],
                    "app":item["app"],"task":item["task"],
                    "language":item["language"],"raw":raw,"parsed":parsed,
                })
                if len(records)%8==0:
                    print("H162_32_REAL_THIRD_APP_FREE_GPU_CALLS",len(records),32,
                          flush=True)
    if counters["before_calls"]!=16 or counters["after_calls"]!=16:
        raise ValueError("H162 paired external 3rd-app GPU calls incomplete")
    summary={
        "schema":"hala.h162.third-original-app-frozen-two-lora-gpu-inference.v1",
        "status":"H162_COMPLETED_32_REAL_FREE_T4_THIRD_APP_CALLS",
        "source_app":"react-html5-board",
        "source_manifest_sha256":SOURCE_SHA,
        "source_zip_sha256":SOURCE_ZIP_SHA,
        "baseline_H157_lora_sha256":BASE_SHA,
        "trained_H161_lora_sha256":TRAINED_SHA,
        "gpu_inference_calls":32,"optimizer_steps":0,
        "h140_blind_calls":0,"paid_GPU":False,
        "Hala35B_unchanged":True,
        "original_app_native_success_measured":False,
        "historically_seen_app_not_sealed_BLIND":True,
        "counts":dict(counters),
        "private_predictions":records,
    }
    output.mkdir(parents=True,exist_ok=False)
    (output/"h162-private-third-app-predictions.json").write_text(
        json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2)+"\n","utf8")
    return {k:v for k,v in summary.items() if k!="private_predictions"}
