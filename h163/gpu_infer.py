"""H163: 32 frozen real-model inference calls on genuinely new original app.

No training, no reference DOM positions, no gold actions on Kaggle.
"""
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from h156.train_gpu import open_media
from h158.gpu_infer import import_adapter
from h161.train_gpu import actions,predict
from h163.pack_private_inference import (
    SOURCE_SHA,GIT_SHA,OLDER_MODEL_SHA,TRAINED_MODEL_SHA,ORIGINAL_ZIP
)

FROZEN_SOURCE_ZIP_SHA="e898cdfd25c814def7d52f92d4d15db1f2d51c685439449ad5ba8f36a4ad4150"
FROZEN_INTERNAL_SHA="8677b0b61779aa2fe842c39cc1e7ae30c3d63ba2d9c8687eb83fc16ef89bf9c7"
FROZEN_TRAINED_ZIP_SHA="8436ab1a9adbdd90b4fe21e6bb389a185994eb0ab631b6408877c2b8c796bb35"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def admit(source):
    source=Path(source)
    if source.is_file() and sha(source.read_bytes())!=FROZEN_SOURCE_ZIP_SHA:
        raise ValueError("H163 owner source uploaded ZIP changed")
    with open_media(source) as z:
        raw=z.read("manifest.json")
        doc=json.loads(raw)
        cases=doc["private_cases"]
        if (sha(raw)!=FROZEN_INTERNAL_SHA or
            doc.get("schema")!="hala.h163.new-original-gridstack-inference-only.v1"
            or doc.get("source_manifest_sha256")!=SOURCE_SHA
            or doc.get("original_app_sha")!=GIT_SHA
            or doc.get("cases")!=16 or len(cases)!=16
            or doc.get("reference_points_uploaded") is not False or
            set(z.namelist())!={"manifest.json"}|
            {"images/"+c["image"] for c in cases}):
            raise ValueError("H163 frozen real original app model inputs invalid")
        known=set()
        for c in cases:
            if (set(c)!={"index","app","task","language","layout",
                         "pre_column","target_column","image","image_sha256",
                         "instruction"} or
                c["image_sha256"] in known or
                c["app"]!="gridstack-original-column-demo" or
                c["task"] not in ("set_column","add_widget") or
                c["language"] not in ("en","ar") or
                sha(z.read("images/"+c["image"]))!=c["image_sha256"]):
                raise ValueError("H163 original app source case tampered or leaked")
            known.add(c["image_sha256"])
    return cases


def run(source,older,newer,output):
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
        raise ValueError("H163 PRIVATE FREE Kaggle T4 inference only")
    cases=admit(source)
    if Path(newer).is_file() and sha(Path(newer).read_bytes())!=FROZEN_TRAINED_ZIP_SHA:
        raise ValueError("H163 trained H161 research adapter ZIP changed")
    prior=import_adapter(older,Path("/kaggle/working/h163-h157-frozen"),OLDER_MODEL_SHA)
    trained=import_adapter(newer,Path("/kaggle/working/h163-h161-frozen"),TRAINED_MODEL_SHA)
    processor=AutoProcessor.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,trust_remote_code=False)
    base=AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,torch_dtype=torch.float16,
        trust_remote_code=False,attn_implementation="sdpa").to("cuda:0")
    model=PeftModel.from_pretrained(base,str(prior),
                                   adapter_name="prior",is_trainable=False)
    model.load_adapter(str(trained),adapter_name="trained",is_trainable=False)
    model.eval()
    counts=Counter()
    predictions=[]
    with open_media(source) as z:
        for role,name in (("before","prior"),("after","trained")):
            model.set_adapter(name)
            for row in cases:
                with Image.open(io.BytesIO(
                    z.read("images/"+row["image"]))) as file:
                    picture=file.convert("RGB")
                if picture.size!=(960,720):
                    raise ValueError("H163 browser screenshot resolution drift")
                raw=predict(processor,model,torch,picture,row)
                parsed=actions(raw)
                counts[f"{role}_calls"]+=1
                counts[f"{role}_valid"]+=int(parsed["valid"])
                counts[f"{role}_click"]+=int(parsed["kind"]=="click")
                predictions.append({
                    "role":role,"image_sha":row["image_sha256"],
                    "app":row["app"],"task":row["task"],
                    "language":row["language"],"raw":raw,"parsed":parsed})
                if len(predictions)%8==0:
                    print("H163_NEW_APP_REAL_T4_CALLS",len(predictions),32,flush=True)
    if counts["before_calls"]!=16 or counts["after_calls"]!=16:
        raise ValueError("H163 inference cohort incomplete")
    doc={"schema":"hala.h163.new-original-gridstack-real-paired-gpu.v1",
         "status":"H163_COMPLETE_32_REAL_FREE_GPU_INFERENCES",
         "original_app_source_manifest_sha256":SOURCE_SHA,
         "original_app_source_zip_sha256":FROZEN_SOURCE_ZIP_SHA,
         "baseline_adapter_sha256":OLDER_MODEL_SHA,
         "trained_adapter_sha256":TRAINED_MODEL_SHA,
         "gpu_calls":32,"optimizer_steps":0,"h140_blind_calls":0,
         "paid_GPU":False,"flagship_Hala35B_unchanged":True,
         "native_browser_success_measured":False,
         "counts":dict(counts),"private_predictions":predictions}
    output.mkdir(parents=True,exist_ok=False)
    (output/"h163-new-original-app-predictions.json").write_text(
        json.dumps(doc,indent=2,sort_keys=True)+"\n","utf8")
    return {k:v for k,v in doc.items() if k!="private_predictions"}
