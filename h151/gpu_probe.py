"""H151 true frozen NEW shifted-layout Qwen3.5-2B inference; NO SGD.

Private owner Kaggle free Nvidia T4. Only SHA-frozen H151 redacted media
and independently SHA-validated H142 research LoRA. Exactly 32 model calls,
no owner DOM gold, no H140 sealed OOD BLIND, no Hala flagship updates.
"""
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from h132.grounder_contract import MODEL_ID,MODEL_SHA
from h139.train_gpu import model_predict
from h141.gpu_visible_probe import visible_prompt,ALLOWED_ADAPTER_FILES
from h151.freeze_eval import EXPORT_SCHEMA,MODEL_FIELDS
from h151.collect_shifted_source import APP_SHA,TRANSFORMS

IMAGE_ARCHIVE_SHA="e1299650b96a4743755eec6504dce5a99e5f4c4901b8a016a86331f2e8c9cf82"
IMAGE_MANIFEST_SHA="a92ea449f902cff11538dac06ee7c301837aa379010f5b28cad661402a34bf4e"
OWNER_SOURCE_MANIFEST_SHA="d2a04db9da8d898eb51c22e015df01baa1e20220e016ab8bb20a18814edf0f75"
ADAPTER_ZIP_SHA="58ec4f6e026b5fc5c8be9a74433d3ca365f1c93680d84b2d148a86b8725e2b8c"
ADAPTER_MODEL_SHA="992fa18fead6724329dbe7ec7b95c8acb0a56229c1fa5b56719e5f4829130a68"
SCHEMA="hala.h151.private-shifted-original-2b-model-predictions.v1"

def sha(raw):return hashlib.sha256(raw).hexdigest()
def storage_names(path):
    path=Path(path)
    if path.is_dir():return {p.name for p in path.iterdir() if p.is_file()}
    with ZipFile(path) as z:return set(z.namelist())
def stored(path,name):
    if (not isinstance(name,str) or "/" in name or "\\" in name or
        ".." in name):
        raise ValueError("H151 flat owner original media filenames only")
    path=Path(path)
    if path.is_dir():return (path/name).read_bytes()
    with ZipFile(path) as z:return z.read(name)
def admit_images(path):
    from PIL import Image
    path=Path(path)
    if path.is_file() and sha(path.read_bytes())!=IMAGE_ARCHIVE_SHA:
        raise ValueError("H151 original independent image archive changed")
    raw=stored(path,"manifest.json")
    if sha(raw)!=IMAGE_MANIFEST_SHA:
        raise ValueError("H151 model-facing source manifest SHA mismatch")
    doc=json.loads(raw)
    rows=doc.get("cases")
    if (doc.get("schema")!=EXPORT_SCHEMA or
        doc.get("role")!="EVAL_ONLY" or
        doc.get("source_app_git_sha")!=APP_SHA or
        doc.get("source_manifest_sha256")!=OWNER_SOURCE_MANIFEST_SHA or
        doc.get("training_eligible") is not False or
        doc.get("model_action_roles")!=["start","visible_column_heading"] or
        not isinstance(rows,list) or len(rows)!=16):
        raise ValueError("H151 only owner-qualified new shifted original evaluation")
    allowable={"manifest.json"}|{r.get("image") for r in rows}
    if len(allowable)!=17 or storage_names(path)!=allowable:
        raise ValueError("H151 model source has extra files or missing originals")
    strata=Counter();layouts=Counter();images=set()
    for r in rows:
        if (set(r)!=set(MODEL_FIELDS) or r["app_git_sha"]!=APP_SHA or
            r["layout"] not in TRANSFORMS or r["language"] not in ("ar","en") or
            r["destination_status"] not in ("doing","done") or
            r["image_sha256"] in images):
            raise ValueError("H151 independent model input source unsupported")
        rawimage=stored(path,r["image"])
        if sha(rawimage)!=r["image_sha256"]:
            raise ValueError("H151 original screenshot bytes mismatched SHA")
        with Image.open(io.BytesIO(rawimage)) as pic:
            if pic.format!="PNG" or pic.size!=(960,720):
                raise ValueError("H151 original browser pixel dimensions changed")
            pic.load()
        images.add(r["image_sha256"])
        strata[(r["language"],r["destination_status"])]+=1
        layouts[r["layout"]]+=1
    if (strata!={(l,d):4 for l in ("ar","en") for d in ("doing","done")}
        or (set(layouts)!=set(TRANSFORMS) or sum(layouts.values())!=16)):
        raise ValueError("H151 shifted holdout role and visual layouts drifted")
    return doc
def admit_adapter(source,output):
    source,output=Path(source),Path(output)
    if output.exists() or (source.is_file() and sha(source.read_bytes())!=ADAPTER_ZIP_SHA):
        raise ValueError("H151 trained H142 adapter archive drift or overwrite")
    if storage_names(source)!=ALLOWED_ADAPTER_FILES:
        raise ValueError("H151 only original rank8 adapter files")
    if sha(stored(source,"adapter_model.safetensors"))!=ADAPTER_MODEL_SHA:
        raise ValueError("H151 only trained H142 adapter SHA allowed")
    config=json.loads(stored(source,"adapter_config.json"))
    if config.get("r")!=8:
        raise ValueError("H151 expected genuine Qwen3.5 rank-8 LoRA")
    output.mkdir()
    for name in sorted(ALLOWED_ADAPTER_FILES):
        (output/name).write_bytes(stored(source,name))
    return output
def run(input_media,adapter_media,dest):
    import torch,transformers
    from PIL import Image
    from transformers import AutoProcessor,AutoModelForImageTextToText
    from peft import PeftModel
    dest=Path(dest)
    if (dest.exists() or
        not str(dest).startswith("/kaggle/working/") or
        transformers.__version__!="5.17.0" or
        not torch.cuda.is_available() or
        not any("T4" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count()))):
        raise ValueError("H151 only owner private free T4 and no existing result")
    cohort=admit_images(input_media)
    weights=admit_adapter(adapter_media,Path("/kaggle/working/h151-private-research-adapter"))
    processor=AutoProcessor.from_pretrained(MODEL_ID,revision=MODEL_SHA,
                                            trust_remote_code=False)
    base=AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,torch_dtype=torch.float16,
        trust_remote_code=False,attn_implementation="sdpa").to("cuda:0")
    model=PeftModel.from_pretrained(base,str(weights))
    model.eval()
    responses=[]
    total=0
    for row in cohort["cases"]:
        with Image.open(io.BytesIO(stored(input_media,row["image"]))) as inp:
            picture=inp.convert("RGB")
        result={"id":row["id"],"app":row["app"],
                "language":row["language"],
                "destination_status":row["destination_status"],
                "layout":row["layout"]}
        for role in ("start","visible_column_heading"):
            prompt=visible_prompt(row["instruction"],role)
            raw,point=model_predict(processor,model,torch,picture,prompt)
            result[role+"_point"]=list(point) if point is not None else None
            result[role+"_output_valid"]=point is not None
            total+=1
            if total%8==0:
                print("H151_REAL_FROZEN_SHIFTED_ORIGINAL_GPU_CALL",total,32,flush=True)
        responses.append(result)
    if total!=32 or len(responses)!=16:
        raise ValueError("H151 independent new screenshot paired inference count mismatch")
    output={"schema":SCHEMA,
            "status":"COMPLETE_REAL_MODEL_SHIFTED_LAYOUT_32_INFERENCE_CALLS",
            "original_new_source_manifest_sha256":OWNER_SOURCE_MANIFEST_SHA,
            "input_manifest_sha256":IMAGE_MANIFEST_SHA,
            "model_checkpoint_sha256":ADAPTER_MODEL_SHA,
            "gpu_calls":32,"optimizer_steps":0,
            "owner_gold_boxes_available_to_model":False,
            "h140_sealed_blind_calls":0,
            "flagship_Hala_35B_updated":False,
            "predictions":responses}
    dest.mkdir(parents=True,exist_ok=False)
    (dest/"private-h151-original-shifted-neural-predictions.json").write_text(
        json.dumps(output,sort_keys=True,indent=2,ensure_ascii=False)+"\n","utf8")
    return {"status":output["status"],"real_gpu_inference_calls":32,
            "new_original_layouts":4,"training_steps":0,
            "gold_seen":False,"h140_blind_calls":0}
