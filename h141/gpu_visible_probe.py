"""H141 owner-private FREE T4 original visual source+column inference ONLY.

Probe targets are VISIBLE on the frozen 960x720 original React screenshot,
not the old invisible DOM drop-slot. The model sees zero owner DOM gold.
This is a new source cohort, not H140 SEALED one-shot BLIND.
"""
import hashlib
import io
import json
import math
from pathlib import Path
from zipfile import ZipFile

from h132.grounder_contract import MODEL_ID,MODEL_SHA,PROMPT_TEMPLATE
from h139.prompt import role_prompt
from h141.qualify_neural_probe import PUBLIC_SCHEMA,MODEL_FIELDS

EVAL_INPUT_SHA="01aeb6d2a95d906e7b7c80087cdf8183a5aeb128e8c83bb8db2b6083061802ba"
EVAL_MANIFEST_SHA="33bfa47da0f3d30e995d19dafdaa16f457bc833ce168bf05f1114b523ff2d438"
ADAPTER_ZIP_SHA="568550c7bf2165821e5898339174e2dfbccd40c117e9a6cda6c3d9879a1042c0"
V2_ADAPTER_SHA="9595e463f2b621e308beaaefffd9ade8c38210bdc59550cb87d8e5f66d2a4bd0"
SOURCE_SHA="fa047f7a32afb57750ed5246456dc282ea7befcd"
ALLOWED_ADAPTER_FILES={
    "README.md","adapter_config.json","adapter_model.safetensors",
    "chat_template.jinja","processor_config.json","tokenizer.json",
    "tokenizer_config.json"
}
STATUS="H141_PRIVATE_VISIBLE_MODEL_2B_PREDICTIONS_NO_GOLD"
SCHEMA="hala.h141.private-original-visible-semantic-target-neural-probe.v1"

def sha(value):
    return hashlib.sha256(value).hexdigest()

def input_admission(path):
    """All model inputs must be allowlisted and EXACTLY the immutable new 16."""
    path=Path(path)
    if sha(path.read_bytes())!=EVAL_INPUT_SHA:
        raise ValueError("H141 frozen model-facing original GUI ZIP SHA drift")
    with ZipFile(path) as z:
        raw=z.read("manifest.json")
        if sha(raw)!=EVAL_MANIFEST_SHA:
            raise ValueError("H141 owner redacted model manifest changed")
        manifest=json.loads(raw)
        rows=manifest.get("cases")
        if (manifest.get("schema")!=PUBLIC_SCHEMA or
            manifest.get("status")!="EVAL_ONLY" or
            manifest.get("source_app_git_sha")!=SOURCE_SHA or
            manifest.get("model_action_roles")!=["start","visible_column_heading"] or
            manifest.get("training_eligible") is not False or
            manifest.get("model_evaluation_eligible") is not True or
            not isinstance(rows,list) or len(rows)!=16):
            raise ValueError("H141 only NEW original photo visual heading probe")
        permitted={"manifest.json"}|{r.get("image") for r in rows}
        if len(permitted)!=17 or set(z.namelist())!=permitted:
            raise ValueError("H141 original model input has unauthorized files")
        languages={"en":0,"ar":0}
        from PIL import Image
        for r in rows:
            if (set(r)!=MODEL_FIELDS or
                r["app_git_sha"]!=SOURCE_SHA or
                r["language"] not in languages or
                not isinstance(r["instruction"],str) or
                "point_2d" in r["instruction"] or
                not isinstance(r["image"],str) or
                not r["image"].endswith(".png") or
                "/" in r["image"] or "\\" in r["image"] or ".." in r["image"]):
                raise ValueError("H141 original visual task contains hidden gold or unsafe path")
            languages[r["language"]]+=1
            data=z.read(r["image"])
            if sha(data)!=r["image_sha256"]:
                raise ValueError("H141 original model-facing screenshot SHA drift")
            with Image.open(io.BytesIO(data)) as pic:
                if pic.format!="PNG" or pic.size!=(960,720):
                    raise ValueError("H141 unchanged native browser screenshot required")
                pic.load()
        if languages!={"en":8,"ar":8}:
            raise ValueError("H141 source original bilingual 16-case audit failed")
    return manifest

def adapter_admission(path,target):
    """Peft source must be exact prior qualified private V2 LoRA weights."""
    path,target=Path(path),Path(target)
    if sha(path.read_bytes())!=ADAPTER_ZIP_SHA or target.exists():
        raise ValueError("H141 source research adapter changed or unsafe overwrite")
    with ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(ALLOWED_ADAPTER_FILES) or set(names)!=ALLOWED_ADAPTER_FILES:
            raise ValueError("H141 LoRA adapter ZIP files are unexpected")
        if sha(z.read("adapter_model.safetensors"))!=V2_ADAPTER_SHA:
            raise ValueError("H141 only original V2 research adapter SHA qualified")
        cfg=json.loads(z.read("adapter_config.json"))
        if cfg.get("r")!=8:
            raise ValueError("H141 expected language-only original rank-8 adapter")
        target.mkdir()
        for name in sorted(ALLOWED_ADAPTER_FILES):
            (target/name).write_bytes(z.read(name))
    return target

def visible_prompt(instruction,role):
    if role=="start":
        return role_prompt(instruction,"start")
    if role=="visible_column_heading":
        # This target is VISUALLY OBSERVABLE in the first screenshot,
        # unlike the previously hidden React invisible drop-zone slot.
        return PROMPT_TEMPLATE.format(instruction=(
            "Locate the CENTER of the visibly labeled DESTINATION COLUMN HEADER "
            "(the heading such as 'Doing' or 'In Progress'), not any invisible "
            "drag-and-drop slot, handle or card. Task: "+instruction))
    raise ValueError("H141 only screenshot-visible START or semantic column heading")

def run(source_zip,adapter_zip,out_dir):
    import torch,transformers
    from PIL import Image
    from transformers import AutoProcessor,AutoModelForImageTextToText
    from peft import PeftModel
    from h139.train_gpu import model_predict
    out_dir=Path(out_dir)
    if (out_dir.exists() or
        not str(out_dir).startswith("/kaggle/working/") or
        transformers.__version__!="5.17.0" or
        not torch.cuda.is_available() or
        not any("T4" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count()))):
        raise ValueError("H141 private free NVIDIA T4 inference only, never training")
    source=input_admission(source_zip)
    adapter=adapter_admission(adapter_zip,
                              Path("/kaggle/working/h141-private-source-v2-adapter"))
    processor=AutoProcessor.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,trust_remote_code=False)
    base=AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,torch_dtype=torch.float16,
        trust_remote_code=False,attn_implementation="sdpa").to("cuda:0")
    model=PeftModel.from_pretrained(base,str(adapter))
    model.eval()
    responses=[]
    total=0
    with ZipFile(source_zip) as z:
        for row in source["cases"]:
            with Image.open(io.BytesIO(z.read(row["image"]))) as file:
                picture=file.convert("RGB")
            response={"id":row["id"],"app":row["app"],
                      "language":row["language"]}
            for role in ("start","visible_column_heading"):
                prompt=visible_prompt(row["instruction"],role)
                raw,point=model_predict(processor,model,torch,picture,prompt)
                response[role+"_point"]=list(point) if point is not None else None
                response[role+"_output_valid"]=point is not None
                total+=1
                if total%8==0:
                    print("H141_ACTUAL_MODEL_VISUAL_GUI_INFERENCE",total,
                          "of",32,flush=True)
            responses.append(response)
    if len(responses)!=16 or total!=32:
        raise ValueError("H141 exact one original source/heading neural pair per frame")
    result={
        "schema":SCHEMA,
        "status":STATUS,
        "original_source_role":"new_h141_neural_probe_not_h140_blind",
        "original_source_git_sha":SOURCE_SHA,
        "input_zip_sha256":EVAL_INPUT_SHA,
        "input_manifest_sha256":EVAL_MANIFEST_SHA,
        "adapter_sha256":V2_ADAPTER_SHA,
        "base_model_sha":MODEL_SHA,
        "model_calls":total,
        "real_optimizer_steps":0,
        "gold_boxes_seen_by_model":False,
        "training_eligible":False,
        "sealed_h140_blind_calls":0,
        "flagship_hala_35B_updated":False,
        "predictions":responses,
    }
    out_dir.mkdir(parents=True,exist_ok=False)
    (out_dir/"h141-private-visible-model-predictions.json").write_text(
        json.dumps(result,sort_keys=True,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    return {"status":STATUS,
            "original_cases":16,"original_neural_calls":32,
            "real_optimizer_steps":0,
            "gold_boxes_seen_by_model":False,
            "sealed_h140_blind_calls":0}
