"""H135 PRIVATE exact 2x2 image-resolution ablation on NEW pilot12.

Frozen pretrained vs H134-trained adapter; same scenes, each at 640x480
and native 960x720 with model processor smart-resize. No training/optimizer.
SEALED18 never attached to this pilot.
"""
import json
from pathlib import Path
import time

from h135.source import SCHEMA,check,sha
from h132.grounder_contract import (MODEL_ID,MODEL_SHA,instruction_for,native_point)
from h134.sealed_gpu import TRAINED_ADAPTER_SHA

SCHEMA_OUT="hala.h135.owner-private-12-scenes-2x2-resolution-pilot.v1"
MODES=("downsample_640","native_960")
MODELS=("pretrained","h134_adapter")

def locate_pilot():
    root=Path("/kaggle/input")
    matched=[]
    for manifest in root.rglob("manifest.json"):
        try:
            doc=json.loads(manifest.read_text("utf8"))
            if doc.get("schema")!=SCHEMA:continue
            if doc.get("split")!="pilot":
                raise ValueError("H135 SEALED data illegally mounted in pilot kernel")
            matched.append((manifest.parent,check(manifest.parent,"pilot")))
        except (ValueError,KeyError,OSError,json.JSONDecodeError) as exc:
            if "SEALED" in str(exc):raise
    if len(matched)!=1:raise ValueError("H135 exactly one pilot12 source required")
    adapters=[]
    for file in root.rglob("adapter_model.safetensors"):
        if sha(file.read_bytes())==TRAINED_ADAPTER_SHA and (file.parent/"adapter_config.json").is_file():
            adapters.append(file.parent)
    if len(adapters)!=1:raise ValueError("H135 exactly one frozen H134 adapter with correct bytes required")
    return matched[0],adapters[0]

def predict(model,processor,torch,image,instruction):
    messages=[{"role":"user","content":[{"type":"image"},{"type":"text","text":instruction}]}]
    prompt=processor.apply_chat_template(messages,tokenize=False,
                  add_generation_prompt=True,enable_thinking=False)
    batch=processor(text=[prompt],images=[image],return_tensors="pt").to("cuda:0")
    model.eval()
    with torch.inference_mode():
        out=model.generate(**batch,max_new_tokens=80,do_sample=False,max_time=45)
    torch.cuda.synchronize(0)
    raw=processor.batch_decode(out[:,batch["input_ids"].shape[1]:],
                               skip_special_tokens=True)[0].strip()
    if type(raw) is not str or len(raw)>300:
        raise ValueError("H135 frozen model raw native format oversize")
    return raw,native_point(raw,960,720)

def execute(output_dir):
    import torch,transformers
    from PIL import Image
    from peft import PeftModel
    from transformers import AutoProcessor,AutoModelForImageTextToText

    output_dir=Path(output_dir)
    if (output_dir.exists() or not str(output_dir).startswith("/kaggle/working/") or
        transformers.__version__!="5.17.0" or
        not torch.cuda.is_available() or
        not any("T4" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count()))):
        raise ValueError("H135 strict private FREE Kaggle T4 pinned model and no overwrite")
    (data_dir,manifest),adapter_dir=locate_pilot()
    torch.manual_seed(135)
    proc=AutoProcessor.from_pretrained(MODEL_ID,revision=MODEL_SHA,
                                        trust_remote_code=False)
    model=AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,torch_dtype=torch.float16,
        trust_remote_code=False,attn_implementation="sdpa").to("cuda:0").eval()
    if "qwen3_5" not in model.config.model_type.lower():
        raise ValueError("H135 fixed pretrained Qwen3.5 model architecture drift")
    start=time.monotonic()
    raw_evidence=[]
    summary={}
    for arm in MODELS:
        if arm=="h134_adapter":
            model=PeftModel.from_pretrained(model,adapter_dir,is_trainable=False).to("cuda:0").eval()
        for mode in MODES:
            metrics={"start":{"valid":0,"hits":0},"end":{"valid":0,"hits":0}}
            for row in manifest["cases"]:
                with Image.open(data_dir/row["image"]) as im:
                    if im.format!="PNG" or im.size!=(960,720):
                        raise ValueError("H135 original owner screenshot dimensions invalid")
                    image=im.convert("RGB")
                # Preserve both visual inputs on same original 960x720 screenshot.
                image_in=image.resize((640,480)) if mode=="downsample_640" else image
                for role in ("start","end"):
                    query=instruction_for(row["instruction"],role)
                    raw,point=predict(model,proc,torch,image_in,query)
                    b=row[role+"_bbox"]
                    hit=(point is not None and b["x"]<=point[0]<=b["x"]+b["width"] and
                         b["y"]<=point[1]<=b["y"]+b["height"])
                    metrics[role]["valid"]+=int(point is not None)
                    metrics[role]["hits"]+=int(hit)
                    raw_evidence.append({
                       "arm":arm,"resolution":mode,
                       "id":row["id"],"image_sha256":row["image_sha256"],
                       "role":role,"raw":raw,"point":point,"hit":bool(hit)})
            summary[arm+"__"+mode]=metrics
    if len(raw_evidence)!=96:
        raise ValueError("H135 required all 96 real frozen pilot model calls")
    data={
        "schema":SCHEMA_OUT,"status":"H135_REAL_PRIVATE_PILOT12_RESOLUTION_PAIRED",
        "pilot_manifest_sha256":sha((data_dir/"manifest.json").read_bytes()),
        "base_id":MODEL_ID,"base_revision":MODEL_SHA,
        "adapter_sha256":TRAINED_ADAPTER_SHA,
        "source_scenes":12,"actual_neural_calls":len(raw_evidence),
        "actual_optimizer_steps":0,"paid_cloud":False,
        "no_sealed_holdout_attached":True,
        "modes":summary,
        "generated_from":"original-h135-workstation-fixture",
        "elapsed_seconds":round(time.monotonic()-start,2),
        "conclusions_are_generalized":False
    }
    output_dir.mkdir(parents=True,exist_ok=False)
    points=output_dir/"private-raw-responses.json"
    points.write_text(json.dumps(raw_evidence,sort_keys=True,indent=2)+"\n","utf8")
    score=output_dir/"private-pilot-score.json"
    score.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n","utf8")
    return {"status":data["status"],"real_neural_calls":96,
       "private_raw_sha256":sha(points.read_bytes()),
       "private_score_sha256":sha(score.read_bytes()),
       "full_source_manifest_sha256":data["pilot_manifest_sha256"],
       "actual_optimizer_steps":0,"paid_cloud":False}
