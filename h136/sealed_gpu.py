"""H136 owner-private ONE-TIME sealed24 pretrained-versus-trained GUI evaluation.

Adapted checkpoint SHA is filled ONLY after verified 64-step training.
This script must never mount DEV/TRAIN sources or execute optimizer steps.
"""
import json
from pathlib import Path
from h136.source import SCHEMA,check,sha
from h136.train_gpu import BASE_MODEL,BASE_REV,model_predict
from h136.prompt import role_prompt

SEALED_MANIFEST_SHA="a8d3ea230af8cab8d0902d1f4dfff2f6eded899a7d357bdbc2d4c4273a9e6244"
FROZEN_ADAPTER_SHA="__TRAINED_ADAPTER_SHA_BEFORE_SEALED__"
FORMAT="hala.h136.once-only-private-sealed24-grounding.v1"

def inputs():
    root=Path("/kaggle/input")
    originals=[]
    adapters=[]
    for file in root.rglob("manifest.json"):
        try:
            record=json.loads(file.read_text("utf8"))
        except (ValueError,OSError,TypeError,json.JSONDecodeError):
            continue
        if record.get("schema")!=SCHEMA:
            continue
        if record.get("split")!="sealed":
            raise ValueError("H136 sealed evaluation must never mount TRAIN or DEV originals")
        originals.append((file.parent,check(file.parent,"sealed")))
    for file in root.rglob("adapter_model.safetensors"):
        if sha(file.read_bytes())==FROZEN_ADAPTER_SHA and (
                file.parent/"adapter_config.json").is_file():
            adapters.append(file.parent)
    if len(originals)!=1 or len(adapters)!=1:
        raise ValueError("H136 exactly one frozen SEALED24 source and genuine trained adapter required")
    path,manifest=originals[0]
    if sha((path/"manifest.json").read_bytes())!=SEALED_MANIFEST_SHA:
        raise ValueError("H136 frozen independent heldout manifest changed")
    if len(manifest["cases"])!=24:
        raise ValueError("H136 sealed24 exact source count")
    return path,manifest,adapters[0]

def run(target):
    import torch,transformers
    from PIL import Image
    from peft import PeftModel
    from transformers import AutoProcessor,AutoModelForImageTextToText

    target=Path(target)
    if (target.exists() or not str(target).startswith("/kaggle/working/") or
        len(FROZEN_ADAPTER_SHA)!=64 or
        transformers.__version__!="5.17.0" or not torch.cuda.is_available() or
        not any("T4" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count()))):
        raise ValueError("H136 once-only nontraining private FREE Kaggle T4")
    source,source_doc,adapter_dir=inputs()
    processor=AutoProcessor.from_pretrained(BASE_MODEL,revision=BASE_REV,trust_remote_code=False)
    model=AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,revision=BASE_REV,torch_dtype=torch.float16,
        trust_remote_code=False,attn_implementation="sdpa").to("cuda:0").eval()
    predictions=[]
    results={}
    for arm in ("base_pretrained","h136_64step_adapter"):
        if arm=="h136_64step_adapter":
            model=PeftModel.from_pretrained(model,adapter_dir,is_trainable=False).to("cuda:0").eval()
        hits={"start":{"valid":0,"hits":0},"end":{"valid":0,"hits":0}}
        for row in source_doc["cases"]:
            with Image.open(source/row["image"]) as im:
                if im.format!="PNG" or im.size!=(960,720):
                    raise ValueError("H136 owner original 960x720 image required")
                frame=im.convert("RGB")
            for role in ("start","end"):
                query=role_prompt(row["instruction"],role)
                raw,point=model_predict(processor,model,torch,frame,query)
                b=row[role+"_bbox"]
                hit=point is not None and (
                    b["x"]<=point[0]<=b["x"]+b["width"] and
                    b["y"]<=point[1]<=b["y"]+b["height"])
                hits[role]["hits"]+=int(hit)
                hits[role]["valid"]+=int(point is not None)
                predictions.append({
                    "arm":arm,"case_id":row["id"],"image_sha256":row["image_sha256"],
                    "role":role,"raw":raw,"point":point,"hit":bool(hit)})
        results[arm]=hits
    if len(predictions)!=96:
        raise ValueError("H136 must finish 96 true frozen neural calls")
    evidence={
        "schema":FORMAT,"status":"H136_REAL_ONCE_ONLY_SEALED24_NEURAL_COMPLETE",
        "frozen_original_source_manifest_sha256":SEALED_MANIFEST_SHA,
        "frozen_adapter_sha256":FROZEN_ADAPTER_SHA,
        "base_model":BASE_MODEL,"base_revision":BASE_REV,
        "sealed_new_scene_count":24,"neural_inference_calls":96,
        "actual_optimizer_steps":0,"paid_compute_used":False,
        "train_or_dev_images_attached":False,
        "flagship_Hala_35B_weights_changed":False,
        "pretrained_and_trained":results,
        "real_browser_successes_pending_owner_replay":True,
        "model_skill_qualified":False}
    target.mkdir(parents=True,exist_ok=False)
    (target/"private-sealed24-raw.json").write_text(
        json.dumps(predictions,indent=2,sort_keys=True)+"\n","utf8")
    (target/"private-sealed24-evidence.json").write_text(
        json.dumps(evidence,indent=2,sort_keys=True)+"\n","utf8")
    return evidence
