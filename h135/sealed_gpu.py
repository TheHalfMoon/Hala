"""H135 once-only SEPARATE FREE T4 sealed18 inference, after pilot choice.

No training, no feedback tuning, original owner sealed screenshots, frozen
pretrained and H134 adapter, exactly ONE preselected image preprocessing mode.
"""
import json
from pathlib import Path
from h135.source import SCHEMA,check,sha
from h135.ablate_gpu import predict
from h132.grounder_contract import MODEL_ID,MODEL_SHA,instruction_for
from h134.sealed_gpu import TRAINED_ADAPTER_SHA

SEALED_MANIFEST_SHA="5dacbcd0dbaaca0f976d0cc342328da9f22a0adf12f3f67401fe88489914ab91"
SELECTED_MODE="native_960"
SELECTED_DECISION_SHA="2574aa16e2a4d112793859852fbe5027f9cb15c2699cdb1712946762dc5857fa"
FORMAT="hala.h135.sealed18-once-frozen-resolution-v1"

def private_inputs():
    root=Path("/kaggle/input")
    sources=[];adapters=[]
    for file in root.rglob("manifest.json"):
        try: m=json.loads(file.read_text("utf8"))
        except (ValueError,TypeError,OSError,json.JSONDecodeError):continue
        if m.get("schema")!=SCHEMA:continue
        if m.get("split")!="sealed":
            raise ValueError("H135 sealed evaluator may not read pilot source")
        sources.append((file.parent,check(file.parent,"sealed")))
    for f in root.rglob("adapter_model.safetensors"):
        if sha(f.read_bytes())==TRAINED_ADAPTER_SHA and (f.parent/"adapter_config.json").is_file():
            adapters.append(f.parent)
    if len(sources)!=1 or len(adapters)!=1:
        raise ValueError("H135 require one owner-only sealed source and immutable trained adapter")
    dirname,manifest=sources[0]
    if sha((dirname/"manifest.json").read_bytes())!=SEALED_MANIFEST_SHA:
        raise ValueError("H135 sealed source preregistered SHA changed")
    return dirname,manifest,adapters[0]

def run(folder):
    import torch,transformers
    from PIL import Image
    from transformers import AutoProcessor,AutoModelForImageTextToText
    from peft import PeftModel

    folder=Path(folder)
    if (folder.exists() or not str(folder).startswith("/kaggle/working/") or
        SELECTED_MODE not in ("native_960","downsample_640") or
        len(SELECTED_DECISION_SHA)!=64 or
        transformers.__version__!="5.17.0" or
        not torch.cuda.is_available() or
        not any("T4" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count()))):
        raise ValueError("H135 one-time free T4 and immutable resolution choice required")
    src,manifest,adapter_dir=private_inputs()
    processor=AutoProcessor.from_pretrained(MODEL_ID,revision=MODEL_SHA,
                                              trust_remote_code=False)
    model=AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,torch_dtype=torch.float16,
        trust_remote_code=False,attn_implementation="sdpa").to("cuda:0").eval()
    evidence=[];scores={}
    for arm in ("pretrained","h134_adapter"):
        if arm=="h134_adapter":
            model=PeftModel.from_pretrained(model,adapter_dir,is_trainable=False).to("cuda:0").eval()
        counts={"start":{"hits":0,"valid":0},"end":{"hits":0,"valid":0}}
        for case in manifest["cases"]:
            with Image.open(src/case["image"]) as im:
                if im.format!="PNG" or im.size!=(960,720):
                    raise ValueError("H135 genuine original Chromium PNG required")
                rgb=im.convert("RGB")
            image=rgb if SELECTED_MODE=="native_960" else rgb.resize((640,480))
            for role in ("start","end"):
                prompt=instruction_for(case["instruction"],role)
                raw,point=predict(model,processor,torch,image,prompt)
                b=case[role+"_bbox"]
                hit=point is not None and b["x"]<=point[0]<=b["x"]+b["width"] and (
                    b["y"]<=point[1]<=b["y"]+b["height"])
                counts[role]["valid"]+=int(point is not None)
                counts[role]["hits"]+=int(hit)
                evidence.append({
                    "arm":arm,"case_id":case["id"],"image_sha256":case["image_sha256"],
                    "role":role,"raw":raw,"point":point,"hit":bool(hit)})
        scores[arm]=counts
    if len(evidence)!=72:raise ValueError("H135 all sealed model calls required before completion")
    result={
      "schema":FORMAT,"status":"H135_PRIVATE_ONCE_SEALED18_72_NEURAL_SCORED",
      "selected_resolution_mode":SELECTED_MODE,
      "frozen_pilot_decision_sha256":SELECTED_DECISION_SHA,
      "sealed_manifest_sha256":SEALED_MANIFEST_SHA,
      "frozen_base_id":MODEL_ID,"frozen_base_revision":MODEL_SHA,
      "immutable_adapter_sha256":TRAINED_ADAPTER_SHA,
      "sealed_cases":18,"actual_neural_calls":72,
      "optimizer_steps":0,"paid_compute":False,
      "no_pilot_data_attached":True,
      "metrics":scores,"real_browser_task_success_not_scored_yet":True,
      "generalized_skill_proven":False,
    }
    folder.mkdir(parents=True,exist_ok=False)
    (folder/"private-sealed-model-points.json").write_text(
       json.dumps(evidence,sort_keys=True,indent=2)+"\n","utf8")
    (folder/"private-sealed-metrics.json").write_text(
       json.dumps(result,sort_keys=True,indent=2)+"\n","utf8")
    return result
