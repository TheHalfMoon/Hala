"""H134 one-time PRIVATE frozen-adapter sealed16 inference, ZERO training.

The source sealed test was never attached to H134 optimizer session.
Single pretrained-vs-adapter comparison; no further model choice or tuning.
"""
import json
from pathlib import Path

from h134.source import SCHEMA,validate,sha
from h134.train_gpu import BASE_MODEL,BASE_REV,model_predict
from h132.grounder_contract import instruction_for

SEALED_MANIFEST_SHA="92cbd879ccb7e44a35c9045a589b5cb95e76af965c95d7c5dacb730c6040a02c"
TRAINED_ADAPTER_SHA="711f9dc76cde160818625c9d154c299ad1e098ae48bd554277634ef4b9defdad"

def admit():
    from pathlib import Path
    root=Path("/kaggle/input")
    sealed=[]
    adapter=[]
    for manifest in root.rglob("manifest.json"):
        try:
            doc=json.loads(manifest.read_text("utf8"))
        except (ValueError,TypeError,OSError,json.JSONDecodeError):
            continue
        if doc.get("schema")==SCHEMA:
            if doc.get("split")!="sealed_test":
                raise ValueError("H134 sealed evaluation cannot mount TRAIN or DEV dataset")
            sealed.append(manifest.parent)
    for file in root.rglob("adapter_model.safetensors"):
        if sha(file.read_bytes())==TRAINED_ADAPTER_SHA and (
               file.parent/"adapter_config.json").is_file():
            adapter.append(file.parent)
    if len(sealed)!=1 or len(adapter)!=1:
        raise ValueError("H134 sealed inference requires exactly one frozen sealed set and adapter")
    manifest=validate(sealed[0],expect="sealed_test")
    if sha((sealed[0]/"manifest.json").read_bytes())!=SEALED_MANIFEST_SHA:
        raise ValueError("H134 one-time heldout source manifest changed")
    return sealed[0],manifest,adapter[0]

def evaluate_model(processor,model,torch,source_dir,manifest):
    from PIL import Image
    stats={"start":{"hit":0,"valid":0},"end":{"hit":0,"valid":0}}
    details=[]
    pair=[]
    for row in manifest["cases"]:
        role_hits=[]
        with Image.open(source_dir/row["image"]) as im:
            image=im.convert("RGB")
        for role in ("start","end"):
            instruction=instruction_for(row["instruction"],role)
            raw,point=model_predict(processor,model,torch,image,instruction)
            bbox=row[role+"_bbox"]
            hit=point is not None and bbox["x"]<=point[0]<=bbox["x"]+bbox["width"] and (
                bbox["y"]<=point[1]<=bbox["y"]+bbox["height"])
            stats[role]["hit"]+=int(hit)
            stats[role]["valid"]+=int(point is not None)
            role_hits.append(bool(hit))
            details.append({
                "source_id":row["source_id"],"image_sha256":row["image_sha256"],
                "role":role,"raw":raw,"prediction":point,"hit":bool(hit),
            })
        pair.append({"source_id":row["source_id"],"both_points_hit":all(role_hits)})
    stats["both_targets_hit"] = sum(x["both_points_hit"] for x in pair)
    return stats,details

def run(output_path):
    import torch,transformers
    from transformers import AutoProcessor,AutoModelForImageTextToText
    from peft import PeftModel
    output_path=Path(output_path)
    if (output_path.exists() or not str(output_path).startswith("/kaggle/working/") or
        transformers.__version__!="5.17.0" or
        not torch.cuda.is_available() or
        not any("T4" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count()))):
        raise ValueError("H134 one-time sealed eval only PRIVATE FREE Kaggle T4")
    source_dir,manifest,adapter_dir=admit()
    proc=AutoProcessor.from_pretrained(BASE_MODEL,revision=BASE_REV,
                                        trust_remote_code=False)
    model=AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,revision=BASE_REV,torch_dtype=torch.float16,
        trust_remote_code=False,attn_implementation="sdpa").to("cuda:0").eval()
    before,before_raw=evaluate_model(proc,model,torch,source_dir,manifest)
    model=PeftModel.from_pretrained(model,adapter_dir,is_trainable=False).to("cuda:0").eval()
    after,after_raw=evaluate_model(proc,model,torch,source_dir,manifest)
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("H134 sealed eval must have all adapter/base weights frozen")
    result={
       "schema":"hala.h134.private-once-only-sealed16-frozen-adapter-comparison.v1",
       "status":"H134_REAL_PRIVATE_ONE_TIME_SEALED16_SCORED",
       "pretrained_model_id":BASE_MODEL,"pretrained_model_sha":BASE_REV,
       "adapter_sha256":TRAINED_ADAPTER_SHA,
       "sealed_source_sha256":SEALED_MANIFEST_SHA,
       "original_holdout_scenes":16,
       "original_train_dev_scenes_in_eval":0,
       "real_neural_calls":64,
       "actual_optimizer_steps":0,
       "pretrained":before,"adapter":after,
       "gpu_compute_paid":False,
       "main_Hala_flagship_weights_updated":False,
       "full_real_browser_task_success_measured":False,
       "human_H127_qualified":False,
       "deployment_approved":False,
    }
    output_path.mkdir(parents=True,exist_ok=False)
    (output_path/"private-sealed16-before.json").write_text(
        json.dumps(before_raw,sort_keys=True,indent=2)+"\n","utf8")
    (output_path/"private-sealed16-after.json").write_text(
        json.dumps(after_raw,sort_keys=True,indent=2)+"\n","utf8")
    (output_path/"private-sealed16-scores.json").write_text(
        json.dumps(result,sort_keys=True,indent=2)+"\n","utf8")
    return result
