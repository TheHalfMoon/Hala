"""PRIVATE one-shot H140 Qwen3.5-2B GUI point inference — NEVER load gold.

Not a training kernel. 32 original full-resolution unseen app screenshots
and owner instructions only. The private owner Windows evaluator will run
the predicted gestures on the pixel-identical original browsers afterward.
"""
import io
import json
import hashlib
from pathlib import Path
from zipfile import ZipFile

from h132.grounder_contract import MODEL_ID,MODEL_SHA
from h139.prompt import role_prompt
from h140.export_blind_redacted import OWNER_BLIND_SHA,SCHEMA as INPUT_SCHEMA,ALLOWED_FIELDS
from h140.train_gpu_v3 import TRAIN_MANIFEST_SHA,DEV_MANIFEST_SHA

PREDICTIONS_SCHEMA="hala.h140.private-once-original-two-app-model-predictions.v1"
INPUT_MANIFEST_SHA="ecde02bd4658a6d7a4302e522064d5ec42066837ee30dc7d93a5dd9aa90b7ef9"
ZIP_SHA="5d12b6435dc61a85bbb261a8211d996fd377f72c49a67bfeba9accd957df4f3d"
ADMISSION_SCHEMA="hala.h140.owner-v3-dev-gate-private-authorization.v1"
SOURCE_SHA={
    "ood-vanilla-complete":"f89d065e2cf16f91918bf1a72ae9ac132150197e",
    "ood-second-drag":"a68c1d9d6281a793835ceaa5e4f620068d95200a",
}

def sha(data):
    return hashlib.sha256(data).hexdigest()

def admit_redacted_source(path):
    """Check every PNG independently, with no bounding boxes or DOM gold."""
    from PIL import Image
    path=Path(path)
    if sha(path.read_bytes())!=ZIP_SHA:
        raise ValueError("H140 redacted sealed inference-only ZIP changed")
    with ZipFile(path) as z:
        entries=z.namelist()
        raw=z.read("manifest.json")
        if sha(raw)!=INPUT_MANIFEST_SHA:
            raise ValueError("H140 unseen original model-input manifest SHA drift")
        doc=json.loads(raw)
        rows=doc.get("cases")
        if (doc.get("schema")!=INPUT_SCHEMA or
            doc.get("owner_original_manifest_sha256")!=OWNER_BLIND_SHA or
            doc.get("training_eligible") is not False or
            doc.get("source_role")!="blind_test" or
            doc.get("source_apps")!=SOURCE_SHA or
            not isinstance(rows,list) or len(rows)!=32):
            raise ValueError("H140 only genuine redacted private independent test source")
        allowed_files={"manifest.json"}|{r.get("image") for r in rows}
        if len(allowed_files)!=33 or set(entries)!=allowed_files or len(entries)!=33:
            raise ValueError("H140 unseen original test ZIP must contain EXACT 32 PNGs")
        for row in rows:
            if set(row)!=ALLOWED_FIELDS:
                raise ValueError("H140 source must not contain model-facing oracle labels")
            if row["app"] not in SOURCE_SHA or row["app_git_sha"]!=SOURCE_SHA[row["app"]]:
                raise ValueError("H140 unseen original application source provenance drift")
            image=row["image"]
            if not isinstance(image,str) or "/" in image or "\\" in image or ".." in image:
                raise ValueError("H140 original screenshot path must be safe basename")
            binary=z.read(image)
            if sha(binary)!=row["image_sha256"]:
                raise ValueError("H140 original native screenshot data corrupted")
            with Image.open(io.BytesIO(binary)) as picture:
                if picture.format!="PNG" or picture.size!=(960,720):
                    raise ValueError("H140 unchanged native browser input PNG required")
                picture.load()
    return doc

def admit_v3_checkpoint(adapter_folder,authorization_file):
    """Only preregistered successful DEV gate unlocks BLIND inference."""
    doc=json.loads(Path(authorization_file).read_text("utf8"))
    if (doc.get("schema")!=ADMISSION_SCHEMA or
        doc.get("status")!="ELIGIBLE_FOR_ONE_FINAL_OFFLINE_BLIND" or
        doc.get("train_manifest_sha256")!=TRAIN_MANIFEST_SHA or
        doc.get("dev_manifest_sha256")!=DEV_MANIFEST_SHA or
        doc.get("owner_original_manifest_sha256")!=OWNER_BLIND_SHA or
        doc.get("private_only") is not True or
        doc.get("neural_blind_calls_before_authorization")!=0):
        raise ValueError("H140 V3 DEV pass certificate absent or invalid; BLIND stays closed")
    path=Path(adapter_folder)/"adapter_model.safetensors"
    if not path.is_file():
        raise ValueError("H140 exact V3 LoRA weights missing")
    actual=sha(path.read_bytes())
    if actual!=doc.get("adapter_sha256") or actual=="":
        raise ValueError("H140 adapter differs from approved V3 owner checkpoint")
    return actual

def run(blind_zip,adapter_directory,dev_authorization,output):
    """64 true vision inference calls, ZERO gold access and ZERO SGD."""
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor,AutoModelForImageTextToText
    from peft import PeftModel
    from h139.train_gpu import model_predict
    output=Path(output)
    if (output.exists() or
        not str(output).startswith("/kaggle/working/") or
        transformers.__version__!="5.17.0" or
        not torch.cuda.is_available() or
        not any("T4" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count()))):
        raise ValueError("H140 one-shot original app inference only on owner FREE private T4")
    checkpoint_sha=admit_v3_checkpoint(adapter_directory,dev_authorization)
    source=admit_redacted_source(blind_zip)
    processor=AutoProcessor.from_pretrained(MODEL_ID,revision=MODEL_SHA,
                                            trust_remote_code=False)
    base=AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,torch_dtype=torch.float16,
        trust_remote_code=False,attn_implementation="sdpa").to("cuda:0")
    model=PeftModel.from_pretrained(base,str(adapter_directory))
    model.eval()
    responses=[]
    total_calls=0
    with ZipFile(blind_zip) as zipfile:
        for row in source["cases"]:
            with Image.open(io.BytesIO(zipfile.read(row["image"]))) as original:
                image=original.convert("RGB")
            answer={"id":row["id"],"app":row["app"],"language":row["language"]}
            for role in ("start","end"):
                prompt=role_prompt(row["instruction"],role)
                _,point=model_predict(processor,model,torch,image,prompt)
                answer[role+"_point"]=list(point) if point is not None else None
                total_calls+=1
            responses.append(answer)
    if total_calls!=64 or len(responses)!=32:
        raise ValueError("H140 exactly 32 two-role original image inferences required")
    result={
        "schema":PREDICTIONS_SCHEMA,
        "owner_original_manifest_sha256":OWNER_BLIND_SHA,
        "redacted_manifest_sha256":INPUT_MANIFEST_SHA,
        "adapter_sha256":checkpoint_sha,
        "model_evaluation_role":"one_shot_blind_inference_only",
        "trained_on_blind":False,
        "model_calls":64,
        "flagship_Hala_35B_updated":False,
        "paid_compute_used":False,
        "predictions":responses,
    }
    output.mkdir(parents=True,exist_ok=False)
    (output/"private-original-unseen-predictions.json").write_text(
        json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+"\n","utf8")
    return {"status":"PRIVATE_BLIND_PREDICTIONS_ONLY_NO_GOLD_OR_SCORE",
            "case_count":32,
            "neural_calls":64,
            "checkpoint_sha256":checkpoint_sha,
            "gold_oracle_available_to_model":False,
            "owner_browser_e2e_still_pending":True}
