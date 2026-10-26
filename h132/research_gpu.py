"""Private free T4 actual independent 2B GUI-grounder, no optimizer steps."""
import json
import time
from pathlib import Path

from h131.contract import verify_manifest,sha
from h132.grounder_contract import (
    MODEL_ID,MODEL_SHA,FORMAT,instruction_for,native_point
)

def find_source():
    root=Path("/kaggle/input")
    admitted=[]
    for file in root.rglob("manifest.json"):
        try:
            manifest=verify_manifest(file.parent,expect_phase="initial")
        except (ValueError,TypeError,KeyError,FileNotFoundError,json.JSONDecodeError):
            continue
        admitted.append((file.parent,manifest))
    if len(admitted)!=1:
        raise ValueError("H132 exactly one private original Chromium dataset is required")
    return admitted[0]

def run(output_dir):
    import torch,transformers
    from PIL import Image
    from transformers import AutoProcessor,AutoModelForImageTextToText

    output_dir=Path(output_dir)
    if (output_dir.exists() or
        not str(output_dir).startswith("/kaggle/working/") or
        not torch.cuda.is_available() or
        not any("T4" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())) or
        transformers.__version__!="5.17.0"):
        raise ValueError("H132 requires fresh private free T4 with pinned Transformers 5.17.0")
    directory,manifest=find_source()
    torch.manual_seed(132)
    proc=AutoProcessor.from_pretrained(MODEL_ID,revision=MODEL_SHA,
                                       trust_remote_code=False)
    model=AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,torch_dtype=torch.float16,
        trust_remote_code=False,attn_implementation="sdpa").to("cuda:0").eval()
    if "qwen3_5" not in model.config.model_type.lower():
        raise ValueError("H132 specialist expected pinned Qwen3.5 model, no alternate")
    results=[]
    count=0
    start=time.monotonic()
    for case in manifest["cases"]:
        with Image.open(directory/case["image"]) as img:
            if img.format!="PNG" or img.size!=(960,720):
                raise ValueError("H132 owner Chromium screenshot is not native 960x720 PNG")
            rgb=img.convert("RGB")
        record={"task":case["task"],"image_sha256":case["image_sha256"]}
        for role in ("start","end"):
            instruction=instruction_for(case["instruction"],role)
            messages=[{"role":"user","content":[
                {"type":"image"},{"type":"text","text":instruction}]}]
            rendered=proc.apply_chat_template(
                messages,tokenize=False,add_generation_prompt=True,
                enable_thinking=False)
            encoded=proc(text=[rendered],images=[rgb],return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                predicted=model.generate(**encoded,max_new_tokens=80,do_sample=False,
                                         max_time=45)
            torch.cuda.synchronize(0)
            raw=proc.batch_decode(predicted[:,encoded.input_ids.shape[1]:],
                                  skip_special_tokens=True)[0].strip()
            record["raw_"+role]=raw
            record[role]=native_point(raw)
            count+=1
            del encoded,predicted
        results.append(record)
    if count!=8 or len(results)!=4:
        raise ValueError("H132 exact 8 pretrained GUI point inference calls required")
    manifest_sha=sha(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode("utf8"))
    evidence={
        "schema":FORMAT,"model_id":MODEL_ID,"model_sha":MODEL_SHA,
        "source_manifest_sha256":manifest_sha,"cases":results,
        "neural_calls":count,"training_steps":0,"browser_oracle_used":False,
        "owner_private":True,"model_skill_qualified":False,
        "elapsed_seconds":round(time.monotonic()-start,2)
    }
    output_dir.mkdir(parents=True,exist_ok=False)
    file=output_dir/"private-point-predictions.json"
    file.write_text(json.dumps(evidence,sort_keys=True,indent=2)+"\n","utf8")
    valid=sum(c["start"] is not None and c["end"] is not None for c in results)
    return {
        "status":"H132_ACTUAL_PRIVATE_SPECIALIST_2B_POINTS_SCORED_NOT_ACTIONS",
        "model_id":MODEL_ID,"model_sha":MODEL_SHA,
        "actual_neural_calls":8,"fully_native_valid_pairs":valid,
        "native_invalid_pairs":4-valid,"private_prediction_sha256":sha(file.read_bytes()),
        "training_steps":0,"paid_compute_used":False,
        "model_improvement_proven":False
    }
