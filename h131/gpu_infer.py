"""H131 owner-private Kaggle free T4: pinned ShowUI pretrained neural GUI points.

This predicts actual START/END from real Chromium screenshots; no DOM, no
oracle selectors/target coordinates, no original heldout train images.
"""
import json
import os
from pathlib import Path
import time
from h125.model_contracts import MODELS, SHOWUI_SYSTEM, prompt_for
from h131.contract import (MODEL_SCHEMA, sha, verify_manifest, parse_showui,
                           TASK_COUNT, WIDTH, HEIGHT)
ARM="showui_2b"

def _infer(model,processor,torch,image,intent,role):
    prompt=prompt_for(ARM,intent,role.upper())
    if not prompt.startswith(SHOWUI_SYSTEM+"\n"):
        raise ValueError("H131 pinned native ShowUI prompt drift")
    message=[{"role":"user","content":[
        {"type":"text","text":SHOWUI_SYSTEM},{"type":"image"},
        {"type":"text","text":prompt[len(SHOWUI_SYSTEM)+1:]}]}]
    prepared=processor.apply_chat_template(message,tokenize=False,add_generation_prompt=True)
    batch=processor(text=[prepared],images=[image],return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        generated=model.generate(**batch,max_new_tokens=64,do_sample=False,max_time=45)
    torch.cuda.synchronize(0)
    raw=processor.batch_decode(generated[:,batch.input_ids.shape[1]:],
                               skip_special_tokens=True)[0]
    if type(raw) is not str or len(raw)>128:
        raise ValueError("H131 invalid model-native text response")
    return raw

def locate_private_input(phase):
    root=Path("/kaggle/input")
    files=list(root.glob("*/manifest.json"))
    valid=[]
    for file in files:
        try:
            manifest=verify_manifest(file.parent,expect_phase=phase)
        except (ValueError,KeyError,TypeError,FileNotFoundError,json.JSONDecodeError):
            continue
        valid.append((file.parent,manifest))
    if len(valid)!=1:
        raise ValueError("H131 exactly one private, SHA-qualified screenshot input required")
    return valid[0]

def run(output,phase):
    from PIL import Image
    import torch,transformers
    from transformers import AutoProcessor,Qwen2VLForConditionalGeneration

    output=Path(output)
    if (phase not in ("initial","feedback") or output.exists() or
        not str(output).startswith("/kaggle/working/") or
        not torch.cuda.is_available() or
        not any("T4" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())) or
        transformers.__version__!="5.16.1"):
        raise ValueError("H131 fresh private free T4 + pinned transformer runtime required")
    directory,manifest=locate_private_input(phase)
    cfg=MODELS[ARM]
    torch.manual_seed(42)
    os.environ.update(HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1")
    processor=AutoProcessor.from_pretrained(
        cfg.model_id,revision=cfg.revision_sha,
        min_pixels=256*28*28,max_pixels=1280*28*28)
    model=Qwen2VLForConditionalGeneration.from_pretrained(
        cfg.model_id,revision=cfg.revision_sha,
        torch_dtype=torch.float16,attn_implementation="sdpa").to("cuda:0").eval()
    if model.config.model_type!="qwen2_vl":
        raise ValueError("H131 frozen pretrained ShowUI architecture mismatch")
    rows=[]; calls=0
    started=time.monotonic()
    for row in manifest["cases"]:
        with Image.open(directory/row["image"]) as im:
            if im.format!="PNG" or im.size!=(WIDTH,HEIGHT):
                raise ValueError("H131 actual Chromium screenshot dimensions incorrect")
            image=im.convert("RGB")
        instruction=row["instruction"]
        if phase=="feedback":
            instruction+=" The preceding mouse drag failed to complete the task. Review the CURRENT screenshot and locate the best START and END points for another attempt."
        output_row={"task":row["task"],"image_sha256":row["image_sha256"]}
        for role in ("start","end"):
            raw=_infer(model,processor,torch,image,instruction,role)
            calls+=1
            point=parse_showui(raw)
            if point is None:
                raise ValueError("H131 pretrained model returned invalid native point, no oracle fallback")
            output_row["raw_"+role]=raw
            output_row[role]=list(point)
        rows.append(output_row)
    if calls!=8 or len(rows)!=TASK_COUNT:
        raise ValueError("H131 exact full neural screenshot stage unfinished")
    manifest_sha=sha(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode("utf8"))
    result={"schema":MODEL_SCHEMA,"phase":phase,
            "input_manifest_sha256":manifest_sha,"model_id":cfg.model_id,
            "model_revision":cfg.revision_sha,"model_license":cfg.license,
            "cases":rows,"real_neural_calls":calls,"optimizer_steps":0,
            "paid_compute_used":False,"source_has_oracle_coordinates":False,
            "evidence_is_real_browser_screenshots":True,
            "model_skill_qualified":False,
            "elapsed_seconds":round(time.monotonic()-started,2)}
    output.mkdir(parents=True,exist_ok=False)
    (output/"private-native-points.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n","utf8")
    return {"status":"PASS_H131_PRIVATE_NEURAL_POINTS_NOT_ACTIONS",
            "phase":phase,"manifest_sha256":manifest_sha,
            "real_model_calls":8,"optimizer_steps":0,
            "original_source_or_evaluation_images_used":False,
            "private_predictions_sha256":sha((output/"private-native-points.json").read_bytes()),
            "model_skill_qualified":False}
