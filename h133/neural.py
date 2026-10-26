"""H133 preregistered paired frozen pretrained GUIs, 8 new screenshots.

Two private FREE Kaggle T4 kernels, different arms, same frozen screenshot
manifest; no oracle geometry, weights updated, H71 source or public results.
"""
import json
import os
import time
from pathlib import Path

from h133.source import validate,sha
from h130.browser_env import WIDTH,HEIGHT
from h125.model_contracts import (SHOWUI_SYSTEM,MODELS,prompt_for)
from h131.contract import parse_showui
from h132.grounder_contract import (MODEL_ID,MODEL_SHA,
                                     instruction_for,native_point)

FORMAT="hala.h133.frozen-unseen-browser8-neural.v1"
ARMS=("showui","specialist")

def source_dir():
    sources=[]
    for path in Path("/kaggle/input").rglob("manifest.json"):
        try: doc=validate(path.parent)
        except (ValueError,KeyError,FileNotFoundError,TypeError,json.JSONDecodeError):
            continue
        sources.append((path.parent,doc))
    if len(sources)!=1:
        raise ValueError("H133 one SHA-qualified private unseen-8 dataset required")
    return sources[0]

def call_showui(processor,model,torch,image,instruction,role):
    prompt=prompt_for("showui_2b",instruction,role.upper())
    if not prompt.startswith(SHOWUI_SYSTEM+"\n"):
        raise ValueError("H133 pinned H125 model-native system prompt changed")
    messages=[{"role":"user","content":[
        {"type":"text","text":SHOWUI_SYSTEM},{"type":"image"},
        {"type":"text","text":prompt[len(SHOWUI_SYSTEM)+1:]}]}]
    text=processor.apply_chat_template(messages,tokenize=False,
                                       add_generation_prompt=True)
    inputs=processor(text=[text],images=[image],return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        tokens=model.generate(**inputs,do_sample=False,max_new_tokens=64,max_time=45)
    torch.cuda.synchronize(0)
    raw=processor.batch_decode(tokens[:,inputs.input_ids.shape[1]:],
                               skip_special_tokens=True)[0].strip()
    return raw, list(parse_showui(raw,WIDTH,HEIGHT)) if parse_showui(raw,WIDTH,HEIGHT) else None

def call_specialist(processor,model,torch,image,instruction,role):
    instruction=instruction_for(instruction,role)
    message=[{"role":"user","content":[{"type":"image"},
                      {"type":"text","text":instruction}]}]
    text=processor.apply_chat_template(message,tokenize=False,
                       add_generation_prompt=True,enable_thinking=False)
    inputs=processor(text=[text],images=[image],return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        tokens=model.generate(**inputs,do_sample=False,max_new_tokens=80,max_time=45)
    torch.cuda.synchronize(0)
    raw=processor.batch_decode(tokens[:,inputs.input_ids.shape[1]:],
                               skip_special_tokens=True)[0].strip()
    return raw,native_point(raw,WIDTH,HEIGHT)

def run(output,arm):
    import torch,transformers
    from PIL import Image
    from transformers import (AutoProcessor,AutoModelForImageTextToText,
                              Qwen2VLForConditionalGeneration)
    output=Path(output)
    if (arm not in ARMS or output.exists() or
        not str(output).startswith("/kaggle/working/") or
        not torch.cuda.is_available() or
        not any("T4" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count()))):
        raise ValueError("H133 fresh owner-private FREE Kaggle T4 only")
    expected_transformers="5.16.1" if arm=="showui" else "5.17.0"
    if transformers.__version__!=expected_transformers:
        raise ValueError("H133 exact package version mismatch for frozen arm")
    data_dir,manifest=source_dir()
    torch.manual_seed(133)
    os.environ.update(HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
    if arm=="showui":
        model_id=MODELS["showui_2b"].model_id
        model_sha=MODELS["showui_2b"].revision_sha
        processor=AutoProcessor.from_pretrained(model_id,revision=model_sha,
                              min_pixels=256*28*28,max_pixels=1280*28*28)
        model=Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,revision=model_sha,torch_dtype=torch.float16,
            attn_implementation="sdpa").to("cuda:0").eval()
        if model.config.model_type!="qwen2_vl":
            raise ValueError("H133 frozen ShowUI architecture unexpected")
        inference=call_showui
    else:
        model_id,model_sha=MODEL_ID,MODEL_SHA
        processor=AutoProcessor.from_pretrained(model_id,revision=model_sha,
                                                trust_remote_code=False)
        model=AutoModelForImageTextToText.from_pretrained(
            model_id,revision=model_sha,torch_dtype=torch.float16,
            trust_remote_code=False,attn_implementation="sdpa").to("cuda:0").eval()
        if "qwen3_5" not in model.config.model_type.lower():
            raise ValueError("H133 frozen grounder architecture unexpected")
        inference=call_specialist
    start=time.monotonic();cases=[];calls=0
    for case in manifest["cases"]:
        file=data_dir/case["image"]
        with Image.open(file) as im:
            if im.format!="PNG" or im.size!=(WIDTH,HEIGHT):
                raise ValueError("H133 screenshot bytes/dimensions differ")
            rgb=im.convert("RGB")
        out={"task":case["task"],"image_sha256":case["image_sha256"]}
        for role in ("start","end"):
            raw,point=inference(processor,model,torch,rgb,case["instruction"],role)
            if not isinstance(raw,str) or not 1<=len(raw)<=300:
                raise ValueError("H133 real model native output malformed/oversized")
            out["raw_"+role]=raw
            out[role]=point
            calls+=1
        cases.append(out)
    if calls!=16 or len(cases)!=8:
        raise ValueError("H133 full eight-scene source paired model run incomplete")
    receipt={
       "schema":FORMAT,"model_arm":arm,"model_id":model_id,
       "model_sha":model_sha,
       "source_manifest_sha256":sha(json.dumps(manifest,sort_keys=True,
                                 separators=(",",":")).encode("utf8")),
       "cases":cases,"real_neural_calls":calls,
       "training_steps":0,"oracle_coordinates_in_model_input":False,
       "private_owner_only":True,"unseen_scene_selection":True,
       "model_skill_qualified":False,
       "elapsed_seconds":round(time.monotonic()-start,2),
    }
    output.mkdir(parents=True,exist_ok=False)
    target=output/"private-native-predictions.json"
    target.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n","utf8")
    return {"status":"H133_PRIVATE_REAL_NEURAL_UNSEEN8_COMPLETED_NOT_SCORED",
       "arm":arm,"real_neural_calls":calls,
       "native_valid_pairs":sum(c["start"] is not None and c["end"] is not None for c in cases),
       "private_json_sha256":sha(target.read_bytes()),"trained":False,
       "paid_compute_used":False}
