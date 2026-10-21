"""H56 genuine unchanged-Qwen native ordered-drag PUBLIC DEV baseline.

Run only interactively on approved FREE Colab Linux CUDA. This module NEVER
loads the H55 golden Edge capture.json, never invents model outputs, never
trains and never touches official benchmark TEST or external accounts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

from training.scripts.baseline import MODEL_ID, MODEL_REVISION, reject_kaggle_runtime

DEV_SCHEMA="hala.h55.original-bilingual-drag-public-dev.v1"
DEV_SOURCE_SHA="707ae19c6baf29d83501c0d7d02ba4c790bc6e40d592363a164bfe41e5b13bc6"
ATTEMPTS_SCHEMA="hala.h55.public-dev-raw-drag-attempts.v1"
MODEL_SCHEMA="hala.h56.real-unchanged-qwen-native-drag-dev.v1"
PROTOCOL="original_image_pixels_1024x640_native_drag_v1"
ALLOWED_GPU_PLATFORMS=("colab",)
MAX_CASES=24
def sha(data):return hashlib.sha256(data).hexdigest()
def canonical(obj):return (json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n").encode("utf8")

def source_inputs(root):
    """Read ONLY prompt, original source provenance and PNGs. No DEV answer access."""
    root=Path(root)
    source=json.loads((root/"source.json").read_text("utf8"))
    plans=json.loads((root/"plans.json").read_text("utf8"))
    raw_source=(root/"source.json").read_bytes()
    if (sha(raw_source)!=DEV_SOURCE_SHA or
        source.get("schema")!=DEV_SCHEMA or source.get("public_dev_only") is not True or
        source.get("blind_test") is not False or source.get("split")!="dev" or
        source.get("rights",{}).get("train_eligible") is not False or
        source.get("plans_sha256")!=sha((root/"plans.json").read_bytes()) or
        len(plans)!=MAX_CASES or len(source.get("scenes",[]))!=MAX_CASES):
        raise ValueError("Frozen first-party PUBLIC DEV provenance mismatch")
    result=[]
    image_digests=set()
    langcounts={"ar":0,"en":0}
    families=set()
    for plan,entry in zip(plans,source["scenes"]):
        ident=plan["id"]
        if (entry.get("id")!=ident or
            plan.get("split")!="dev" or plan.get("viewport")!=[1024,640] or
            plan.get("language") not in langcounts or
            type(plan.get("goal_percent")) is not int or not 15<=plan["goal_percent"]<=95):
            raise ValueError("Unexpected H55 source plan or DEV metadata")
        html_path=root/"html"/(ident+".html")
        image_path=root/"png"/(ident+".png")
        if sha(html_path.read_bytes())!=entry["html_sha256"]:
            raise ValueError("Authored public DEV source HTML changed")
        png=image_path.read_bytes()
        if (not png.startswith(b"\x89PNG\r\n\x1a\n") or
            len(png)<128 or len(png)>5_000_000):
            raise ValueError("Original real Edge screenshot is not a bounded PNG")
        # Verify PNG native dimensions directly (not image-content oracle).
        import struct
        if struct.unpack(">II",png[16:24])!=(1024,640):
            raise ValueError("Incorrect original screenshot dimensions")
        image_hash=sha(png)
        if image_hash in image_digests:raise ValueError("Duplicate original screenshot")
        image_digests.add(image_hash)
        langcounts[plan["language"]]+=1
        families.add(plan["family"])
        target=plan["goal_percent"]
        request=("اسحب مؤشر الواجهة الافتراضية إلى "+str(target)+"٪. "
                 "أعد فعل سحب واحد فقط بإحداثيات الصورة الأصلية." if plan["language"]=="ar"
                 else "Drag the fictional interface slider to "+str(target)+
                   "%. Return only ONE ordered drag action in original-image pixel coordinates.")
        request+=("\nThe image is 1024x640 pixels. Return ONLY a JSON object with "
                  "type=drag, start_point=[x,y], end_point=[x,y], "
                  "coordinate_space=source_image_pixels, ordered=true. "
                  "Coordinates must be integer pixels of the ORIGINAL screenshot; "
                  "do not click, navigate, or execute anything.")
        result.append(dict(id=ident,language=plan["language"],family=plan["family"],
                           image_path=str(image_path),image_sha256=image_hash,prompt=request))
    if langcounts!={"ar":12,"en":12} or len(families)!=6:
        raise ValueError("Missing full disjoint bilingual public DEV inventory")
    return result

def validate_remote():
    # CLI is a strictly interactive notebook run, not a background shell or
    # disguised Kaggle commercial GPU session.
    if (platform.system()!="Linux" or not os.environ.get("COLAB_RELEASE_TAG") or
        os.environ.get("HALA_REMOTE_BASELINE_ALLOWED")!="1"):
        raise ValueError("Only explicitly authorized interactive FREE Colab Linux session may run weights")
    reject_kaggle_runtime()
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        raise ValueError("Kaggle GPU training/inference not authorized")
    if not Path("/content").exists():
        raise ValueError("Missing interactive Colab workspace; no remote weights elsewhere")
    return True

def run(root,output,*,max_cases=24,max_new_tokens=160):
    validate_remote()
    if (type(max_cases) is not int or not 1<=max_cases<=MAX_CASES or
        type(max_new_tokens) is not int or not 32<=max_new_tokens<=192):
        raise ValueError("Bounded actual model run parameters required")
    cases=source_inputs(root)
    output=Path(output)
    if output.exists():raise FileExistsError("Create-only baseline output folder")
    output.mkdir(parents=True)
    # Output never includes a DEV answer, only raw model generations.
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    if (not torch.cuda.is_available() or transformers.__version__!="5.16.1"):
        raise ValueError("Unqualified installed CUDA/transformers. No silent fallback")
    gpu=torch.cuda.get_device_properties(0)
    info=dict(schema=MODEL_SCHEMA,model_id=MODEL_ID,revision=MODEL_REVISION,
      evidence="real_inference",source_sha256=DEV_SOURCE_SHA,
      coordinate_protocol=PROTOCOL,split="public_dev",blind_test=False,
      model_training_executed=False,adapters_loaded=False,
      provider="interactive_approved_free_colab",
      gpu=gpu.name,vram_bytes=gpu.total_memory,
      package_versions={"torch":torch.__version__,"transformers":transformers.__version__},
      real_generated_cases=[],source_cases=len(cases),max_cases=max_cases,
      status="loading_model",inference_failure=None,
      independent_model_performance_qualified=False)
    def save_partial():
        (output/"run-partial.json").write_bytes(canonical(info))
    save_partial()
    model=None
    try:
        os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
        os.environ["DO_NOT_TRACK"]="1"
        os.environ["HF_HOME"]="/tmp/hala-h56-model-cache"
        torch.manual_seed(42)
        processor=AutoProcessor.from_pretrained(MODEL_ID,revision=MODEL_REVISION,
                                                  trust_remote_code=False)
        model=Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,revision=MODEL_REVISION,trust_remote_code=False,
            torch_dtype=torch.float16,attn_implementation="sdpa").to("cuda:0").eval()
        info["status"]="inferencing"
        save_partial()
        for scene in cases[:max_cases]:
            with Image.open(scene["image_path"]) as image:
                if image.format!="PNG" or image.size!=(1024,640):
                    raise ValueError("Source screenshot changed since frozen preflight")
                picture=image.convert("RGB")
            messages=[{"role":"user","content":[{"type":"image"},{"type":"text","text":scene["prompt"]}]}]
            template=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inputs=processor(text=[template],images=[picture],return_tensors="pt").to("cuda:0")
            before=time.perf_counter()
            with torch.inference_mode():
                token_ids=model.generate(**inputs,max_new_tokens=max_new_tokens,
                                           do_sample=False,max_time=60)
            torch.cuda.synchronize(0)
            generated=token_ids[:,inputs.input_ids.shape[1]:]
            raw=processor.batch_decode(generated,skip_special_tokens=True)[0]
            if not isinstance(raw,str) or len(raw)>8192:
                raise ValueError("Unbounded original neural text output")
            info["real_generated_cases"].append(dict(
                 id=scene["id"],image_sha256=scene["image_sha256"],
                 input_prompt_sha256=sha(scene["prompt"].encode("utf8")),
                 language=scene["language"],output_raw=raw,
                 output_sha256=sha(raw.encode("utf8")),
                 elapsed_seconds=time.perf_counter()-before,
                 output_tokens=int(generated.shape[1]),status="real_neural_raw"))
            save_partial()
            print("H56_ACTUAL_QWEN_RAW_CASE",len(info["real_generated_cases"]),scene["id"],flush=True)
            del inputs,token_ids,generated
        info["status"]="completed" if max_cases==24 else "diagnostic_partial_not_baseline"
        info["peak_vram_bytes"]=int(torch.cuda.max_memory_allocated(0))
        if max_cases==24:
            predictions=dict(schema=ATTEMPTS_SCHEMA,source_sha256=DEV_SOURCE_SHA,
                origin="provider_raw_model_outputs",
                cases=[dict(id=c["id"],image_sha256=c["image_sha256"],
                            raw_output=c["output_raw"]) for c in info["real_generated_cases"]])
            (output/"raw-attempts.json").write_bytes(canonical(predictions))
            info["raw_attempts_sha256"]=sha((output/"raw-attempts.json").read_bytes())
        (output/"run-result.json").write_bytes(canonical(info))
        return info
    except Exception as exc:
        info["status"]="blocked"
        info["inference_failure"]=type(exc).__name__
        save_partial()
        raise
    finally:
        if model is not None:
            del model
        torch.cuda.empty_cache()

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",default="h55/assets")
    parser.add_argument("--output",required=True)
    parser.add_argument("--max-cases",type=int,default=24)
    a=parser.parse_args()
    res=run(a.source,a.output,max_cases=a.max_cases)
    print(json.dumps({k:res[k] for k in ("status","source_cases","max_cases","model_id","revision")},sort_keys=True))
