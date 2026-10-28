"""H172 paired fresh DEV inference: H164 frozen-language vs H171 language+vision.

Never opens training images, owner gold, previous DEV/OOD. Private no-optimizer,
no fine tuning or selected weights. Exactly 32 pairs frozen before starting.
"""
import hashlib,importlib.metadata,io,json,os,zipfile
from collections import Counter
from pathlib import Path
from h132.grounder_contract import MODEL_ID,MODEL_SHA
from h164.native_action_contract import parse_action
from h166.vision_probe import require_free_t4,MAX_PEAK_GIB
from h167.combined_probe import (
    admit_frozen_language_directory,LANGUAGE_WEIGHTS_SHA)
from h168.train_gpu import sha as sha_bytes
from h172.source_dev import cases

DEV_ZIP_SHA="0be919e841d737b16a86875e81ab7c640d430a2baceb05596d94e0611a79b231"
VISION_WEIGHTS_SHA="8796e964b9f830d353daf8bc21d90543b9b792931fe7733c6f443ba97057622e"
DEV_SCHEMA="hala.h172.private-new-original-dev-only-gpu-input.v1"
EXPECTED_COUNT=32

def sha_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as stream:
        for chunk in iter(lambda:stream.read(2**20),b""):
            h.update(chunk)
    return h.hexdigest()

def admit_dev(path):
    if sha_file(path)!=DEV_ZIP_SHA:
        raise ValueError("H172 private unseen DEV source ZIP pin mismatch")
    with zipfile.ZipFile(path) as z:
        doc=json.loads(z.read("manifest.json"))
        rows=doc.get("private_cases",[])
        spec=cases()
        if (doc.get("schema")!=DEV_SCHEMA or doc.get("split")!="DEV"
            or doc.get("count")!=EXPECTED_COUNT
            or doc.get("reference_gold_uploaded") is not False
            or doc.get("previous_train_and_consumed_eval_excluded") is not True
            or len(rows)!=EXPECTED_COUNT
            or set(z.namelist())!={"manifest.json"}|{
                "images/"+str(r["index"])+".png" for r in spec}):
            raise ValueError("H172 original-dev-only frozen input contract failed")
        seen=set()
        for row,expected in zip(rows,spec):
            if (not all(row.get(key)==value for key,value in expected.items())
                or row.get("image")!="images/"+str(row["index"])+".png"
                or row["image_sha256"] in seen
                or hashlib.sha256(z.read(row["image"])).hexdigest()!=row["image_sha256"]
                or any(k in row for k in ("training_supervision","reference_point",
                     "owner_gold","owner_reference_action","gold"))
                or len(row.get("instruction",""))<8):
                raise ValueError("H172 DEV screenshot or model-only instruction changed")
            seen.add(row["image_sha256"])
    return rows

def admit_vision(path):
    folder=Path(path)
    required={"adapter_model.safetensors","adapter_config.json"}
    if {x.name for x in folder.iterdir()}!=required:
        raise ValueError("H172 private frozen vision-only adapter contains unexpected files")
    if sha_file(folder/"adapter_model.safetensors")!=VISION_WEIGHTS_SHA:
        raise ValueError("H172 H171 actual visual LoRA SHA mismatch")
    config=json.loads((folder/"adapter_config.json").read_text("utf8"))
    if (config.get("base_model_name_or_path")!=MODEL_ID or config.get("r")!=16
        or sorted(config.get("target_modules",[]))!=[
           "model.visual.merger.linear_fc1","model.visual.merger.linear_fc2"]):
        raise ValueError("H172 vision rank4 merger config unqualified")
    return folder

def run(dev_zip,lang_dir,vision_dir,output):
    import torch
    from PIL import Image
    from peft import PeftModel
    from transformers import AutoProcessor,AutoModelForImageTextToText
    from h164.train_gpu import predict
    from h167.combined_probe import activate_frozen_language_and_trainable_vision

    output=Path(output)
    if (output.exists() or not str(output).startswith("/kaggle/working/")
        or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
        or os.environ.get("HALA_NONCOMMERCIAL_RESEARCH")!="1"):
        raise ValueError("H172 one-shot private free Kaggle model evaluation only")
    versions={n:importlib.metadata.version(n).split("+")[0]
              for n in ("torch","transformers","peft","Pillow")}
    gpu=require_free_t4(torch,versions)
    rows=admit_dev(dev_zip)
    original=admit_frozen_language_directory(
        lang_dir,Path("/kaggle/tmp/h172-verified-frozen-h164"))
    saved_vision=admit_vision(vision_dir)
    processor=AutoProcessor.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,trust_remote_code=False)
    base=AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,torch_dtype=torch.float16,
        attn_implementation="sdpa",trust_remote_code=False).to("cuda:0")
    model=PeftModel.from_pretrained(base,str(original),
              adapter_name="language",is_trainable=False)
    model.load_adapter(str(saved_vision),adapter_name="vision",is_trainable=False)
    model.eval()
    predictions=[]
    counts=Counter()
    torch.cuda.reset_peak_memory_stats(0)
    with zipfile.ZipFile(dev_zip) as z:
        for role in ("before","after"):
            if role=="before":
                model.set_adapter("language")
                if list(model.active_adapters)!=["language"]:
                    raise ValueError("H172 model before dual-LoRA not isolated")
            else:
                both=activate_frozen_language_and_trainable_vision(model)
                if both!=["language","vision"]:
                    raise ValueError("H172 after model failed H164+H171 activation")
            for row in rows:
                with Image.open(io.BytesIO(z.read(row["image"]))) as raw:
                    picture=raw.convert("RGB")
                proposal=predict(processor,model,torch,picture,row)
                parsed=parse_action(proposal)
                predictions.append({
                   "role":role,"case_index":row["index"],
                   "image_sha256":row["image_sha256"],
                   "app":row["app"],"language":row["language"],
                   "action_type":row["action_type"],
                   "raw":proposal,"parsed":parsed})
                counts[role+"/calls"]+=1
                counts[role+"/valid"]+=int(parsed["valid"] is True)
                if len(predictions)%8==0:
                    print("H172_FROZEN_DEV_GPU",len(predictions),64,flush=True)
    if len(predictions)!=64 or counts["before/calls"]!=32 or counts["after/calls"]!=32:
        raise ValueError("H172 paired 32-case evaluations incomplete")
    peak=torch.cuda.max_memory_allocated(0)/2**30
    if peak>MAX_PEAK_GIB:
        raise ValueError("H172 FREE T4 peak exceeded")
    output.mkdir(parents=True,exist_ok=False)
    result=dict(
        schema="hala.h172.frozen-paired-original-dev-predictions.v1",
        status="H172_PRIVATE_64_REAL_GPU_DEV_PREDICTIONS_COMPLETE",
        source_dev_zip_sha256=DEV_ZIP_SHA,
        frozen_language_sha256=LANGUAGE_WEIGHTS_SHA,
        trained_vision_sha256=VISION_WEIGHTS_SHA,
        base_model=MODEL_ID,base_revision=MODEL_SHA,
        real_gpu_calls=64,paired_dev_cases=32,
        peak_allocated_gib=round(peak,3),gpu=gpu,
        versions=versions,optimizer_steps=0,
        owner_gold_on_gpu=False,previous_consumed_ood_used=False,
        private_owner_browser_reward_measured=False,
        paid_gpu=False,weights_published=False,flagship_35b_unchanged=True,
        counts=dict(counts),predictions=predictions)
    (output/"h172-paired-dev-predictions.json").write_text(
       json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+"\n","utf8")
    return {k:v for k,v in result.items() if k!="predictions"}
