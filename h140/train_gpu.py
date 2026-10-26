"""H140 PRIVATE original GUI multi-app geometry LoRA experiment, FREE T4 ONLY.

External pinned Qwen3.5-2B research baseline, never Hala flagship 35B.
Exactly one new source-qualified 256 TRAIN + 64 DEV, no blind mount.
All 512 original role targets are sampled exactly once across 128 steps.
"""
import json
import os
import random
import time
from collections import Counter,defaultdict
from pathlib import Path

from h140.qualification import load_split,digest,SCHEMA
from h139.train_gpu import (
    assemble_example,make_prompt_and_gold,model_predict,find_lora_targets,
    BASE_MODEL,BASE_REV,
)

DATA_ROOT=Path("/kaggle/input")
TRAIN_MANIFEST_SHA="a59438ca178c3b86da721c164e3ae653566c2001c25bb9208ff42afed0c14f38"
DEV_MANIFEST_SHA="433f4001481d5db704e3b8b19b47154d5637ffc570533e6223880ecdc172d87a"
STEPS=128
GRAD_ACCUM=4
LORA_RANK=8
SEED=140
EXPECTED_APP_SET={"vanilla-native-board","react-native-board","react-dndkit-board"}

def private_splits():
    manifests=[]
    for file in DATA_ROOT.rglob("manifest.json"):
        try:
            candidate=json.loads(file.read_text("utf8"))
        except (OSError,ValueError):
            continue
        if candidate.get("schema")!=SCHEMA:
            continue
        manifests.append((file.parent,candidate))
    roles=Counter(d.get("split") for folder,d in manifests)
    if roles!={"train":1,"dev":1}:
        raise ValueError("H140 EXACTLY one private TRAIN and one DEV; refuse BLIND mount")
    selected={d["split"]:(folder,d) for folder,d in manifests}
    for role,expected_hash,expected_count in (
        ("train",TRAIN_MANIFEST_SHA,256),("dev",DEV_MANIFEST_SHA,64)
    ):
        folder,_=selected[role]
        actual=digest((folder/"manifest.json").read_bytes())
        if actual!=expected_hash:
            raise ValueError("H140 owner-frozen original source manifest SHA mismatch")
        doc=load_split(folder,role)
        if len(doc["cases"])!=expected_count or set(doc["app_sources"])!=EXPECTED_APP_SET:
            raise ValueError("H140 original three-app source size/provenance wrong")
        selected[role]=(folder,doc)
    images={role:{row["image_sha256"] for row in doc["cases"]}
            for role,(folder,doc) in selected.items()}
    if images["train"]&images["dev"] or len(images["train"])!=256 or len(images["dev"])!=64:
        raise ValueError("H140 frozen source train/dev image SHA leak")
    return selected

def summarize_native(rows,manifest):
    """Predeclared paired point-box hits and anti-collapse on 16 per original app."""
    truth={row["id"]:row for row in manifest["cases"]}
    hits=defaultdict(lambda:{"start":0,"end":0,"both":0,"cases":0})
    pred={}
    seen=set()
    for item in rows:
        key=(item["case_id"],item["role"])
        if key in seen or item["case_id"] not in truth or item["role"] not in ("start","end"):
            raise ValueError("H140 wrong or duplicated frozen DEV model response")
        seen.add(key)
        original=truth[item["case_id"]]
        raw_point=item["point"]
        rect=original[item["role"]+"_bbox"]
        hit=(raw_point is not None
             and rect["x"]<=raw_point[0]<=rect["x"]+rect["width"]
             and rect["y"]<=raw_point[1]<=rect["y"]+rect["height"])
        if bool(hit)!=item["hit"]:
            raise ValueError("H140 source native point scoring drift")
        pred[key]=tuple(raw_point) if raw_point is not None else None
    if len(seen)!=128:
        raise ValueError("H140 full original DEV64 x START/END = 128 calls")
    for original in manifest["cases"]:
        key=original["id"]
        app=original["app"];lang=original["language"]
        s=next(x for x in rows if x["case_id"]==key and x["role"]=="start")
        e=next(x for x in rows if x["case_id"]==key and x["role"]=="end")
        for name in ("__all__",app,app+"/"+lang):
            v=hits[name]
            v["cases"]+=1
            v["start"]+=int(s["hit"])
            v["end"]+=int(e["hit"])
            v["both"]+=int(s["hit"] and e["hit"])
    diversity={}
    for app in sorted(EXPECTED_APP_SET):
        sixteen=[r for r in manifest["cases"] if r["app"]==app][:16]
        if len(sixteen)!=16:
            raise ValueError("H140 per-app 16-case dynamic geometry slice absent")
        per_role={}
        for role in ("start","end"):
            positions=[pred[(r["id"],role)] for r in sixteen]
            frequencies=Counter(positions)
            valid=sum(point is not None for point in positions)
            uniques=len({point for point in positions if point is not None})
            mode=max(frequencies.values())
            per_role[role]={"valid_points":valid,"unique_points":uniques,
                            "largest_point_count":mode,
                            "passed":valid==16 and uniques>=8 and mode<=4}
        diversity[app]=per_role
    return {"point_hits":dict(hits),"point_diversity_by_app":diversity,
            "all_apps_and_roles_diverse":all(
                z["passed"] for v in diversity.values() for z in v.values())}

def score(processor,model,torch,folder,doc):
    from PIL import Image
    records=[]
    for row in doc["cases"]:
        with Image.open(folder/row["image"]) as file:
            img=file.convert("RGB")
        for role in ("start","end"):
            prompt,_=make_prompt_and_gold(row,role)
            raw,point=model_predict(processor,model,torch,img,prompt)
            rect=row[role+"_bbox"]
            hit=bool(point is not None and
                     rect["x"]<=point[0]<=rect["x"]+rect["width"] and
                     rect["y"]<=point[1]<=rect["y"]+rect["height"])
            records.append({"case_id":row["id"],"app":row["app"],
                            "language":row["language"],"role":role,
                            "raw":raw,"point":point,"hit":hit})
    return summarize_native(records,doc),records

def run(destination):
    import torch,transformers
    from PIL import Image
    from transformers import AutoProcessor,AutoModelForImageTextToText
    from peft import LoraConfig,get_peft_model
    destination=Path(destination)
    if (destination.exists() or not str(destination).startswith("/kaggle/working/")
        or transformers.__version__!="5.17.0"
        or not torch.cuda.is_available()
        or not any("T4" in torch.cuda.get_device_name(i)
                   for i in range(torch.cuda.device_count()))):
        raise ValueError("H140 FREE T4, pinned 2B runtime, private owner-only output")
    allowed=private_splits()
    train_root,train_doc=allowed["train"]
    dev_root,dev_doc=allowed["dev"]
    random.seed(SEED)
    torch.manual_seed(SEED)
    os.environ.update(HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
    processor=AutoProcessor.from_pretrained(BASE_MODEL,revision=BASE_REV,
                                           trust_remote_code=False)
    model=AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,revision=BASE_REV,torch_dtype=torch.float16,
        trust_remote_code=False,attn_implementation="sdpa").to("cuda:0")
    if "qwen3_5" not in model.config.model_type.lower():
        raise ValueError("H140 expected pinned pretrained external 2B model")
    start_clock=time.monotonic()
    before,before_raw=score(processor,model,torch,dev_root,dev_doc)
    for param in model.parameters():
        param.requires_grad_(False)
    targets=find_lora_targets(model,torch)
    lora=LoraConfig(r=LORA_RANK,lora_alpha=16,lora_dropout=0.05,bias="none",
                    task_type="CAUSAL_LM",target_modules=targets)
    model=get_peft_model(model,lora)
    if hasattr(model,"enable_input_require_grads"):
        model.enable_input_require_grads()
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if (not trainable or any("lora_" not in n for n,p in trainable)
        or any("vision" in n or "visual" in n for n,p in trainable)):
        raise ValueError("H140 only explicitly selected language LoRA trainable")
    if hasattr(model,"gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config,"use_cache"):
        model.config.use_cache=False
    optimizer=torch.optim.AdamW([p for n,p in trainable],lr=7.5e-5,
                                weight_decay=0.01)
    examples=[(i,role) for i in range(256) for role in ("start","end")]
    random.shuffle(examples)
    if len(examples)!=STEPS*GRAD_ACCUM:
        raise ValueError("H140 original TRAIN256 x two roles must be one exact epoch")
    loss_trace=[]
    steps_taken=0
    for step in range(STEPS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for k in range(GRAD_ACCUM):
            row_id,role=examples[step*GRAD_ACCUM+k]
            row=train_doc["cases"][row_id]
            with Image.open(train_root/row["image"]) as file:
                image=file.convert("RGB")
            batch=assemble_example(processor,image,row,role,"cuda:0")
            output=model(**batch)
            loss=output.loss/GRAD_ACCUM
            if not torch.isfinite(loss):
                raise ValueError("H140 actual optimizer loss nonfinite")
            loss.backward()
            loss_trace.append(round(float(loss.detach().cpu())*GRAD_ACCUM,6))
            del loss,output,batch
        torch.nn.utils.clip_grad_norm_([p for n,p in trainable],1.0)
        optimizer.step()
        steps_taken+=1
        if step%8==0:
            print(f"H140 REAL FREE T4 DYNAMIC GUI LoRA {steps_taken}/{STEPS}",
                  flush=True)
        torch.cuda.empty_cache()
    if steps_taken!=128:
        raise ValueError("H140 incomplete real optimization must not be promoted")
    model.eval()
    if hasattr(model.config,"use_cache"):
        model.config.use_cache=True
    after,after_raw=score(processor,model,torch,dev_root,dev_doc)
    destination.mkdir(parents=True,exist_ok=False)
    adapter_dir=destination/"private-h140-original-geometry-lora-rank8"
    model.save_pretrained(adapter_dir,safe_serialization=True)
    processor.save_pretrained(adapter_dir)
    model_file=adapter_dir/"adapter_model.safetensors"
    if not model_file.is_file():
        raise ValueError("H140 optimizer reported success but weights absent")
    # Critically, qualified as a research experiment, NOT independent transfer.
    receipt={
        "schema":"hala.h140.original-three-app-position-diverse-2b-lora.v1",
        "status":"H140_COMPLETED_128_REAL_GPU_OPTIMIZER_STEPS",
        "frozen_train_original_screens":256,"frozen_dev_original_screens":64,
        "real_optimizer_steps":steps_taken,
        "original_start_examples_seen":256,"original_end_examples_seen":256,
        "lora_rank":LORA_RANK,"trainable_parameter_count":sum(
             p.numel() for n,p in trainable),
        "loss_trace":loss_trace,
        "pretrained_dev":before,"h140_adapter_dev":after,
        "train_manifest_sha256":TRAIN_MANIFEST_SHA,
        "dev_manifest_sha256":DEV_MANIFEST_SHA,
        "adapter_sha256":digest(model_file.read_bytes()),
        "elapsed_seconds":round(time.monotonic()-start_clock,2),
        "real_blind_neural_calls":0,
        "paid_compute_used":False,
        "blind_gold_attached":False,
        "published_model_weights":False,
        "flagship_Hala_35B_updated":False,
        "independent_app_transfer_proven":False}
    (destination/"private-h140-learning-evidence.json").write_text(
        json.dumps(receipt,sort_keys=True,indent=2)+"\n",encoding="utf8")
    for name,items in (("dev-before",before_raw),("dev-after",after_raw)):
        (destination/("private-h140-"+name+".json")).write_text(
            json.dumps(items,sort_keys=True,indent=2)+"\n",encoding="utf8")
    return receipt
