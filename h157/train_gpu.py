"""H157 multi-task spatial GUI research-LoRA continuation, free T4 only.

Reuses the verified H156 image/text multimodal completion-only masking
and native JSON prompt. Genuinely trains 280 new independent source
screenshots, with 24 retained dropdown examples repeated once.

NO H156 consumed DEV, H140 blind, flagship Hala 35B, or paid GPU.
"""
import io
import json
import math
import os
import random
import time
from collections import Counter,defaultdict
from pathlib import Path

from h156.train_gpu import open_media,sha,assemble,inference
from h157.action_contract_v3 import SEQUENCES,parse_v3
from h157.collect_spatial_source import COUNTS,SCHEMA,APP_SHA
from h157.audit_spatial_source import verify_png

SOURCE_SHA="a28b845b212ba7b745f1b155944174278b8aac69e69ad0f0eee3ca3acd8b7f78"
SOURCE_ZIP_SHA="711f6eaa71646e6862ce92852fd8545b9101a45a36ee9689668d223467c900bf"
START_ADAPTER_SHA="732f54893172262348f7ef802a3d645626d6721f2214f28944740f0f07c91ac6"
START_ADAPTER_ZIP_SHA="b10e3e32fc4bfd098d551b9b93e14b4bd21d87a9ed5a245975c17d404b47136b"
STEPS=152
GRAD_ACCUM=2
RANDOM_SEED=157
MODEL_FILES={"README.md","adapter_config.json","adapter_model.safetensors"}


def read_source(source):
    source=Path(source)
    if not (source.is_dir() or source.is_file()):
        raise ValueError("H157 owner private frozen source missing")
    if source.is_file() and sha(source.read_bytes())!=SOURCE_ZIP_SHA:
        raise ValueError("H157 source ZIP itself altered")
    with open_media(source) as reader:
        names=reader.namelist()
        if "manifest.json" not in names:
            raise ValueError("H157 frozen original manifest absent")
        raw=reader.read("manifest.json")
        if sha(raw)!=SOURCE_SHA:
            raise ValueError("H157 original action source manifest changed")
        doc=json.loads(raw)
        cases=doc.get("private_cases")
        if (doc.get("schema")!=SCHEMA or doc.get("original_upstream_sha")!=APP_SHA
            or doc.get("train")!=280 or doc.get("dev")!=72
            or doc.get("task_counts")!=COUNTS or
            doc.get("h140_sealed_blind_calls")!=0 or
            not isinstance(cases,list) or len(cases)!=352):
            raise ValueError("H157 private source training/dev split changed")
        required={"manifest.json"}|{"images/"+row["image"] for row in cases}
        if len(names)!=353 or set(names)!=required:
            raise ValueError("H157 private source contains extra or missing files")
        seen=set()
        counted=Counter()
        for row in cases:
            if (row.get("split") not in COUNTS or
                row.get("task") not in COUNTS[row["split"]] or
                row.get("language")!="en" or
                row["image_sha256"] in seen or
                row.get("original_positive_native_reward") is not True or
                row.get("model_predictions_used") is not False):
                raise ValueError("H157 original case role or SHA drift")
            pixels=reader.read("images/"+row["image"])
            verify_png(pixels)
            if sha(pixels)!=row["image_sha256"]:
                raise ValueError("H157 original source pixel bytes changed")
            try:
                acts=parse_v3(json.dumps(row["training_supervision"]),
                              task=row["task"],model_checkpoint_sha="a"*64)
            except ValueError as exc:
                raise ValueError("H157 original structured action gold invalid") from exc
            if acts[0].kind=="abstain":
                raise ValueError("H157 source reference cannot be an abstention")
            counted[(row["split"],row["task"])]+=1
            seen.add(row["image_sha256"])
        if counted!={(split,task):quota for split,tasks in COUNTS.items()
                    for task,quota in tasks.items()}:
            raise ValueError("H157 source task quotas changed")
    return cases


def import_adapter(source,destination):
    source,destination=Path(source),Path(destination)
    if (destination.exists() or
        (source.is_file() and sha(source.read_bytes())!=START_ADAPTER_ZIP_SHA)):
        raise ValueError("H157 checkpoint archive altered or output consumed")
    with open_media(source) as container:
        if set(container.namelist())!=MODEL_FILES:
            raise ValueError("H157 checkpoint contains untrusted files")
        weights=container.read("adapter_model.safetensors")
        conf=json.loads(container.read("adapter_config.json"))
        if sha(weights)!=START_ADAPTER_SHA or conf.get("r")!=8:
            raise ValueError("H157 must continue exact H156 trained rank8 research adapter")
        destination.mkdir(parents=True,exist_ok=False)
        for name in sorted(MODEL_FILES):
            (destination/name).write_bytes(container.read(name))
    return destination


def train_schedule(train):
    if len(train)!=280:
        raise ValueError("H157 only frozen 280 original TRAIN sources")
    ids=list(range(280))
    # Retain 24 hard-earned dropdown examples without replaying H156 DEV.
    ids.extend(i for i,row in enumerate(train) if row["task"]=="choose-list")
    random.Random(RANDOM_SEED).shuffle(ids)
    if (len(ids)!=STEPS*GRAD_ACCUM or
        Counter(train[i]["task"] for i in ids)!={
            "click-test":64,"click-button":64,
            "enter-text":64,"enter-text-dynamic":64,"choose-list":48}):
        raise ValueError("H157 action roles and source training schedule drift")
    return ids


def score_dev(processor,model,torch,source,dev,sha_model):
    from PIL import Image
    stats=Counter()
    rows=[]
    with open_media(source) as stored:
        for i,row in enumerate(dev):
            with Image.open(io.BytesIO(stored.read("images/"+row["image"]))) as im:
                picture=im.convert("RGB")
            raw=inference(processor,model,torch,picture,row)
            accepted=False
            non_abstain=False
            text_correct=False
            try:
                acts=parse_v3(raw,task=row["task"],model_checkpoint_sha=sha_model)
                accepted=True
                non_abstain=acts[0].kind!="abstain"
                texts=[a.text for a in acts if a.text is not None]
                targets=[a["text"] for a in row["training_supervision"]["actions"]
                         if "text" in a]
                text_correct=non_abstain and texts==targets
            except ValueError:
                pass
            needs_text=row["task"] in (
                "enter-text","enter-text-dynamic","choose-list")
            stats["dev"]+=1
            stats["valid_json"]+=int(accepted)
            stats["valid_action_sequences"]+=int(non_abstain)
            stats["requires_text"]+=int(needs_text)
            stats["correct_text_values"]+=int(needs_text and text_correct)
            stats[row["task"]+"_n"]+=1
            stats[row["task"]+"_valid"]+=int(accepted)
            rows.append({
                "task":row["task"],"image_sha":row["image_sha256"],
                "raw":raw,"valid":accepted,"sequence":non_abstain,
                "text_correct":text_correct if needs_text else None,
            })
            if (i+1)%24==0:
                print("H157_REAL_GPU_MODEL_DEV_CALLS",i+1,len(dev),flush=True)
    if stats["dev"]!=72 or stats["requires_text"]!=40:
        raise ValueError("H157 DEV role coverage incorrect")
    return dict(stats),rows


def run(source_archive,adapter_archive,output):
    import torch,transformers
    from PIL import Image
    from transformers import AutoProcessor,AutoModelForImageTextToText
    from peft import PeftModel
    from h132.grounder_contract import MODEL_ID,MODEL_SHA

    output=Path(output)
    if (output.exists() or
        not str(output).startswith("/kaggle/working/") or
        transformers.__version__!="5.17.0" or
        not torch.cuda.is_available() or
        not any("T4" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count()))):
        raise ValueError("H157 only private free Nvidia Tesla T4 research")
    sources=read_source(source_archive)
    train=[r for r in sources if r["split"]=="TRAIN"]
    dev=[r for r in sources if r["split"]=="DEV"]
    if len(train)!=280 or len(dev)!=72 or ({r["image_sha256"] for r in train}
                                                 & {r["image_sha256"] for r in dev}):
        raise ValueError("H157 leaked or mismatched frozen TRAIN/DEV images")
    order=train_schedule(train)
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    os.environ.update(HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
    old=import_adapter(adapter_archive,
                       Path("/kaggle/working/h157-starting-h156-research-lora"))
    processor=AutoProcessor.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,trust_remote_code=False)
    base=AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,torch_dtype=torch.float16,
        trust_remote_code=False,attn_implementation="sdpa").to("cuda:0")
    model=PeftModel.from_pretrained(base,str(old),is_trainable=True)
    model.eval()
    before,before_predictions=score_dev(
        processor,model,torch,source_archive,dev,START_ADAPTER_SHA)
    for name,param in model.named_parameters():
        param.requires_grad_("lora_" in name)
    trainable=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
    if (not trainable or any("lora_" not in name or
                           "visual" in name or "vision" in name
                            for name,p in trainable)):
        raise ValueError("H157 only language rank8 LoRA can be retrained")
    if hasattr(model,"enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model,"gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config,"use_cache"):
        model.config.use_cache=False
    optimizer=torch.optim.AdamW([p for _,p in trainable],
                                lr=4e-5,weight_decay=.01)
    losses=[]
    started=time.monotonic()
    with open_media(source_archive) as originals:
        for step in range(STEPS):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for j in range(GRAD_ACCUM):
                row=train[order[step*GRAD_ACCUM+j]]
                with Image.open(io.BytesIO(
                        originals.read("images/"+row["image"]))) as im:
                    screenshot=im.convert("RGB")
                batch=assemble(processor,screenshot,row,"cuda:0")
                result=model(**batch)
                loss=result.loss/GRAD_ACCUM
                if not torch.isfinite(loss):
                    raise ValueError("H157 genuine optimizer loss not finite")
                loss.backward()
                losses.append(round(float(loss.detach().cpu())*GRAD_ACCUM,6))
                del batch,result,loss
            torch.nn.utils.clip_grad_norm_([p for _,p in trainable],1.0)
            optimizer.step()
            if (step+1)%16==0:
                print("H157_GENUINE_SPATIAL_LO_RA_STEPS",step+1,STEPS,flush=True)
            torch.cuda.empty_cache()
    model.eval()
    if hasattr(model.config,"use_cache"):
        model.config.use_cache=True
    output.mkdir(parents=True,exist_ok=False)
    weights=output/"private-h157-spatial-research-lora"
    model.save_pretrained(weights,safe_serialization=True)
    trained_file=weights/"adapter_model.safetensors"
    if not trained_file.is_file():
        raise ValueError("H157 no actual updated LoRA checkpoint saved")
    new_sha=sha(trained_file.read_bytes())
    if new_sha==START_ADAPTER_SHA:
        raise ValueError("H157 new training did not update research adapter")
    after,after_predictions=score_dev(
        processor,model,torch,source_archive,dev,new_sha)
    receipt={
        "schema":"hala.h157.private-original-spatial-action-2b-lora.v1",
        "status":"H157_GENUINE_152_FREE_T4_OPTIMIZER_STEPS_COMPLETE",
        "optimizer_steps":STEPS,"gradient_accumulation":GRAD_ACCUM,
        "train_unique_original":280,"train_weighted_action_examples":304,
        "dev_unique_original":72,
        "source_manifest_sha256":SOURCE_SHA,
        "initial_trained_h156_adapter_sha256":START_ADAPTER_SHA,
        "new_trained_h157_adapter_sha256":new_sha,
        "before_dev":before,"after_dev":after,
        "loss_trace":losses,
        "elapsed_seconds":round(time.monotonic()-started,1),
        "dev_not_blind":True,"true_native_task_completion_measured":False,
        "independent_cross_app_transfer_proven":False,
        "h140_sealed_blind_calls":0,"paid_compute":False,
        "Hala35B_unchanged":True,"weights_published":False,
        "release_qualified":False,
    }
    (output/"private-h157-learning-receipt.json").write_text(
        json.dumps(receipt,sort_keys=True,indent=2)+"\n","utf8")
    for name,data in (("before",before_predictions),("after",after_predictions)):
        (output/("private-h157-dev-"+name+".json")).write_text(
            json.dumps(data,sort_keys=True,indent=2)+"\n","utf8")
    return {k:v for k,v in receipt.items() if k!="loss_trace"}
