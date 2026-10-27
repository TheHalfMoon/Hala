"""H142 genuine 64-step continuation LoRA on VISIBLE React column labels.

External Qwen3.5-2B research adapter only, no Hala 35B update.
Owner source TRAIN64+DEV16 are new and SHA-pinned. START remains supervised;
HEAD goal is visibly labeled Doing/Done, NEVER invisible .hide_drop.
FREE private Kaggle T4; no GPU fallback and no H140/H141 holdout input.
"""
import json
import os
import random
import time
from collections import Counter,defaultdict
from pathlib import Path

from h139.train_gpu import (
    BASE_MODEL,BASE_REV,find_lora_targets,model_predict
)
from h141.gpu_visible_probe import visible_prompt
from h140.qualification import digest
from h142.collect_visible_source import SCHEMA,COUNTS,APP_SHA,DESTINATIONS

DATA_ROOT=Path("/kaggle/input")
TRAIN_SHA="988e4f0b716c574c6fa11a70519102003d4230f66bb0a6f89402ea877b8cc664"
DEV_SHA="22776558cf8c18865b715faaa33ea3b6d0ee8f58e4049a9151222f86dedaa285"
PRETRAINED_V2_ADAPTER_SHA="9595e463f2b621e308beaaefffd9ade8c38210bdc59550cb87d8e5f66d2a4bd0"
STEPS=64
ACCUM=3
SEED=142
START_ROLE="start"
HEADING_ROLE="visible_column_heading"

def find_input_sources():
    result={}
    for file in DATA_ROOT.rglob("manifest.json"):
        try:doc=json.loads(file.read_text("utf8"))
        except (OSError,ValueError):continue
        if doc.get("schema")!=SCHEMA:continue
        split=doc.get("split")
        if split not in ("train","dev") or split in result:
            raise ValueError("H142 only one frozen genuine TRAIN and DEV split")
        if digest(file.read_bytes())!={"train":TRAIN_SHA,"dev":DEV_SHA}[split]:
            raise ValueError("H142 original source manifest SHA changed on Kaggle")
        if (doc.get("private") is not True or
            doc.get("blind_eligible") is not False or
            doc.get("app_sources")!={"react-native-board":{
                "sha":APP_SHA,"license":"MIT"}} or
            doc.get("training_eligible") is not (split=="train") or
            doc.get("evaluation_eligible") is not (split=="dev") or
            len(doc.get("cases",[]))!=COUNTS[split]):
            raise ValueError("H142 incorrect owner-private source role")
        image_shas=set()
        strata=Counter()
        for row in doc["cases"]:
            if (row.get("app_git_sha")!=APP_SHA or row.get("split")!=split or
                row.get("language") not in ("ar","en") or
                row.get("destination_status") not in DESTINATIONS or
                row.get("reference_drag_persisted") is not True or
                row.get("reference_survived_page_reload") is not True or
                row.get("target_is_visible_at_rest") is not True or
                row.get("hidden_exact_drop_point_is_not_output_target") is not True):
                raise ValueError("H142 original native visible-target source not eligible")
            img=file.parent/row["image"]
            if (not img.is_file() or digest(img.read_bytes())!=row["image_sha256"]
                or row["image_sha256"] in image_shas):
                raise ValueError("H142 owner original train-dev media SHA mismatch")
            image_shas.add(row["image_sha256"])
            strata[(row["language"],row["destination_status"])]+=1
        if strata!={(a,b):COUNTS[split]//4 for a in ("ar","en") for b in DESTINATIONS}:
            raise ValueError("H142 train/dev original bilingual goal stratification drift")
        result[split]=(file.parent,doc)
    if set(result)!={"train","dev"}:
        raise ValueError("H142 exactly original TRAIN64 and DEV16 datasets required")
    if ({r["image_sha256"] for r in result["train"][1]["cases"]} &
        {r["image_sha256"] for r in result["dev"][1]["cases"]}):
        raise ValueError("H142 frozen screenshot SHA leak between train and dev")
    return result

def role_prompt_and_gold(row,role):
    if role not in (START_ROLE,HEADING_ROLE):
        raise ValueError("H142 cannot optimize hidden drop role")
    prompt=visible_prompt(row["instruction"],role)
    rect=(row["start_bbox"] if role==START_ROLE
          else row["visible_destination_column_heading_bbox"])
    x=rect["x"]+rect["width"]/2
    y=rect["y"]+rect["height"]/2
    if not (0<=x<960 and 0<=y<720):
        raise ValueError("H142 only visible native target inside original viewport")
    gold=json.dumps({"point_2d":[round(x/960*1000),round(y/720*1000)]},
                    separators=(",",":"))
    return prompt,gold

def assemble(processor,image,row,role,device):
    prompt,gold=role_prompt_and_gold(row,role)
    if image.size!=(960,720):raise ValueError("H142 original image dimensions differ")
    messages=[{"role":"user","content":[{"type":"image"},
                  {"type":"text","text":prompt}]}]
    context=processor.apply_chat_template(
        messages,tokenize=False,add_generation_prompt=True,
        enable_thinking=False)
    full=processor.apply_chat_template(
        messages+[{"role":"assistant","content":[{"type":"text","text":gold}]}],
        tokenize=False,add_generation_prompt=False,enable_thinking=False)
    if not full.startswith(context):
        raise ValueError("H142 Qwen3.5 completion-only source token template mismatch")
    prompt_tokens=processor(text=[context],images=[image],return_tensors="pt")
    combined=processor(text=[full],images=[image],return_tensors="pt")
    prefix=int(prompt_tokens["input_ids"].shape[1])
    if not 0<prefix<int(combined["input_ids"].shape[1]):
        raise ValueError("H142 malformed completion-only supervisory sequence")
    labels=combined["input_ids"].clone()
    labels[:,:prefix]=-100
    if "attention_mask" in combined:
        labels[combined["attention_mask"]==0]=-100
    if (labels!=-100).sum().item()<5:
        raise ValueError("H142 original visible heading completion labels missing")
    combined["labels"]=labels
    return {k:(v.to(device) if hasattr(v,"to") else v)
            for k,v in combined.items()}

def schedule(rows):
    if len(rows)!=64:raise ValueError("H142 only original TRAIN64")
    # Every START once (protect source skill), every visible HEADING twice
    # (fix H141 y≈441 rather than visible heading y≈182), 64*3 examples.
    examples=[(i,START_ROLE) for i in range(64)]
    examples.extend((i,HEADING_ROLE) for i in range(64) for _ in range(2))
    shuffle=random.Random(SEED)
    shuffle.shuffle(examples)
    if (len(examples)!=STEPS*ACCUM or
        Counter(role for _,role in examples)!={START_ROLE:64,HEADING_ROLE:128}):
        raise ValueError("H142 balanced real GUI source schedule drift")
    return examples

def score(processor,model,torch,root,doc):
    from PIL import Image
    table=defaultdict(Counter)
    cases=[]
    for row in doc["cases"]:
        with Image.open(root/row["image"]) as picture:
            image=picture.convert("RGB")
        answers={}
        for role in (START_ROLE,HEADING_ROLE):
            prompt,_=role_prompt_and_gold(row,role)
            raw,point=model_predict(processor,model,torch,image,prompt)
            rect=(row["start_bbox"] if role==START_ROLE
                  else row["visible_destination_column_heading_bbox"])
            hit=bool(point and rect["x"]<=point[0]<=rect["x"]+rect["width"]
                     and rect["y"]<=point[1]<=rect["y"]+rect["height"])
            answers[role]=hit
            for name in ("all",row["destination_status"],
                         row["language"],row["destination_status"]+"/"+row["language"]):
                table[name][role+"_hit"]+=int(hit)
                table[name][role+"_valid"]+=int(point is not None)
            cases.append({"id":row["id"],"role":role,
                          "point":point,"raw":raw,"hit":hit})
        for name in ("all",row["destination_status"],
                     row["language"],row["destination_status"]+"/"+row["language"]):
            table[name]["both_hit"]+=int(all(answers.values()))
            table[name]["cases"]+=1
    if len(cases)!=32:
        raise ValueError("H142 DEV16 must score both visible original roles")
    return {a:dict(b) for a,b in table.items()},cases

def run(destination,adapter_directory):
    import torch,transformers
    from PIL import Image
    from transformers import AutoProcessor,AutoModelForImageTextToText
    from peft import PeftModel
    destination=Path(destination)
    adapter_directory=Path(adapter_directory)
    if (destination.exists() or
        not str(destination).startswith("/kaggle/working/") or
        transformers.__version__!="5.17.0" or
        not torch.cuda.is_available() or
        not any("T4" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count()))):
        raise ValueError("H142 only NEW owner private FREE T4, no paid fallback")
    if (not (adapter_directory/"adapter_model.safetensors").is_file() or
        digest((adapter_directory/"adapter_model.safetensors").read_bytes())
        !=PRETRAINED_V2_ADAPTER_SHA):
        raise ValueError("H142 requires original previously qualified 2B research LoRA")
    sources=find_input_sources()
    train_dir,train_doc=sources["train"]
    dev_dir,dev_doc=sources["dev"]
    random.seed(SEED)
    torch.manual_seed(SEED)
    os.environ.update(HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
    processor=AutoProcessor.from_pretrained(
        BASE_MODEL,revision=BASE_REV,trust_remote_code=False)
    base=AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,revision=BASE_REV,torch_dtype=torch.float16,
        trust_remote_code=False,attn_implementation="sdpa").to("cuda:0")
    if "qwen3_5" not in base.config.model_type.lower():
        raise ValueError("H142 original Qwen3.5 GUI grounding base changed")
    model=PeftModel.from_pretrained(base,str(adapter_directory),is_trainable=True)
    model.eval()
    start=time.monotonic()
    before,before_rows=score(processor,model,torch,dev_dir,dev_doc)
    for name,p in model.named_parameters():
        p.requires_grad_("lora_" in name)
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if (not trainable or any("lora_" not in n or "vision" in n or
                            "visual" in n for n,p in trainable)):
        raise ValueError("H142 previous external 2B adapter language LoRA only")
    if hasattr(model,"enable_input_require_grads"):model.enable_input_require_grads()
    if hasattr(model,"gradient_checkpointing_enable"):model.gradient_checkpointing_enable()
    if hasattr(model.config,"use_cache"):model.config.use_cache=False
    optimizer=torch.optim.AdamW([p for _,p in trainable],lr=8e-5,
                                weight_decay=.01)
    examples=schedule(train_doc["cases"])
    losses=[]
    for i in range(STEPS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for j in range(ACCUM):
            case_index,role=examples[i*ACCUM+j]
            row=train_doc["cases"][case_index]
            with Image.open(train_dir/row["image"]) as pic:
                image=pic.convert("RGB")
            batch=assemble(processor,image,row,role,"cuda:0")
            output=model(**batch)
            loss=output.loss/ACCUM
            if not torch.isfinite(loss):
                raise ValueError("H142 true original visible GUI optimization loss nonfinite")
            loss.backward()
            losses.append(round(float(loss.detach().cpu())*ACCUM,6))
            del batch,output,loss
        torch.nn.utils.clip_grad_norm_([p for _,p in trainable],1.0)
        optimizer.step()
        if i%8==0:print("H142_REAL_FREE_T4_VISIBLE_HEADER_OPTIMIZER",
                         i+1,"/",STEPS,flush=True)
        torch.cuda.empty_cache()
    model.eval()
    if hasattr(model.config,"use_cache"):model.config.use_cache=True
    after,after_rows=score(processor,model,torch,dev_dir,dev_doc)
    destination.mkdir(parents=True,exist_ok=False)
    weights=destination/"private-h142-visible-column-lora"
    model.save_pretrained(weights,safe_serialization=True)
    processor.save_pretrained(weights)
    adapter=weights/"adapter_model.safetensors"
    if not adapter.is_file():
        raise ValueError("H142 no actual trained LoRA model persisted")
    report={
        "schema":"hala.h142.fresh-original-visible-column-2b-adapter.v1",
        "status":"COMPLETED_64_REAL_FREE_T4_TRAIN_STEPS",
        "steps":STEPS,"accum":ACCUM,
        "source_train_original":64,"source_dev_original":16,
        "train_manifest_sha256":TRAIN_SHA,"dev_manifest_sha256":DEV_SHA,
        "start_examples_seen":64,"visible_heading_examples_seen":128,
        "language_only_lora_trainable_parameters":sum(p.numel() for _,p in trainable),
        "loss_trace":losses,
        "v2_checkpoint_before":before,"h142_checkpoint_after":after,
        "original_v2_adapter_sha256":PRETRAINED_V2_ADAPTER_SHA,
        "h142_adapter_sha256":digest(adapter.read_bytes()),
        "training_seconds":round(time.monotonic()-start,1),
        "gpu_t4_only":True,"paid_compute_used":False,
        "h140_blind_model_calls":0,
        "h141_consumed_probe_train_or_dev_used":False,
        "flagship_Hala_35B_updated":False,
        "independent_app_transfer_proven":False,
        "release_qualified":False,
    }
    (destination/"private-h142-learning-evidence.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n","utf8")
    for name,rows in (("before",before_rows),("after",after_rows)):
        (destination/("private-h142-"+name+"-dev-predictions.json")).write_text(
            json.dumps(rows,sort_keys=True,indent=2)+"\n","utf8")
    return report
