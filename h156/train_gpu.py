"""H156 free-T4 pilot of typed-native GUI action SFT on NEW MiniWoB sources.

All 48 TRAIN examples are distinct from 12 DEV and 81 previous pilot images.
Research LoRA continuation only (external pretrained Qwen3.5-2B specialist);
never treat a MiniWoB DEV score as cross-app or flagship Hala 35B progress.
"""
import hashlib
import io
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

SOURCE_SHA="fe274bc48225685ea135137e1607169ef9481903c5848f9bb1e2ede78f509562"
SCHEMA="hala.h156.private-action-train-dev-v2.v1"
ADAPTER_SHA="992fa18fead6724329dbe7ec7b95c8acb0a56229c1fa5b56719e5f4829130a68"
ADAPTER_ZIP_SHA="58ec4f6e026b5fc5c8be9a74433d3ca365f1c93680d84b2d148a86b8725e2b8c"
STEPS=24
SEED=156
GRAD_ACCUM=2
APP_SHA="7fd85d71a4b60325c6585396ec4f48377d049838"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_source(archive):
    from h156.action_contract_v2 import action_from_dict_v2
    archive=Path(archive)
    if not archive.is_file():
        raise ValueError("H156 private source must be a frozen ZIP")
    with ZipFile(archive) as z:
        names=z.namelist()
        if "manifest.json" not in names:
            raise ValueError("H156 source manifest missing")
        raw=z.read("manifest.json")
        if sha(raw)!=SOURCE_SHA:
            raise ValueError("H156 source SHA changed")
        doc=json.loads(raw)
        rows=doc.get("private_cases")
        if (doc.get("schema")!=SCHEMA or
            doc.get("source_app_git_sha")!=APP_SHA or
            doc.get("train")!=48 or doc.get("dev")!=12 or
            doc.get("h140_sealed_blind_calls")!=0 or
            not isinstance(rows,list) or len(rows)!=60):
            raise ValueError("H156 manifest source roles invalid")
        expected={"manifest.json"}|{"images/"+r["image"] for r in rows}
        if set(names)!=expected or len(names)!=61:
            raise ValueError("H156 private media extra/missing paths")
        counts=Counter()
        seen=set()
        for r in rows:
            if (r["split"] not in ("TRAIN","DEV") or
                r["task"] not in ("click-test","enter-text","choose-list") or
                r["image_sha256"] in seen):
                raise ValueError("H156 source split/action/duplicate")
            b=z.read("images/"+r["image"])
            if sha(b)!=r["image_sha256"]:
                raise ValueError("H156 original PNG image mismatch")
            seen.add(r["image_sha256"])
            counts[(r["split"],r["task"])]+=1
            actions=r["training_supervision"]["actions"]
            if not actions or any(action_from_dict_v2(a).kind=="abstain"
                                  for a in actions):
                raise ValueError("H156 genuine native action label required")
        if counts!={(s,t):n for s,n in (("TRAIN",16),("DEV",4))
                    for t in ("click-test","enter-text","choose-list")}:
            raise ValueError("H156 unbalanced independent train/dev source")
    return rows


def prompt(row):
    return (
        "You are Hala, a visual computer-use agent. See this 960x720 "
        "screenshot from an original website. Execute the task using "
        "only visible information. Return one JSON object with EXACTLY "
        "an actions array; no markdown or reasoning. Actions are "
        "click(point:[x,y]), type(text), select(point:[x,y],text), "
        "submit(point:[x,y]) or abstain. Coordinates are absolute pixels "
        "in the screenshot. For typing, click the input first, type the "
        "requested text, and click submit. For dropdown, select its "
        "visible center point and the visibly requested option text, "
        "then submit. No DOM selectors. Task: "+row["instruction"]
    )


def label(row):
    return json.dumps(row["training_supervision"],
                      ensure_ascii=False,separators=(",",":"),sort_keys=True)


def assemble(processor,image,row,device):
    import torch
    if image.size!=(960,720):
        raise ValueError("H156 original image must be native 960x720")
    request=[{"role":"user","content":[{"type":"image"},
                                         {"type":"text","text":prompt(row)}]}]
    pre=processor.apply_chat_template(
        request,tokenize=False,add_generation_prompt=True,
        enable_thinking=False)
    full=processor.apply_chat_template(
        request+[{"role":"assistant","content":[
            {"type":"text","text":label(row)}]}],
        tokenize=False,add_generation_prompt=False,enable_thinking=False)
    if not full.startswith(pre):
        raise ValueError("H156 full completion not an exact prompt extension")
    a=processor(text=[pre],images=[image],return_tensors="pt")
    b=processor(text=[full],images=[image],return_tensors="pt")
    prefix=int(a["input_ids"].shape[1])
    if not 0<prefix<int(b["input_ids"].shape[1]):
        raise ValueError("H156 multimodal completion supervision absent")
    ids=b["input_ids"].clone()
    ids[:,:prefix]=-100
    if "attention_mask" in b:
        ids[b["attention_mask"]==0]=-100
    if (ids!=-100).sum().item()<5:
        raise ValueError("H156 native action output not supervised")
    b["labels"]=ids
    return {k:v.to(device) if hasattr(v,"to") else v for k,v in b.items()}


def inference(processor,model,torch,image,row):
    model.eval()
    message=[{"role":"user","content":[{"type":"image"},
                                          {"type":"text","text":prompt(row)}]}]
    pre=processor.apply_chat_template(message,tokenize=False,
          add_generation_prompt=True,enable_thinking=False)
    batch=processor(text=[pre],images=[image],
                    return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        output=model.generate(**batch,max_new_tokens=176,
                              do_sample=False,max_time=50)
    torch.cuda.synchronize(0)
    raw=processor.batch_decode(
        output[:,batch["input_ids"].shape[1]:],
        skip_special_tokens=True)[0].strip()
    return raw


def score(processor,model,torch,archive,dev,checkpoint):
    from PIL import Image
    from h156.action_contract_v2 import parse_v2
    stats=Counter()
    results=[]
    with ZipFile(archive) as z:
        for row in dev:
            with Image.open(io.BytesIO(z.read("images/"+row["image"]))) as im:
                picture=im.convert("RGB")
            raw=inference(processor,model,torch,picture,row)
            stats["cases"]+=1
            valid=False;exact_sequence=False;typed_text=False
            try:
                actions=parse_v2(raw,task=row["task"],
                                 model_checkpoint_sha=checkpoint)
                valid=True
                exact_sequence=actions[0].kind!="abstain"
                target=row["training_supervision"]["actions"]
                predicted=[a.text for a in actions if a.text is not None]
                reference=[a["text"] for a in target if "text" in a]
                typed_text=(predicted==reference)
            except ValueError:
                pass
            stats["valid_json_actions"]+=int(valid)
            stats["non_abstain_valid_sequences"]+=int(exact_sequence)
            stats["correct_visible_text_values"]+=int(typed_text)
            results.append({"task":row["task"],"image_sha":row["image_sha256"],
                            "raw":raw,"valid":valid,
                            "sequence":exact_sequence,"text_correct":typed_text})
    return dict(stats),results


def run(archive,adapter_zip,output):
    import torch,transformers
    from PIL import Image
    from transformers import AutoProcessor,AutoModelForImageTextToText
    from peft import PeftModel
    from h132.grounder_contract import MODEL_ID,MODEL_SHA
    from h153.gpu_probe import admit_adapter
    output=Path(output)
    if (output.exists() or
        not str(output).startswith("/kaggle/working/") or
        transformers.__version__!="5.17.0" or
        not torch.cuda.is_available() or
        not any("T4" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count()))):
        raise ValueError("H156 only NEW FREE PRIVATE Nvidia T4 research worker")
    rows=load_source(archive)
    train=[r for r in rows if r["split"]=="TRAIN"]
    dev=[r for r in rows if r["split"]=="DEV"]
    if len(train)!=48 or len(dev)!=12:
        raise ValueError("H156 source roles mismatch")
    random.seed(SEED)
    torch.manual_seed(SEED)
    os.environ.update(HF_HUB_DISABLE_TELEMETRY="1",DO_NOT_TRACK="1")
    adapter=admit_adapter(adapter_zip,Path("/kaggle/working/h156-old-research-lora"))
    processor=AutoProcessor.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,trust_remote_code=False)
    base=AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,torch_dtype=torch.float16,
        trust_remote_code=False,attn_implementation="sdpa").to("cuda:0")
    model=PeftModel.from_pretrained(base,str(adapter),is_trainable=True)
    model.eval()
    before,before_rows=score(processor,model,torch,archive,dev,ADAPTER_SHA)
    for name,param in model.named_parameters():
        param.requires_grad_("lora_" in name)
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if (not trainable or any("lora_" not in n or
                           "vision" in n or "visual" in n
                            for n,p in trainable)):
        raise ValueError("H156 research may optimize language LoRA only")
    if hasattr(model,"enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model,"gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config,"use_cache"):
        model.config.use_cache=False
    optimizer=torch.optim.AdamW(
        [p for _,p in trainable],lr=6e-5,weight_decay=.01)
    schedule=list(range(48))
    random.Random(SEED).shuffle(schedule)
    loss_trace=[]
    start=time.monotonic()
    with ZipFile(archive) as z:
        for step in range(STEPS):
            optimizer.zero_grad(set_to_none=True)
            model.train()
            for j in range(GRAD_ACCUM):
                row=train[schedule[step*GRAD_ACCUM+j]]
                with Image.open(io.BytesIO(
                     z.read("images/"+row["image"]))) as img:
                    picture=img.convert("RGB")
                batch=assemble(processor,picture,row,"cuda:0")
                out=model(**batch)
                loss=out.loss/GRAD_ACCUM
                if not torch.isfinite(loss):
                    raise ValueError("H156 nonfinite genuine optimizer loss")
                loss.backward()
                loss_trace.append(round(float(loss.detach().cpu())*GRAD_ACCUM,5))
                del batch,out,loss
            torch.nn.utils.clip_grad_norm_([p for _,p in trainable],1.0)
            optimizer.step()
            if (step+1)%4==0:
                print("H156_GENUINE_LO_RA_OPTIMIZER_STEPS",step+1,STEPS,
                      flush=True)
            torch.cuda.empty_cache()
    model.eval()
    if hasattr(model.config,"use_cache"):
        model.config.use_cache=True
    output.mkdir(parents=True,exist_ok=False)
    checkpoint=output/"private-h156-typed-action-lora"
    model.save_pretrained(checkpoint,safe_serialization=True)
    final_file=checkpoint/"adapter_model.safetensors"
    if not final_file.is_file():
        raise ValueError("H156 LoRA weights missing after optimization")
    result_sha=sha(final_file.read_bytes())
    after,after_rows=score(processor,model,torch,archive,dev,result_sha)
    report={
        "schema":"hala.h156.first-genuine-typed-action-2b-lora-pilot.v1",
        "status":"H156_COMPLETED_REAL_TYPED_ACTION_OPTIMIZER_STEPS",
        "actual_optimizer_steps":STEPS,
        "train_cases":48,"dev_cases":12,
        "source_manifest_sha256":SOURCE_SHA,
        "frozen_initial_adapter_sha256":ADAPTER_SHA,
        "updated_action_adapter_sha256":result_sha,
        "loss_trace":loss_trace,
        "dev_before":before,"dev_after":after,
        "source_app_count":1,"source_languages":["en"],
        "evaluation": "DEV STRUCTURED OUTPUT ONLY; NOT NATIVE TASK SCORE",
        "elapsed_seconds":round(time.monotonic()-start,1),
        "paid_compute":False,"h140_sealed_blind_calls":0,
        "Hala_flagship_35B_updated":False,
        "independent_multiapp_accuracy_proven":False,
        "weights_published":False,"release_qualified":False,
    }
    (output/"private-h156-train-receipt.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n","utf8")
    for name,results in (("before",before_rows),("after",after_rows)):
        (output/("private-h156-dev-"+name+".json")).write_text(
            json.dumps(results,indent=2,sort_keys=True)+"\n","utf8")
    return {k:v for k,v in report.items() if k not in ("loss_trace",)}
