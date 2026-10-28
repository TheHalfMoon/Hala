"""H171 bounded bilingual 4-app native click/type/drag visual-LoRA SFT.

Real original-browser TRAIN images only. Never DEV/OOD. Free T4, new
rank16 visual merger on frozen H164 language LoRA, fixed 128 steps.
"""
import hashlib,io,importlib.metadata,json,math,os,random,zipfile
from collections import Counter
from pathlib import Path
from h132.grounder_contract import MODEL_ID,MODEL_SHA
from h164.native_action_contract import supervision,parse_action
from h166.vision_probe import require_free_t4,visual_merger_targets,MAX_PEAK_GIB
from h167.combined_probe import (admit_frozen_language_directory,
   activate_frozen_language_and_trainable_vision,is_visual_trainable,
   tensor_digest,LANGUAGE_WEIGHTS_SHA)
from h168.train_gpu import admit_train as admit_h168_train,TRAIN_ZIP_SHA

NEW_TRAIN_SHA="8690b8bf40db1de82dec3620cdbb02ca1ea7fca953f253b8e4208dd559c81235"
NEW_TRAIN_SCHEMA="hala.h171.private-original-click-train-32.v1"
SEED=171
EPOCHS=2
STEPS=128
RANK=16
LR=2e-5

def sha(blob):return hashlib.sha256(blob).hexdigest()

def admit_new_click(path):
    if sha(Path(path).read_bytes())!=NEW_TRAIN_SHA:
        raise ValueError("H171 new original native click TRAIN zip SHA mismatch")
    with zipfile.ZipFile(path) as z:
        doc=json.loads(z.read("manifest.json"))
        rows=doc.get("private_cases")
        expected={"manifest.json"}|{"images/"+str(i)+".png" for i in
            list(range(171000,171016))+list(range(171100,171116))}
        if (doc.get("schema")!=NEW_TRAIN_SCHEMA or doc.get("split")!="TRAIN"
            or doc.get("count")!=32 or len(rows)!=32
            or doc.get("reference_gold_uploaded_for_train_only") is not True
            or doc.get("previous_consumed_dev_ood_used") is not False
            or doc.get("original_upstream_mit_verified") is not True
            or set(z.namelist())!=expected):
            raise ValueError("H171 TRAIN-only new source not qualified")
        seen=set()
        for row in rows:
            if (row.get("split")!="TRAIN" or row.get("action_type")!="click"
                or row.get("app") not in ("flatpickr","awesomplete")
                or row.get("language") not in ("ar","en")
                or row["image_sha256"] in seen
                or row["image"]!="images/"+str(row["index"])+".png"
                or sha(z.read(row["image"]))!=row["image_sha256"]
                or parse_action(supervision(row))["kind"]!="click"):
                raise ValueError("H171 original screenshot/click action invalid")
            seen.add(row["image_sha256"])
        if (Counter(r["app"] for r in rows)!={"flatpickr":16,"awesomplete":16}
            or Counter(r["language"] for r in rows)!={"ar":16,"en":16}):
            raise ValueError("H171 32 new real UI click TRAIN distribution changed")
    return rows

def admit_combined(old,new):
    h168=admit_h168_train(old)
    click=admit_new_click(new)
    rows=[(0,row) for row in h168]+[(1,row) for row in click]
    images={r["image_sha256"] for _,r in rows}
    if (len(rows)!=64 or len(images)!=64
        or len({r["index"] for _,r in rows})!=64
        or Counter(r["language"] for _,r in rows)!={"ar":32,"en":32}
        or Counter(r["action_type"] for _,r in rows)!={
           "click":32,"type":16,"drag":16}
        or len({r["app"] for _,r in rows})!=4):
        raise ValueError("H171 original click/type/drag four-family bilingual TRAIN not qualified")
    return rows

def run(old,new,language,output):
    import torch
    from PIL import Image
    from transformers import AutoProcessor,AutoModelForImageTextToText
    from peft import PeftModel,LoraConfig
    from h164.train_gpu import assemble
    output=Path(output)
    if (output.exists() or not str(output).startswith("/kaggle/working/")
        or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
        or os.environ.get("HALA_NONCOMMERCIAL_RESEARCH")!="1"):
        raise ValueError("H171 free PRIVATE Kaggle research only")
    installed={k:importlib.metadata.version(k).split("+")[0]
               for k in ("torch","transformers","peft","Pillow")}
    gpu=require_free_t4(torch,installed)
    rows=admit_combined(old,new)
    lang=admit_frozen_language_directory(
        language,Path("/kaggle/tmp/h171-frozen-h164-language"))
    torch.manual_seed(SEED);random.seed(SEED)
    processor=AutoProcessor.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,trust_remote_code=False)
    base=AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,revision=MODEL_SHA,torch_dtype=torch.float16,
        attn_implementation="sdpa",trust_remote_code=False).to("cuda:0")
    targets=visual_merger_targets(base.named_modules())
    model=PeftModel.from_pretrained(base,str(lang),
        adapter_name="language",is_trainable=False)
    model.add_adapter("vision",LoraConfig(
        r=RANK,lora_alpha=32,lora_dropout=0.0,
        target_modules=targets,bias="none",task_type="CAUSAL_LM"))
    active=activate_frozen_language_and_trainable_vision(model)
    for name,p in model.named_parameters():
        p.requires_grad_(is_visual_trainable(name))
    language_params=[(n,p) for n,p in model.named_parameters()
                     if ".lora_A.language." in n or ".lora_B.language." in n]
    visual=[(n,p) for n,p in model.named_parameters()
            if is_visual_trainable(n)]
    if (len(visual)!=4 or len(language_params)==0 or any(
            p.requires_grad for _,p in language_params)):
        raise ValueError("H171 visual-only trainability partition malformed")
    frozen_digest=tensor_digest(language_params)
    if hasattr(model,"enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model,"gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config,"use_cache"):model.config.use_cache=False
    optim=torch.optim.AdamW([p for _,p in visual],lr=LR,weight_decay=0)
    schedule=list(range(64))*EPOCHS
    random.Random(SEED).shuffle(schedule)
    receipts=[];exposures=Counter()
    torch.cuda.reset_peak_memory_stats(0)
    with zipfile.ZipFile(old) as first,zipfile.ZipFile(new) as second:
        zips=(first,second)
        for ix,rid in enumerate(schedule):
            origin,row=rows[rid]
            with Image.open(io.BytesIO(zips[origin].read(row["image"]))) as im:
                img=im.convert("RGB")
            batch=assemble(processor,img,row,"cuda:0")
            model.train();optim.zero_grad(set_to_none=True)
            loss=model(**batch).loss
            if loss is None or not torch.isfinite(loss):
                raise ValueError("H171 real original app training loss nonfinite")
            value=float(loss.detach().item())
            if not 0<value<500:raise ValueError("H171 real training loss unsafe")
            loss.backward()
            gradients=sum(int(p.grad is not None
                         and torch.isfinite(p.grad).all().item()
                         and torch.count_nonzero(p.grad).item()>0)
                          for _,p in visual)
            if gradients<1:raise ValueError("H171 no visual LoRA gradients")
            optim.step();torch.cuda.synchronize(0)
            peak=torch.cuda.max_memory_allocated(0)/2**30
            if peak>MAX_PEAK_GIB:raise ValueError("H171 T4 memory exceeded 13 GiB")
            exposures[row["index"]]+=1
            receipts.append({"step":ix+1,"index":row["index"],
                "app":row["app"],"kind":row["action_type"],
                "language":row["language"],"finite_loss":True,
                "loss":round(value,6),"nonzero_visual_gradients":gradients})
            if ix%16==0 or ix==STEPS-1:
                print("H171_REAL_ORIGINAL_TRAIN",ix+1,STEPS,
                      round(peak,3),round(value,4),flush=True)
    if (len(receipts)!=STEPS or len(exposures)!=64
        or set(exposures.values())!={EPOCHS}
        or tensor_digest(language_params)!=frozen_digest
        or not all(optim.state.get(p) for _,p in visual)):
        raise ValueError("H171 actual bilingual 4-app optimizer/frozen language failed")
    output.mkdir(parents=True,exist_ok=False)
    folder=output/"private-rank16-vision"
    model.save_pretrained(str(folder),selected_adapters=["vision"],
                          safe_serialization=True)
    files=list(folder.rglob("adapter_model.safetensors"))
    if len(files)!=1 or files[0].stat().st_size<1024:
        raise ValueError("H171 only a saved private vision adapter allowed")
    receipt=dict(
        schema="hala.h171.64-original-4-app-two-epoch-real-t4-train.v1",
        status="H171_FOUR_APP_128_REAL_ORIGINAL_OPTIMIZER_STEPS_PASSED",
        original_h168_train_zip_sha256=TRAIN_ZIP_SHA,
        new_h171_click_train_zip_sha256=NEW_TRAIN_SHA,
        h164_frozen_language_sha256=LANGUAGE_WEIGHTS_SHA,
        base_model=MODEL_ID,base_revision=MODEL_SHA,
        active_adapters=active,vision_rank=RANK,
        visual_target_modules=targets,real_optimizer_steps=STEPS,
        unique_original_train_cases=64,exposures_per_case=EPOCHS,
        train_click_cases=32,train_type_cases=16,train_drag_cases=16,
        train_arabic_cases=32,train_english_cases=32,
        distinct_original_app_families=4,
        trainable_visual_parameters=sum(p.numel() for _,p in visual),
        frozen_language_parameters=sum(p.numel() for _,p in language_params),
        frozen_language_unchanged=True,
        private_vision_weights_sha256=sha(files[0].read_bytes()),
        gpu=gpu,installed=installed,
        peak_allocated_gib=round(torch.cuda.max_memory_allocated(0)/2**30,3),
        peak_reserved_gib=round(torch.cuda.max_memory_reserved(0)/2**30,3),
        memory_cap_gib=MAX_PEAK_GIB,
        optimizer_trace=receipts,
        dev_ood_model_calls=0,old_consumed_eval_used=False,
        private_weights_only=True,paid_gpu=False,
        real_browser_quality_measured=False,flagship_35b_unchanged=True)
    (output/"h171-training-receipt.json").write_text(
        json.dumps(receipt,indent=2,sort_keys=True)+"\n","utf8")
    return {k:v for k,v in receipt.items() if k!="optimizer_trace"}
