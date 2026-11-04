"""H417 private source-bound bilingual native-DRAG LoRA from 12K originals.

Sources: 10,000 unique original DragOn TRAIN JPEGs and 2,000 truly
first-party Microsoft Edge rendered TRAIN scenes (1K Arabic, 1K English).
48 first-party images are reserved from TRAIN for development, as are 96
unused-original TRAIN images across 4 domains. These are author-visible DEV,
NEVER independent external OOD. No click- or browser-action execution.
"""
from __future__ import annotations
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import zipfile

BASE = "Qwen/Qwen3-VL-2B-Instruct"
REV = "89644892e4d85e24eaac8bacfd4f463576704203"
SOURCE_REV = "3451c1fc3543c18fc3361a32bfe1f12cc69b64b5"
DATASET_SHA = "294cfb953457549a4315819557d9346e6bb81a243657bd940fd654cc29474f23"
FIRSTPARTY_ZIP_SHA = "6c17d2f312709ec693666027b103b35a628811db8fb433f5f32a35e068faedba"
FIRSTPARTY_LABEL_SHA = "94c9b943f79264ceccf5d04a2daf8278e7cb02589ce5edbddee8a3e557daa4e6"
SOURCE_SHAS = {
    "sheet": ("h60-dragon-train-sheet-0000.hala.bin",
              "09635ac0410418ef19b6dd64d878469500d56d94cb362047b6019d2e4b811924"),
    "slider": ("h60-dragon-train-slider-0005.hala.bin",
               "cbd55b0a003d0c6aafef2460ce5bc8b0321e8a4237032053a53a077901483c96"),
    "slide_resize": ("h60-dragon-train-slide_resize-0000.hala.bin",
                     "3e66f34b9a1228b99c7c8866885f63eaba3848b7973b187089cf055e8d2cbf7a"),
    "text_highlight": ("h60-dragon-train-text_highlight-0000.hala.bin",
                       "848c0db0f4ec02036f0e5290ec3da862e3e937e301ffdd9c1e06baa8956a8117"),
}
BIL_FILE = "original-bilingual-drag.hala.bin"
CURRIC_FILE = "h61_drag_curriculum.jsonl"
DOMAINS = tuple(SOURCE_SHAS)
OUT = Path("/kaggle/working/h417-owner-private-native-drag")
N_SOURCE_DRAGON = 10000
N_SOURCE_FIRSTPARTY = 2000
DEV_DRAGON_PER_DOMAIN = 24
DEV_FIRSTPARTY_PER_LANG = 24
TRAIN_UNIQUE = 11952
AR_REPEAT = 3
GRAD_ACCUM = 4
EXPECTED_PASSES = 14880
STEPS = EXPECTED_PASSES // GRAD_ACCUM
SEED = 417
TOL_NORM = 50.0


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    dig = hashlib.sha256()
    with Path(path).open("rb") as f:
        for blob in iter(lambda: f.read(1024*1024), b""):
            dig.update(blob)
    return dig.hexdigest()


def only_input_file(name: str) -> Path:
    found = list(Path("/kaggle/input").rglob(name))
    if len(found) != 1 or found[0].name != name:
        raise ValueError("H417 private input missing or ambiguous: "+name)
    return found[0]


def strict_load(value: str):
    def pairs(items):
        got = {}
        for k,v in items:
            if k in got:
                raise ValueError("Duplicate JSON key")
            got[k] = v
        return got
    def nonfinite(x):
        raise ValueError("Nonfinite JSON numeric")
    return json.loads(value, object_pairs_hook=pairs, parse_constant=nonfinite)


def valid_coords(point):
    return (type(point) is list and len(point) == 2 and
            all(type(v) in (int,float) and math.isfinite(v) and 0 <= v <= 1000
                for v in point))


def parse_drag(raw: str):
    try:
        data = strict_load(raw.strip())
        if (type(data) is not dict or set(data) !=
            {"type","start_point","end_point","ordered","coordinate_space"} or
            data["type"] != "drag" or
            data["coordinate_space"] != "normalized_1000" or
            type(data["ordered"]) is not bool or
            not valid_coords(data["start_point"]) or
            not valid_coords(data["end_point"])):
            return None
        return data
    except (TypeError, ValueError, AttributeError):
        return None


def get_research_sources():
    originals = {}
    zipfiles = {}
    for domain,(name,expected) in SOURCE_SHAS.items():
        archive_path = only_input_file(name)
        if sha_file(archive_path) != expected:
            raise ValueError("H417 original H60 archive SHA changed: "+domain)
        archive = zipfile.ZipFile(archive_path)
        manifest = strict_load(archive.read("manifest.json").decode("utf8"))
        if (manifest.get("schema") != "hala.h60.dragon-train-shard.v1" or
            manifest.get("source_commit_sha") != SOURCE_REV or
            manifest.get("source_domain") != domain or
            manifest.get("split") != "train" or
            manifest.get("official_eval_or_test_used") is not False or
            manifest.get("benchmark_answer_leakage") is not False or
            manifest.get("source_license") != "Apache-2.0"):
            raise ValueError("H417 original source manifest not TRAIN qualified")
        labels_bytes = archive.read("labels.jsonl")
        if sha_bytes(labels_bytes) != manifest.get("labels_sha256"):
            raise ValueError("H417 original domain labels digest drift")
        originals[domain] = [strict_load(line)
                             for line in labels_bytes.decode("utf8").splitlines()
                             if line.strip()]
        if len(originals[domain]) != manifest["native_drag_actions"]:
            raise ValueError("H417 original H60 label count drift")
        zipfiles[domain] = archive

    c_path = only_input_file(CURRIC_FILE)
    if sha_file(c_path) != DATASET_SHA:
        raise ValueError("H417 frozen original 10k curriculum SHA mismatch")
    curriculum = [strict_load(line)
                  for line in c_path.read_text("utf8").splitlines() if line.strip()]
    if len(curriculum) != N_SOURCE_DRAGON:
        raise ValueError("H417 original 10k frozen source count mismatch")

    p_path = only_input_file(BIL_FILE)
    if sha_file(p_path) != FIRSTPARTY_ZIP_SHA:
        raise ValueError("H417 original 2k first-party source ZIP hash mismatch")
    firstparty = zipfile.ZipFile(p_path)
    receipt = strict_load(firstparty.read("receipt.json").decode("utf8"))
    bytes_labels = firstparty.read("labels.jsonl")
    if (sha_bytes(bytes_labels) != FIRSTPARTY_LABEL_SHA or
        receipt.get("labels_sha256") != FIRSTPARTY_LABEL_SHA or
        receipt.get("scenes_accepted") != N_SOURCE_FIRSTPARTY or
        receipt.get("scenes_rejected") != 0 or
        receipt.get("by_language") != {"ar":1000,"en":1000} or
        receipt.get("benchmark_answer_leakage") is not False or
        receipt.get("public_dev_families_used") is not False):
        raise ValueError("H417 original first-party browser evidence mismatch")
    firstparty_rows = [strict_load(line)
                       for line in bytes_labels.decode("utf8").splitlines()
                       if line.strip()]
    if len(firstparty_rows) != N_SOURCE_FIRSTPARTY:
        raise ValueError("H417 first-party scene count drift")
    zipfiles["firstparty"] = firstparty
    return zipfiles, originals, curriculum, firstparty_rows


def case_from_dragon(r):
    if (r.get("split") != "train" or r.get("language") != "en" or
        r.get("image_format", "JPEG") != "JPEG" or
        type(r.get("image_path")) is not str or
        type(r.get("image_sha256")) is not str):
        raise ValueError("H417 one original DragOn TRAIN row is not qualified")
    parts = r["image_path"].split("/")
    if len(parts) != 3 or parts[0] != "images" or parts[1] not in DOMAINS:
        raise ValueError("H417 original image path must be bounded by domain")
    data = r["completion"] if type(r.get("completion")) is dict else strict_load(r["completion"])
    if (type(data) is not dict or data.get("type") != "drag" or
        data.get("coordinate_space") != "source_image_pixels" or
        type(data.get("ordered")) is not bool):
        raise ValueError("H417 missing original native drag completion")
    # Raw H60 rows carry `ordered`; H61 curated rows store the same
    # trusted original boolean inside completion and capability. Require
    # agreement with each available independent source field.
    if "ordered" in r and (type(r["ordered"]) is not bool or
                           r["ordered"] is not data["ordered"]):
        raise ValueError("H417 original native drag ordered flag drift")
    if "capability" in r:
        cap = r["capability"]
        if (type(cap) is not str or len(cap.split("/")) < 2 or
            cap.split("/")[1] != ("ordered" if data["ordered"] else "unordered")):
            raise ValueError("H417 curated original drag capability flag drift")
    for field in ("start_point","end_point"):
        pair = data.get(field)
        if (type(pair) is not list or len(pair) != 2 or
            any(type(v) is not int or v < 0 for v in pair)):
            raise ValueError("H417 native drag source coordinates changed")
    body = (r.get("intent") if type(r.get("intent")) is str else
            r["prompt"].split("\nReturn only ONE ",1)[0]).strip()
    if len(body) < 4 or len(body) > 1400:
        raise ValueError("H417 task body bounds")
    return {
       "id":r["id"],"domain":parts[1],"language":"en","kind":"original",
       "image_path":r["image_path"],"image_sha256":r["image_sha256"],
       "body":body,"start":data["start_point"],"end":data["end_point"],
       "ordered":data["ordered"]
    }


def case_from_firstparty(r):
    if (r.get("split") != "train" or r.get("language") not in ("en","ar") or
        r.get("kind") not in DOMAINS or type(r.get("ordered")) is not bool or
        type(r.get("image_path")) is not str or
        not r["image_path"].startswith("png/") or
        "/" in r["image_path"][4:]):
        raise ValueError("H417 original browser image path or provenance invalid")
    def centre(key):
        bbox = r[key]
        if (type(bbox) is not list or len(bbox)!=4 or
            any(type(x) is not int for x in bbox)):
            raise ValueError("H417 browser capture bbox invalid")
        x0,y0,x1,y1=bbox
        if not (0<=x0<x1<=1024 and 0<=y0<y1<=640):
            raise ValueError("H417 browser capture bbox outside original source")
        return [(x0+x1)//2,(y0+y1)//2]
    return {
       "id":r["id"],"domain":r["kind"],"language":r["language"],
       "kind":"firstparty","image_path":r["image_path"],
       "image_sha256":r["image_sha256"],
       "body":r["instruction"],"start":centre("start_bbox"),
       "end":centre("end_bbox"),"ordered":r["ordered"]
    }


def partition(originals,cur,firstparty):
    training = [case_from_dragon(r) for r in cur]
    if len({r["image_sha256"] for r in training})!=10000:
        raise ValueError("H417 original screenshot duplicates in 10k source")
    source_ids = {r["image_sha256"] for r in training}
    development = []
    from collections import Counter
    for dom in DOMAINS:
        seen = set()
        pool = sorted(originals[dom],
            key=lambda row: sha_bytes(row["id"].encode()))
        for row in pool:
            if row["image_sha256"] in source_ids or row["image_sha256"] in seen:
                continue
            r = case_from_dragon(row)
            development.append(r)
            seen.add(r["image_sha256"])
            if len(seen)==DEV_DRAGON_PER_DOMAIN:
                break
        if len(seen) != DEV_DRAGON_PER_DOMAIN:
            raise ValueError("H417 insufficient source-disjoint original DEV")
    source_ids.update(r["image_sha256"] for r in development)
    from_fp = [case_from_firstparty(row) for row in firstparty]
    if len({r["image_sha256"] for r in from_fp}) != 2000:
        raise ValueError("H417 repeated original Edge screenshot")
    withheld = set()
    for language in ("en","ar"):
        vals = sorted((r for r in from_fp if r["language"]==language),
              key=lambda row: sha_bytes(row["id"].encode()))
        chosen = vals[:DEV_FIRSTPARTY_PER_LANG]
        if len(chosen) != DEV_FIRSTPARTY_PER_LANG:
            raise ValueError("H417 insufficient first-party bilingual DEV")
        development.extend(chosen)
        withheld.update(r["image_sha256"] for r in chosen)
    training.extend(r for r in from_fp if r["image_sha256"] not in withheld)
    if len(training) != TRAIN_UNIQUE or len(development) != 144:
        raise ValueError("H417 source image-level holdout count changed")
    if source_ids & {r["image_sha256"] for r in from_fp}:
        raise ValueError("H417 unrelated source family image collision")
    if {r["image_sha256"] for r in training} & {
             r["image_sha256"] for r in development}:
        raise ValueError("H417 original TRAIN/DEV source pixels overlap")
    n_ar = sum(r["language"]=="ar" for r in training)
    n_en = sum(r["language"]=="en" for r in training)
    if (n_ar,n_en) != (976,10976):
        raise ValueError("H417 original Arabic/English training balance drift")
    order = list(range(len(training)))
    order += [i for i,r in enumerate(training) if r["language"]=="ar"]*AR_REPEAT
    if len(order) != EXPECTED_PASSES or len(order)%GRAD_ACCUM:
        raise ValueError("H417 full gradient-bearing data schedule mismatch")
    random.Random(SEED).shuffle(order)
    return training,development,order


def source_image(case,zips):
    from PIL import Image
    source = zips["firstparty" if case["kind"]=="firstparty" else case["domain"]]
    b = source.read(case["image_path"])
    if sha_bytes(b)!=case["image_sha256"]:
        raise ValueError("H417 original image media SHA drift")
    with Image.open(io.BytesIO(b)) as frame:
        if frame.format!=("PNG" if case["kind"]=="firstparty" else "JPEG"):
            raise ValueError("H417 original image encoding drift")
        w,h = frame.size
        for point in (case["start"],case["end"]):
            if not (0<=point[0]<=w and 0<=point[1]<=h):
                raise ValueError("H417 original native coordinates offscreen")
        im=frame.convert("RGB")
    im.thumbnail((640,400),Image.Resampling.LANCZOS)
    return im,w,h


def target(case,w,h):
    def norm(p):
        return [round(p[0]/w*1000),round(p[1]/h*1000)]
    return {
       "type":"drag","start_point":norm(case["start"]),
       "end_point":norm(case["end"]),"ordered":case["ordered"],
       "coordinate_space":"normalized_1000"
    }


def prompt(case):
    if case["language"]=="ar":
        suffix=(" أجب فقط بكائن JSON type=drag مع start_point وend_point "
                "وordered وcoordinate_space=normalized_1000. الإحداثيات 0 إلى 1000. لا تنفذ السحب.")
    else:
        suffix=(" Return ONLY JSON {type:drag,start_point:[x,y],end_point:[x,y],"
                "ordered:boolean,coordinate_space:normalized_1000};"
                " coordinates are 0-1000. Do not act.")
    return case["body"] + "\n" + suffix


def to_inputs(processor,case,zips):
    im,w,h=source_image(case,zips)
    answer=json.dumps(target(case,w,h),separators=(",",":"),ensure_ascii=False)
    user=[{"role":"user","content":[{"type":"image"},{"type":"text","text":prompt(case)}]}]
    assistant={"role":"assistant","content":[{"type":"text","text":answer}]}
    head=processor.apply_chat_template(user,tokenize=False,
                                       add_generation_prompt=True)
    full=processor.apply_chat_template(user+[assistant],tokenize=False,
                                       add_generation_prompt=False)
    header=processor(text=[head],images=[im],return_tensors="pt")
    batch=processor(text=[full],images=[im],return_tensors="pt")
    head_len=int(header["input_ids"].shape[1])
    if head_len<=0 or head_len>=int(batch["input_ids"].shape[1]):
        raise ValueError("H417 supervised target label boundary invalid")
    labels=batch["input_ids"].clone()
    labels[:,:head_len]=-100
    if "attention_mask" in batch:
        labels[batch["attention_mask"]==0]=-100
    if (labels!=-100).sum().item()<10:
        raise ValueError("H417 native drag supervised tokens missing")
    return {k:v.to("cuda:0") for k,v in batch.items()},labels.to("cuda:0")


def coordinate_weighted_loss(output,labels,processor):
    import torch
    logits=output.logits[:,:-1,:]
    target_ids=labels[:,1:]
    selected=target_ids!=-100
    if not bool(selected.any()):
        raise ValueError("H417 empty drag loss")
    ids=target_ids[selected]
    scores=logits[selected].float()
    ce=torch.nn.functional.cross_entropy(scores,ids,reduction="none")
    weights=torch.tensor([
        8.0 if any(c.isdigit() for c in processor.tokenizer.decode([int(v)]))
        else 1.0 for v in ids.tolist()],device=ce.device,dtype=torch.float32)
    if not bool((weights>0).all()) or not bool((weights==8).any()):
        raise ValueError("H417 no supervised numeric grounding tokens")
    return (ce*weights).sum()/weights.sum()


def score_predicted(pred,target_gold):
    doc=parse_drag(pred)
    if doc is None:
        return {"valid":False,"correct":False,"error":"schema"}
    if doc["ordered"] is not target_gold["ordered"]:
        return {"valid":True,"correct":False,"error":"ordered_mismatch"}
    def near(a,b):
        return (abs(a[0]-b[0])<=TOL_NORM and
                abs(a[1]-b[1])<=TOL_NORM)
    straight=(near(doc["start_point"],target_gold["start_point"])
              and near(doc["end_point"],target_gold["end_point"]))
    reversed_hit=(near(doc["end_point"],target_gold["start_point"])
                  and near(doc["start_point"],target_gold["end_point"]))
    success=straight or (not target_gold["ordered"] and reversed_hit)
    return {"valid":True,"correct":bool(success),
            "error":None if success else "coordinates"}


def evaluate(model,processor,dev,zips,label):
    import torch
    stats={}
    records=[]
    model.eval()
    with torch.inference_mode():
        for i,case in enumerate(dev):
            im,w,h=source_image(case,zips)
            msg=[{"role":"user","content":[{"type":"image"},
                    {"type":"text","text":prompt(case)}]}]
            formatted=processor.apply_chat_template(msg,tokenize=False,
                                                    add_generation_prompt=True)
            batch=processor(text=[formatted],images=[im],
                            return_tensors="pt").to("cuda:0")
            tokens=model.generate(**batch,max_new_tokens=120,do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id)
            raw=processor.batch_decode(tokens[:,batch["input_ids"].shape[1]:],
                                       skip_special_tokens=True)[0].strip()
            score=score_predicted(raw,target(case,w,h))
            for key in ("all",case["language"],case["domain"],case["kind"]):
                x=stats.setdefault(key,{"n":0,"correct":0,"valid":0})
                x["n"]+=1;x["correct"]+=int(score["correct"])
                x["valid"]+=int(score["valid"])
            records.append({"id":case["id"],"family":case["kind"],
                  "domain":case["domain"],"language":case["language"],
                  "image_sha256":case["image_sha256"],
                  "raw_prediction":raw,**score})
            if (i+1)%24==0:
                print("H417_IMAGE_CONDITIONED_EVAL",label,i+1,"/",len(dev),flush=True)
    return stats,records


def receipt(state):
    raw=json.dumps(state,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False).encode()
    (OUT/"receipt.json").write_bytes(raw)
    (OUT/"receipt.sha256").write_text(sha_bytes(raw))


def run():
    import torch
    st={
      "schema":"hala.h417.private-12k-native-drag.v1",
      "status":"NOT_STARTED",
      "official_base":BASE,"official_revision":REV,
      "source_revision":SOURCE_REV,
      "source_curriculum_sha256":DATASET_SHA,
      "bilingual_labels_sha256":FIRSTPARTY_LABEL_SHA,
      "private_sources_only":True,
      "train_only_sources":True,
      "benchmark_answers_used":False,
      "full_upstream_tar_sha_verified":False,
      "independent_external_ood":False,
      "weights_public":False,
      "model_release_qualified":False,
      "actual_optimizer_steps":0,
    }
    if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not torch.cuda.is_available():
        raise ValueError("H417 FREE Kaggle T4 only, no paid computation")
    if OUT.exists():
        raise ValueError("H417 private output collision")
    OUT.mkdir(parents=True)
    started=time.monotonic()
    zips=None
    try:
        subprocess.run([sys.executable,"-m","pip","install","-q",
           "transformers==4.57.3","peft==0.18.0","accelerate==1.12.0",
           "Pillow==11.3.0"],check=True,timeout=330)
        from transformers import AutoProcessor,AutoModelForImageTextToText
        from peft import LoraConfig,get_peft_model
        zips,originals,curriculum,firstparty=get_research_sources()
        train,dev,order=partition(originals,curriculum,firstparty)
        st.update(train_unique_images=len(train),train_example_passes=len(order),
                  dev_source_disjoint_images=len(dev),
                  dev_languages={lang:sum(c["language"]==lang for c in dev)
                                 for lang in ("en","ar")},
                  train_languages={lang:sum(c["language"]==lang for c in train)
                                   for lang in ("en","ar")})
        if len(train)!=TRAIN_UNIQUE or len(order)!=EXPECTED_PASSES:
            raise ValueError("H417 true 12K source size incomplete")
        torch.manual_seed(SEED)
        proc=AutoProcessor.from_pretrained(BASE,revision=REV,
                                            trust_remote_code=False)
        pretrained=AutoModelForImageTextToText.from_pretrained(
            BASE,revision=REV,dtype=torch.float16,trust_remote_code=False,
            attn_implementation="sdpa").to("cuda:0")
        st["base_dev"],st["base_dev_exact"]=evaluate(
            pretrained,proc,dev,zips,"FROZEN_BASE")
        model=get_peft_model(pretrained,LoraConfig(
            r=8,lora_alpha=16,lora_dropout=0.0,bias="none",
            target_modules=["q_proj","k_proj","v_proj","o_proj"],
            task_type="CAUSAL_LM"))
        active=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if not active or any("lora_" not in n for n,p in active):
            raise ValueError("H417 only dedicated LoRA tensors trainable")
        before={n:p.detach().float().cpu().clone() for n,p in active}
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        if hasattr(model.config,"use_cache"):model.config.use_cache=False
        optim=torch.optim.AdamW([p for _,p in active],lr=0.000018)
        losses=[]
        for step in range(STEPS):
            model.train();optim.zero_grad(set_to_none=True)
            micros=[]
            for micro in range(GRAD_ACCUM):
                c=train[order[step*GRAD_ACCUM+micro]]
                batch,labels=to_inputs(proc,c,zips)
                labels=batch.pop("labels",labels)
                out=model(**batch)
                loss=coordinate_weighted_loss(out,labels,proc)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("H417 nonfinite weighted gradient loss")
                (loss/GRAD_ACCUM).backward()
                micros.append(float(loss.detach().cpu()))
                del out,batch,loss,labels
            torch.nn.utils.clip_grad_norm_([p for _,p in active],1.0)
            optim.step()
            losses.append(round(sum(micros)/len(micros),6))
            st["actual_optimizer_steps"]=step+1
            if (step+1)%100==0:
                print("H417_GENUINE_NATIVE_DRAG_GRADIENT",step+1,
                      "examples_seen",(step+1)*GRAD_ACCUM,
                      "weighted_loss",losses[-1],flush=True)
            if (step+1)%600==0:
                path=OUT/"private-checkpoints"/("step-"+str(step+1))
                model.save_pretrained(path,safe_serialization=True)
                st["last_private_checkpoint_step"]=step+1
        st["changed_lora_tensors"]=sum(int(bool(
            (p.detach().float().cpu()!=before[n]).any())) for n,p in active)
        if st["changed_lora_tensors"]<1:
            raise ValueError("H417 no actual trained LoRA weight change")
        model.gradient_checkpointing_disable()
        if hasattr(model.config,"use_cache"):model.config.use_cache=True
        st["adapted_dev"],st["adapted_dev_exact"]=evaluate(
            model,proc,dev,zips,"NATIVE_DRAG_ADAPTED")
        target=OUT/"private-native-drag-adapter"
        model.save_pretrained(target,safe_serialization=True)
        f=target/"adapter_model.safetensors"
        if not f.exists() or f.stat().st_size<1000:
            raise ValueError("H417 real private model checkpoint missing")
        st.update(status="COMPLETE_PRIVATE_NATIVE_DRAG_14880_PASSES",
             processed_training_examples=EXPECTED_PASSES,
             unique_train_images=TRAIN_UNIQUE,
             adapter_sha256=sha_file(f),adapter_bytes=f.stat().st_size,
             weighted_train_losses=losses,completed_steps=STEPS,
             device_name=torch.cuda.get_device_name(0),
             elapsed_seconds=round(time.monotonic()-started,2),
             source_only_dev_improved=(
               st["adapted_dev"]["all"]["correct"] >
               st["base_dev"]["all"]["correct"]))
    except Exception as e:
        st["status"]="FAILED_PRIVATE_NATIVE_DRAG_NO_RELEASE"
        st["failure_class"]=type(e).__name__
        raise
    finally:
        if zips:
            for z in zips.values():
                z.close()
        receipt(st)
        print("H417_PRIVATE_NATIVE_DRAG_SUMMARY",
              json.dumps({k:v for k,v in st.items()
                          if not k.endswith("_exact") and k!="weighted_train_losses"},
                         sort_keys=True),flush=True)


if __name__=="__main__":
    run()
