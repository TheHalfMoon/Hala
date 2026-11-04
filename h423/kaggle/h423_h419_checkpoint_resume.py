"""H419 private continuation of REAL H417 LoRA on 30K NEW original drags.

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
OUT = Path("/kaggle/working/h419-owner-private-native-drag")
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
    # H61 frozen selection has 10k original *identifiers* but four pairs of
    # pixel-identical slider images. Only 9996 unique JPEG source bytes.
    original = [case_from_dragon(r) for r in cur]
    chosen = {}
    for row in original:
        chosen.setdefault(row["image_sha256"], row)
    if len(original) != 10000 or len(original)-len(chosen) != 4:
        raise ValueError("H417 frozen four same-pixel collisions changed")

    development=[]
    dev_hashes=set()
    # The selected sheet shard has EXACTLY 2361 distinct images and the
    # original H61 curriculum selected ALL 2361. To obtain a truly
    # disjoint sheet DEV set, deliberately withhold 24 source images from
    # this selected pool rather than inventing an unseen sheet holdout.
    # For other domains, prefer unused original source images.
    for dom in DOMAINS:
        candidates = sorted(originals[dom],
                 key=lambda row: sha_bytes(row["id"].encode("utf8")))
        count=0
        for raw in candidates:
            digest=raw["image_sha256"]
            if digest in chosen or digest in dev_hashes:
                continue
            dev=case_from_dragon(raw)
            development.append(dev)
            dev_hashes.add(digest)
            count+=1
            if count==DEV_DRAGON_PER_DOMAIN:
                break
        if count < DEV_DRAGON_PER_DOMAIN:
            # Non-independent, but image-disjoint source holdout from
            # existing curated TRAIN. Remove it BEFORE selecting new TRAIN.
            candidates_in_train=sorted(
                (v for v in chosen.values() if v["domain"]==dom),
                key=lambda v: sha_bytes(v["id"].encode("utf8")))
            for dev in candidates_in_train:
                digest=dev["image_sha256"]
                if digest in dev_hashes:
                    continue
                development.append(dev)
                dev_hashes.add(digest)
                del chosen[digest]
                count+=1
                if count==DEV_DRAGON_PER_DOMAIN:
                    break
        if count!=DEV_DRAGON_PER_DOMAIN:
            raise ValueError("H417 insufficient original source disjoint DEV")

    # We dropped four duplicate original pixels and reserved 24 source
    # sheet images, so refill exactly 28 unique originals from the same
    # already-authenticated owner-private slider TRAIN shard. No DEV
    # fingerprint can be admitted as TRAIN.
    repairs=[]
    for raw in sorted(originals["slider"],
                      key=lambda row: sha_bytes(row["id"].encode("utf8"))):
        if len(chosen)==N_SOURCE_DRAGON:
            break
        digest=raw["image_sha256"]
        if digest in chosen or digest in dev_hashes:
            continue
        replacement=case_from_dragon(raw)
        replacement["source_repair"]=True
        chosen[digest]=replacement
        repairs.append(replacement["id"])
    if len(chosen)!=N_SOURCE_DRAGON or len(repairs)!=28:
        raise ValueError("H417 28 original source holdout/collision repairs unavailable")
    training=list(chosen.values())
    source_ids=set(chosen)
    if dev_hashes & source_ids:
        raise ValueError("H417 original TRAIN/DEV pixel overlap")

    from_fp = [case_from_firstparty(row) for row in firstparty]
    if len({r["image_sha256"] for r in from_fp})!=N_SOURCE_FIRSTPARTY:
        raise ValueError("H417 repeated original Edge screenshot")
    withheld=set()
    for lang in ("en","ar"):
        candidates=sorted((r for r in from_fp if r["language"]==lang),
                          key=lambda row: sha_bytes(row["id"].encode("utf8")))
        holdout=candidates[:DEV_FIRSTPARTY_PER_LANG]
        if len(holdout)!=DEV_FIRSTPARTY_PER_LANG:
            raise ValueError("H417 insufficient original first-party bilingual DEV")
        development.extend(holdout)
        withheld.update(r["image_sha256"] for r in holdout)
    training.extend(r for r in from_fp if r["image_sha256"] not in withheld)
    if len(training)!=TRAIN_UNIQUE or len(development)!=144:
        raise ValueError("H417 exact original source train/development size drift")
    if source_ids & {r["image_sha256"] for r in from_fp}:
        raise ValueError("H417 source family fingerprint collision")
    if len({r["image_sha256"] for r in training})!=TRAIN_UNIQUE:
        raise ValueError("H417 source TRAIN image-level uniqueness violated")
    if {r["image_sha256"] for r in training}&{
             r["image_sha256"] for r in development}:
        raise ValueError("H417 original TRAIN/DEV source pixels overlap")
    n_ar=sum(r["language"]=="ar" for r in training)
    n_en=sum(r["language"]=="en" for r in training)
    if (n_ar,n_en)!=(976,10976):
        raise ValueError("H417 original bilingual TRAIN source composition drift")
    order=list(range(len(training)))
    order += [i for i,r in enumerate(training) if r["language"]=="ar"]*AR_REPEAT
    if len(order)!=EXPECTED_PASSES or len(order)%GRAD_ACCUM:
        raise ValueError("H417 actual gradient-bearing 14,880 passes unavailable")
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



# H419 original source contract: all nine H60 TRAIN ZIPs are SHA-pinned.
H419_EXTRA_SHAS={
    "h60-dragon-train-slide_resize-0001.hala.bin":
      "6e5ad052a460fe23eb865a15fbf83694d248970c3eb158118bb83feff4c696db",
    "h60-dragon-train-text_highlight-0001.hala.bin":
      "c07191900e1a4f17f65795381a57b12d91071d4dc580b87a6e4875293f5160af",
    "h60-dragon-train-text_highlight-0002.hala.bin":
      "ae07e67f085d1749f19747af25fc3571768cc28bfda1e4ebf980ec0f207a3005",
    "h60-dragon-train-text_highlight-0003.hala.bin":
      "4170b4687e462c045ecbfb7198324110a3db54d5b2b774b3fdb6bdbce3586b74",
    "h60-dragon-train-text_highlight-0004.hala.bin":
      "c4824f510cb9d27bec60bed536aba81fd3fdfb50ee827c9e2a121f730a16d782",
}
H419_ALL_SOURCE_SHAS={n:d for n,d in SOURCE_SHAS.values()}
H419_ALL_SOURCE_SHAS.update(H419_EXTRA_SHAS)
H418_ROWS="h418_fresh_original_30k_drag.jsonl"
H418_ROWS_SHA="e7fce082d6ccd931515beacbc6cfc2455a6683c7c92566a245fda9f2182a1605"
H418_RECEIPT_FILE="h418_fresh_original_30k_manifest.json"
H418_RECEIPT_SHA="b7b75d6aa2587c3eba83a3934fe4cdf3cacd48a5b970b5a60a640c89ee3054cf"
PARENT_RECEIPT_FILE="h417_original_receipt.json"
PARENT_RECEIPT_SHA="227967230362eea759553ae5af892865bae4a3625947c45f30c0d9cb36f9f76a"
PARENT_ADAPTER_FILE="adapter_model.safetensors"
PARENT_ADAPTER_SHA="11f06aac6c225b86d1f30272eb852bca49fd82ce6e8553113b39b4cba4216df1"
H419_NEW_IMAGE_N=30000
H419_AR_N=976
H419_AR_ADDITIONAL_PASSES=7
H419_UNIQUE_IMAGES=H419_NEW_IMAGE_N+H419_AR_N
H419_TRAINING_PASSES=H419_UNIQUE_IMAGES+H419_AR_N*H419_AR_ADDITIONAL_PASSES
H419_STEPS=H419_TRAINING_PASSES//GRAD_ACCUM
H419_NEW_DEV_PER_DOMAIN=32
H419_NEW_DEV_N=3*H419_NEW_DEV_PER_DOMAIN
H419_OLD_DEV_N=144
H419_DEV_N=H419_NEW_DEV_N+H419_OLD_DEV_N
H419_SEED=41919
H419_PRIVATE_STATUS="COMPLETE_PRIVATE_30000_NEW_ORIGINAL_NATIVE_DRAG"
H419_FAILURE_STATUS="FAILED_PRIVATE_30000_NATIVE_DRAG_NO_RELEASE"
assert (H419_NEW_IMAGE_N,H419_UNIQUE_IMAGES,H419_TRAINING_PASSES,H419_STEPS)==(30000,30976,37808,9452)


# Keep the original source SHA-aware and native-drag implementations.
h417_source_image=source_image


def source_image(case,zips):
    """Original H60 ZIP SHA checked at actual training/evaluation decode."""
    if "source_zip" not in case:
        return h417_source_image(case,zips)
    from PIL import Image
    filename=case["source_zip"]
    if filename not in H419_ALL_SOURCE_SHAS:
        raise ValueError("H419 original source ZIP outside verified allowlist")
    source=zips.get("_by_file")
    if type(source) is not dict or filename not in source:
        raise ValueError("H419 source ZIP not opened with private original SHA proof")
    raw=source[filename].read(case["image_path"])
    if sha_bytes(raw)!=case["image_sha256"]:
        raise ValueError("H419 selected ORIGINAL JPEG SHA mismatch")
    with Image.open(io.BytesIO(raw)) as frame:
        if frame.format!="JPEG":
            raise ValueError("H419 source is not ORIGINAL JPEG")
        w,h=frame.size
        if ("image_width" in case and
            (case["image_width"],case["image_height"])!=(w,h)):
            raise ValueError("H419 native original image dimensions altered")
        if not all(0<=v[0]<=w and 0<=v[1]<=h
                   for v in (case["start"],case["end"])):
            raise ValueError("H419 start/end outside source ORIGINAL JPEG")
        im=frame.convert("RGB")
    im.thumbnail((640,400),Image.Resampling.LANCZOS)
    return im,w,h


def parent_model_source():
    path=only_input_file(PARENT_ADAPTER_FILE)
    if sha_file(path)!=PARENT_ADAPTER_SHA:
        raise ValueError("H419 real H417 private parent LoRA SHA mismatch")
    conf=path.parent/"adapter_config.json"
    if not conf.is_file():
        raise ValueError("H419 parent LoRA config not bundled")
    receipt_path=only_input_file(PARENT_RECEIPT_FILE)
    raw=receipt_path.read_bytes()
    if sha_bytes(raw)!=PARENT_RECEIPT_SHA:
        raise ValueError("H419 H417 original 3720-step parent receipt SHA mismatch")
    sidecar=only_input_file("h417_original_receipt.sha256")
    if sidecar.read_text("ascii").strip()!=PARENT_RECEIPT_SHA:
        raise ValueError("H419 original parent training receipt sidecar mismatch")
    receipt=strict_load(raw.decode("utf8"))
    if (receipt.get("status")!="COMPLETE_PRIVATE_NATIVE_DRAG_14880_PASSES" or
        receipt.get("official_base")!=BASE or
        receipt.get("official_revision")!=REV or
        receipt.get("actual_optimizer_steps")!=3720 or
        receipt.get("processed_training_examples")!=14880 or
        receipt.get("train_unique_images")!=11952 or
        receipt.get("adapter_sha256")!=PARENT_ADAPTER_SHA or
        receipt.get("independent_external_ood") is not False or
        receipt.get("model_release_qualified") is not False):
        raise ValueError("H419 parent native drag trained-source lineage invalid")
    return path


def open_extra_sources(zips):
    byfile={name:zips[domain] for domain,(name,_) in SOURCE_SHAS.items()}
    extra=[]
    for filename,expected_sha in H419_EXTRA_SHAS.items():
        path=only_input_file(filename)
        if sha_file(path)!=expected_sha:
            raise ValueError("H419 new genuine original H60 TRAIN ZIP SHA changed")
        archive=zipfile.ZipFile(path)
        extra.append(archive)
        manifest=strict_load(archive.read("manifest.json").decode("utf8"))
        domain=filename[len("h60-dragon-train-"):].rsplit("-",1)[0]
        if (manifest.get("schema")!="hala.h60.dragon-train-shard.v1" or
            manifest.get("source_commit_sha")!=SOURCE_REV or
            manifest.get("source_domain")!=domain or
            manifest.get("split")!="train" or
            manifest.get("source_license")!="Apache-2.0" or
            manifest.get("official_eval_or_test_used") is not False or
            manifest.get("benchmark_answer_leakage") is not False):
            raise ValueError("H419 new original H60 manifest NOT qualified as TRAIN")
        labels=archive.read("labels.jsonl")
        if sha_bytes(labels)!=manifest.get("labels_sha256"):
            raise ValueError("H419 original H60 TRAIN native labels byte SHA drift")
        byfile[filename]=archive
    if len(byfile)!=9:
        raise ValueError("H419 missing original 9 TRAIN ZIPs")
    zips["_by_file"]=byfile
    return extra


def case_from_new_original(row):
    if (type(row) is not dict or
        row.get("schema")!="hala.h418.original-native-drag-train.v1" or
        row.get("split")!="train" or
        row.get("language")!="en" or
        row.get("source_domain") not in
                ("slider","slide_resize","text_highlight") or
        row.get("source_zip") not in H419_ALL_SOURCE_SHAS or
        row.get("source_zip_sha256")!=H419_ALL_SOURCE_SHAS[row["source_zip"]] or
        type(row.get("image_sha256")) is not str or
        len(row["image_sha256"])!=64 or
        type(row.get("image_width")) is not int or
        type(row.get("image_height")) is not int or
        row["image_width"]<=0 or row["image_height"]<=0 or
        type(row.get("original_source_task_sha256")) is not str or
        len(row["original_source_task_sha256"])!=64):
        raise ValueError("H419 source-original 30k TRAIN row provenance changed")
    if type(row.get("native_drag_completion")) is not dict:
        raise ValueError("H419 native drag original training target missing")
    original={
      "id":row["id"],"split":"train","language":"en",
      "image_path":row["image_path"],
      "image_sha256":row["image_sha256"],
      "intent":row["intent"],"completion":row["native_drag_completion"],
      "ordered":row["ordered"],
    }
    case=case_from_dragon(original)
    if case["domain"]!=row["source_domain"]:
        raise ValueError("H419 original image source domain shifted")
    for field in ("start","end"):
        p=case[field]
        if not (0<=p[0]<=row["image_width"] and
                0<=p[1]<=row["image_height"]):
            raise ValueError("H419 original native endpoint outside raw JPEG")
    case.update(source_zip=row["source_zip"],
                image_width=row["image_width"],
                image_height=row["image_height"])
    return case


def new_verified_images(original_train,original_dev):
    rows_path=only_input_file(H418_ROWS)
    if sha_file(rows_path)!=H418_ROWS_SHA:
        raise ValueError("H419 genuine new original 30k source rows SHA mismatch")
    receipt_path=only_input_file(H418_RECEIPT_FILE)
    if sha_file(receipt_path)!=H418_RECEIPT_SHA:
        raise ValueError("H419 true H418 private source receipt SHA mismatch")
    manifest=strict_load(receipt_path.read_text("utf8"))
    if (manifest.get("schema")!="hala.h418.owner-private-30k-fresh-original-drags.v1" or
        manifest.get("status")!="ORIGINAL_SOURCE_BYTES_VERIFIED_MODEL_NOT_TRAINED" or
        manifest.get("source_revision")!=SOURCE_REV or
        manifest.get("source_split")!="train" or
        manifest.get("original_9_archive_sha256")!=H419_ALL_SOURCE_SHAS or
        manifest.get("selected_unique_original_train_images")!=H419_NEW_IMAGE_N or
        manifest.get("selected_training_rows")!=H419_NEW_IMAGE_N or
        manifest.get("selected_bytes_sha256")!=H418_ROWS_SHA or
        manifest.get("actual_all_30000_original_jpeg_bytes_sha_verified") is not True or
        manifest.get("h417_train_image_sha_disjoint") is not True or
        manifest.get("h417_dev144_image_sha_disjoint") is not True or
        manifest.get("model_training_executed") is not False or
        manifest.get("external_independent_ood") is not False):
        raise ValueError("H419 source receipt cannot prove disjoint original 30k")
    cases=[case_from_new_original(strict_load(line))
            for line in rows_path.read_text("utf8").splitlines() if line.strip()]
    shas={r["image_sha256"] for r in cases}
    ids={r["id"] for r in cases}
    if (len(cases)!=H419_NEW_IMAGE_N or len(shas)!=H419_NEW_IMAGE_N or
        len(ids)!=H419_NEW_IMAGE_N or
        shas&original_train or shas&original_dev):
        raise ValueError("H419 genuine new 30k source includes prior H417 TRAIN/DEV")
    domain_count={d:sum(c["domain"]==d for c in cases)
                  for d in ("slider","slide_resize","text_highlight")}
    if domain_count!={d:10000 for d in domain_count}:
        raise ValueError("H419 10K per original source domain expected")
    return cases


def new_source_development(zips,already_used):
    """96 NEW source images not used in H417 or H418 actual TRAIN.

    These remain author-visible same-source research. Not external OOD.
    H417 original sheet domain is exhausted, so new groups are three
    original source domains with 32 unique source images each.
    """
    byfile=zips.get("_by_file")
    if type(byfile) is not dict or len(byfile)!=9:
        raise ValueError("H419 missing true original source ZIP registry")
    domain_seen={domain:{} for domain in
                 ("slider","slide_resize","text_highlight")}
    for filename,archive in byfile.items():
        domain=strict_load(archive.read("manifest.json").decode("utf8"))["source_domain"]
        if domain not in domain_seen:continue
        for line in archive.read("labels.jsonl").decode("utf8").splitlines():
            if not line.strip():continue
            raw=strict_load(line)
            digest=raw.get("image_sha256")
            if digest in already_used:continue
            case=case_from_dragon(raw)
            case["source_zip"]=filename
            rank=sha_bytes((str(H419_SEED)+"|"+case["id"]).encode("utf8"))
            prior=domain_seen[domain].get(digest)
            if prior is None or rank<prior[0]:
                domain_seen[domain][digest]=(rank,case)
    dev=[]
    for domain,shas in domain_seen.items():
        ranked=sorted(shas.values(),key=lambda x:x[0])
        if len(ranked)<H419_NEW_DEV_PER_DOMAIN:
            raise ValueError("H419 not enough genuinely NEW image-disjoint source DEV")
        dev.extend(case for _,case in ranked[:H419_NEW_DEV_PER_DOMAIN])
    if (len(dev)!=H419_NEW_DEV_N or
        len({r["image_sha256"] for r in dev})!=H419_NEW_DEV_N or
        {r["image_sha256"] for r in dev}&already_used):
        raise ValueError("H419 fresh source dev not source-image-disjoint")
    return dev


def h419_prepare_sources():
    zips,original,curr,firstparty=get_research_sources()
    extra=[]
    try:
        old_train,old_dev,old_order=partition(original,curr,firstparty)
        if (len(old_train),len(old_dev),len(old_order))!=(11952,144,14880):
            raise ValueError("H419 prior original H417 exact source frozen partition shifted")
        prior_train={c["image_sha256"] for c in old_train}
        prior_dev={c["image_sha256"] for c in old_dev}
        extra=open_extra_sources(zips)
        new_train=new_verified_images(prior_train,prior_dev)
        new_shas={c["image_sha256"] for c in new_train}
        fresh_dev=new_source_development(zips,prior_train|prior_dev|new_shas)
        ar=[r for r in old_train if r["language"]=="ar"]
        if (len(ar)!=H419_AR_N or
            len({c["image_sha256"] for c in ar})!=H419_AR_N):
            raise ValueError("H419 original browser Arabic TRAIN preservation failed")
        train=new_train+ar
        old_h417_dev=old_dev
        dev=old_h417_dev+fresh_dev
        train_sha={r["image_sha256"] for r in train}
        dev_sha={r["image_sha256"] for r in dev}
        if (len(train)!=H419_UNIQUE_IMAGES or
            len(train_sha)!=H419_UNIQUE_IMAGES or
            len(dev)!=H419_DEV_N or
            len(dev_sha)!=H419_DEV_N or
            train_sha&dev_sha):
            raise ValueError("H419 actual 30K source-new + 976 AR TRAIN/240 DEV partition drift")
        order=list(range(len(train)))
        order += [i for i,r in enumerate(train) if r["language"]=="ar"]*H419_AR_ADDITIONAL_PASSES
        if (len(order)!=H419_TRAINING_PASSES or
            len(order)%GRAD_ACCUM or len(train)!=H419_UNIQUE_IMAGES):
            raise ValueError("H419 planned real gradient-bearing examples incomplete")
        random.Random(H419_SEED).shuffle(order)
        return zips,extra,train,dev,order,old_h417_dev,fresh_dev
    except Exception:
        for z in extra:z.close()
        for value in zips.values():
            if isinstance(value,zipfile.ZipFile):value.close()
        raise


def native_summary(stats):
    return {
      "all":stats["all"],
      "en":stats["en"],
      "ar":stats.get("ar",{"n":0,"correct":0,"valid":0}),
      "original":stats["original"],
      "firstparty":stats["firstparty"],
      "slider":stats["slider"],
      "slide_resize":stats["slide_resize"],
      "text_highlight":stats["text_highlight"],
    }



import re
import struct
from hashlib import sha256

H419_KERNEL="abdulazizshehri/hala-h419-private-30k-native-drag-continuation"
H419_TOTAL=9452
H419_MICRO=4
H419_INTERVAL=1500
H419_SOURCE_ROWS_SHA="e7fce082d6ccd931515beacbc6cfc2455a6683c7c92566a245fda9f2182a1605"
H419_SOURCE_RECEIPT_SHA="b7b75d6aa2587c3eba83a3934fe4cdf3cacd48a5b970b5a60a640c89ee3054cf"
H417_TRAINED_PARENT_SHA="11f06aac6c225b86d1f30272eb852bca49fd82ce6e8553113b39b4cba4216df1"
H417_PARENT_RECEIPT_SHA="227967230362eea759553ae5af892865bae4a3625947c45f30c0d9cb36f9f76a"
BASE="Qwen/Qwen3-VL-2B-Instruct"
BASE_REV="89644892e4d85e24eaac8bacfd4f463576704203"
OUT_SCHEMA="hala.h423.interrupted-h419-actual-private-checkpoint-qualification.v1"
CHECKPOINT_LOG_RE=re.compile(
    r"H419_REAL_ORIGINAL_30000_NATIVE_GRADIENT\s+(\d+)\s*/\s*9452\s+"
    r"image_conditioned_samples_seen\s+(\d+)\s+loss\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


def file_sha(path):
    d=sha256()
    with Path(path).open("rb") as src:
        for piece in iter(lambda:src.read(1024*1024),b""):
            d.update(piece)
    return d.hexdigest()


def exact_json(value):
    def no_dupe(items):
        obj={}
        for k,v in items:
            if k in obj:raise ValueError("H423 duplicate JSON keys in original model source")
            obj[k]=v
        return obj
    def finite_only(x):
        raise ValueError("H423 nonfinite original loss")
    return json.loads(value,object_pairs_hook=no_dupe,parse_constant=finite_only)


def bounded_step(step):
    if type(step) is not int or step not in range(H419_INTERVAL,H419_TOTAL,H419_INTERVAL):
        raise ValueError("H423 only original 1500-step private checkpoint boundaries accepted")
    return step


def actual_log_progress(raw,checkpoint_step):
    """Require CONTIGUOUS real original 150-step GPU progress through boundary."""
    bounded_step(checkpoint_step)
    if type(raw) is bytes:raw=raw.decode("utf8",errors="replace")
    if type(raw) is not str or len(raw)>8_000_000:
        raise ValueError("H423 original Kaggle provider run log absent or too large")
    observed={}
    for m in CHECKPOINT_LOG_RE.finditer(raw):
        step=int(m.group(1));examples=int(m.group(2))
        if (step%150 or step<=0 or step>H419_TOTAL
            or examples!=step*H419_MICRO):
            raise ValueError("H423 original provider log inconsistent GPU step/examples")
        loss=float(m.group(3))
        if not (0<=loss<1_000_000):
            raise ValueError("H423 original provider log has invalid native DRAG loss")
        if step in observed:
            raise ValueError("H423 original Kaggle progress row duplicated")
        observed[step]=loss
    wanted=set(range(150,checkpoint_step+1,150))
    if not wanted.issubset(observed):
        missing=sorted(wanted-set(observed))
        raise ValueError("H423 original provider log has not independently shown all steps: "+str(missing[:6]))
    return {"last_logged_step":max(observed),"verified_checkpoint_step":checkpoint_step,
            "log_150_step_records":len(observed),
            "loss_at_verified_boundary":observed[checkpoint_step]}


def safe_adapter_headers(path):
    """Inspect adapter's real immutable safetensors layout without loading weights."""
    path=Path(path)
    size=path.stat().st_size
    if size<100_000 or size>100_000_000:
        raise ValueError("H423 checkpoint adapter safetensors filesize implausible")
    with path.open("rb") as f:
        prefix=f.read(8)
        if len(prefix)!=8:
            raise ValueError("H423 missing original safetensors header")
        n=struct.unpack("<Q",prefix)[0]
        if not 32<=n<=2_000_000 or size<8+n:
            raise ValueError("H423 original model safetensors header bounds invalid")
        header=exact_json(f.read(n).decode("utf8"))
    if type(header) is not dict:
        raise ValueError("H423 original model adapter not a tensor dictionary")
    entries={k:v for k,v in header.items() if k!="__metadata__"}
    if len(entries)<16 or len(entries)>4096:
        raise ValueError("H423 original LoRA tensor count implausible")
    intervals=[]
    for k,v in entries.items():
        if "lora_" not in k or type(v) is not dict or set(v)!={"dtype","shape","data_offsets"}:
            raise ValueError("H423 original model has non-LoRA or malformed tensor")
        dims=v["shape"]
        offsets=v["data_offsets"]
        if (v["dtype"] not in ("F16","F32","BF16")
            or type(dims) is not list or not dims
            or any(type(z) is not int or z<=0 for z in dims)
            or type(offsets) is not list or len(offsets)!=2
            or any(type(z) is not int for z in offsets)
            or not 0<=offsets[0]<offsets[1]<=size-8-n):
            raise ValueError("H423 original model safetensors offsets/dtype invalid")
        bytes_per=2 if v["dtype"]!="F32" else 4
        cells=1
        for dim in dims:cells*=dim
        if cells*bytes_per!=offsets[1]-offsets[0]:
            raise ValueError("H423 original tensor byte length inconsistent with shape")
        intervals.append(tuple(offsets))
    if sorted(intervals)!=[(a,b) for a,b in sorted(intervals)]:
        raise ValueError("H423 unexpected safetensors interval sorting")
    intervals.sort()
    if intervals[0][0]!=0 or intervals[-1][1]!=size-8-n:
        raise ValueError("H423 safetensors not fully byte-covered by valid tensors")
    for before,after in zip(intervals,intervals[1:]):
        if before[1]!=after[0]:
            raise ValueError("H423 overlapping or missing original adapter tensor bytes")
    return {"real_lora_tensor_count":len(entries),
            "real_lora_bytes":size,"exact_safetensors_tensors_cover_all_bytes":True}


def qualify(checkpoint_dir,original_provider_log,source_version):
    checkpoint_dir=Path(checkpoint_dir)
    if type(source_version) is not str or not re.fullmatch(r"\d+",source_version) or int(source_version)<1:
        raise ValueError("H423 exact original private Kaggle source version required")
    path=checkpoint_dir/"adapter_model.safetensors"
    cfg=checkpoint_dir/"adapter_config.json"
    if path.parent.name not in ("step-1500","step-3000","step-4500","step-6000","step-7500","step-9000"):
        raise ValueError("H423 original H419 checkpoint folder is not from 1500-step evidence")
    step=bounded_step(int(path.parent.name.removeprefix("step-")))
    if not path.is_file() or not cfg.is_file():
        raise ValueError("H423 original Kaggle provider LoRA weights/config missing")
    adapter=safe_adapter_headers(path)
    c=exact_json(cfg.read_text("utf8"))
    if (type(c) is not dict or c.get("peft_type")!="LORA"
        or c.get("base_model_name_or_path")!=BASE
        or type(c.get("r")) is not int or not 1<=c["r"]<=256):
        raise ValueError("H423 original LoRA config not exact qualified Qwen base")
    logs=Path(original_provider_log)
    progress=actual_log_progress(logs.read_bytes(),step)
    return {
       "schema":OUT_SCHEMA,
       "status":"PARTIAL_H419_PRIVATE_CHECKPOINT_ONLY_NOT_FULLY_TRAINED",
       "source_notebook_slug":H419_KERNEL,
       "source_notebook_version":source_version,
       "original_log_sha256":file_sha(logs),
       "original_model_checkpoint_step":step,
       "verified_original_source_gradient_examples":step*H419_MICRO,
       "remaining_optimizer_steps_to_original_h419_plan":H419_TOTAL-step,
       "remaining_grad_examples_to_original_plan":(H419_TOTAL-step)*H419_MICRO,
       "exact_shuffled_order_resume_offset":step*H419_MICRO,
       "checkpoint_adapter_model_sha256":file_sha(path),
       "checkpoint_adapter_config_sha256":file_sha(cfg),
       "original_source_30k_rows_sha256":H419_SOURCE_ROWS_SHA,
       "original_source_30k_receipt_sha256":H419_SOURCE_RECEIPT_SHA,
       "h417_real_parent_adapter_sha256":H417_TRAINED_PARENT_SHA,
       "h417_real_parent_receipt_sha256":H417_PARENT_RECEIPT_SHA,
       "base_model":BASE,
       "base_revision":BASE_REV,
       **adapter,**progress,
       "optimizer_state_saved_in_original_h419":False,
       "optimizer_reset_required_for_resumed_training":True,
       "uninterrupted_9452_step_run":False,
       "source_only_original_heldout_evaluation_completed":False,
       "genuine_independent_external_ood":False,
       "original_source_rights_independently_signed":False,
       "model_release_qualified":False,
       "weights_public":False,
    }


def store(receipt_root,doc):
    root=Path(receipt_root);root.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(doc,sort_keys=True,indent=2,allow_nan=False).encode("utf8")
    rec=root/"receipt.json";sha_file=root/"receipt.sha256"
    if rec.exists() or sha_file.exists():
        raise FileExistsError("H423 original interrupted private checkpoint receipt CREATE ONLY")
    try:
        with rec.open("xb") as f:f.write(payload)
        with sha_file.open("x",encoding="ascii",newline="\n") as f:f.write(sha256(payload).hexdigest())
    except BaseException:
        rec.unlink(missing_ok=True);sha_file.unlink(missing_ok=True)
        raise
    return sha256(payload).hexdigest()



# H423 standalone private Kaggle rescue. NO full-model or external OOD claim.
# There is intentionally no optimizer state in the original H419 saved
# LoRA-only checkpoint. Resume uses FRESH AdamW and MUST say so.
import shutil

OUT=Path("/kaggle/working/h423-owner-private-interrupted-native-drag")
PARTIAL_ADAPTER_FILE="h419_private_partial_adapter.safetensors"
PARTIAL_CONFIG_FILE="h419_private_partial_adapter_config.json"
PARTIAL_GPU_LOG="h419_original_provider.log"
PARTIAL_EVIDENCE="h423_original_checkpoint_qualification.json"
PARTIAL_EVIDENCE_SHA="h423_original_checkpoint_qualification.sha256"
H423_T4_SEGMENT_STEPS=1500
H423_STEP_STATUS="COMPLETE_H423_PRIVATE_SOURCE_ONLY_RESET_ADAMW_SEGMENT"
H423_STEP_FAILURE="FAILED_H423_PRIVATE_SOURCE_ONLY_RESUME_NO_RELEASE"


def exact_real_original_checkpoint():
    source=only_input_file(PARTIAL_ADAPTER_FILE)
    config=only_input_file(PARTIAL_CONFIG_FILE)
    log=only_input_file(PARTIAL_GPU_LOG)
    owner_receipt=only_input_file(PARTIAL_EVIDENCE)
    owner_sidecar=only_input_file(PARTIAL_EVIDENCE_SHA)
    if owner_sidecar.read_text("ascii").strip()!=sha_file(owner_receipt):
        raise ValueError("H423 original source-only checkpoint receipt sidecar mismatch")
    saved=strict_load(owner_receipt.read_text("utf8"))
    if (saved.get("schema")!=OUT_SCHEMA or
        saved.get("status")!="PARTIAL_H419_PRIVATE_CHECKPOINT_ONLY_NOT_FULLY_TRAINED"
        or saved.get("source_notebook_slug")!=H419_KERNEL
        or saved.get("original_source_30k_rows_sha256")!=H419_SOURCE_ROWS_SHA
        or saved.get("original_source_30k_receipt_sha256")!=H419_SOURCE_RECEIPT_SHA
        or saved.get("h417_real_parent_adapter_sha256")!=PARENT_ADAPTER_SHA
        or saved.get("h417_real_parent_receipt_sha256")!=PARENT_RECEIPT_SHA
        or saved.get("base_model")!=BASE
        or saved.get("base_revision")!=REV
        or saved.get("optimizer_state_saved_in_original_h419") is not False
        or saved.get("optimizer_reset_required_for_resumed_training") is not True
        or saved.get("uninterrupted_9452_step_run") is not False
        or saved.get("source_only_original_heldout_evaluation_completed") is not False
        or saved.get("model_release_qualified") is not False
        or saved.get("genuine_independent_external_ood") is not False
        or saved.get("weights_public") is not False):
        raise ValueError("H423 source-qualified original Kaggle partial checkpoint invalid")
    step=bounded_step(saved["original_model_checkpoint_step"])
    if (sha_file(source)!=saved["checkpoint_adapter_model_sha256"] or
        sha_file(config)!=saved["checkpoint_adapter_config_sha256"] or
        sha_file(log)!=saved["original_log_sha256"]):
        raise ValueError("H423 original checkpoint bytes/config/GPU progress log SHA drift")
    parent_dir=OUT/"owner-original-checkpoint-input"/("step-"+str(step))
    parent_dir.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(source,parent_dir/"adapter_model.safetensors")
    shutil.copyfile(config,parent_dir/"adapter_config.json")
    recomputed=qualify(parent_dir,log,saved["source_notebook_version"])
    if recomputed!=saved:
        raise ValueError("H423 original interrupted T4 log and actual saved LoRA receipt forged")
    return saved,owner_sidecar.read_text("ascii").strip(),parent_dir


def run_h423():
    import torch
    if OUT.exists():
        raise ValueError("H423 private interrupted GPU output path collision")
    OUT.mkdir(parents=True,exist_ok=False)
    st={
        "schema":"hala.h423.private-1500-step-segmented-native-drag.v1",
        "status":"NOT_STARTED",
        "source_h419_kernel":H419_KERNEL,
        "original_model_source":"Hcompany/DragOn original H60 TRAIN-only plus owner original AR screenshots",
        "official_base":BASE,
        "official_revision":REV,
        "h417_original_trained_parent_adapter_sha256":PARENT_ADAPTER_SHA,
        "h418_frozen_new_30000_source_sha256":H418_ROWS_SHA,
        "h418_original_source_receipt_sha256":H418_RECEIPT_SHA,
        "uninterrupted_9452_step_optimizer_history":False,
        "optimizer_moments_restored_from_H419":False,
        "AdamW_optimizer_reset_on_resume":True,
        "actual_optimizer_steps_this_segment":0,
        "actual_grad_image_passes_this_segment":0,
        "independent_external_ood":False,
        "model_release_qualified":False,
        "weights_public":False,
    }
    zips=None
    extras=[]
    started=time.monotonic()
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or not torch.cuda.is_available():
            raise ValueError("H423 only PRIVATE free Kaggle T4 original checkpoints allowed")
        if torch.cuda.get_device_name(0)!="Tesla T4":
            raise ValueError("H423 exact private FREE Tesla T4 only, never paid compute")
        subprocess.run([sys.executable,"-m","pip","install","-q",
            "transformers==4.57.3","peft==0.18.0",
            "accelerate==1.12.0","Pillow==11.3.0"],
            check=True,timeout=330)
        from transformers import AutoProcessor,AutoModelForImageTextToText
        from peft import PeftModel
        before_receipt,original_receipt_sha,checkpoint_dir=exact_real_original_checkpoint()
        begin=before_receipt["original_model_checkpoint_step"]
        end=min(begin+H423_T4_SEGMENT_STEPS,H419_STEPS)
        if not begin<end<=H419_STEPS:
            raise ValueError("H423 source original planned gradient order already exhausted")
        zips,extras,train,dev,order,old_dev,fresh_dev=h419_prepare_sources()
        if (len(train),len(dev),len(order),len(old_dev),len(fresh_dev))!=(
               H419_UNIQUE_IMAGES,H419_DEV_N,H419_TRAINING_PASSES,
               H419_OLD_DEV_N,H419_NEW_DEV_N):
            raise ValueError("H423 original frozen H419 native source order changed")
        st.update(
          genuine_original_checkpoint_step=begin,
          segment_end_optimizer_step=end,
          original_h419_checkpoint_qualification_receipt_sha256=original_receipt_sha,
          original_h419_checkpoint_adapter_sha256=before_receipt["checkpoint_adapter_model_sha256"],
          real_source_frozen_30000_native_sha256=H418_ROWS_SHA,
          actual_source_dev_images=len(dev),
          source_train_unique_original_and_ar_images=len(train),
          segment_sample_order_start_offset=begin*GRAD_ACCUM,
          planned_segment_end_offset=end*GRAD_ACCUM,
          full_original_H419_9452_steps_not_uninterrupted=True,
          original_source_zip_sha256=H419_ALL_SOURCE_SHAS,
        )
        torch.manual_seed(H419_SEED+begin)
        processor=AutoProcessor.from_pretrained(BASE,revision=REV,
                                                trust_remote_code=False)
        pretrained=AutoModelForImageTextToText.from_pretrained(
            BASE,revision=REV,dtype=torch.float16,trust_remote_code=False,
            attn_implementation="sdpa").to("cuda:0")
        model=PeftModel.from_pretrained(pretrained,str(checkpoint_dir),
                                        is_trainable=True)
        if hasattr(model.config,"use_cache"):model.config.use_cache=True
        st["starting_source_dev"],st["starting_source_dev_exact"]=evaluate(
            model,processor,dev,zips,"ORIGINAL_H419_PARTIAL_REAL_CHECKPOINT")
        active=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if not active or any("lora_" not in name for name,p in active):
            raise ValueError("H423 only actual original private LoRA checkpoint tensors may train")
        prior={n:p.detach().float().cpu().clone() for n,p in active}
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        if hasattr(model.config,"use_cache"):model.config.use_cache=False
        optimizer=torch.optim.AdamW([p for _,p in active],lr=0.000008)
        losses=[]
        for absolute_step in range(begin,end):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            micro_losses=[]
            for micro in range(GRAD_ACCUM):
                index=order[absolute_step*GRAD_ACCUM+micro]
                row=train[index]
                batch,labels=to_inputs(processor,row,zips)
                labels=batch.pop("labels",labels)
                output=model(**batch)
                loss=coordinate_weighted_loss(output,labels,processor)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("H423 original continued native DRAG loss nonfinite")
                (loss/GRAD_ACCUM).backward()
                micro_losses.append(float(loss.detach().cpu()))
                del output,loss,labels,batch
            torch.nn.utils.clip_grad_norm_([p for _,p in active],1.0)
            optimizer.step()
            losses.append(round(sum(micro_losses)/len(micro_losses),6))
            st["actual_optimizer_steps_this_segment"]+=1
            st["actual_grad_image_passes_this_segment"]+=GRAD_ACCUM
            if (absolute_step+1)%150==0:
                print("H423_ACTUAL_INTERRUPTED_SOURCE_RECOVERY_STEP",
                      absolute_step+1,"/",H419_STEPS,
                      "true_grad_examples_this_segment",
                      st["actual_grad_image_passes_this_segment"],
                      "reset_adamw_loss",losses[-1],flush=True)
        st["changed_checkpoint_lora_tensors"]=sum(
            int(bool((p.detach().float().cpu()!=prior[n]).any()))
            for n,p in active)
        if st["changed_checkpoint_lora_tensors"]<1:
            raise ValueError("H423 original model partial LoRA did not update")
        model.gradient_checkpointing_disable()
        if hasattr(model.config,"use_cache"):model.config.use_cache=True
        st["after_segment_source_dev"],st["after_segment_source_dev_exact"]=evaluate(
            model,processor,dev,zips,"H423_REAL_SOURCE_SEGMENT_AFTER_ADAMW_RESET")
        checkpoint=OUT/"private-source-authentic-segment-adapter"
        model.save_pretrained(checkpoint,safe_serialization=True)
        safe=checkpoint/"adapter_model.safetensors"
        if not safe.is_file() or safe.stat().st_size<1000:
            raise ValueError("H423 actual resumed LoRA final private safetensors absent")
        st.update(
           status=H423_STEP_STATUS,
           actual_end_optimizer_step=end,
           actual_cumulative_optimizer_steps_including_owner_evidenced_original=(
               begin+st["actual_optimizer_steps_this_segment"]),
           original_grad_examples_before_resume=begin*GRAD_ACCUM,
           actual_cumulative_grad_examples_with_optimizer_reset=(
               begin*GRAD_ACCUM+st["actual_grad_image_passes_this_segment"]),
           exact_model_adapter_sha256=sha_file(safe),
           exact_model_adapter_bytes=safe.stat().st_size,
           original_old_h417_dev_at_start=sum(r["correct"] for r in
               st["starting_source_dev_exact"][:H419_OLD_DEV_N]),
           original_old_h417_dev_after=sum(r["correct"] for r in
               st["after_segment_source_dev_exact"][:H419_OLD_DEV_N]),
           original_new_h419_dev_at_start=sum(r["correct"] for r in
               st["starting_source_dev_exact"][H419_OLD_DEV_N:]),
           original_new_h419_dev_after=sum(r["correct"] for r in
               st["after_segment_source_dev_exact"][H419_OLD_DEV_N:]),
           elapsed_seconds=round(time.monotonic()-started,2),
           device_name=torch.cuda.get_device_name(0),
           true_reset_optimizer_weighted_losses=losses,
        )
    except Exception as exc:
        st["status"]=H423_STEP_FAILURE
        st["failure_class"]=type(exc).__name__
        raise
    finally:
        if extras:
            for z in extras:z.close()
        if zips:
            for z in zips.values():
                if isinstance(z,zipfile.ZipFile):z.close()
        original=json.dumps(st,sort_keys=True,indent=2,allow_nan=False,
                            ensure_ascii=False).encode("utf8")
        (OUT/"receipt.json").write_bytes(original)
        (OUT/"receipt.sha256").write_text(sha_bytes(original))
        print("H423_ORIGINAL_PARTIAL_GPU_RESCUE_GENUINE_STATUS",
           json.dumps({k:v for k,v in st.items() if
                       not k.endswith("_exact") and
                       k not in ("true_reset_optimizer_weighted_losses",
                                 "original_source_zip_sha256")},sort_keys=True),
           flush=True)


if __name__=="__main__":
    run_h423()
