"""Pack H168 true original-browser TRAIN-only data into owner-private Kaggle ZIP.

This is not a model, and NEVER packages DEV, OOD, or earlier consumed images.
"""
import argparse
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

from h164.native_action_contract import supervision,parse_action
from h168.source import (PINS,APPS_FOR_PILOT,N_PER_APP,
   SCHEMA,sha,validate_pilot,historical_image_shas)

ZIP_SCHEMA="hala.h168.owner-private-new-original-train-32.v1"

def pack(owner,source,output):
    owner=Path(owner)
    source=Path(source)
    output=Path(output)
    if output.exists() or source==output or output.suffix!=".zip":
        raise ValueError("H168 private TRAIN-only archive must not preexist")
    meta=json.loads((source/"manifest.json").read_text("utf8"))
    gold=json.loads((source/"train-supervision-private.json").read_text("utf8"))
    historical=historical_image_shas(owner)
    quality=validate_pilot(meta,gold,historical)
    if (meta.get("schema")!=SCHEMA or len(meta["private_cases"])!=32
        or quality["cases"]!=32
        or {r["app"] for r in meta["private_cases"]}!=set(APPS_FOR_PILOT)
        or Counter(r["language"] for r in meta["private_cases"])!={"en":16,"ar":16}
        or Counter(r["action_type"] for r in meta["private_cases"])!={"type":16,"drag":16}):
        raise ValueError("H168 incomplete or imbalanced original TRAIN pilot")
    records=[]
    images={}
    for scene,reference in zip(meta["private_cases"],
                               gold["private_training_actions"]):
        entry={k:v for k,v in reference.items()
               if k not in ("index","native_success")}
        row=dict(scene,split="TRAIN",training_supervision={"actions":[entry]})
        encoded=supervision(row)
        parsed=parse_action(encoded)
        if parsed["valid"] is not True or parsed["kind"]!=scene["action_type"]:
            raise ValueError("H168 native user-action gold format not qualified")
        name=scene["image"]
        if not name.startswith("images/") or "/" in name[7:] or name in images:
            raise ValueError("H168 private app image path unsafe or repeated")
        pixels=(source/name).read_bytes()
        if sha(pixels)!=scene["image_sha256"]:
            raise ValueError("H168 original browser image changed since replay")
        with Image.open(io.BytesIO(pixels)) as image:
            if image.size!=(960,720) or image.format!="PNG":
                raise ValueError("H168 original Chrome TRAIN screenshot not PNG 960x720")
        images[name]=pixels
        records.append(row)
    doc=dict(schema=ZIP_SCHEMA,split="TRAIN",count=32,
             data_role="H168_NEW_SOURCE_TRAIN_PILOT_ONLY",
             app_count=2,dev_ood_count=0,
             historical_consumed_images_excluded=True,
             reference_gold_uploaded_for_train_only=True,
             cases=records)
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,"x",compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json",json.dumps(
            doc,sort_keys=True,separators=(",",":"),ensure_ascii=False))
        for name in sorted(images):
            z.writestr(name,images[name])
    receipt=dict(status="H168_PRIVATE_REAL_ORIGINAL_32_TRAIN_ONLY_PACKED",
                 training_zip_sha256=sha(output.read_bytes()),
                 split="TRAIN",original_browser_cases=32,
                 distinct_app_families=2,arabic_cases=16,english_cases=16,
                 native_drag_cases=16,native_type_cases=16,
                 historical_source_excluded=True,dev_ood_scenes_in_zip=0,
                 paid_gpu=False,optimizer_steps=0)
    return receipt

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner",required=True)
    p.add_argument("--source",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    print(json.dumps(pack(args.owner,args.source,args.output),
                     indent=2,sort_keys=True))
