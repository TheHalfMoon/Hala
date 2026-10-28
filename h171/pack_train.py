"""Owner-private H171 new original click-only training ZIP; no DEV/OOD."""
import argparse
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image
from h171.source import SCHEMA,COUNT,cases,prior_hashes,validate
from h164.native_action_contract import supervision,parse_action

ZIP_SCHEMA="hala.h171.private-original-click-train-32.v1"

def sha(blob):
    return hashlib.sha256(blob).hexdigest()

def pack(owner,source,h168_train,h169_dev,output):
    owner=Path(owner);source=Path(source);output=Path(output)
    if output.exists() or output.suffix!=".zip":
        raise ValueError("H171 new owner-private TRAIN ZIP must not preexist")
    doc=json.loads((source/"manifest.json").read_text("utf8"))
    gold=json.loads((source/"train-actions-private.json").read_text("utf8"))
    previous=prior_hashes(owner,h168_train,h169_dev)
    admitted=validate(doc,gold,previous)
    if admitted["cases"]!=32:
        raise ValueError("H171 incomplete native-original click TRAIN admission")
    rows=[];pics={}
    for scene,reference in zip(doc["private_cases"],gold["owner_train_actions"]):
        row=dict(scene,split="TRAIN",
                 training_supervision={"actions":[{"kind":"click",
                                            "point":reference["point"]}]})
        encoded=supervision(row)
        if parse_action(encoded)["kind"]!="click":
            raise ValueError("H171 real owner click supervised action invalid")
        name=row["image"]
        if (name!="images/"+str(row["index"])+".png" or name in pics
            or any(key in row for key in ("owner_gold","reference_point"))):
            raise ValueError("H171 image path or source supervision contaminated")
        pic=(source/name).read_bytes()
        if sha(pic)!=row["image_sha256"]:
            raise ValueError("H171 frozen original Chrome image SHA mismatched")
        with Image.open(io.BytesIO(pic)) as im:
            if im.format!="PNG" or im.size!=(960,720):
                raise ValueError("H171 original screenshot 960x720 RGB PNG required")
        pics[name]=pic
        rows.append({k:v for k,v in row.items() if k in (
            "index","split","app","variant","action_type","task","language",
            "image","image_sha256","instruction","training_supervision")})
    if (Counter(r["app"] for r in rows)!={"flatpickr":16,"awesomplete":16}
        or Counter(r["language"] for r in rows)!={"ar":16,"en":16}):
        raise ValueError("H171 click TRAIN app or language distribution changed")
    manifest={"schema":ZIP_SCHEMA,"split":"TRAIN","count":32,
              "previous_consumed_dev_ood_used":False,
              "reference_gold_uploaded_for_train_only":True,
              "original_upstream_mit_verified":True,
              "private_cases":rows}
    output.parent.mkdir(exist_ok=True,parents=True)
    with zipfile.ZipFile(output,"x",compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json",json.dumps(
            manifest,ensure_ascii=False,sort_keys=True,separators=(",",":")))
        for name in sorted(pics):
            z.writestr(name,pics[name])
    return {"status":"H171_OWNER_PRIVATE_32_FRESH_ORIGINAL_CLICK_TRAIN_PACKED",
      "source_training_cases":32,"click":32,"arabic":16,"english":16,
      "private_train_zip_sha256":sha(output.read_bytes()),
      "prior_dev_ood_used":False,"optimizer_steps":0}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for flag in ("owner","source","h168-train","h169-dev","output"):
        p.add_argument("--"+flag,required=True)
    a=p.parse_args()
    print(json.dumps(pack(a.owner,a.source,a.h168_train,a.h169_dev,a.output),
                     indent=2,sort_keys=True))
