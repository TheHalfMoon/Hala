"""H164 preregistered diverse-application dataset admission (NO training yet).

H163 new app was consumed with 0/16 native success. Neither its source images
nor any prior owner-only frozen benchmark image may enter future optimization.
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from h161.original_apps_source import prior_shas

SCHEMA="hala.h164.multi-original-application-geometry.v1"
SOURCE_REGISTRY={
    "private-h161/two-real-app-source-v1/manifest.json":72,
    "private-h162/original-html5-board-source-once-v1/manifest.json":16,
    "private-h163/source-once-v1/manifest.json":16,
}
ALLOWED_ACTIONS={"click","drag","type"}
REQUIRED_TRAIN_APPS=4
REQUIRED_DEV_APPS=2
REQUIRED_NEW_OOD_APPS=2
MIN_TRAIN=384
MIN_DEV=96
MIN_OOD=64


def locked_prior_shas(owner):
    owner=Path(owner)
    previous=set(prior_shas(owner))
    for relative,count in SOURCE_REGISTRY.items():
        file=owner/relative
        data=json.loads(file.read_text("utf8"))
        rows=data.get("private_cases")
        if not isinstance(rows,list) or len(rows)!=count:
            raise ValueError("H164 frozen consumed original-app registry changed")
        for row in rows:
            value=row["image_sha256"]
            if (not re.fullmatch(r"[a-f0-9]{64}",value) or
                value in previous):
                raise ValueError("H164 original image SHA invalid/duplicated")
            previous.add(value)
    if len(previous)<120:
        raise ValueError("H164 cannot prove historical frozen app images excluded")
    return previous


def validate_source_splits(owner,train,dev,ood,*,admission="final"):
    """All arguments are only metadata; never load/modify sealed H140 images."""
    if admission not in ("pilot","final"):
        raise ValueError("H164 source admission must declare pilot/final")
    history=locked_prior_shas(owner)
    groups={"TRAIN":train,"DEV":dev,"OOD":ood}
    seen={}
    apps={}
    totals={}
    for split,items in groups.items():
        if not isinstance(items,list):
            raise ValueError("H164 source split must be list")
        totals[split]=len(items)
        apps[split]=set()
        for row in items:
            if not isinstance(row,dict) or not all(
                key in row for key in (
                    "app","task","language","image_sha256","action_type"
                )
            ):
                raise ValueError("H164 source manifest missing provenance")
            image=row["image_sha256"]
            if not isinstance(image,str) or not re.fullmatch(r"[a-f0-9]{64}",image):
                raise ValueError("H164 not exact SHA256 source data")
            if image in history or image in seen:
                raise ValueError(
                    "H164 consumed previous evaluation / cross-split image leakage"
                )
            if (not isinstance(row["app"],str) or not row["app"].strip()
                or row["language"] not in ("en","ar")
                or row["action_type"] not in ALLOWED_ACTIONS):
                raise ValueError("H164 native app/task/language contract invalid")
            if not isinstance(row["task"],str) or not row["task"].strip():
                raise ValueError("H164 no native task label")
            if split!="TRAIN" and (
                "training_supervision" in row or "owner_gold_click" in row or
                "reference_point" in row or "owner_reference_action" in row):
                raise ValueError("H164 DEV/OOD gold labels may not enter model inputs")
            seen[image]=split
            apps[split].add(row["app"])
    if apps["OOD"]&(apps["TRAIN"]|apps["DEV"]):
        raise ValueError("H164 new-application OOD app cannot be in TRAIN/DEV")
    if admission=="final":
        if (totals["TRAIN"]<MIN_TRAIN or totals["DEV"]<MIN_DEV
            or totals["OOD"]<MIN_OOD
            or len(apps["TRAIN"])<REQUIRED_TRAIN_APPS
            or len(apps["DEV"])<REQUIRED_DEV_APPS
            or len(apps["OOD"])<REQUIRED_NEW_OOD_APPS):
            raise ValueError("H164 independent multiactor app coverage not sufficient")
        for split in groups:
            langs=Counter(x["language"] for x in groups[split])
            if min(langs.values() or [0])<totals[split]//3 or len(langs)!=2:
                raise ValueError("H164 must include balanced Arabic/English source")
        if not ALLOWED_ACTIONS.issubset({r["action_type"] for r in train}):
            raise ValueError("H164 train must cover native click, drag and text input")
    return {"status":"H164_SOURCE_SPLITS_ADMISSIBLE_"+admission.upper(),
            "old_consumed_images_excluded":len(history),
            "counts":totals,
            "distinct_original_apps":{k:len(v) for k,v in apps.items()},
            "optimizer_steps":0,
            "paid_GPU":False,"h140_blind_calls":0}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner",required=True)
    p.add_argument("--train",required=True)
    p.add_argument("--dev",required=True)
    p.add_argument("--ood",required=True)
    p.add_argument("--admission",choices=("pilot","final"),default="final")
    args=p.parse_args()
    data=[json.loads(Path(x).read_text("utf8"))["private_cases"]
          for x in (args.train,args.dev,args.ood)]
    print(json.dumps(validate_source_splits(args.owner,*data,
                     admission=args.admission),indent=2,sort_keys=True))
