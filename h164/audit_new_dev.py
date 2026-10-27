"""Independent owner-private H164 DEV 96 admission with separated gold."""
import argparse,json
from collections import Counter
from pathlib import Path
from h164.new_dev_96 import SCHEMA,PRIVATE,cases,used
from h164.expand_288_train import digest
from h164.protocol import validate_source_splits

def audit(owner):
    owner=Path(owner);root=owner/PRIVATE
    raw=(root/"manifest.json").read_bytes()
    manifest=json.loads(raw)
    gold=json.loads((root/"owner-gold-not-for-GPU.json").read_text("utf8"))
    rows=manifest.get("private_cases")
    refs=gold.get("owner_reference_actions")
    expected=cases()
    if (manifest.get("schema")!=SCHEMA or
        manifest.get("dev_only") is not True or
        manifest.get("gold_model_inputs") is not False or
        not isinstance(rows,list) or len(rows)!=96 or
        not isinstance(refs,list) or len(refs)!=96 or
        gold.get("source_manifest_sha256")!=digest(raw)):
        raise ValueError("H164 DEV source/gold isolation broken")
    prior=used(owner)
    unique=set();names=set()
    counts=Counter();languages=Counter();kinds=Counter()
    for row,reference,case in zip(rows,refs,expected):
        if (any(row.get(k)!=v for k,v in case.items()) or
            row.get("training_supervision") is not None or
            "owner_reference_action" in row or
            "owner_gold_click" in row or "reference_point" in row):
            raise ValueError("H164 DEV model input has gold or case drift")
        key=row.get("image_sha256")
        if (not isinstance(key,str) or key in prior or key in unique or
            not isinstance(row.get("instruction"),str) or
            reference.get("image_sha256")!=key or
            reference.get("native_reference_success") is not True or
            reference.get("owner_supervision",{}).get("actions") is None):
            raise ValueError("H164 DEV references/image duplicated or incomplete")
        png=row.get("image")
        if (not isinstance(png,str) or Path(png).name!=png or
            digest((root/"images"/png).read_bytes())!=key):
            raise ValueError("H164 DEV source PNG hash collision")
        native=reference["owner_supervision"]["actions"]
        if len(native)!=1 or native[0]["kind"]!=row["action_type"]:
            raise ValueError("H164 DEV owner native oracle mismatched")
        unique.add(key);names.add(png)
        counts[row["app"]]+=1;languages[row["language"]]+=1
        kinds[row["action_type"]]+=1
    if (counts!={"simple-table":48,"calculator":48}
        or languages!={"en":48,"ar":48}
        or names!={x.name for x in (root/"images").glob("*.png")}):
        raise ValueError("H164 DEV bilingual/source quotas incomplete")
    train1=json.loads((owner/"private-h164/four-app-train-pilot-v1/manifest.json").read_text("utf8"))["private_cases"]
    train2=json.loads((owner/"private-h164/additional-three-app-288-train-v1/manifest.json").read_text("utf8"))["private_cases"]
    report=validate_source_splits(owner,train1+train2,rows,[],admission="pilot")
    if report["counts"]!={"TRAIN":384,"DEV":96,"OOD":0}:
        raise ValueError("H164 TRAIN/DEV split count mismatch")
    return {"status":"H164_384_TRAIN_96_DEV_NO_GOLD_MODEL_INPUTS_AUDITED",
            "dev_source_manifest_sha256":digest(raw),
            "app_quotas":dict(counts),"languages":dict(languages),
            "native_actions":dict(kinds),"owner_native_reference_success":96,
            "train_count":384,"dev_count":96,"ood_count":0,
            "no_DEV_gold_in_model_manifest":True,
            "optimizer_steps":0,"paid_GPU":False,"h140_blind_calls":0}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--owner",required=True)
    args=p.parse_args()
    print(json.dumps(audit(args.owner),indent=2,sort_keys=True))
