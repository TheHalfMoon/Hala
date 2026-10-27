"""H164 independent strict final 384 TRAIN/96 DEV/64 OOD admission audit."""
import argparse,json
from pathlib import Path
from collections import Counter
from h164.new_ood_64 import PRIVATE,SCHEMA,cases,old_hashes,PINS
from h164.expand_288_train import digest
from h164.protocol import validate_source_splits
from h164.native_action_contract import parse_action

def audit(owner):
    owner=Path(owner)
    root=owner/PRIVATE
    raw=(root/"manifest.json").read_bytes()
    doc=json.loads(raw)
    gold=json.loads((root/"owner-gold-not-for-model.json").read_text("utf8"))
    qa=json.loads((owner/"private-h164/H164_OOD_64_PIXEL_QA.json").read_text("utf8"))
    rows=doc.get("private_cases")
    refs=gold.get("reference_actions")
    if (doc.get("schema")!=SCHEMA or doc.get("ood_only") is not True or
        doc.get("gold_model_inputs") is not False or
        doc.get("git_source_shas")!=PINS or
        not isinstance(rows,list) or len(rows)!=64 or
        not isinstance(refs,list) or len(refs)!=64 or
        gold.get("source_manifest_sha256")!=digest(raw) or
        qa.get("status")!="H164_OOD_64_NEW_APP_EXACT_BROWSER_PIXEL_REPLAY"):
        raise ValueError("H164 two fresh original-app OOD model input not qualified")
    prior=old_hashes(owner)
    unique=set();files=set();apps=Counter();langs=Counter()
    for row,reference,case in zip(rows,refs,cases()):
        if (any(row.get(k)!=v for k,v in case.items()) or
            "training_supervision" in row or "owner_supervision" in row or
            "reference_point" in row or
            "owner_gold_click" in row):
            raise ValueError("H164 OOD owner gold leaked into model input")
        h=row.get("image_sha256")
        if (not isinstance(h,str) or h in unique or h in prior or
            reference.get("image_sha256")!=h or
            reference.get("native_reference_success") is not True):
            raise ValueError("H164 original OOD cases overlap prior/each other")
        target=reference.get("owner_supervision")
        if (not isinstance(target,dict) or
            not parse_action(json.dumps(target))["valid"] or
            target["actions"][0]["kind"]!=row.get("action_type")):
            raise ValueError("H164 owner reference real original OOD mouse action missing")
        name=row.get("image")
        if (not isinstance(name,str) or Path(name).name!=name or
            digest((root/"images"/name).read_bytes())!=h):
            raise ValueError("H164 OOD image cannot be verified")
        if (not isinstance(row.get("instruction"),str) or
            (row["language"]=="ar" and not any(
                "\u0600"<=x<="\u06ff" for x in row["instruction"])) or
            (row["language"]=="en" and not row["instruction"].isascii())):
            raise ValueError("H164 OOD bilingual prompt invalid")
        unique.add(h);files.add(name)
        apps[row["app"]]+=1;langs[row["language"]]+=1
    if (apps!={"pell":32,"pikaday":32} or
        langs!={"ar":32,"en":32} or
        files!={p.name for p in (root/"images").glob("*.png")}):
        raise ValueError("H164 original OOD quotas/images incomplete")
    def read(path):return json.loads((owner/path/"manifest.json").read_text("utf8"))["private_cases"]
    train=(read("private-h164/four-app-train-pilot-v1")+
           read("private-h164/additional-three-app-288-train-v1"))
    dev=read("private-h164/new-dev-96-v1")
    result=validate_source_splits(owner,train,dev,rows,admission="final")
    if result["counts"]!={"TRAIN":384,"DEV":96,"OOD":64}:
        raise ValueError("H164 final frozen split mismatch")
    return {"status":"H164_FINAL_384_TRAIN_96_DEV_64_TRUE_TWO_NEW_APP_OOD_ADMITTED",
            "OOD_source_manifest_sha256":digest(raw),
            "new_app_count":2,"new_app_quotas":dict(apps),
            "language_quotas":dict(langs),
            "gold_in_GPU_model_inputs":False,
            "trained_model_calls":0,"optimizer_steps":0,
            "app_train_count":result["distinct_original_apps"]["TRAIN"],
            "h140_blind_calls":0,"paid_GPU":False}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner",required=True)
    args=p.parse_args()
    print(json.dumps(audit(args.owner),indent=2,sort_keys=True))
