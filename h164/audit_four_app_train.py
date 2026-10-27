"""Independent owner-only H164 96 TRAIN original-app media and action audit."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from h164.protocol import locked_prior_shas,validate_source_splits
from h164.four_app_train_pilot import APPS,SCHEMA,N_PER_APP,specs,PRIVATE

def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def audit(owner):
    owner=Path(owner)
    root=owner/PRIVATE
    raw=(root/"manifest.json").read_bytes()
    doc=json.loads(raw)
    cases=doc.get("private_cases")
    frozen=specs()
    if (doc.get("schema")!=SCHEMA or doc.get("train_only") is not True
        or doc.get("train_count")!=96
        or doc.get("native_reference_success")!=96
        or not isinstance(cases,list) or len(cases)!=96):
        raise ValueError("H164 four-original-app TRAIN source incomplete")
    original=locked_prior_shas(owner)
    seen=set()
    counts=Counter()
    acts=Counter()
    files=set()
    for c,proposal in zip(cases,frozen):
        if (not isinstance(c,dict) or
            any(c.get(k)!=v for k,v in proposal.items())
            or c.get("native_reference_success") is not True
            or c.get("split")!="TRAIN"
            or not isinstance(c.get("instruction"),str)
            or not c["instruction"].strip()
            or not isinstance(c.get("training_supervision"),dict)):
            raise ValueError("H164 source train app/task/reference mismatch")
        action=c["training_supervision"].get("actions")
        if not isinstance(action,list) or len(action)!=1:
            raise ValueError("H164 native action supervision incomplete")
        act=action[0]
        if set(act)!=(
            {"kind","point","text"} if act.get("kind")=="type"
            else {"kind","point"} if act.get("kind")=="click"
            else {"kind","start","end"}):
            raise ValueError("H164 typed click/drag supervision schema invalid")
        points=[act["point"]] if act["kind"] in ("type","click") else (
            [act["start"],act["end"]])
        for point in points:
            if (not isinstance(point,list) or len(point)!=2 or
                any(type(x) not in (int,float) or not 0<=x<upper
                    for x,upper in zip(point,(960,720)))):
                raise ValueError("H164 original owner native action outside image")
        if act["kind"]=="type" and (
            not isinstance(act["text"],str) or not act["text"]):
            raise ValueError("H164 text input label empty")
        if c["language"]=="ar" and not any(
            "\u0600"<=s<="\u06ff" for s in c["instruction"]
        ):
            raise ValueError("H164 Arabic original app instruction absent")
        if c["language"]=="en" and not c["instruction"].isascii():
            raise ValueError("H164 English original app task contamination")
        if not isinstance(c.get("image"),str) or Path(c["image"]).name!=c["image"]:
            raise ValueError("H164 unsafe original image filename")
        path=root/"images"/c["image"]
        blob=path.read_bytes()
        h=hash_bytes(blob)
        if h!=c["image_sha256"] or h in seen or h in original:
            raise ValueError("H164 duplicated/consumed source image")
        if blob[:8]!=b"\x89PNG\r\n\x1a\n":
            raise ValueError("H164 original screenshot not PNG")
        files.add(c["image"])
        seen.add(h)
        counts[c["app"]]+=1
        acts[act["kind"]]+=1
    if (counts!={app:N_PER_APP for app in APPS}
        or files!={x.name for x in (root/"images").glob("*.png")}):
        raise ValueError("H164 source TRAIN images/app quota altered")
    pilot=validate_source_splits(owner,cases,[],[],admission="pilot")
    if pilot["counts"]["TRAIN"]!=96:
        raise ValueError("H164 original source admission total mismatch")
    return {"status":"H164_FOUR_REAL_APPS_96_TRAIN_REFERENCES_AUDITED",
            "manifest_sha256":hash_bytes(raw),
            "source_images":96,"native_source_success":96,
            "apps":dict(counts),"native_actions":dict(acts),
            "old_consumed_image_shas_excluded":len(original),
            "new_dev_cases":0,"new_ood_cases":0,
            "model_training_started":False,"h140_blind_calls":0,
            "paid_GPU":False}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner",required=True)
    a=p.parse_args()
    print(json.dumps(audit(a.owner),indent=2,sort_keys=True))
