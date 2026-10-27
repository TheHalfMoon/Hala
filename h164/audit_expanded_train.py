"""Independent owner-private audit for scaled H164 288 actual original TRAIN frames."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
from h164.expand_288_train import SCHEMA,APPS,PRIVATE,specs,prior,PINS,digest
from h164.native_action_contract import supervision,parse_action
from h164.protocol import validate_source_splits

def audit(owner):
    owner=Path(owner)
    root=owner/PRIVATE
    raw=(root/"manifest.json").read_bytes()
    doc=json.loads(raw)
    rows=doc.get("private_cases")
    if (doc.get("schema")!=SCHEMA or doc.get("train_only") is not True or
        doc.get("train_count")!=288 or doc.get("native_reference_success")!=288 or
        doc.get("source_git_shas")!=PINS or
        not isinstance(rows,list) or len(rows)!=288):
        raise ValueError("H164 new TRAIN source frozen header incomplete")
    expected=specs()
    history=prior(owner)
    seen=set();files=set()
    counts=Counter();actions=Counter();languages=Counter()
    for row,contract in zip(rows,expected):
        if (not isinstance(row,dict) or
            any(row.get(k)!=v for k,v in contract.items()) or
            row.get("native_reference_success") is not True):
            raise ValueError("H164 expanded TRAIN source manifest task mismatch")
        image=row.get("image")
        if (not isinstance(image,str) or
            Path(image).name!=image or not image.endswith(".png")):
            raise ValueError("H164 unsafe original source image name")
        content=(root/"images"/image).read_bytes()
        sha=digest(content)
        if (sha!=row.get("image_sha256") or sha in seen or sha in history or
            content[:8]!=b"\x89PNG\r\n\x1a\n"):
            raise ValueError("H164 original 288 picture SHA duplicate/leak")
        result=parse_action(supervision(row))
        if result["kind"]!=row["action_type"]:
            raise ValueError("H164 actual native action schema invalid")
        if row["language"]=="ar" and not any(
            "\u0600"<=c<="\u06ff" for c in row["instruction"]):
            raise ValueError("H164 missing bilingual TRAIN task")
        if row["language"]=="en" and not row["instruction"].isascii():
            raise ValueError("H164 English source task unexpectedly non-ASCII")
        seen.add(sha);files.add(image)
        counts[row["app"]]+=1;actions[result["kind"]]+=1
        languages[row["language"]]+=1
    if (counts!={app:96 for app in APPS} or languages!={"en":144,"ar":144} or
        files!={p.name for p in (root/"images").glob("*.png")}):
        raise ValueError("H164 288 source quotas/files language not exact")
    old=json.loads((owner/"private-h164/four-app-train-pilot-v1/manifest.json").read_text("utf8"))
    train=old["private_cases"]+rows
    if len(train)!=384:
        raise ValueError("H164 first two TRAIN cohorts cannot total 384")
    admission=validate_source_splits(owner,train,[],[],admission="pilot")
    if admission["counts"]["TRAIN"]!=384 or admission["distinct_original_apps"]["TRAIN"]!=5:
        raise ValueError("H164 TRAIN combined true-original-app coverage drift")
    return {"status":"H164_384_INDEPENDENT_ORIGINAL_TRAIN_SOURCES_AUDITED",
            "new_manifest_sha256":digest(raw),
            "earlier_train":96,"added_train":288,"combined_train":384,
            "distinct_original_apps":5,
            "new_actions":dict(actions),"new_languages":dict(languages),
            "previous_images_excluded":len(history),
            "model_optimizer_steps":0,"fresh_dev":0,"fresh_ood":0,
            "h140_blind_calls":0,"paid_GPU":False}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner",required=True)
    args=p.parse_args()
    print(json.dumps(audit(args.owner),indent=2,sort_keys=True))
