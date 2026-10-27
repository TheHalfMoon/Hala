"""Independent H161 two MIT source-app provenance, split and PNG auditor."""
import argparse
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path

from h161.original_apps_source import (
    SCHEMA, APPS_USED, QUOTAS, all_specs, prior_shas,
)

REQUIRED={
    "split","app","task","language","index","image",
    "image_sha256","instruction","training_supervision",
    "original_native_reference_success","model_calls",
}


def audit(owner):
    owner=Path(owner)
    folder=owner/"private-h161"/"two-real-app-source-v1"
    raw=(folder/"manifest.json").read_bytes()
    doc=json.loads(raw)
    if (doc.get("schema")!=SCHEMA or doc.get("training_source_only") is not True or
        doc.get("source_reference_success")!=72 or
        doc.get("train")!=48 or doc.get("dev")!=24 or
        doc.get("quotas")!=QUOTAS or
        doc.get("dev_is_not_training") is not True or
        doc.get("h140_blind_calls")!=0 or doc.get("paid_gpu") is not False):
        raise ValueError("H161 frozen two-app source header invalid")
    cases=doc.get("private_cases")
    expected=all_specs()
    if not isinstance(cases,list) or len(cases)!=len(expected):
        raise ValueError("H161 original source expected 72 episodes")
    earlier=prior_shas(owner)
    digests=set()
    counts=Counter()
    accepted_paths=set()
    for actual, spec in zip(cases,expected):
        if not isinstance(actual,dict) or set(actual)!=REQUIRED:
            raise ValueError("H161 original native episode schema mismatched")
        if any(actual[k]!=spec[k] for k in spec):
            raise ValueError("H161 frozen source app/split/source index changed")
        if (actual.get("original_native_reference_success") is not True or
            actual.get("model_calls")!=0 or
            not isinstance(actual.get("instruction"),str) or
            len(actual["instruction"])<12 or
            not isinstance(actual.get("training_supervision"),dict)):
            raise ValueError("H161 native source/reward evidence incomplete")
        actions=actual["training_supervision"].get("actions")
        if not isinstance(actions,list) or len(actions)!=1:
            raise ValueError("H161 typed supervised action missing")
        action=actions[0]
        if actual["app"]=="todomvc-es6":
            if set(action)!={"kind","point"} or action["kind"]!="click":
                raise ValueError("H161 original click schema altered")
            points=[action["point"]]
        else:
            if set(action)!={"kind","start","end"} or action["kind"]!="drag":
                raise ValueError("H161 original drag schema altered")
            points=[action["start"],action["end"]]
        for xy in points:
            if (not isinstance(xy,list) or len(xy)!=2 or
                any(type(z) not in (int,float) or
                    z<0 or z>=lim for z,lim in zip(xy,(960,720)))):
                raise ValueError("H161 native points out of original viewport")
        if actual["language"]=="ar" and not any(
            "\u0600"<=c<="\u06ff" for c in actual["instruction"]
        ):
            raise ValueError("H161 missing real Arabic instruction")
        if actual["language"]=="en" and not actual["instruction"].isascii():
            raise ValueError("H161 English instruction contaminated")
        name=actual["image"]
        if (not isinstance(name,str) or
            Path(name).name!=name or
            not name.endswith(".png") or name in accepted_paths):
            raise ValueError("H161 source PNG filename unsafe or duplicated")
        contents=(folder/"images"/name).read_bytes()
        if (contents[:8]!=b"\x89PNG\r\n\x1a\n" or
            struct.unpack(">II",contents[16:24])!=(960,720)):
            raise ValueError("H161 native browser original image invalid")
        h=hashlib.sha256(contents).hexdigest()
        if (h!=actual["image_sha256"] or h in digests or
            h in earlier):
            raise ValueError("H161 consumed original UI image collision")
        digests.add(h)
        accepted_paths.add(name)
        counts[(actual["split"],actual["app"])]+=1
    if (set(p.name for p in (folder/"images").glob("*.png"))!=accepted_paths or
        counts!={(split,app):n for split,a in QUOTAS.items()
                 for app,n in a.items()}):
        raise ValueError("H161 source dataset incomplete or extra images")
    return {
        "status":"H161_72_UNIQUE_ORIGINAL_MIT_NATIVE_REFERENCE_EPISODES_AUDITED",
        "source_manifest_sha256":hashlib.sha256(raw).hexdigest(),
        "source_apps":list(APPS_USED),
        "train_images":48, "dev_images":24,
        "native_reference_success":72, "model_inference_calls":0,
        "original_source_images_excluded":len(earlier),
        "h140_blind_calls":0,"paid_gpu":False,
        "cross_app_blind_is_NOT_yet_qualified":True,
    }


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--owner",required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.owner),sort_keys=True,indent=2))
