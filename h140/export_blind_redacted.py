"""Redact ALL owner gold GUI geometry from sealed H140 unseen original apps.

Create an inference-only private ZIP with 32 frozen native screenshots and
task text. No DOM boxes, reference mouse coordinates or localStorage state.
Never mount this ZIP in the TRAIN/DEV Kaggle SFT kernel.
"""
import argparse
import io
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from PIL import Image

from h140.qualification import load_split,digest,WIDTH,HEIGHT
from h140.freeze_blind import SOURCES

SCHEMA="hala.h140.redacted-owner-private-one-shot-unseen-originals.v1"
OWNER_BLIND_SHA="6de30cff6ef31b0095c30a73bbe3dff3c4d98fbc72679d1d684f75f50ed6e162"
ALLOWED_FIELDS={
    "id","app","app_git_sha","language",
    "instruction","image","image_sha256"
}

def redacted_cases(doc):
    if (doc.get("split")!="blind_test" or
        doc.get("training_eligible") is not False or
        set(doc.get("app_sources",{}))!=set(SOURCES) or
        len(doc.get("cases",[]))!=32):
        raise ValueError("H140 only frozen independent 2-app source blind, no TRAIN or DEV")
    count=Counter()
    seen=set()
    rows=[]
    for source in doc["cases"]:
        app=source.get("app")
        if app not in SOURCES or source.get("app_git_sha")!=SOURCES[app]["sha"]:
            raise ValueError("H140 original independent source SHA drift")
        if source.get("language") not in ("en","ar"):
            raise ValueError("H140 original bilingual model input missing")
        if source.get("image_sha256") in seen:
            raise ValueError("H140 duplicate original blind input image")
        seen.add(source["image_sha256"])
        count[(app,source["language"])]+=1
        row={k:source[k] for k in ALLOWED_FIELDS}
        if set(row)!=ALLOWED_FIELDS:
            raise ValueError("H140 redacted native model input fields incomplete")
        rows.append(row)
    if any(count[(app,lang)]!=8 for app in SOURCES for lang in ("en","ar")):
        raise ValueError("H140 private one-shot independent-app EN/AR role balance drift")
    return rows

def create(owner_root,destination):
    owner_root=Path(owner_root)
    destination=Path(destination)
    source=owner_root/"frozen-source-v1"/"blind_test"
    if (destination.exists() or
        destination.with_suffix(".incomplete").exists() or
        not any(p.name=="private-h140" for p in destination.parents)):
        raise ValueError("H140 private redacted model input must be create-only")
    original=(source/"manifest.json").read_bytes()
    if digest(original)!=OWNER_BLIND_SHA:
        raise ValueError("H140 sealed owner-original SHA mismatch")
    receipt=json.loads((source/"OWNER_BLIND_SEAL_RECEIPT.json").read_text("utf8"))
    if (receipt.get("status")!="SEALED_OOD_SOURCE_ONLY_NEURAL_NOT_YET_RUN"
        or receipt.get("owner_manifest_sha256")!=OWNER_BLIND_SHA
        or receipt.get("new_original_blind_cases")!=32
        or receipt.get("real_neural_calls")!=0):
        raise ValueError("H140 one-shot owner premodel BLIND seal changed")
    doc=load_split(source,"blind_test")
    safe=redacted_cases(doc)
    model_manifest={
        "schema":SCHEMA,
        "source_role":"blind_test",
        "private":True,
        "training_eligible":False,
        "owner_original_manifest_sha256":OWNER_BLIND_SHA,
        "source_apps":{app:SOURCES[app]["sha"] for app in SOURCES},
        "case_count":32,"cases":safe,
    }
    raw=(json.dumps(model_manifest,ensure_ascii=False,sort_keys=True,
                    indent=2)+"\n").encode("utf8")
    forbidden=(b"start_bbox",b"end_bbox",b"true_start_point",
               b"true_end_point",b"localStorage",b"reference_drag",
               b"source_task_id",b"source_task_state")
    if any(key in raw for key in forbidden):
        raise ValueError("H140 private BLIND raw model manifest contains oracle data")
    temp=destination.with_suffix(".incomplete")
    with ZipFile(temp,"w",ZIP_DEFLATED,compresslevel=2) as z:
        z.writestr("manifest.json",raw)
        for row in safe:
            image=(source/row["image"]).read_bytes()
            if digest(image)!=row["image_sha256"]:
                raise ValueError("H140 source PNG drift during blind redaction")
            with Image.open(io.BytesIO(image)) as decoded:
                if decoded.format!="PNG" or decoded.size!=(WIDTH,HEIGHT):
                    raise ValueError("H140 never resize/paint original blinded screenshot")
                decoded.load()
            z.writestr(row["image"],image)
    with ZipFile(temp) as z:
        entries=z.namelist()
        if (len(entries)!=33 or len(set(entries))!=33 or
            set(entries)!={"manifest.json"}|{x["image"] for x in safe}
            or z.testzip() is not None
            or digest(z.read("manifest.json"))!=digest(raw)):
            raise ValueError("H140 one-shot model input ZIP integrity error")
        actual=json.loads(z.read("manifest.json"))
        if any(set(x)!=ALLOWED_FIELDS for x in actual["cases"]):
            raise ValueError("H140 original unseen model input payload leaked owner gold")
    temp.rename(destination)
    return {"schema":"hala.h140.one-shot-owner-model-input-export-receipt.v1",
            "status":"REDACTED_PRIVATE_BLIND32_NOT_MODEL_EVALUATED",
            "owner_source_sha256":OWNER_BLIND_SHA,
            "redacted_manifest_sha256":digest(raw),
            "private_zip_sha256":digest(destination.read_bytes()),
            "case_count":32,
            "original_images_unmodified":32,
            "owner_gold_coordinates_in_model_input":False,
            "real_model_calls":0,
            "trained_on_blind":False,
            "flagship_Hala_35B_updated":False}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--private-root",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    print(json.dumps(create(a.private_root,a.output),sort_keys=True,indent=2))
