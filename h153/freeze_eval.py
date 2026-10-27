"""H153 completely NEW shifted-layout original GUI images, no trained-media leakage.

Owner-only TRUE rectangles stay on Windows. Kaggle receives exactly 16
original screenshots + language/instruction/source Git SHA, NEVER DOM gold.
"""
import argparse
import io
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from PIL import Image

from h140.qualification import digest
from h140.vanilla_pilot import visible_box
from h142.qualify import history,FOLDERS
from h153.collect_shifted_source import (
    SCHEMA,CASES,APP_SHA,TRANSFORMS,style,source_case
)

FOLDER="shifted-eval16-v1"
EXPORT_SCHEMA="hala.h153.private-original-shifted-vision-probe-input.v1"
MODEL_FIELDS=("id","app","app_git_sha","language","destination_status",
              "layout","instruction","image","image_sha256")
EXPECTED={"ar/doing":4,"ar/done":4,"en/doing":4,"en/done":4}
def historical_shas(root):
    root=Path(root)
    old=history(root)
    # All earlier model-observed and consumed H143 screenshots must be
    # excluded from H153 before any new model inference.
    earlier=root/"private-h143"/"shifted-eval16-v1"/"manifest.json"
    prev=json.loads(earlier.read_text("utf8"))
    if len(prev.get("cases",[]))!=16:
        raise ValueError("H153 prior H143 evaluation cohort missing")
    old.update(r["image_sha256"] for r in prev["cases"])
    # Previously evaluated model sources are permanently excluded.
    prior=root/"private-h151"/"shifted-eval16-v1"/"manifest.json"
    earlier=json.loads(prior.read_text("utf8"))
    if len(earlier.get("cases",[]))!=16:
        raise ValueError("H153 previous real neural cohort missing")
    old.update(row["image_sha256"] for row in earlier["cases"])
    for role,folder in FOLDERS.items():
        dir=root/"private-h142"/folder
        obj=json.loads((dir/"manifest.json").read_text("utf8"))
        if obj.get("split")!=role:
            raise ValueError("H153 prior train/dev original role drift")
        old.update(x["image_sha256"] for x in obj["cases"])
    return old

def audit(root):
    root=Path(root)
    folder=root/"private-h153"/FOLDER
    raw=(folder/"manifest.json").read_bytes()
    doc=json.loads(raw)
    cases=doc.get("cases")
    if (doc.get("schema")!=SCHEMA or
        doc.get("status")!="NEW_SHIFTED_LAYOUT_ORIGINALS_NO_NEURAL_TEST" or
        doc.get("private") is not True or
        doc.get("training_eligible") is not False or
        doc.get("model_evaluation_eligible") is not True or
        doc.get("h140_blind_eligible") is not False or
        doc.get("source_app_sha")!=APP_SHA or
        doc.get("case_count")!=CASES or
        doc.get("neural_calls_before_freeze")!=0 or
        not isinstance(cases,list) or len(cases)!=CASES):
        raise ValueError("H153 independent original shifted layout manifest invalid")
    old=historical_shas(root)
    seen=set();case_ids=set();attempts=set();counts=Counter();layouts=Counter()
    for item in cases:
        attempt=item.get("attempt_index")
        if (item.get("id") in case_ids or
            not isinstance(attempt,int) or attempt in attempts or
            item.get("app_git_sha")!=APP_SHA or
            item.get("layout") not in TRANSFORMS or
            item.get("language") not in ("ar","en") or
            item.get("destination_status") not in ("doing","done") or
            item.get("css_in_browser_only") is not True or
            item.get("training_eligible") is not False or
            item.get("model_evaluation_eligible") is not True or
            item.get("h140_blind_eligible") is not False or
            item.get("source_reference_drag_persisted") is not True or
            item.get("reference_survived_reload") is not True or
            item.get("neural_calls_before_freeze")!=0 or
            not visible_box(item.get("start_bbox")) or
            not visible_box(item.get("visible_destination_column_heading_bbox"))):
            raise ValueError("H153 shifted source owner gold/provenance invalid")
        origin=source_case(attempt)
        if (item["language"]!=origin["language"] or
            item["destination_status"]!=origin["destination_status"] or
            item["instruction"]!=origin["instruction"] or
            item["layout"]!=origin["layout"]):
            raise ValueError("H153 original source seed changed")
        h=item.get("image_sha256")
        fn=item.get("image")
        if (not isinstance(h,str) or len(h)!=64 or
            h in old or h in seen or
            not isinstance(fn,str) or not fn.endswith(".png") or
            "/" in fn or "\\" in fn or ".." in fn):
            raise ValueError("H153 screenshot historical leak or duplicate")
        img=(folder/fn).read_bytes()
        if digest(img)!=h:
            raise ValueError("H153 native image SHA corrupted")
        with Image.open(io.BytesIO(img)) as p:
            if p.format!="PNG" or p.size!=(960,720):
                raise ValueError("H153 original viewport not native 960x720")
            p.load()
        box=item["visible_destination_column_heading_bbox"]
        baseline_x=480 if item["destination_status"]=="doing" else 749
        if (abs(box["y"]+box["height"]/2-181)<18 and
            abs(box["x"]+box["width"]/2-baseline_x)<15):
            raise ValueError("H153 original heading did not shift away from training")
        case_ids.add(item["id"]);attempts.add(attempt);seen.add(h)
        counts[item["language"]+"/"+item["destination_status"]]+=1
        layouts[item["layout"]]+=1
    if counts!=EXPECTED or set(layouts)!=set(TRANSFORMS) or sum(layouts.values())!=16:
        raise ValueError("H153 original shifted layouts and bilingual goals unbalanced")
    return {"schema":"hala.h153.owner-independent-shifted-source-audit.v1",
            "status":"FROZEN_NEW_SHIFTED_ORIGINALS_NOT_TRAINED",
            "manifest_sha256":digest(raw),"cases":16,
            "historical_excluded_sha":len(old),
            "fresh_unique_png":len(seen),
            "new_layouts":dict(layouts),
            "real_native_reference_drags_persisted":16,
            "training_eligible":False,
            "h140_blind_neural_calls":0}
def export(root,output):
    root=Path(root);output=Path(output)
    if (output.exists() or output.with_suffix(".incomplete").exists()
        or not any(x.name=="private-h153" for x in output.parents)):
        raise ValueError("H153 private original media create-only")
    result=audit(root)
    folder=root/"private-h153"/FOLDER
    doc=json.loads((folder/"manifest.json").read_text("utf8"))
    safe=[{k:r[k] for k in MODEL_FIELDS} for r in doc["cases"]]
    source={"schema":EXPORT_SCHEMA,
            "source_manifest_sha256":result["manifest_sha256"],
            "source_app_git_sha":APP_SHA,"role":"EVAL_ONLY",
            "training_eligible":False,
            "model_action_roles":["start","visible_column_heading"],
            "cases":safe}
    raw=(json.dumps(source,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode()
    if any(word in raw for word in (
        b"start_bbox",b"heading_bbox",b"end_bbox",b"reference_drag",
        b"localStorage",b"true_start_point",b"source_index")):
        raise ValueError("H153 owner gold leaked to inference archive")
    part=output.with_suffix(".incomplete")
    with ZipFile(part,"w",ZIP_DEFLATED,compresslevel=2) as z:
        z.writestr("manifest.json",raw)
        for r in safe:
            pixels=(folder/r["image"]).read_bytes()
            if digest(pixels)!=r["image_sha256"]:
                raise ValueError("H153 original PNG changed during model archive creation")
            z.writestr(r["image"],pixels)
    with ZipFile(part) as z:
        if (z.testzip() is not None or len(z.namelist())!=17 or
            digest(z.read("manifest.json"))!=digest(raw)):
            raise ValueError("H153 original redacted model zip invalid")
    part.rename(output)
    return {"status":"PRIVATE_H153_ORIGINAL_REDACTED_INFERENCE_READY",
            "zip_sha256":digest(output.read_bytes()),
            "model_manifest_sha256":digest(raw),
            "source_manifest_sha256":result["manifest_sha256"],
            "cases":16,
            "owner_oracle_boxes_uploaded":False,
            "h140_blind_neural_calls":0}
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner-root",required=True)
    p.add_argument("--export")
    args=p.parse_args()
    print(json.dumps(export(args.owner_root,args.export) if args.export
                     else audit(args.owner_root),
                     indent=2,sort_keys=True))
