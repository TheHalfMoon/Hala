"""H48 offline validation of original Chromium training screenshots/labels.

No H47 DEV or benchmark trajectories used for H48 training projection.
DOM target provides a supervised teacher *only* within new authored train set.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from h37.freeze import checked_png
from h48.factory import source_cases,page,canon,sha,SCHEMA,FAMILIES

def verify(root:Path)->tuple[dict,dict,dict]:
    root=Path(root)
    original=(root/"source_manifest.json").read_bytes()
    manifest=json.loads(original)
    captured=(root/"capture.json").read_bytes()
    receipt=json.loads(captured)
    expected=source_cases()
    if (manifest.get("schema")!=SCHEMA or manifest.get("train_only") is not True
        or manifest.get("external_benchmark") is not False
        or manifest.get("blind_holdout") is not False
        or manifest.get("model_qualified") is not False
        or len(manifest.get("sources",[]))!=128):
        raise ValueError("Invalid H48 original source split and claim")
    if (receipt.get("schema")!="hala.h48.actual-chromium-128-train-images.v1"
        or receipt.get("source_manifest_sha256")!=sha(original)
        or receipt.get("puppeteer_core")!="25.12.0"
        or not str(receipt.get("chromium_version","")).startswith("Edg/155.")
        or (receipt.get("width"),receipt.get("height"))!=(960,540)
        or receipt.get("train_only") is not True
        or receipt.get("benchmark") is not False
        or receipt.get("model_inference") is not False
        or receipt.get("native_actions") is not False
        or receipt.get("external_network_requests") is not False
        or len(receipt.get("records",[]))!=128):
        raise ValueError("Browser capture does not match source or honest scope")
    input_only=[]
    train_labels=[]
    all_families=set()
    language={"en":0,"ar":0}
    for idx,(case,source,rec) in enumerate(zip(expected,manifest["sources"],receipt["records"])):
        expected_p=page(case)
        if (source.get("id")!=case["id"] or source.get("goal")!=case["goal"]
            or source.get("language")!=case["language"] or source.get("layout")!=case["layout"]
            or source.get("family")!=case["family"] or source.get("labels")!=case["labels"]
            or source.get("target_option_index")!=case["target_option_index"]
            or source.get("source_sha256")!=sha(expected_p)
            or rec.get("id")!=case["id"]
            or rec.get("source_sha256")!=sha(expected_p)
            or rec.get("language")!=case["language"]):
            raise ValueError("Original target/labels/source mismatch")
        if (root/"pages"/(case["id"]+".html")).read_bytes()!=expected_p:
            raise ValueError("HTML changed from original source")
        img=(root/"screenshots"/(case["id"]+".png")).read_bytes()
        checked_png(img)
        if sha(img)!=rec.get("image_sha256") or len(img)!=rec.get("image_bytes"):
            raise ValueError("Original browser screenshot mutated")
        controls=rec.get("controls")
        if type(controls) is not list or len(controls)!=4:
            raise ValueError("Missing browser controls")
        allrect=[]
        for i,(ctrl,correct_name) in enumerate(zip(controls,case["labels"])):
            if (type(ctrl) is not dict or set(ctrl)!={"option_index","name","rect","visible"}
                or ctrl["option_index"]!=i or ctrl["name"]!=correct_name
                or ctrl["visible"] is not True or
                type(ctrl["rect"]) is not list or len(ctrl["rect"])!=4):
                raise ValueError("Browser label mismatch or malformed")
            x,y,w,h=ctrl["rect"]
            if not all(type(v) in (int,float) and math.isfinite(v) for v in (x,y,w,h)):
                raise ValueError("Invalid real browser rectangle")
            if x<0 or y<0 or w<72 or h<39 or x+w>960 or y+h>540:
                raise ValueError("Bounding rectangle outside screenshot")
            allrect.append((x,y,w,h))
        for j,(x,y,w,h) in enumerate(allrect):
            for k,(a,b,c,e) in enumerate(allrect):
                if j<k and max(x,a)<min(x+w,a+c) and max(y,b)<min(y+h,b+e):
                    raise ValueError("Browser actionable controls overlap")
        target=controls[case["target_option_index"]]
        r=target["rect"]
        center=[round((r[0]+r[2]/2)/960,7),round((r[1]+r[3]/2)/540,7)]
        if not(0<center[0]<1 and 0<center[1]<1):
            raise ValueError("Target point outside image")
        all_families.add(case["family"])
        language[case["language"]]+=1
        path="screenshots/"+case["id"]+".png"
        input_only.append({"id":case["id"],"goal":case["goal"],
                           "language":case["language"],"family":case["family"],
                           "screenshot":path,"image_sha256":rec["image_sha256"],
                           "width":960,"height":540,"split":"train_only"})
        train_labels.append({"id":case["id"],"family":case["family"],
                             "language":case["language"],"split":"train_only",
                             "screenshot_sha256":rec["image_sha256"],
                             "target_label":case["labels"][case["target_option_index"]],
                             "target_point":center,"target_rect":r,
                             "all_controls":controls,"provenance":"new_original_first_party_synthetic",
                             "license":"CC0-1.0"})
    if len(all_families)!=8 or language!={"en":64,"ar":64}:
        raise ValueError("Original bilingual training count and family coverage invalid")
    return (
        {"schema":"hala.h48.image-only-train-observations.v1",
         "split":"train_only","model_qualified":False,"cases":input_only},
        {"schema":"hala.h48.supervised-train-only-targets.v1",
         "split":"train_only","model_qualified":False,"cases":train_labels},
        {"schema":"hala.h48.actual-browser-training-audit.v1",
         "source_manifest_sha256":sha(original),"browser_capture_sha256":sha(captured),
         "original_chromium_pngs":128,"control_boxes_checked":512,
         "english":64,"arabic":64,"original_families":sorted(all_families),
         "genuine_training_samples":128,
         "real_app_or_benchmark_results":False,
         "independent_blind_test":False,"model_trained":False}
    )
