"""Independent offline-only H49 source, browser pixels, semantic target and SFT audit.

This is original first-party public synthetic DEV, NOT independently blind TEST.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import io
from PIL import Image
from h49.factory import (RIGHTS, SCHEMA, SOURCES, WIDTH, HEIGHT, scenes, validate,
                         render, canonical, sha)
from training.artifacts import validate_action_completion

DATASET = "authoring_sft.jsonl"
MANIFEST = "sft_manifest.json"
INPUTS = "model_inputs.json"
EVALUATOR = "evaluator_only.json"
INTEGRITY = "integrity.json"

def inspect(root: Path, *, capture_override=None):
    root=Path(root)
    source_path=root/"source.json"
    source=json.loads(source_path.read_text("utf-8"))
    cases=scenes()
    validate(cases)
    rows=[]
    for row in cases:
        payload=render(row)
        rows.append(dict(id=row["id"],family=row["family"],split=row["split"],
            language=row["language"],goal=row["goal"],target_index=row["target_index"],
            target_key=row["target_key"],html_sha256=sha(payload)))
    expected_source=dict(schema=SCHEMA,rights=RIGHTS,sources_for_vocabulary_only=SOURCES,
        count=96,train=64,public_dev=32,independent_blind_test=False,
        real_service_test=False,model_inference=False,rows=rows)
    if source!=expected_source or source_path.read_bytes()!=canonical(expected_source):
        raise ValueError("Original H49 public domain source changed")
    capture=(json.loads((root/"capture.json").read_text("utf-8")) if capture_override is None
             else capture_override)
    if (capture.get("schema")!="hala.h49.actual-offline-edge-capture.v1"
        or capture.get("source_sha256")!=sha(source_path.read_bytes())
        or capture.get("puppeteer_core")!="25.12.0"
        or not str(capture.get("browser_version","")).startswith("Edg/")
        or capture.get("screenshots")!=96
        or (capture.get("width"),capture.get("height"))!=(WIDTH,HEIGHT)
        or any(capture.get(field) is not False for field in
            ("real_service_access","network_used","native_actions","model_inference"))
        or type(capture.get("external_requests_rejected")) is not int
        or type(capture.get("file_requests_permitted")) is not int
        or len(capture.get("records",[]))!=96):
        raise ValueError("H49 genuine browser capture admission failed")
    model_inputs=[]
    sft=[]
    media=[]
    oracle=[]
    for expected,found in zip(cases,capture["records"]):
        if (found.get("id")!=expected["id"] or
            found.get("html_sha256")!=rows[len(model_inputs)]["html_sha256"] or
            found.get("language")!=expected["language"] or
            found.get("family")!=expected["family"] or
            found.get("split")!=expected["split"]):
            raise ValueError("H49 browser evidence mismatched row/order")
        path=root/"html"/(expected["id"]+".html")
        payload=render(expected)
        if path.read_bytes()!=payload:
            raise ValueError("H49 source HTML modified")
        png_bytes=(root/"png"/(expected["id"]+".png")).read_bytes()
        if (not 1024 <= len(png_bytes) <= 400000 or
            png_bytes[:8]!=b"\x89PNG\r\n\x1a\n" or
            sha(png_bytes)!=found.get("png_sha256") or
            len(png_bytes)!=found.get("png_bytes")):
            raise ValueError("Original H49 rendered PNG digest/size mismatch")
        with Image.open(io.BytesIO(png_bytes)) as img:
            if img.format!="PNG" or img.size!=(WIDTH,HEIGHT):
                raise ValueError("Wrong H49 screenshot size/format")
            img.verify()
        actual=found.get("controls")
        if type(actual) is not list or len(actual)!=6:
            raise ValueError("Missing original H49 browser controls")
        boxes=[]
        for i,(c,item) in enumerate(zip(actual,expected["choices"])):
            verb="اختر" if expected["language"]=="ar" else "Select"
            if (set(c)!={"index","key","label","text","bounds","visible"}
                or c["index"]!=i or c["key"]!=item["key"]
                or c["label"]!=f"{verb}: {item['label']}"
                or c["text"]!=verb or c["visible"] is not True
                or type(c["bounds"]) is not list or len(c["bounds"])!=4):
                raise ValueError("Browser control identity/display mismatch")
            x,y,w,h=c["bounds"]
            if (not all(type(q) in (int,float) and math.isfinite(q)
                        for q in (x,y,w,h))
                or x<0 or y<0 or w<50 or h<35
                or x+w>WIDTH or y+h>610):
                raise ValueError("H49 UI control not uniquely onscreen")
            boxes.append((x,y,w,h))
        for i,(x,y,w,h) in enumerate(boxes):
            for a,b,p,q in boxes[i+1:]:
                if max(x,a)<min(x+w,a+p) and max(y,b)<min(y+h,b+q):
                    raise ValueError("H49 controls visually overlapping")
        actual_goal=found.get("golden")
        wrong=found.get("negative")
        expected_index=expected["target_index"]
        expected_key=expected["target_key"]
        if (actual_goal!={"selected":expected_key,"clicks":1} or
            type(wrong) is not dict or set(wrong)!={"selected","clicks","wrong_index"} or
            wrong["wrong_index"]!=(expected_index+1)%6 or
            wrong["selected"]!=expected["choices"][wrong["wrong_index"]]["key"] or
            wrong["selected"]==expected_key or wrong["clicks"]!=1):
            raise ValueError("Independent click postcondition/genuine wrong control failed")
        x,y,w,h=boxes[expected_index]
        px,py=round(x+w/2),round(y+h/2)
        if not (0<=px<WIDTH and 0<=py<HEIGHT):
            raise ValueError("Model click target outside frame")
        if sum(a<=px<a+p and b<=py<b+q for a,b,p,q in boxes)!=1:
            raise ValueError("Training label not unique DOM control")
        input_row=dict(id=expected["id"],family=expected["family"],
            split=expected["split"],language=expected["language"],goal=expected["goal"],
            image="png/"+expected["id"]+".png",image_sha256=found["png_sha256"],
            width=WIDTH,height=HEIGHT,independent_blind_test=False)
        model_inputs.append(input_row)
        oracle.append(dict(id=expected["id"],split=expected["split"],
            family=expected["family"],language=expected["language"],
            target_index=expected_index,target_key=expected_key,
            target_bounds=list(boxes[expected_index]),image_sha256=found["png_sha256"],
            original_postcondition_verified=True,independent_blind_test=False))
        if expected["split"]=="train":
            action=dict(type="click",point=[px,py],coordinate_space="window",
                        image_scale=1,element_ref_hint="visual_target")
            completion=json.dumps(action,sort_keys=True,ensure_ascii=False)
            validate_action_completion(completion)
            sft.append(dict(id=expected["id"],split="train",family=expected["family"],
                language=expected["language"],prompt=(
                    "Select the single intended visible control from this screenshot. "
                    "Return only one JSON window-pixel click proposal; do not act.\n"
                    "Screenshot dimensions: 1024 by 640.\nGoal: "+expected["goal"]),
                completion=completion,image_path=input_row["image"],
                image_sha256=input_row["image_sha256"]))
            media.append(dict(path=input_row["image"],sha256=found["png_sha256"],
                              family=expected["family"],provenance=RIGHTS))
    training_families=sorted({r["family"] for r in sft})
    dev_families=sorted({r["family"] for r in oracle if r["split"]=="dev"})
    if (len(sft)!=64 or len(model_inputs)!=96 or len(oracle)!=96
        or len({r["id"] for r in sft})!=64 or
        len(training_families)!=4 or len(dev_families)!=2 or
        set(training_families)&set(dev_families) or
        sum(r["language"]=="ar" for r in sft)!=32 or
        sum(r["language"]=="en" for r in sft)!=32):
        raise ValueError("H49 SFT authoring/DEV split invalid")
    dataset="".join(json.dumps(r,ensure_ascii=False,sort_keys=True,allow_nan=False)+"\n"
                    for r in sft).encode("utf-8")
    manifest=dict(schema="hala.training.manifest.v1",source_schema=SCHEMA,
        split="train",source_kind="synthetic",rights_cleared=True,
        source_sha256=sha(source_path.read_bytes()),
        capture_sha256=sha((root/"capture.json").read_bytes()),
        dataset_sha256=sha(dataset),families=training_families,
        held_out_families=dev_families,token_budget=4096,media=media,
        language_policy=dict(arabic=["justice_execution","traffic_services"],
                             english=["fhir_resources","fhir_bundles"]),
        source_rights=RIGHTS,real_service_access=False,
        official_benchmark_test=False,independent_blind_test=False,
        training_executed=False,adapter_qualified=False)
    projections={
        DATASET:dataset,MANIFEST:canonical(manifest),
        INPUTS:canonical(dict(schema="hala.h49.model-only.v1",cases=model_inputs,
                              independent_blind_test=False)),
        EVALUATOR:canonical(dict(schema="hala.h49.evaluator-only.v1",cases=oracle,
                                 independent_blind_test=False)),
        INTEGRITY:canonical(dict(schema="hala.h49.original-edge-audit.v1",
            screenshots=len(oracle),train_examples=len(sft),public_dev=32,
            arabic=sum(r["language"]=="ar" for r in oracle),
            english=sum(r["language"]=="en" for r in oracle),
            visible_buttons=576,correct_click_replays=96,
            wrong_target_rejections=96,real_service_access=False,
            source_sha256=manifest["source_sha256"],
            capture_sha256=manifest["capture_sha256"],
            dataset_sha256=manifest["dataset_sha256"],
            model_trained=False,independent_blind_test=False))}
    return projections,manifest

def load_export(path,manifest):
    path=Path(path)
    if path.name!=DATASET or path.is_symlink():
        raise ValueError("Require original H49 training export path")
    projections,expected=inspect(path.parent)
    if (manifest!=expected or
        any((path.parent/name).read_bytes()!=payload for name,payload in projections.items())):
        raise ValueError("Original H49 signed source/data/labels changed")
    rows=[json.loads(line) for line in projections[DATASET].decode("utf-8").splitlines()]
    for row in rows:
        validate_action_completion(row["completion"])
    return rows

def export(root):
    root=Path(root)
    projections,_=inspect(root)
    if any((root/name).exists() for name in projections):
        raise FileExistsError("H49 export is strictly create-only")
    created=[]
    try:
        for name,data in projections.items():
            dest=root/name
            with dest.open("xb") as file:
                file.write(data)
            created.append(dest)
        manifest=json.loads((root/MANIFEST).read_text("utf-8"))
        load_export(root/DATASET,manifest)
    except Exception:
        for name in created:
            name.unlink()
        raise
    return dict(outputs=list(projections),manifest_sha256=sha(projections[MANIFEST]),
                dataset_sha256=sha(projections[DATASET]))
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--root",default="h49/assets")
    print(json.dumps(export(p.parse_args().root)))
