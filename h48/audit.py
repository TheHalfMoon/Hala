"""H48 independent integrity replay of first-party Edge pixels and real DOM boxes.

Evaluator learns correct code by original canonical source, not by model guess.
No native/real-app clicks. Keeps public DEV completely out of training export.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import io
from PIL import Image
from h48.factory import scenes,validate,render,canonical,sha,PROVENANCE,WIDTH,HEIGHT,SCHEMA

def verify(root:Path, *, raw_override:dict|None=None)->tuple[dict,dict,dict,dict]:
    root=Path(root)
    src=root/"source_manifest.json"
    source=json.loads(src.read_text(encoding="utf-8"))
    cases=scenes()
    validate(cases)
    if (source.get("schema")!=SCHEMA or source.get("provenance")!=PROVENANCE or
        source.get("count")!=96 or source.get("authoring_count")!=64 or
        source.get("public_dev_count")!=32 or
        source.get("independent_blind_test") is not False or
        source.get("actual_model_result") is not False or len(source.get("rows",[]))!=96):
        raise ValueError("Original H48 source manifest differs")
    capture=json.loads((root/"capture.json").read_text("utf-8")) if raw_override is None else raw_override
    if (capture.get("schema")!="hala.h48.original-edge-pixels.v1" or
        not str(capture.get("browser_version","")).startswith("Edg/155.") or
        capture.get("puppeteer_core")!="25.12.0" or
        (capture.get("width"),capture.get("height"))!=(WIDTH,HEIGHT) or
        capture.get("source_manifest_sha256")!=sha(src.read_bytes()) or
        capture.get("screenshots")!=96 or
        capture.get("network_used") is not False or
        capture.get("native_actions") is not False or
        capture.get("model_inference") is not False or
        len(capture.get("records",[]))!=96):
        raise ValueError("Actual Chromium screen evidence differs")
    model_inputs=[]
    train=[]
    evaluator=[]
    for i,(scene,provenance,original) in enumerate(zip(cases,source["rows"],capture["records"])):
        if (provenance["id"]!=scene["id"] or provenance["target_index"]!=scene["target_index"] or
            provenance["goal"]!=scene["goal"] or provenance["split"]!=scene["split"] or
            provenance["language"]!=scene["language"]):
            raise ValueError("Scene source/oracle provenance mismatch")
        html_payload=(root/"html"/(scene["id"]+".html")).read_bytes()
        if (html_payload!=render(scene) or sha(html_payload)!=provenance["html_sha256"]):
            raise ValueError("Original H48 HTML changed")
        if (original.get("id")!=scene["id"] or original.get("source_sha256")!=sha(html_payload) or
            original.get("language")!=scene["language"] or original.get("split")!=scene["split"] or
            original.get("family")!=scene["family"]):
            raise ValueError("Actual capture/source row mismatch")
        data=(root/"png"/(scene["id"]+".png")).read_bytes()
        if not 1024 <= len(data) <= 400_000:
            raise ValueError("H48 screenshot PNG outside file-size bound")
        with Image.open(io.BytesIO(data)) as image:
            if image.format!="PNG" or image.size!=(WIDTH,HEIGHT):
                raise ValueError("H48 genuine PNG wrong format/dimensions")
            image.verify()
        if (data[:8]!=b"\x89PNG\r\n\x1a\n" or
            int.from_bytes(data[16:20],"big")!=WIDTH or
            int.from_bytes(data[20:24],"big")!=HEIGHT or
            sha(data)!=original["png_sha256"] or
            len(data)!=original["png_bytes"]):
            raise ValueError("Recorded original browser PNG altered")
        observed=original.get("controls")
        if type(observed) is not list or len(observed)!=6:
            raise ValueError("Expected 6 genuine DOM buttons")
        boxes=[]
        for j,(label,ctrl) in enumerate(zip(scene["items"],observed)):
            if (set(ctrl)!={"index","name","text","bounds","visible"} or
                ctrl["index"]!=j or ctrl["visible"] is not True or
                ctrl["name"]!=scene["verb"]+" "+label or ctrl["text"]!=scene["verb"] or
                type(ctrl["bounds"]) is not list or len(ctrl["bounds"])!=4):
                raise ValueError("Original observed control identity incorrect")
            x,y,w,h=ctrl["bounds"]
            if (not all(type(z) in (int,float) and math.isfinite(z) for z in (x,y,w,h))
                or x<0 or y<0 or w<65 or h<32 or x+w>WIDTH or y+h>HEIGHT):
                raise ValueError("DOM control offscreen/invalid")
            boxes.append([float(x),float(y),float(w),float(h)])
        for a,(x,y,w,h) in enumerate(boxes):
            for b,(i2,j2,p2,q2) in enumerate(boxes):
                if a<b and max(x,i2)<min(x+w,i2+p2) and max(y,j2)<min(y+h,j2+q2):
                    raise ValueError("Observed DOM target boxes overlap")
        target=boxes[scene["target_index"]]
        cx=(target[0]+target[2]/2)/WIDTH
        cy=(target[1]+target[3]/2)/HEIGHT
        normalized=[round(cx,8),round(cy,8)]
        input_row={"id":scene["id"],"family":scene["family"],"split":scene["split"],
              "language":scene["language"],"goal":scene["goal"],
              "image":"png/"+scene["id"]+".png",
              "image_sha256":original["png_sha256"],"width":WIDTH,"height":HEIGHT,
              "independent_test":False}
        model_inputs.append(input_row)
        if scene["split"]=="authoring":
            train.append({**input_row,"labelled_point":normalized,
                          "labelled_target_code":scene["target_code"],
                          "source_rights":PROVENANCE})
        evaluator.append({"id":scene["id"],"family":scene["family"],
              "split":scene["split"],"language":scene["language"],
              "expected_index":scene["target_index"],
              "expected_ref_code":scene["target_code"],
              "target_bounds":target,"labelled_point":normalized,
              "observed_buttons":observed,"image_sha256":original["png_sha256"],
              "qualified_independent_test":False})
    if (len(model_inputs),len(train),len(evaluator))!=(96,64,96):
        raise ValueError("H48 split audit wrong")
    dev={x["family"] for x in evaluator if x["split"]=="dev"}
    train_groups={x["family"] for x in train}
    if dev & train_groups or len(dev)!=2 or len(train_groups)!=4:
        raise ValueError("Leak from H48 public DEV into training export")
    return (
        {"schema":"hala.h48.model-only-pixels.v1","source_manifest_sha256":sha(src.read_bytes()),
         "independent_blind_test":False,"cases":model_inputs},
        {"schema":"hala.h48.authoring-labelled.v1","provenance":PROVENANCE,
         "independent_blind_test":False,"examples":train},
        {"schema":"hala.h48.evaluator-only.v1","independent_blind_test":False,
         "cases":evaluator},
        {"schema":"hala.h48.browser-original-integrity-audit.v1",
         "source_sha256":sha(src.read_bytes()),
         "capture_sha256":sha((root/"capture.json").read_bytes()),
         "original_browser_screenshots":96,"original_authoring":64,"public_dev":32,
         "arabic":48,"english":48,
         "visible_controls_verified":576,"model_trained":False,
         "actual_model_results":False,"independent_blind_test":False,
         "external_benchmark":False,"native_actions":False}
    )
