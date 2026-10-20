"""H44 independent CPU replay of authentic Edge PNG bytes + DOM bounding boxes.

Model-only screenshot/goal input is strictly separate from evaluator-only target
and bounds. This is first-party browser-rendered synthetic data, NOT external.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from h37.freeze import checked_png
from h43.factory import author_cases,canonical,digest
from h44.render import page

def sha(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


def verify(root:Path, *, capture_override:dict|None=None)->tuple[dict,dict,dict]:
    root=Path(root)
    source=author_cases()
    plan=root/"scenes.json"
    raw=plan.read_bytes()
    plan_obj=json.loads(raw)
    cases=source["cases"]
    if (plan_obj.get("schema")!="hala.h44.original-html-plan.v1"
        or plan_obj.get("h43_manifest_sha256")!=digest(source)
        or len(plan_obj.get("scenes",[]))!=24):
        raise ValueError("Invalid original H44 source manifest")
    record=json.loads((root/"capture.json").read_text("utf-8")) if capture_override is None else capture_override
    if (record.get("schema")!="hala.h44.edge-browser-capture.v1"
        or record.get("puppeteer_core")!="25.12.0"
        or not str(record.get("browser_version","")).startswith("Edg/155.")
        or record.get("viewport")!={"width":960,"height":540,"device_scale_factor":1}
        or record.get("scene_plan_sha256")!=sha(raw)
        or record.get("original_h43_manifest_sha256")!=digest(source)
        or record.get("model_inference") is not False
        or record.get("no_external_network") is not True
        or record.get("native_actions") is not False
        or record.get("model_oracle_exposure") is not False
        or len(record.get("rendered_records",[]))!=24):
        raise ValueError("Actual browser capture metadata mismatched")
    projection=[]
    evaluator=[]
    for case,scene,rec in zip(cases,plan_obj["scenes"],record["rendered_records"]):
        if any(rec.get(k)!=case[k] for k in ("id","language","layout","split","family")):
            raise ValueError("Scene and capture row differ")
        if (scene["id"]!=case["id"] or scene["goal"]!=case["goal"]
            or scene["family"]!=case["family"] or scene["split"]!=case["split"]
            or scene["rtl"]!=case["rtl"]):
            raise ValueError("H43 source provenance mismatched")
        html_data=(root/"pages"/(case["id"]+".html")).read_bytes()
        html_hash=sha(html_data)
        if (html_data!=page(case).encode("utf-8") or
            html_hash!=scene["page_sha256"] or html_hash!=rec["page_sha256"]):
            raise ValueError("HTML source was changed or injected")
        png=(root/"screenshots"/(case["id"]+".png")).read_bytes()
        checked_png(png)
        if sha(png)!=rec["image_sha256"] or len(png)!=rec["image_bytes"]:
            raise ValueError("PNG hash/bytes differ from observed browser record")
        controls=rec["controls"]
        expected=case["controls"]
        if type(controls) is not list or len(controls)!=len(expected):
            raise ValueError("Actual accessibility snapshot incomplete")
        bounds=[]
        for a,b in zip(controls,expected):
            if (set(a)!={"ref","name","role","visible","bounds"}
                or any(a.get(k)!=b[k] for k in ("ref","name","role"))
                or a["visible"] is not True or
                type(a["bounds"]) is not list or len(a["bounds"])!=4):
                raise ValueError("Browser DOM control name/role mismatch")
            x,y,w,h=a["bounds"]
            if (not all(type(v) in (int,float) and __import__("math").isfinite(v) for v in (x,y,w,h))
                or x<0 or y<0 or w<60 or h<30 or x+w>960 or y+h>540):
                raise ValueError("Browser target not uniquely visible/in viewport")
            bounds.append((x,y,w,h))
        for i,(x,y,w,h) in enumerate(bounds):
            for k,(a,b,c,e) in enumerate(bounds):
                if i<k and max(x,a)<min(x+w,a+c) and max(y,b)<min(y+h,b+e):
                    raise ValueError("Observed browser controls overlap")
        target=next(v for v in controls if v["ref"]=="row_target")
        projection.append({"id":case["id"],"language":case["language"],
                           "split":case["split"],"family":case["family"],
                           "goal":case["goal"],"screenshot":"screenshots/"+case["id"]+".png",
                           "image_sha256":rec["image_sha256"],"width":960,"height":540,
                           "model_trained":False})
        evaluator.append({"id":case["id"],"language":case["language"],
                          "split":case["split"],"family":case["family"],
                          "expected_ref":"row_target",
                          "target_bounds":target["bounds"],
                          "controls":controls,"image_sha256":rec["image_sha256"],
                          "model_not_shown_oracle":True})
    assert {q["id"] for q in projection}=={q["id"] for q in evaluator}
    return (
        {"schema":"hala.h44.model-only-projection.v1","cases":projection,
         "independent_blind_test":False,"model_qualified":False},
        {"schema":"hala.h44.evaluator-only-oracle.v1","cases":evaluator,
         "independent_blind_test":False,"model_qualified":False},
        {"schema":"hala.h44.browser-evidence-audit.v1",
         "h43_manifest_sha256":digest(source),"h44_source_plan_sha256":sha(raw),
         "h44_capture_sha256":sha((root/"capture.json").read_bytes()),
         "actual_chromium_screenshots":24,"english":12,"arabic":12,
         "public_authoring":16,"public_dev":8,
         "model_inference":False,"independent_test":False,
         "native_actions":False,"network_requests":False,
         "all_controls_unique_visible_no_overlap":True}
    )
