"""H134 original, fully owner-rendered Chromium GUI click-target corpus.

Only TRAIN split may be mounted in the training kernel. DEV is for choices
after adapter fit; SEALED_TEST is owner-side locked until final evaluation.
No H71 images, external training data or previous synthetic eval screenshots.
"""
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import quote
from h130.browser_env import WIDTH,HEIGHT,_chromium_binary

SCHEMA="hala.h134.original-chromium-start-end-source.v1"
SPLITS={"train":(1000,128),"dev":(20000,16),"sealed_test":(30000,16)}
FAMILIES=("slider","resize","column","drop")

def sha(data):
    return hashlib.sha256(data).hexdigest()

def seeds(split):
    if split not in SPLITS: raise ValueError("H134 exact split required")
    start,n=SPLITS[split]
    return range(start,start+n)

def point(box):
    if not isinstance(box,dict) or any(type(box.get(k)) not in (int,float)
                    or not math.isfinite(box[k]) for k in ("x","y","width","height")):
        raise ValueError("H134 actual browser target bbox absent/invalid")
    if box["width"]<=0 or box["height"]<=0:
        raise ValueError("H134 target bbox empty")
    x=round(box["x"]+box["width"]/2)
    y=round(box["y"]+box["height"]/2)
    if not (0<=x<WIDTH and 0<=y<HEIGHT):
        raise ValueError("H134 browser target outside actual viewport")
    return [x,y]

def norm(point_xy):
    x,y=point_xy
    return [round(x/WIDTH*1000),round(y/HEIGHT*1000)]

def render(split,output,prior_shas=()):
    if split not in SPLITS: raise ValueError("H134 invalid split")
    path=Path(output)
    if path.exists(): raise ValueError("H134 all source artifacts create-only")
    source=Path(__file__).with_name("fixture.html").resolve()
    if not source.is_file(): raise ValueError("H134 original GUI source missing")
    from playwright.sync_api import sync_playwright
    path.mkdir(parents=True)
    rows=[]
    with sync_playwright() as playwright:
        opts={"headless":True}
        executable=_chromium_binary()
        if executable:opts["executable_path"]=executable
        browser=playwright.chromium.launch(**opts)
        try:
            context=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                                        device_scale_factor=1,service_workers="block")
            page=context.new_page()
            for seed in seeds(split):
                page.goto(source.as_uri()+"?seed="+str(seed),wait_until="load")
                page.locator('[data-role="handle"]').wait_for()
                instruction=page.locator("#instruction").inner_text()
                handle_box=page.locator('[data-role="handle"]').bounding_box()
                target_box=page.locator("#target").bounding_box()
                start=point(handle_box);end=point(target_box)
                if len(instruction)<12:raise ValueError("H134 original instruction missing")
                picture=page.screenshot(type="png")
                if not picture.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError("H134 genuine Chromium PNG needed")
                image_name=f"h134-{split}-{seed}.png"
                (path/image_name).write_bytes(picture)
                page.mouse.move(*start)
                page.mouse.down()
                page.mouse.move(*end,steps=12)
                page.mouse.up()
                if page.locator("#stage").get_attribute("data-success")!="true":
                    raise ValueError(f"H134 actual oracle GUI task did not succeed: {seed}")
                rows.append({
                   "source_id":f"h134-{seed}",
                   "task_family":FAMILIES[seed%4],
                   "language":"ar" if seed%5==0 else "en",
                   "image":image_name,"image_sha256":sha(picture),
                   "instruction":instruction,
                   "start_px":start,"end_px":end,
                   "start_grid_1000":norm(start),
                   "end_grid_1000":norm(end),
                   "start_bbox":handle_box,"end_bbox":target_box,
                   "actual_browser_oracle_passed":True,
                })
            context.close()
        finally:
            browser.close()
    manifest={"schema":SCHEMA,"split":split,"case_count":len(rows),
              "viewport_width":WIDTH,"viewport_height":HEIGHT,
              "cases":rows,"created_from":"owner-original-self-authored-h134-offline-fixture",
              "external_content_or_past_H133_images_used":False,
              "oracle_labels_for_training_or_owner_scoring_only":True,
              "license_for_self_created_screenshots":"owner-original",
              "optimizer_steps":0}
    (path/"manifest.json").write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n","utf8")
    return validate(path,expect=split,prior_shas=prior_shas)

def validate(path,expect=None,prior_shas=()):
    path=Path(path)
    doc=json.loads((path/"manifest.json").read_text("utf8"))
    split=doc.get("split")
    if (doc.get("schema")!=SCHEMA or split not in SPLITS or
        (expect is not None and split!=expect) or
        doc.get("case_count")!=SPLITS[split][1] or
        doc.get("viewport_width")!=WIDTH or
        doc.get("viewport_height")!=HEIGHT or
        doc.get("external_content_or_past_H133_images_used") is not False):
        raise ValueError("H134 source provenance/split/count mismatch")
    cases=doc.get("cases")
    if not isinstance(cases,list) or len(cases)!=SPLITS[split][1]:
        raise ValueError("H134 source row count incomplete")
    previous=set(prior_shas)
    seen=set()
    for seed,case in zip(seeds(split),cases):
        if (case.get("source_id")!=f"h134-{seed}" or
            case.get("image")!=f"h134-{split}-{seed}.png" or
            case.get("task_family")!=FAMILIES[seed%4] or
            case.get("language")!=("ar" if seed%5==0 else "en") or
            case.get("actual_browser_oracle_passed") is not True):
            raise ValueError("H134 original screen row identity / outcome mismatch")
        file=path/case["image"]
        if not file.is_file() or file.is_symlink():raise ValueError("H134 image not regular file")
        image=file.read_bytes();actual=sha(image)
        if (not image.startswith(b"\x89PNG\r\n\x1a\n") or
            actual!=case.get("image_sha256") or actual in seen or actual in previous):
            raise ValueError("H134 screenshot missing/corrupt/overlap")
        seen.add(actual)
        for role in ("start","end"):
            point_data=case.get(role+"_px")
            bbox=case.get(role+"_bbox")
            if type(point_data) is not list or len(point_data)!=2 or point(bbox)!=point_data:
                raise ValueError("H134 gold label does not match actual DOM target center")
            if case.get(role+"_grid_1000")!=norm(point_data):
                raise ValueError("H134 1000-grid target inconsistency")
        if not isinstance(case.get("instruction"),str) or len(case["instruction"])<12:
            raise ValueError("H134 original visible instruction invalid")
    if len(seen)!=SPLITS[split][1]:raise ValueError("H134 unique image source count mismatch")
    return doc

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--split",choices=list(SPLITS),required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    out=render(args.split,args.output)
    print(json.dumps({"status":"H134_ORIGINAL_REAL_GUI_SOURCE_PASS",
        "split":args.split,"verified_screenshots":out["case_count"]}))
