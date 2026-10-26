"""New owner-authored H136 desktop GUI scenes for controlled image-resolution ablation."""
import hashlib
import json
import math
from pathlib import Path
from h130.browser_env import WIDTH,HEIGHT,_chromium_binary

SCHEMA="hala.h136.owner-original-workstation-gui-scenes.v1"
SPLITS={"train":(730000,288),"dev":(830000,24),"sealed":(930000,24)}
FAMILIES=("timeline","pane_resize","table_divider","card_drop","audio_slider","schedule_drop")

def sha(data):return hashlib.sha256(data).hexdigest()
def seeds(split):
    if split not in SPLITS:raise ValueError("H136 only frozen pilot / sealed")
    start,n=SPLITS[split]
    return range(start,start+n)
def center(box):
    if not isinstance(box,dict) or any(
        type(box.get(k)) not in (int,float) or not math.isfinite(box[k])
        for k in ("x","y","width","height")):
        raise ValueError("H136 actual browser bbox missing")
    if box["width"]<=0 or box["height"]<=0:raise ValueError("H136 empty element bbox")
    p=[round(box["x"]+box["width"]/2),round(box["y"]+box["height"]/2)]
    if not 0<=p[0]<WIDTH or not 0<=p[1]<HEIGHT:
        raise ValueError("H136 target outside viewport")
    return p

def check(path,split):
    path=Path(path)
    manifest=json.loads((path/"manifest.json").read_text("utf8"))
    if (manifest.get("schema")!=SCHEMA or manifest.get("split")!=split or
        manifest.get("viewport")!=[WIDTH,HEIGHT] or
        manifest.get("source")!="original-h136-workstation-fixture" or
        manifest.get("no_H134_or_H133_screenshot_reuse") is not True):
        raise ValueError("H136 source/split/provenance invalid")
    records=manifest.get("cases")
    if not isinstance(records,list) or len(records)!=SPLITS[split][1]:
        raise ValueError("H136 exact scene count wrong")
    seen=set()
    for i,row in zip(seeds(split),records):
        if (row.get("id")!=f"h136-{i}" or row.get("image")!=f"h136-{i}.png" or
            row.get("family")!=FAMILIES[i%6] or not row.get("gold_real_browser_success")):
            raise ValueError("H136 scene ID/type/gold execution invalid")
        f=path/row["image"]
        if not f.is_file() or f.is_symlink():
            raise ValueError("H136 screenshot missing")
        data=f.read_bytes();digest=sha(data)
        if not data.startswith(b"\x89PNG\r\n\x1a\n") or row.get("image_sha256")!=digest or digest in seen:
            raise ValueError("H136 screenshot SHA duplicated/corrupt")
        seen.add(digest)
        for role in ("start","end"):
            if row[role+"_px"]!=center(row[role+"_bbox"]):
                raise ValueError("H136 source true DOM box does not agree with pixel center")
        if not isinstance(row.get("instruction"),str) or len(row["instruction"])<15:
            raise ValueError("H136 real instruction absent")
    return manifest

def capture(output,split):
    from playwright.sync_api import sync_playwright
    if split not in SPLITS:raise ValueError("H136 split unapproved")
    path=Path(output)
    if path.exists():raise ValueError("H136 create-only private image source")
    path.mkdir(parents=True)
    fixture=Path(__file__).with_name("fixture.html").resolve()
    rows=[]
    with sync_playwright() as pw:
        kwargs={"headless":True}
        browser_path=_chromium_binary()
        if browser_path:kwargs["executable_path"]=browser_path
        browser=pw.chromium.launch(**kwargs)
        try:
            ctx=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                                     device_scale_factor=1,service_workers="block")
            ctx.route("http://**/*",lambda r:r.abort())
            ctx.route("https://**/*",lambda r:r.abort())
            page=ctx.new_page()
            for i in seeds(split):
                page.goto(fixture.as_uri()+"?id="+str(i),wait_until="load")
                page.locator('[data-role="handle"]').wait_for()
                instruction=page.locator("#instruction").inner_text()
                handle=page.locator('[data-role="handle"]').bounding_box()
                target=page.locator("#target").bounding_box()
                start,end=center(handle),center(target)
                picture=page.screenshot(type="png")
                image_name=f"h136-{i}.png"
                (path/image_name).write_bytes(picture)
                page.mouse.move(*start);page.mouse.down()
                page.mouse.move(*end,steps=12);page.mouse.up()
                if page.locator("#canvas").get_attribute("data-success")!="true":
                    raise ValueError(f"H136 actual Chromium fixture cannot complete task {i}")
                rows.append({"id":f"h136-{i}","image":image_name,
                             "image_sha256":sha(picture),
                             "family":FAMILIES[i%6],"instruction":instruction,
                             "start_px":start,"end_px":end,
                             "start_bbox":handle,"end_bbox":target,
                             "gold_real_browser_success":True})
            ctx.close()
        finally:browser.close()
    manifest={"schema":SCHEMA,"split":split,"viewport":[WIDTH,HEIGHT],
              "source":"original-h136-workstation-fixture",
              "no_H134_or_H133_screenshot_reuse":True,
              "cases":rows}
    (path/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n","utf8")
    return check(path,split)

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--split",choices=list(SPLITS),required=True)
    parser.add_argument("--output",required=True)
    a=parser.parse_args()
    r=capture(a.output,a.split)
    print(f"H136 ORIGINAL {a.split.upper()} REAL BROWSER QA PASS {len(r['cases'])}",flush=True)
