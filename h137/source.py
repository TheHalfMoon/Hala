"""H137 cross-library zero-training browser benchmark.

Real MIT noUiSlider, SortableJS and Split.js run offline from owner-private
npm binaries, not the H136 hand-built drag renderer. Application wrappers
are explicitly owner-authored, NOT independent production apps.
All scenes are EVAL ONLY; gold coordinates are never model prompts.
"""
import hashlib
import json
import shutil
from pathlib import Path

from h130.browser_env import WIDTH,HEIGHT,_chromium_binary

SCHEMA="hala.h137.offline-three-independent-MIT-interaction-libraries.v1"
FAMILIES=("nouislider","splitjs","sortablejs")
IDS=range(570000,570036)
VENDOR_SOURCES={
    "nouislider.min.js":("nouislider","15.8.1","dist/nouislider.min.js"),
    "nouislider.min.css":("nouislider","15.8.1","dist/nouislider.min.css"),
    "sortable.min.js":("sortablejs","1.15.6","Sortable.min.js"),
    "split.min.js":("split.js","1.6.5","dist/split.min.js"),
}

def digest(blob):return hashlib.sha256(blob).hexdigest()
def center(box):
    if not isinstance(box,dict) or any(type(box.get(k)) not in (int,float)
                                   for k in ("x","y","width","height")):
        raise ValueError("H137 genuine DOM bbox required")
    x=round(box["x"]+box["width"]/2); y=round(box["y"]+box["height"]/2)
    if not 0<=x<WIDTH or not 0<=y<HEIGHT or box["width"]<=0 or box["height"]<=0:
        raise ValueError("H137 actual tiny-control bbox invalid")
    return [x,y]

def prepare_runtime(runtime, vendor_root):
    runtime=Path(runtime)
    vendor_root=Path(vendor_root)
    if runtime.exists():raise ValueError("H137 private offline runtime create-only")
    runtime.mkdir(parents=True)
    owner=Path(__file__).resolve().parent
    for file in ("fixture.html","app.js"):
        shutil.copyfile(owner/file,runtime/file)
    folder=runtime/"vendor";folder.mkdir()
    pinned={}
    for filename,(name,version,rel) in VENDOR_SOURCES.items():
        package=vendor_root/"node_modules"/name
        meta=json.loads((package/"package.json").read_text("utf8"))
        if meta.get("name")!=name or meta.get("version")!=version or meta.get("license")!="MIT":
            raise ValueError("H137 external MIT library exact package version mismatch")
        original=package/rel
        if not original.is_file():raise ValueError("H137 real third-party runtime missing")
        shutil.copyfile(original,folder/filename)
        pinned[filename]={"package":name,"version":version,
                          "sha256":digest(original.read_bytes()),"license":"MIT"}
    (runtime/"provenance.json").write_text(
        json.dumps(pinned,indent=2,sort_keys=True)+"\n","utf8")
    return pinned

def capture(output,vendor_root):
    from playwright.sync_api import sync_playwright
    out=Path(output)
    if out.exists():raise ValueError("H137 original library evaluation create-only")
    out.mkdir(parents=True)
    runtime=out/"_runtime"
    vendors=prepare_runtime(runtime,vendor_root)
    rows=[]
    errors=[]
    with sync_playwright() as playwright:
        opts={"headless":True}
        binary=_chromium_binary()
        if binary:opts["executable_path"]=binary
        browser=playwright.chromium.launch(**opts)
        try:
            ctx=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                                    device_scale_factor=1,service_workers="block")
            ctx.route("http://**/*",lambda r:r.abort())
            ctx.route("https://**/*",lambda r:r.abort())
            page=ctx.new_page()
            page.on("pageerror",lambda error:errors.append(str(error)))
            for seed in IDS:
                page.goto((runtime/"fixture.html").as_uri()+"?id="+str(seed),
                          wait_until="load")
                page.locator("#start-target").wait_for()
                if errors:raise ValueError("H137 vendor UI runtime exception: "+errors[-1][:300])
                family=FAMILIES[seed%3]
                lang="ar" if seed%2 else "en"
                if page.locator("#workspace").get_attribute("data-vendor")!=family:
                    raise ValueError("H137 genuine independent JS library family not mounted")
                instruction=page.locator("#instruction").inner_text()
                if len(instruction)<15:raise ValueError("H137 visible task instruction absent")
                start_box=page.locator("#start-target").bounding_box()
                end_box=page.locator("#end-target").bounding_box()
                start,end=center(start_box),center(end_box)
                png=page.screenshot(type="png")
                if not png.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError("H137 genuine Chromium PNG required")
                fname=f"h137-{seed}.png"
                (out/fname).write_bytes(png)
                page.mouse.move(*start);page.mouse.down()
                page.mouse.move(*end,steps=24);page.mouse.up()
                page.wait_for_timeout(75)
                success=page.locator("#workspace").get_attribute("data-success")=="true"
                if not success:
                    raise ValueError(f"H137 {family} genuine vendor-backed oracle drag failed {seed}")
                rows.append({
                    "case_id":f"h137-{seed}","family":family,"lang":lang,
                    "image":fname,"image_sha256":digest(png),
                    "instruction":instruction,"start_bbox":start_box,
                    "end_bbox":end_box,"start_px":start,"end_px":end,
                    "actual_third_party_browser_gold_drag_passed":True,
                })
            ctx.close()
        finally:browser.close()
    manifest={
       "schema":SCHEMA,"evaluation_only":True,
       "training_or_selection_eligible":False,
       "scene_count":len(rows),"viewport":[WIDTH,HEIGHT],
       "source":"owner-authored-wrappers-around-three-real-MIT-JS-libraries",
       "not_production_applications":True,
       "external_runtime_assets":vendors,
       "cases":rows}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n","utf8")
    return check(out)

def check(output):
    output=Path(output)
    source=json.loads((output/"manifest.json").read_text("utf8"))
    if (source.get("schema")!=SCHEMA or source.get("evaluation_only") is not True or
        source.get("training_or_selection_eligible") is not False or
        source.get("not_production_applications") is not True or
        source.get("viewport")!=[WIDTH,HEIGHT] or
        source.get("scene_count")!=36):
        raise ValueError("H137 source evaluation strict metadata absent")
    seen=set()
    for seed,case in zip(IDS,source.get("cases",[])):
        if (case.get("case_id")!=f"h137-{seed}" or
            case.get("family")!=FAMILIES[seed%3] or
            case.get("lang")!=("ar" if seed%2 else "en") or
            case.get("actual_third_party_browser_gold_drag_passed") is not True):
            raise ValueError("H137 real vendor eval provenance/label mismatch")
        file=output/case["image"]
        if (not file.is_file() or file.is_symlink() or
            digest(file.read_bytes())!=case["image_sha256"] or
            case["image_sha256"] in seen or
            center(case["start_bbox"])!=case["start_px"] or
            center(case["end_bbox"])!=case["end_px"]):
            raise ValueError("H137 source screenshot bytes/labels invalid")
        seen.add(case["image_sha256"])
    if len(seen)!=36:raise ValueError("H137 exact 36 unique library scenes missing")
    return source

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    p.add_argument("--vendor-root",required=True)
    a=p.parse_args()
    data=capture(a.output,a.vendor_root)
    print("H137 MIT THIRD PARTY LIBRARY GOLD SUCCESS",data["scene_count"])
