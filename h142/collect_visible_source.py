"""H142 original-React visible destination COLUMN, not invisible drop slot.

New source-only TRAIN64 and DEV16. Native MIT upstream unchanged, genuine
Chromium screenshots, persistent real Playwright drag proof. Arabic/English
AND Doing/Done goal column balanced. Never reuse H140 or H141 screenshots.
This is an owner-private SFT experiment, NOT Hala flagship 35B.
"""
import argparse
import io
import json
from collections import Counter,defaultdict
from pathlib import Path

from PIL import Image

from h130.browser_env import WIDTH,HEIGHT,_chromium_binary
from h138.probe import serve
from h140.qualification import digest
from h140.vanilla_pilot import visible_box
from h141.interactive_source import APP,APP_SHA,verify_owner_original,original_case

SCHEMA="hala.h142.original-react-visible-column-sft-source.v1"
COUNTS={"train":64,"dev":16}
OFFSETS={"train":52000,"dev":62000}
DESTINATIONS={"doing":1,"done":2}

def seed_scene(attempt):
    """Use fresh post-H141 native seeded tasks, balance TWO visible goals."""
    if attempt<52000:raise ValueError("H142 historical source attempt forbidden")
    original=original_case(attempt)
    destination="done" if attempt%4 in (2,3) else "doing"
    label={"doing":"Doing","done":"Done"}[destination]
    original["instruction"]=(
        "اسحب البطاقة "+original["title"]+" إلى عمود "+
        ("قيد التنفيذ" if destination=="doing" else "مكتمل")
        if original["language"]=="ar" else
        "Drag the task "+original["title"]+" to the "+label+" column"
    )
    original["destination_status"]=destination
    original["destination_column"]=DESTINATIONS[destination]
    # Original Done column has only one seeded card/two native slots;
    # Doing has N seeded cards. Never index an absent hidden drop slot.
    if destination=="done":
        original["destination_index"]=attempt%2
    return original

def collect(source,output,split,attempt_offset=None,max_attempts=450):
    if split not in COUNTS:raise ValueError("H142 requires train or dev")
    count=COUNTS[split]
    attempt_offset=OFFSETS[split] if attempt_offset is None else attempt_offset
    output=Path(output)
    if (attempt_offset<OFFSETS[split] or attempt_offset%4 or
        max_attempts<count or output.exists() or
        output.with_name(output.name+".incomplete").exists() or
        not any(p.name=="private-h142" for p in output.parents)):
        raise ValueError("H142 private source create only, preregistered fresh ids")
    built=verify_owner_original(source)
    from playwright.sync_api import sync_playwright
    partial=output.with_name(output.name+".incomplete")
    partial.mkdir(parents=True)
    attempts=set();images=set();rows=[]
    rejects=defaultdict(int)
    quota=count//4
    with sync_playwright() as pw:
        opts={"headless":True}
        binary=_chromium_binary()
        if binary:opts["executable_path"]=binary
        browser=pw.chromium.launch(**opts,timeout=25000)
        try:
            with serve(built) as url:
                for attempt in range(attempt_offset,attempt_offset+max_attempts):
                    if len(rows)>=count:break
                    original=seed_scene(attempt)
                    key=(original["language"],original["destination_status"])
                    if sum((r["language"],r["destination_status"])==key for r in rows)>=quota:
                        rejects["stratified_role_cap"]+=1
                        continue
                    ctx=browser.new_context(
                        viewport={"width":WIDTH,"height":HEIGHT},
                        device_scale_factor=1,service_workers="block")
                    ctx.add_init_script(
                        "if(localStorage.getItem('tasks')===null){"+
                        "localStorage.setItem('tasks',"+
                        json.dumps(json.dumps(original["rows"]))+");}")
                    ctx.route("https://**/*",lambda route:route.abort())
                    try:
                        page=ctx.new_page()
                        page.goto(url,wait_until="commit",timeout=18000)
                        start=page.locator(".task_column").nth(0).locator(
                            ".task_card").nth(original["source_index"])
                        column=page.locator(".task_column").nth(
                            original["destination_column"])
                        heading=column.locator(".task_column_heading").first
                        hidden=column.locator(".hide_drop").nth(
                            original["destination_index"])
                        start.wait_for(timeout=16000)
                        heading.wait_for(timeout=16000)
                        hidden.wait_for(timeout=16000)
                        sb,gb=start.bounding_box(),heading.bounding_box()
                        if not visible_box(sb) or not visible_box(gb):
                            rejects["card_or_heading_not_originally_visible"]+=1
                            continue
                        style=heading.evaluate(
                            "(el)=>({opacity:Number(getComputedStyle(el).opacity),"+
                            "visibility:getComputedStyle(el).visibility})")
                        hidden_opacity=hidden.evaluate(
                            "(el)=>Number(getComputedStyle(el).opacity)")
                        if (style["opacity"]<.99 or
                            style["visibility"]!="visible" or hidden_opacity!=0):
                            rejects["heading_or_hidden_target_style_invalid"]+=1
                            continue
                        photo=page.screenshot(type="png")
                        with Image.open(io.BytesIO(photo)) as pic:
                            if pic.format!="PNG" or pic.size!=(WIDTH,HEIGHT):
                                raise ValueError("H142 unmodified native original screenshot only")
                            pic.load()
                        image_hash=digest(photo)
                        if image_hash in images:
                            rejects["duplicate_original_pixel_sha"]+=1
                            continue
                        try:
                            start.drag_to(hidden,force=True,timeout=16000)
                        except Exception:
                            rejects["original_native_drag_failed"]+=1
                            continue
                        page.wait_for_timeout(90)
                        now=json.loads(page.evaluate("localStorage.getItem('tasks')"))
                        changed=sum(t["task"]==original["title"] and
                                    t["status"]==original["destination_status"] for t in now)==1
                        page.reload(wait_until="commit",timeout=16000)
                        after=json.loads(page.evaluate("localStorage.getItem('tasks')"))
                        persisted=sum(t["task"]==original["title"] and
                                      t["status"]==original["destination_status"]
                                      for t in after)==1
                        if not (changed and persisted):
                            rejects["native_move_or_reload_persistence_failed"]+=1
                            continue
                        name=f"h142-{split}-{len(rows):03d}.png"
                        (partial/name).write_bytes(photo)
                        images.add(image_hash)
                        attempts.add(attempt)
                        rows.append({
                            "id":f"h142-{split}-{len(rows):03d}",
                            "split":split,
                            "app":APP,"app_git_sha":APP_SHA,
                            "attempt_index":attempt,
                            "language":original["language"],
                            "destination_status":original["destination_status"],
                            "instruction":original["instruction"],
                            "image":name,"image_sha256":image_hash,
                            "start_bbox":sb,
                            "visible_destination_column_heading_bbox":gb,
                            "target_is_visible_at_rest":True,
                            "hidden_exact_drop_point_is_not_output_target":True,
                            "reference_drag_persisted":True,
                            "reference_survived_page_reload":True,
                            "model_neural_calls":0,
                            "training_eligible":split=="train",
                            "evaluation_eligible":split=="dev",
                            "blind_eligible":False,
                        })
                        if len(rows)%8==0:
                            print("H142_GENUINE_ORIGINAL_SOURCE_CAPTURE",split,
                                  len(rows),"/",count,flush=True)
                    finally:
                        ctx.close()
        finally:
            browser.close()
    if len(rows)!=count:
        raise ValueError("H142 original app source failed count; keep incomplete evidence")
    strata=Counter((x["language"],x["destination_status"]) for x in rows)
    if (len(images)!=count or len(attempts)!=count or
        strata!={(l,d):quota for l in ("ar","en") for d in DESTINATIONS}):
        raise ValueError("H142 source language/goal balance or unique SHA failed")
    doc={
        "schema":SCHEMA,"status":"H142_ORIGINAL_PERSISTED_UI_SOURCE_READY",
        "split":split,"private":True,
        "app_sources":{APP:{"sha":APP_SHA,"license":"MIT"}},
        "training_eligible":split=="train",
        "evaluation_eligible":split=="dev",
        "blind_eligible":False,
        "target_role":"visible_destination_column_heading",
        "original_screenshot_dimensions":[WIDTH,HEIGHT],
        "real_neural_calls":0,
        "cases":rows,
    }
    (partial/"manifest.json").write_text(
        json.dumps(doc,sort_keys=True,indent=2,ensure_ascii=False)+"\n",
        encoding="utf8")
    partial.rename(output)
    return {"status":doc["status"],"split":split,
            "cases":count,"strata":{"/".join(k):v for k,v in strata.items()},
            "unique_original_png":len(images),
            "real_reference_drag_persisted":count,
            "refusals":dict(reject),"model_calls":0}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--split",required=True,choices=tuple(COUNTS))
    p.add_argument("--max-attempts",type=int,default=450)
    p.add_argument("--attempt-offset",type=int)
    args=p.parse_args()
    print(json.dumps(collect(args.source,args.output,args.split,
                             attempt_offset=args.attempt_offset,
                             max_attempts=args.max_attempts),
                     sort_keys=True,indent=2))
