"""H141 original at-rest visible-column target source audit, not neural training.

For visually grounded *coarse destination column* localization, NOT
invisible direct mouse release. Every example uses real upstream React
app pixels and a real persisted reference drag, no synthetic target box.
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
from h141.interactive_source import (
    APP_SHA,APP,verify_owner_original,original_case
)

SCHEMA="hala.h141.visible-at-rest-column-original-source-pilot.v1"

def collect(source,folder,count=16,attempt_offset=44000,max_attempts=160):
    folder=Path(folder)
    if (not 4<=count<=32 or count%2 or
        attempt_offset<44000 or attempt_offset%2 or max_attempts<count or
        folder.exists() or folder.with_name(folder.name+".incomplete").exists() or
        not any(p.name=="private-h141" for p in folder.parents)):
        raise ValueError("H141 owner-private native at-rest column source create-only")
    # CPU-only CI must reject invalid requests BEFORE importing the
    # optional browser/Playwright runtime.
    built=verify_owner_original(source)
    from playwright.sync_api import sync_playwright
    temp=folder.with_name(folder.name+".incomplete")
    temp.mkdir(parents=True)
    seen=set();rows=[];reject=defaultdict(int)
    with sync_playwright() as pw:
        opts={"headless":True}
        path=_chromium_binary()
        if path:opts["executable_path"]=path
        browser=pw.chromium.launch(**opts,timeout=20000)
        try:
            with serve(built) as url:
                for attempt in range(attempt_offset,attempt_offset+max_attempts):
                    if len(rows)>=count:break
                    example=original_case(attempt)
                    lang=example["language"]
                    if sum(x["language"]==lang for x in rows)>=count//2:
                        reject["balanced_language_limit"]+=1
                        continue
                    ctx=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                                            device_scale_factor=1,service_workers="block")
                    ctx.add_init_script(
                        "if(localStorage.getItem('tasks')===null){"+
                        "localStorage.setItem('tasks',"+
                        json.dumps(json.dumps(example["rows"]))+");}")
                    ctx.route("https://**/*",lambda route:route.abort())
                    try:
                        page=ctx.new_page()
                        page.goto(url,wait_until="commit",timeout=18000)
                        start=page.locator(".task_column").nth(0).locator(
                            ".task_card").nth(example["source_index"])
                        goal=page.locator(".task_column").nth(1).locator(
                            ".task_column_heading").first
                        hidden=page.locator(".task_column").nth(1).locator(
                            ".hide_drop").nth(example["destination_index"])
                        start.wait_for(timeout=18000)
                        goal.wait_for(timeout=18000)
                        hidden.wait_for(timeout=18000)
                        sb,gb=start.bounding_box(),goal.bounding_box()
                        if not visible_box(sb) or not visible_box(gb):
                            reject["source_card_or_visible_column_offscreen"]+=1
                            continue
                        state=goal.evaluate(
                            "(el)=>({opacity:Number(getComputedStyle(el).opacity),"+
                            "visibility:getComputedStyle(el).visibility})")
                        invis=hidden.evaluate(
                            "(el)=>Number(getComputedStyle(el).opacity)")
                        if state["opacity"]<0.99 or state["visibility"]!="visible" or invis!=0:
                            reject["visible_column_or_hidden_drop_state_invalid"]+=1
                            continue
                        photo=page.screenshot(type="png")
                        with Image.open(io.BytesIO(photo)) as im:
                            if im.format!="PNG" or im.size!=(WIDTH,HEIGHT):
                                raise ValueError("H141 native original viewport changed")
                            im.load()
                        sha=digest(photo)
                        if sha in seen:
                            reject["duplicate_source_png"]+=1
                            continue
                        hidden_target=page.locator(".task_column").nth(1).locator(
                            ".hide_drop").nth(example["destination_index"])
                        start.drag_to(hidden_target,force=True,timeout=18000)
                        page.wait_for_timeout(130)
                        new=json.loads(page.evaluate("localStorage.getItem('tasks')"))
                        moved=sum(t["task"]==example["title"] and
                                  t["status"]=="doing" for t in new)==1
                        page.reload(wait_until="commit",timeout=18000)
                        last=json.loads(page.evaluate("localStorage.getItem('tasks')"))
                        persistent=sum(t["task"]==example["title"] and
                                       t["status"]=="doing" for t in last)==1
                        if not moved or not persistent:
                            reject["persisted_source_reference_drag_failed"]+=1
                            continue
                        fn=f"original-visible-goal-{len(rows):03d}.png"
                        (temp/fn).write_bytes(photo)
                        seen.add(sha)
                        rows.append({
                            "id":f"original-visible-goal-{len(rows):03d}",
                            "app":APP,"app_git_sha":APP_SHA,
                            "attempt_index":attempt,"language":lang,
                            "instruction":example["instruction"],
                            "image":fn,"image_sha256":sha,
                            "start_bbox":sb,
                            "visible_destination_column_heading_bbox":gb,
                            "target_is_visible_at_rest":True,
                            "hidden_exact_drop_point_is_not_output_target":True,
                            "training_eligible":False,
                            "blind_eligible":False,
                            "model_evaluation_eligible":False,
                            "reference_drag_persisted":True,
                            "reference_survived_page_reload":True,
                            "model_input_contains_oracle_coordinates":False,
                        })
                        if len(rows)%4==0:
                            print("H141_VISIBLE_SEMANTIC_TARGET_NATIVE_SOURCE_QA",
                                  len(rows),count,flush=True)
                    finally:
                        ctx.close()
        finally:browser.close()
    if len(rows)!=count:
        raise ValueError("H141 visible semantic column source incomplete; preserved partial")
    if Counter(row["language"] for row in rows)!={"ar":count//2,"en":count//2}:
        raise ValueError("H141 visible original column source bilingual role cap failed")
    doc={
        "schema":SCHEMA,"status":"VISIBLE_ORIGINAL_COLUMN_DIAGNOSTIC_ONLY",
        "private":True,"source_original_mit_sha":APP_SHA,
        "target_role":"visible_destination_column_approach_not_release",
        "training_eligible":False,"blind_eligible":False,
        "model_evaluation_eligible":False,
        "real_neural_calls":0,"original_cards_reloaded":count,
        "cases":rows,
    }
    (temp/"manifest.json").write_text(
        json.dumps(doc,sort_keys=True,indent=2,ensure_ascii=False)+"\n","utf8")
    temp.rename(folder)
    return {
        "status":doc["status"],
        "cases":count,
        "genuine_original_visible_column_targets":count,
        "real_native_persisted_drags":count,
        "unique_at_rest_png":len(seen),
        "language_counts":dict(Counter(x["language"] for x in rows)),
        "training_eligible":False,"blind_eligible":False,
        "refusals":dict(reject),
        "flagship_Hala_35B_updated":False,
    }

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--count",type=int,default=16)
    p.add_argument("--attempt-offset",type=int,default=44000)
    p.add_argument("--max-attempts",type=int,default=160)
    a=p.parse_args()
    print(json.dumps(collect(a.source,a.output,count=a.count,
                             attempt_offset=a.attempt_offset,
                             max_attempts=a.max_attempts),
                     sort_keys=True,indent=2))
