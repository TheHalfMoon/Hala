"""H153 frozen ORIGINAL React independent shifted-layout browser challenge.

Source changes are prohibited: CSS is injected only into the owner test
browser, never written to upstream. Fresh IDs, original native screenshots,
persisted reference drags, and a new visual layout unseen in H142 SFT/DEV.
NO model feedback, training, or sealed H140 blind data.
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
from h141.interactive_source import verify_owner_original
from h142.collect_visible_source import seed_scene,APP,APP_SHA

SCHEMA="hala.h153.owner-original-shifted-layout-evaluation.v1"
CASES=16
OFFSET=184000
# All layouts shift BOTH SOURCE and target columns, so pixel geometry
# differs from every fixed H142 header and source coordinate.
TRANSFORMS={
    "up":(-27,-48),
    "down":(22,52),
    "far_down":(-19,80),
    "left_up":(-38,-26),
}
def layout_name(attempt):
    return tuple(TRANSFORMS)[(attempt//4)%4]

def style(layout):
    if layout not in TRANSFORMS:raise ValueError("H153 only frozen shifted layouts")
    x,y=TRANSFORMS[layout]
    return ".task_column { transform:translate("+str(x)+"px,"+str(y)+"px); }"

def source_case(attempt):
    if attempt<OFFSET:raise ValueError("H153 never reuse previously evaluated source IDs")
    case=seed_scene(attempt)
    case["layout"]=layout_name(attempt)
    return case

def collect(source,output,max_attempts=360):
    output=Path(output)
    if (output.exists() or
        output.with_name(output.name+".incomplete").exists() or
        not any(x.name=="private-h153" for x in output.parents) or
        max_attempts<CASES):
        raise ValueError("H153 source must be owner-private and write-once")
    built=verify_owner_original(source)
    from playwright.sync_api import sync_playwright
    intermediate=output.with_name(output.name+".incomplete")
    intermediate.mkdir(parents=True)
    rows=[];sha_seen=set();refusals=defaultdict(int)
    quota=CASES//4
    with sync_playwright() as pw:
        options={"headless":True}
        binary=_chromium_binary()
        if binary:options["executable_path"]=binary
        browser=pw.chromium.launch(**options,timeout=25000)
        try:
            with serve(built) as url:
                for attempt in range(OFFSET,OFFSET+max_attempts):
                    if len(rows)>=CASES:break
                    case=source_case(attempt)
                    key=(case["language"],case["destination_status"])
                    if sum((r["language"],r["destination_status"])==key for r in rows)>=quota:
                        refusals["balanced_stratum_full"]+=1
                        continue
                    ctx=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                                            device_scale_factor=1,service_workers="block")
                    ctx.add_init_script(
                        "if(localStorage.getItem('tasks')===null){"+
                        "localStorage.setItem('tasks',"+
                        json.dumps(json.dumps(case["rows"]))+");}")
                    ctx.route("https://**/*",lambda route:route.abort())
                    try:
                        page=ctx.new_page()
                        page.goto(url,wait_until="commit",timeout=18000)
                        src=page.locator(".task_column").nth(0).locator(
                            ".task_card").nth(case["source_index"])
                        col=page.locator(".task_column").nth(
                            case["destination_column"])
                        heading=col.locator(".task_column_heading").first
                        hidden=col.locator(".hide_drop").nth(
                            case["destination_index"])
                        src.wait_for(timeout=16000)
                        heading.wait_for(timeout=16000)
                        hidden.wait_for(timeout=16000)
                        page.add_style_tag(content=style(case["layout"]))
                        start,goal=src.bounding_box(),heading.bounding_box()
                        if not visible_box(start) or not visible_box(goal):
                            refusals["original_shifted_source_or_heading_offscreen"]+=1
                            continue
                        if (goal["y"]<95 or goal["y"]>WIDTH):
                            refusals["heading_unexpected_position"]+=1
                            continue
                        photo=page.screenshot(type="png")
                        with Image.open(io.BytesIO(photo)) as pic:
                            if pic.format!="PNG" or pic.size!=(WIDTH,HEIGHT):
                                raise ValueError("H153 original native screen changed")
                            pic.load()
                        image_sha=digest(photo)
                        if image_sha in sha_seen:
                            refusals["duplicate_original_photo"]+=1
                            continue
                        try:
                            src.drag_to(hidden,force=True,timeout=16000)
                        except Exception:
                            refusals["real_drag_could_not_complete"]+=1
                            continue
                        page.wait_for_timeout(130)
                        current=json.loads(page.evaluate("localStorage.getItem('tasks')"))
                        moved=sum(r["task"]==case["title"] and
                                  r["status"]==case["destination_status"]
                                  for r in current)==1
                        page.reload(wait_until="commit",timeout=16000)
                        last=json.loads(page.evaluate("localStorage.getItem('tasks')"))
                        persisted=sum(r["task"]==case["title"] and
                                      r["status"]==case["destination_status"]
                                      for r in last)==1
                        if not moved or not persisted:
                            refusals["real_persisted_reference_action_failed"]+=1
                            continue
                        idx=len(rows)
                        name=f"h153-new-shifted-{idx:03d}.png"
                        (intermediate/name).write_bytes(photo)
                        sha_seen.add(image_sha)
                        rows.append({
                            "id":f"h153-new-shifted-{idx:03d}",
                            "app":APP,"app_git_sha":APP_SHA,
                            "attempt_index":attempt,"language":case["language"],
                            "destination_status":case["destination_status"],
                            "layout":case["layout"],
                            "viewport":[WIDTH,HEIGHT],
                            "instruction":case["instruction"],
                            "image":name,"image_sha256":image_sha,
                            "start_bbox":start,
                            "visible_destination_column_heading_bbox":goal,
                            "source_reference_drag_persisted":True,
                            "reference_survived_reload":True,
                            "css_in_browser_only":True,
                            "training_eligible":False,
                            "model_evaluation_eligible":True,
                            "h140_blind_eligible":False,
                            "neural_calls_before_freeze":0,
                        })
                        if len(rows)%4==0:
                            print("H153_SHIFTED_ORIGINAL_REFERENCE_QA",len(rows),CASES,
                                  flush=True)
                    finally:
                        ctx.close()
        finally:
            browser.close()
    if len(rows)!=CASES:raise ValueError("H153 incomplete native original source, leave failed partial")
    strata=Counter((r["language"],r["destination_status"]) for r in rows)
    if strata!={(a,b):4 for a in ("ar","en") for b in ("doing","done")}:
        raise ValueError("H153 bilingual two-goal source role drift")
    if len({r["layout"] for r in rows})<3:
        raise ValueError("H153 lacks multiple distinct new shifted layouts")
    manifest={
        "schema":SCHEMA,"status":"NEW_SHIFTED_LAYOUT_ORIGINALS_NO_NEURAL_TEST",
        "private":True,"training_eligible":False,
        "model_evaluation_eligible":True,"h140_blind_eligible":False,
        "source_app_sha":APP_SHA,
        "case_count":CASES,"cases":rows,
        "neural_calls_before_freeze":0,
    }
    (intermediate/"manifest.json").write_text(
        json.dumps(manifest,indent=2,sort_keys=True,ensure_ascii=False)+"\n","utf8")
    intermediate.rename(output)
    return {"cases":len(rows),"new_native_sha":len(sha_seen),
            "layouts":dict(Counter(r["layout"] for r in rows)),
            "genuine_persisted_reference_drags":len(rows),
            "strata":{str(k):v for k,v in strata.items()},
            "refusals":dict(refusals),"model_neural_calls":0}
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--max-attempts",type=int,default=360)
    args=p.parse_args()
    print(json.dumps(collect(args.source,args.output,args.max_attempts),
                     sort_keys=True,indent=2))
