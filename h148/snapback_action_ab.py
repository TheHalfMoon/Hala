"""H148 confirmatory NEW native React visible-drop release snapback study.

H147 consumed 16 original tasks; their diagnostic suggested overshoot
after detecting drop entry. H148 uses entirely NEW IDs >=124000 and
predeclares +18px snap-back from the noisy visible destination header
after first screenshot-observed visual entry. Control uses original H147
visual-entry policy. Never replay the consumed H147 cases or H140 BLIND.
No hidden drop DOM boxes are passed to either mouse controller.
"""
import argparse
import io
import json
from collections import Counter,defaultdict
from pathlib import Path

from h138.probe import serve
from h130.browser_env import WIDTH,HEIGHT,_chromium_binary
from h140.vanilla_pilot import visible_box
from h141.coarse_column_action import visible_center
from h141.interactive_source import verify_owner_original
from h145.noisy_point_action_ab import jittered_points
from h147.visual_drag_entry_ab import (
    changed_samples,one_arm as original_visual_arm,
    OFFSETS,MIN_VISIBLE_CHANGED_SAMPLES
)
from h143.collect_shifted_source import style
from h145.noisy_point_action_ab import source as make_scene

OFFSET=124000
N=16
SNAP_TO_HEADING_OFFSET=18
SCHEMA="hala.h148.original-independent-drop-entry-snapback-confirm.v1"

def new_case(i):
    if i<OFFSET:
        raise ValueError("H148 cannot use previously consumed source episodes")
    return make_scene(i)

def snapback_controller(page,start,goal):
    sx,sy=start;gx,gy=goal
    if not all(0<=v<limit for v,limit in
               ((sx,WIDTH),(sy,HEIGHT),(gx,WIDTH),(gy,HEIGHT))):
        raise ValueError("H148 only screen-visible mouse points")
    pre=page.screenshot(type="png")
    page.mouse.move(sx,sy)
    page.mouse.down()
    page.mouse.move(sx+6,sy+18,steps=8)
    page.mouse.move(gx,gy-38,steps=24)
    detected=False
    first_signal=None
    retained_signal=False
    try:
        for delta in OFFSETS:
            y=gy+delta
            if not 0<=y<HEIGHT:continue
            page.mouse.move(gx,y,steps=4)
            page.wait_for_timeout(60)
            screenshot=page.screenshot(type="png")
            if (delta>=18 and
                changed_samples(pre,screenshot,[gx,gy])>=MIN_VISIBLE_CHANGED_SAMPLES):
                detected=True
                first_signal=delta
                break
        # EXACT precommitted difference from H147: after detection,
        # normalize mouse-up to +18px from visible predicted heading.
        # A controller never reads true/native hidden drop rectangles.
        page.mouse.move(gx,gy+SNAP_TO_HEADING_OFFSET,steps=6)
        page.wait_for_timeout(85)
        retained_signal=(
            changed_samples(pre,page.screenshot(type="png"),[gx,gy])
            >=MIN_VISIBLE_CHANGED_SAMPLES
        )
    finally:
        page.mouse.up()
    return {"visual_changed_during_entry":detected,
            "first_signal_offset":first_signal,
            "visual_signal_after_snapback":retained_signal,
            "release_y_offset":SNAP_TO_HEADING_OFFSET}

def corrected_arm(url,browser,seed):
    ctx=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                            device_scale_factor=1,service_workers="block")
    ctx.add_init_script(
        "if(localStorage.getItem('tasks')===null){"+
        "localStorage.setItem('tasks',"+
        json.dumps(json.dumps(seed["rows"]))+");}")
    ctx.route("https://**/*",lambda route:route.abort())
    try:
        page=ctx.new_page()
        page.goto(url,wait_until="commit",timeout=18000)
        src=page.locator(".task_column").nth(0).locator(
            ".task_card").nth(seed["source_index"])
        dest=page.locator(".task_column").nth(seed["destination_column"])
        header=dest.locator(".task_column_heading").first
        hidden=dest.locator(".hide_drop").nth(seed["destination_index"])
        src.wait_for(timeout=16000)
        header.wait_for(timeout=16000)
        hidden.wait_for(timeout=16000)
        page.add_style_tag(content=style(seed["layout"]))
        start_box,head_box=src.bounding_box(),header.bounding_box()
        if not visible_box(start_box) or not visible_box(head_box):
            return {"qualified":False,"reason":"original card/header offscreen"}
        start=visible_center(start_box)
        _,heading_point=jittered_points(
            start_box,head_box,seed["attempt_index"])
        try:
            feedback=snapback_controller(page,start,heading_point)
        except (ValueError,TimeoutError):
            return {"qualified":False,"reason":"invalid native mouse release"}
        page.wait_for_timeout(130)
        state=json.loads(page.evaluate("localStorage.getItem('tasks')"))
        changed=sum(x["task"]==seed["title"] and
                    x["status"]==seed["destination_status"] for x in state)==1
        page.reload(wait_until="commit",timeout=18000)
        persisted=json.loads(page.evaluate("localStorage.getItem('tasks')"))
        saved=sum(x["task"]==seed["title"] and
                  x["status"]==seed["destination_status"] for x in persisted)==1
        return {"qualified":True,"persisted":bool(changed and saved),**feedback}
    finally:ctx.close()

def evaluate(owner_root,output,max_attempts=250):
    from playwright.sync_api import sync_playwright
    owner_root,output=Path(owner_root),Path(output)
    if output.exists() or not any(p.name=="private-h148" for p in output.parents):
        raise ValueError("H148 fresh original paired owner study create-only")
    original=verify_owner_original(
        owner_root/"private-h140"/"upstream"/"react-native-board")
    records=[];refused=defaultdict(int)
    with sync_playwright() as pw:
        opts={"headless":True}
        binary=_chromium_binary()
        if binary:opts["executable_path"]=binary
        browser=pw.chromium.launch(**opts,timeout=20000)
        try:
            with serve(original) as url:
                for attempt in range(OFFSET,OFFSET+max_attempts):
                    if len(records)>=N:break
                    seed=new_case(attempt)
                    seed["attempt_index"]=attempt
                    key=(seed["language"],seed["destination_status"])
                    if sum((r["language"],r["destination_status"])==key
                           for r in records)>=4:
                        refused["stratum_full"]+=1
                        continue
                    baseline=original_visual_arm(
                        url,browser,seed,"visual_entry")
                    correction=corrected_arm(url,browser,seed)
                    if not baseline["qualified"] or not correction["qualified"]:
                        refused["unqualified_pair"]+=1
                        continue
                    records.append({
                        "attempt_index":attempt,
                        "language":seed["language"],
                        "destination_status":seed["destination_status"],
                        "layout":seed["layout"],
                        "baseline_persisted":baseline["persisted"],
                        "snapback_persisted":correction["persisted"],
                        "baseline_screenshot_signal":baseline["image_feedback_activated"],
                        "snapback_screenshot_signal":correction["visual_changed_during_entry"],
                        "snapback_signal_after_normalize":correction["visual_signal_after_snapback"],
                        "baseline_release_offset":baseline["release_offset"],
                        "snapback_initial_trigger":correction["first_signal_offset"],
                        "both_get_same_noisy_heading_proposal":True,
                    })
                    if len(records)%4==0:
                        print("H148_NEW_INDEPENDENT_SNAPBACK_AB",
                              len(records),N,flush=True)
        finally:browser.close()
    groups=Counter((x["language"],x["destination_status"]) for x in records)
    layouts=Counter(x["layout"] for x in records)
    if (len(records)!=N or
        groups!={(a,b):4 for a in ("ar","en") for b in ("doing","done")} or
        len(layouts)!=4):
        raise ValueError("H148 native original cohort incomplete")
    summary={
        "schema":SCHEMA,"status":"H148_NEW_NATIVE_SNAPBACK_COMPARISON_COMPLETE",
        "new_original_task_cases":N,"real_mouse_drags":N*2,
        "h147_frozen_visual_entry_on_new_cases":sum(r["baseline_persisted"]
                                                   for r in records),
        "h148_precommitted_snapback_persisted":sum(r["snapback_persisted"]
                                                  for r in records),
        "both_success":sum(r["baseline_persisted"] and r["snapback_persisted"]
                           for r in records),
        "snapback_only_success":sum(not r["baseline_persisted"] and r["snapback_persisted"]
                                    for r in records),
        "baseline_only_success":sum(r["baseline_persisted"] and not r["snapback_persisted"]
                                    for r in records),
        "snapback_feedback_retained":sum(r["snapback_signal_after_normalize"]
                                         for r in records),
        "reference_goal_noise_inside_visible_heading":True,
        "hidden_drop_dom_provided_to_mouse":False,
        "release_offset_predeclared":SNAP_TO_HEADING_OFFSET,
        "layouts":dict(layouts),"refusals":dict(refused),
        "model_neural_calls":0,"optimizer_steps":0,
        "h140_sealed_blind_calls":0,"Hala35B_unchanged":True,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf8") as f:
        f.write(json.dumps({"summary":summary,"owner_private_cases":records},
                           indent=2,sort_keys=True)+"\n")
    return summary
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner-root",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    print(json.dumps(evaluate(a.owner_root,a.output),sort_keys=True,indent=2))
