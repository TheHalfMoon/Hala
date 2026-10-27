"""H147 original React mouse drag visual-activation controller.

New owner-only original task IDs >=114000 and four CSS-shifted layouts.
The controller's ONLY target input is an imprecise VISIBLE heading point
inside the user-visible heading. It scans top-to-bottom from above the
heading, observes screenshots, detects a new highlighted drop region by
pixel difference (not DOM), and releases only after entering it.
Compares against direct release at the same noisy heading proposal.
No models, no H140 sealed BLIND, no old test reuse, no donor code edits.
"""
import argparse
import io
import json
from collections import Counter,defaultdict
from pathlib import Path

from PIL import Image,ImageChops

from h130.browser_env import WIDTH,HEIGHT,_chromium_binary
from h138.probe import serve
from h140.vanilla_pilot import visible_box
from h141.coarse_column_action import (
    move_card_by_coarse_visible_goal,visible_center
)
from h141.interactive_source import verify_owner_original
from h145.noisy_point_action_ab import jittered_points,source as seeded

SCHEMA="hala.h147.independent-visual-drag-enter-ab.v1"
OFFSET=114000
CASES=16
ARMS=("direct","visual_entry")
# Frozen controller signal and scan, before inspecting NEW cohort.
MIN_VISIBLE_CHANGED_SAMPLES=20
OFFSETS=(-24,-16,-8,0,8,14,18,22,26,30,36,42,50,58,70)

def case(index):
    if index<OFFSET:
        raise ValueError("H147 never reuse previously consumed cohorts")
    return seeded(index)

def changed_samples(before,after,goal):
    """Screenshot-only local destination change; no DOM or OCR allowed."""
    box=(max(0,int(goal[0]-105)),
         max(0,int(goal[1]+18)),
         min(WIDTH,int(goal[0]+105)),
         min(HEIGHT,int(goal[1]+145)))
    if box[2]<=box[0] or box[3]<=box[1]:
        raise ValueError("H147 screenshot destination ROI offscreen")
    with Image.open(io.BytesIO(before)) as old:
        with Image.open(io.BytesIO(after)) as new:
            if old.size!=(WIDTH,HEIGHT) or new.size!=(WIDTH,HEIGHT):
                raise ValueError("H147 only original 960x720 native images")
            delta=ImageChops.difference(old.convert("RGB"),new.convert("RGB"))
            region=delta.crop(box)
            width,height=region.size
            pixels=region.load()
            count=0
            for y in range(0,height,3):
                for x in range(0,width,3):
                    color=pixels[x,y]
                    if max(color)>60:
                        count+=1
            return count

def visually_observed_drop(page,start,heading_point):
    """Image-observed native drag activation, no hidden drop DOM access."""
    sx,sy=start
    gx,gy=heading_point
    if not all(0<=p<bound for p,bound in
               ((sx,WIDTH),(sy,HEIGHT),(gx,WIDTH),(gy,HEIGHT))):
        raise ValueError("H147 native source or goal model point invalid")
    before=page.screenshot(type="png")
    page.mouse.move(sx,sy)
    page.mouse.down()
    page.mouse.move(sx+6,sy+18,steps=8)
    page.mouse.move(gx,gy-38,steps=24)
    activated=False
    release_offset=None
    threshold_count=0
    try:
        for dy in OFFSETS:
            if not 0<=gy+dy<HEIGHT:continue
            page.mouse.move(gx,gy+dy,steps=4)
            page.wait_for_timeout(60)
            observation=page.screenshot(type="png")
            changed=changed_samples(before,observation,[gx,gy])
            # Avoid interpretation of an outline when cursor has not
            # entered the below-heading drop target. The test uses no
            # DOM element presence, true slot bbox, or CSS class.
            if dy>=18 and changed>=MIN_VISIBLE_CHANGED_SAMPLES:
                activated=True
                release_offset=dy
                threshold_count=changed
                break
        if not activated:
            # Explicitly a no-visual-signal fallback, never use oracle.
            page.mouse.move(gx,min(HEIGHT-1,gy+30),steps=6)
            release_offset=30
    finally:
        page.mouse.up()
    return {"image_feedback_activated":activated,
            "release_offset":release_offset,
            "change_pixel_samples":threshold_count,
            "hidden_drop_dom_read_by_controller":False}

def one_arm(url,browser,original,arm):
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
        src=page.locator(".task_column").nth(0).locator(
            ".task_card").nth(original["source_index"])
        dest=page.locator(".task_column").nth(original["destination_column"])
        header=dest.locator(".task_column_heading").first
        hidden=dest.locator(".hide_drop").nth(original["destination_index"])
        src.wait_for(timeout=16000)
        header.wait_for(timeout=16000)
        hidden.wait_for(timeout=16000)
        from h143.collect_shifted_source import style
        page.add_style_tag(content=style(original["layout"]))
        a,b=src.bounding_box(),header.bounding_box()
        if not visible_box(a) or not visible_box(b):
            return {"qualified":False,"reason":"reference_card_or_heading_offscreen"}
        sx,sy=visible_center(a)
        _,point=jittered_points(a,b,original["attempt_index"])
        try:
            if arm=="direct":
                fake={"x":point[0]-60,"y":point[1]-15.5,
                      "width":120,"height":31}
                move_card_by_coarse_visible_goal(page,[sx,sy],fake)
                feedback={"image_feedback_activated":False,
                          "release_offset":27.5,
                          "change_pixel_samples":0}
            elif arm=="visual_entry":
                feedback=visually_observed_drop(page,[sx,sy],point)
            else:
                raise ValueError("H147 undefined comparison arm")
        except (ValueError,TimeoutError):
            return {"qualified":False,"reason":"browser_mouse_invalid"}
        page.wait_for_timeout(130)
        current=json.loads(page.evaluate("localStorage.getItem('tasks')"))
        changed=sum(x["task"]==original["title"] and
                    x["status"]==original["destination_status"]
                    for x in current)==1
        page.reload(wait_until="commit",timeout=18000)
        persisted_rows=json.loads(page.evaluate("localStorage.getItem('tasks')"))
        persisted=sum(x["task"]==original["title"] and
                      x["status"]==original["destination_status"]
                      for x in persisted_rows)==1
        return {"qualified":True,"persisted":bool(changed and persisted),
                **feedback}
    finally:ctx.close()

def run(owner_root,output,max_attempts=260):
    from playwright.sync_api import sync_playwright
    owner_root,output=Path(owner_root),Path(output)
    if (output.exists() or
        not any(p.name=="private-h147" for p in output.parents) or
        max_attempts<CASES):
        raise ValueError("H147 create-only new original visual action source")
    original=verify_owner_original(
        owner_root/"private-h140"/"upstream"/"react-native-board")
    results=[];rejects=defaultdict(int)
    with sync_playwright() as pw:
        opts={"headless":True}
        bin=_chromium_binary()
        if bin:opts["executable_path"]=bin
        browser=pw.chromium.launch(**opts,timeout=20000)
        try:
            with serve(original) as url:
                for attempt in range(OFFSET,OFFSET+max_attempts):
                    if len(results)>=CASES:break
                    seed=case(attempt)
                    seed["attempt_index"]=attempt
                    key=(seed["language"],seed["destination_status"])
                    if sum((r["language"],r["destination_status"])==key
                           for r in results)>=4:
                        rejects["stratum_full"]+=1
                        continue
                    arms={name:one_arm(url,browser,seed,name)
                          for name in ARMS}
                    if not all(x["qualified"] for x in arms.values()):
                        rejects["both_arms_not_qualified"]+=1
                        continue
                    results.append({
                        "attempt_index":attempt,
                        "language":seed["language"],
                        "destination_status":seed["destination_status"],
                        "layout":seed["layout"],
                        "same_noisy_visible_heading_point_per_arm":True,
                        "direct_persisted":arms["direct"]["persisted"],
                        "visual_entry_persisted":arms["visual_entry"]["persisted"],
                        "visual_feedback_activated":arms["visual_entry"]["image_feedback_activated"],
                        "visual_release_offset":arms["visual_entry"]["release_offset"],
                        "visual_change_pixel_samples":arms["visual_entry"]["change_pixel_samples"],
                    })
                    if len(results)%4==0:
                        print("H147_NEW_NATIVE_IMAGE_FEEDBACK_DRAG_AB",
                              len(results),CASES,flush=True)
        finally:browser.close()
    counts=Counter((x["language"],x["destination_status"]) for x in results)
    layouts=Counter(x["layout"] for x in results)
    if (len(results)!=CASES or
        counts!={(a,b):4 for a in ("ar","en") for b in ("doing","done")}
        or len(layouts)!=4):
        raise ValueError("H147 original image-feedback experiment incomplete")
    summary={
        "schema":SCHEMA,
        "status":"H147_NEW_INDEPENDENT_TRUE_NATIVE_VISUAL_ENTRY_AB_COMPLETE",
        "original_new_task_count":CASES,
        "real_mouse_actions":CASES*2,
        "direct_persisted":sum(r["direct_persisted"] for r in results),
        "visual_entry_persisted":sum(r["visual_entry_persisted"] for r in results),
        "visual_feedback_signaled":sum(r["visual_feedback_activated"] for r in results),
        "visual_only_success":sum(not r["direct_persisted"] and r["visual_entry_persisted"]
                                  for r in results),
        "direct_only_success":sum(r["direct_persisted"] and not r["visual_entry_persisted"]
                                  for r in results),
        "strata":{"/".join(k):v for k,v in counts.items()},
        "layouts":dict(layouts),"source_refusals":dict(rejects),
        "source_card_center_reference_only":True,
        "heading_input_noisy_inside_visible_header":True,
        "invisible_drop_dom_given_to_controller":False,
        "model_neural_calls":0,"real_optimizer_steps":0,
        "sealed_h140_blind_calls":0,"Hala_35B_changed":False,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf8") as f:
        f.write(json.dumps({"summary":summary,"owner_private_cases":results},
                           sort_keys=True,indent=2)+"\n")
    return summary
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner-root",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    print(json.dumps(run(a.owner_root,a.output),indent=2,sort_keys=True))
