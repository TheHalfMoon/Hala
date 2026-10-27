"""H150 independent vision-guided drop release using real image differences.

NEW original task IDs >=154000, no old model-evaluation reuse.
Controller gets only visible source point and noisy visible heading point.
It compares destination screenshots during drag to identify newly drawn
drop-zone pixels and chooses the CENTER of that VISIBLE region. It never
reads hidden drop DOM, event log, CSS class, card state, or oracle bbox.
The owner evaluates persisted original React task state after mouse-up.
"""
import argparse
import io
import json
from collections import Counter
from pathlib import Path

from PIL import Image,ImageChops

from h130.browser_env import WIDTH,HEIGHT,_chromium_binary
from h138.probe import serve
from h140.vanilla_pilot import visible_box
from h141.coarse_column_action import visible_center
from h141.interactive_source import verify_owner_original
from h145.noisy_point_action_ab import jittered_points,source as original_case
from h143.collect_shifted_source import style
from h147.visual_drag_entry_ab import (
    one_arm as prior_visual_entry, OFFSETS,MIN_VISIBLE_CHANGED_SAMPLES,
    changed_samples
)

SCHEMA="hala.h150.new-original-visual-zone-internal-release.v1"
OFFSET=154000
N=16
MIN_CHANGED=35
PIXEL_STEP=2
PIXEL_DIFF=19
DROP_ENTRY_OFFSETS=(-24,-16,-8,0,8,14,18,22,26,30,36,42,50,58,70)

def case(i):
    if i<OFFSET:raise ValueError("H150 must use entirely fresh original tasks")
    row=original_case(i)
    row["attempt_index"]=i
    return row

def locate_visible_change(before,after,heading):
    """Only screen pixels. No DOM bounding-box reads in actuator."""
    gx,gy=heading
    x0=max(0,int(gx-110));x1=min(WIDTH,int(gx+110))
    y0=max(0,int(gy+12));y1=min(HEIGHT,int(gy+178))
    if x1-x0<60 or y1-y0<40:
        raise ValueError("H150 visible destination ROI unavailable")
    with Image.open(io.BytesIO(before)) as old:
        with Image.open(io.BytesIO(after)) as new:
            if old.size!=(WIDTH,HEIGHT) or new.size!=(WIDTH,HEIGHT):
                raise ValueError("H150 original native image size mismatch")
            crop=ImageChops.difference(
                old.convert("RGB").crop((x0,y0,x1,y1)),
                new.convert("RGB").crop((x0,y0,x1,y1)))
            pix=crop.load()
            candidates=[]
            for y in range(0,y1-y0,PIXEL_STEP):
                for x in range(0,x1-x0,PIXEL_STEP):
                    if max(pix[x,y])>=PIXEL_DIFF:
                        candidates.append((x+x0,y+y0))
    if len(candidates)<MIN_CHANGED:
        return None
    ys=sorted(p[1] for p in candidates)
    q_low=ys[len(ys)//10]
    q_high=ys[len(ys)*9//10]
    if q_high-q_low<18:
        return None
    # Destination centered by visible diff's central vertical region.
    # Horizontal point comes exclusively from noisy visible heading.
    result=[gx,(q_low+q_high)/2]
    if not (0<=result[0]<WIDTH and 0<=result[1]<HEIGHT):
        return None
    return result

def visually_center_drop(page,start,heading):
    """Screenshot-observed activation + infer safe interior before mouse-up."""
    sx,sy=start;gx,gy=heading
    if not all(0<=v<bound for v,bound in
               ((sx,WIDTH),(sy,HEIGHT),(gx,WIDTH),(gy,HEIGHT))):
        raise ValueError("H150 requires proposed visible image points")
    baseline=page.screenshot(type="png")
    page.mouse.move(sx,sy)
    page.mouse.down()
    page.mouse.move(sx+6,sy+18,steps=8)
    page.mouse.move(gx,gy-38,steps=24)
    found=None;validated=False;triggered=False
    try:
        for delta in DROP_ENTRY_OFFSETS:
            y=gy+delta
            if not 0<=y<HEIGHT:continue
            page.mouse.move(gx,y,steps=4)
            page.wait_for_timeout(70)
            now=page.screenshot(type="png")
            signal=changed_samples(baseline,now,[gx,gy])
            if delta<18 or signal<MIN_VISIBLE_CHANGED_SAMPLES:
                continue
            triggered=True
            page.wait_for_timeout(210)
            current=page.screenshot(type="png")
            found=locate_visible_change(baseline,current,[gx,gy])
            if found is not None:
                page.mouse.move(*found,steps=14)
                page.wait_for_timeout(110)
                again=page.screenshot(type="png")
                validated=changed_samples(baseline,again,[gx,gy])>=MIN_VISIBLE_CHANGED_SAMPLES
                if validated:break
        if not validated:
            # A missing visible region must not be replaced by DOM oracle.
            # Report as failed action, release where already hovering.
            pass
    finally:page.mouse.up()
    return {"visible_region_found":found is not None,
            "retained_visual_change_at_release":validated,
            "first_visual_signal_seen":triggered,
            "inferred_release_point":found,
            "controller_read_hidden_drop_dom":False}

def center_arm(url,browser,seed):
    ctx=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                            device_scale_factor=1,service_workers="block")
    ctx.add_init_script(
        "if(localStorage.getItem('tasks')===null){localStorage.setItem('tasks',"+
        json.dumps(json.dumps(seed["rows"]))+");}")
    ctx.route("https://**/*",lambda req:req.abort())
    try:
        page=ctx.new_page()
        page.goto(url,wait_until="commit",timeout=18000)
        card=page.locator(".task_column").nth(0).locator(
            ".task_card").nth(seed["source_index"])
        col=page.locator(".task_column").nth(seed["destination_column"])
        heading=col.locator(".task_column_heading").first
        hidden=col.locator(".hide_drop").nth(seed["destination_index"])
        card.wait_for(timeout=16000)
        heading.wait_for(timeout=16000)
        hidden.wait_for(timeout=16000)
        page.add_style_tag(content=style(seed["layout"]))
        sr,hr=card.bounding_box(),heading.bounding_box()
        if not visible_box(sr) or not visible_box(hr):
            return {"qualified":False}
        source_point=visible_center(sr)
        _,goal_point=jittered_points(sr,hr,seed["attempt_index"])
        feedback=visually_center_drop(page,source_point,goal_point)
        page.wait_for_timeout(120)
        status=json.loads(page.evaluate("localStorage.getItem('tasks')"))
        changed=any(x["task"]==seed["title"] and
                    x["status"]==seed["destination_status"] for x in status)
        page.reload(wait_until="commit",timeout=18000)
        persisted=json.loads(page.evaluate("localStorage.getItem('tasks')"))
        success=changed and any(
            x["task"]==seed["title"] and
            x["status"]==seed["destination_status"] for x in persisted)
        return {"qualified":True,"persisted":bool(success),**feedback}
    finally:ctx.close()

def run(owner_root,output,max_attempts=250):
    from playwright.sync_api import sync_playwright
    owner_root,output=Path(owner_root),Path(output)
    if (output.exists() or
        not any(x.name=="private-h150" for x in output.parents) or
        max_attempts<N):
        raise ValueError("H150 create-once owner private evaluation only")
    original=verify_owner_original(
        owner_root/"private-h140"/"upstream"/"react-native-board")
    examples=[]
    with sync_playwright() as pw:
        opts={"headless":True}
        binary=_chromium_binary()
        if binary:opts["executable_path"]=binary
        browser=pw.chromium.launch(**opts,timeout=20000)
        try:
            with serve(original) as url:
                for i in range(OFFSET,OFFSET+max_attempts):
                    if len(examples)>=N:break
                    seed=case(i)
                    group=(seed["language"],seed["destination_status"])
                    if sum((r["language"],r["destination_status"])==group
                           for r in examples)>=4:
                        continue
                    baseline=prior_visual_entry(url,browser,seed,"visual_entry")
                    vision=center_arm(url,browser,seed)
                    if not baseline["qualified"] or not vision["qualified"]:
                        continue
                    examples.append({
                        "attempt":i,"language":seed["language"],
                        "destination_status":seed["destination_status"],
                        "layout":seed["layout"],
                        "old_visual_persisted":baseline["persisted"],
                        "new_zone_center_persisted":vision["persisted"],
                        "visual_region_found":vision["visible_region_found"],
                        "retained_signal":vision["retained_visual_change_at_release"],
                        "release_point":vision["inferred_release_point"],
                    })
                    if len(examples)%4==0:
                        print("H150_FRESH_PIXELS_ONLY_DROP_INTERIOR_EVAL",
                              len(examples),N,flush=True)
        finally:browser.close()
    groups=Counter((x["language"],x["destination_status"]) for x in examples)
    layouts=Counter(x["layout"] for x in examples)
    if (len(examples)!=N or
        groups!={(l,d):4 for l in ("ar","en") for d in ("doing","done")}
        or len(layouts)!=4):
        raise ValueError("H150 independent genuine task corpus incomplete")
    results={
        "schema":SCHEMA,
        "status":"H150_COMPLETE_NEW_NATIVE_IMAGE_DIFF_DROP_AB",
        "new_original_tasks":N,
        "real_original_native_mouse_drags":2*N,
        "prior_visual_persisted":sum(x["old_visual_persisted"] for x in examples),
        "new_image_diff_persisted":sum(x["new_zone_center_persisted"] for x in examples),
        "new_only_successes":sum(x["new_zone_center_persisted"] and not x["old_visual_persisted"] for x in examples),
        "old_only_successes":sum(x["old_visual_persisted"] and not x["new_zone_center_persisted"] for x in examples),
        "visual_regions_localized":sum(x["visual_region_found"] for x in examples),
        "visual_signal_retained":sum(x["retained_signal"] for x in examples),
        "layouts":dict(layouts),
        "neural_model_calls":0,"new_optimizer_steps":0,
        "h140_sealed_blind_calls":0,"Hala35B_unchanged":True,
        "controller_saw_hidden_drop_dom":False
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf8") as f:
        f.write(json.dumps({"summary":results,"owner_private_cases":examples},
                           sort_keys=True,indent=2)+"\n")
    return results

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner-root",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    print(json.dumps(run(a.owner_root,a.output),sort_keys=True,indent=2))
