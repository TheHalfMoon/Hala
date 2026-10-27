"""H152 independent owner-only horizontal visual drop search A/B.

No reused H151 neural model evaluation source or older QA episodes.
Sixteen brand-new original React task IDs 174000+, balanced bilingual
Doing/Done across four shifted layouts. Both arms receive the same
visible source card center and artificially displaced VISIBLE heading
point (+/- 38 px x; zero y): H150 prior visual center vs predeclared
multi-lane screenshot-only horizontal approach, without hidden DOM access.

This is source-reference controller robustness, not neural success.
"""
import argparse
import json
from collections import Counter,defaultdict
from pathlib import Path

from h130.browser_env import WIDTH,HEIGHT,_chromium_binary
from h138.probe import serve
from h140.vanilla_pilot import visible_box
from h141.coarse_column_action import visible_center
from h141.interactive_source import verify_owner_original
from h143.collect_shifted_source import style
from h145.noisy_point_action_ab import source as original_case
from h150.visual_region_controller import (
    visually_center_drop,locate_visible_change,
    changed_samples,MIN_VISIBLE_CHANGED_SAMPLES,
    DROP_ENTRY_OFFSETS
)

SCHEMA="hala.h152.independent-horizontal-visual-drop-search.v1"
OFFSET=174000
N=16
LANES=(0,-40,40,-80,80)
HORIZONTAL_PERTURB=38

def new_source(attempt):
    if attempt<OFFSET:raise ValueError("H152 must not reuse prior original episode")
    row=original_case(attempt)
    row["attempt_index"]=attempt
    return row

def displaced_visible_heading(box,attempt):
    if not visible_box(box):raise ValueError("H152 original visible header unavailable")
    x,y=visible_center(box)
    offset=HORIZONTAL_PERTURB*(1 if attempt%2 else -1)
    candidate=[x+offset,y]
    if not (box["x"]<=candidate[0]<=box["x"]+box["width"] and
            box["y"]<=candidate[1]<=box["y"]+box["height"]):
        raise ValueError("H152 perturbed model-like x outside visible header")
    return candidate

def search_horizontal_visible_release(page,start,heading):
    """Scan SCREEN observations around the imperfect visible x point."""
    sx,sy=start;gx,gy=heading
    if not all(0<=v<limit for v,limit in
               ((sx,WIDTH),(sy,HEIGHT),(gx,WIDTH),(gy,HEIGHT))):
        raise ValueError("H152 screenshot-space point invalid")
    reference=page.screenshot(type="png")
    page.mouse.move(sx,sy)
    page.mouse.down()
    page.mouse.move(sx+6,sy+18,steps=8)
    selected=None
    lane_used=None
    retained=False
    visual_signal=False
    try:
        for lane in LANES:
            x=gx+lane
            if not 0<=x<WIDTH:continue
            page.mouse.move(x,gy-38,steps=24)
            for delta in DROP_ENTRY_OFFSETS:
                y=gy+delta
                if not 0<=y<HEIGHT:continue
                page.mouse.move(x,y,steps=4)
                page.wait_for_timeout(50)
                image=page.screenshot(type="png")
                if (delta<18 or
                    changed_samples(reference,image,[x,gy])<
                    MIN_VISIBLE_CHANGED_SAMPLES):
                    continue
                visual_signal=True
                page.wait_for_timeout(150)
                screenshot=page.screenshot(type="png")
                proposed=locate_visible_change(reference,screenshot,[x,gy])
                if proposed is None:continue
                page.mouse.move(*proposed,steps=14)
                page.wait_for_timeout(90)
                stable=page.screenshot(type="png")
                retained=(changed_samples(reference,stable,[x,gy])
                          >=MIN_VISIBLE_CHANGED_SAMPLES)
                if retained:
                    selected=proposed
                    lane_used=lane
                    break
            if retained:break
        # No owner hidden DOM geometry, no DOM event telemetry, and no
        # source state read is ever supplied to the mouse controller.
    finally:page.mouse.up()
    return {"visible_region_found":selected is not None,
            "visual_signal_retained":retained,
            "visual_signal_seen":visual_signal,
            "used_horizontal_lane":lane_used,
            "controller_read_hidden_target_dom":False}

def one_arm(url,browser,task,arm):
    ctx=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                            device_scale_factor=1,service_workers="block")
    ctx.add_init_script(
        "if(localStorage.getItem('tasks')===null){"+
        "localStorage.setItem('tasks',"+
        json.dumps(json.dumps(task["rows"]))+");}")
    ctx.route("https://**/*",lambda request:request.abort())
    try:
        page=ctx.new_page()
        page.goto(url,wait_until="commit",timeout=18000)
        card=page.locator(".task_column").nth(0).locator(
            ".task_card").nth(task["source_index"])
        column=page.locator(".task_column").nth(task["destination_column"])
        heading=column.locator(".task_column_heading").first
        hidden=column.locator(".hide_drop").nth(task["destination_index"])
        card.wait_for(timeout=16000)
        heading.wait_for(timeout=16000)
        hidden.wait_for(timeout=16000)
        page.add_style_tag(content=style(task["layout"]))
        source_box,heading_box=card.bounding_box(),heading.bounding_box()
        if not visible_box(source_box) or not visible_box(heading_box):
            return {"qualified":False}
        source=visible_center(source_box)
        goal=displaced_visible_heading(heading_box,task["attempt_index"])
        if arm=="prior_h150":
            feedback=visually_center_drop(page,source,goal)
            found=feedback["visible_region_found"]
            stable=feedback["retained_visual_change_at_release"]
            lane=None
        elif arm=="horizontal_lanes":
            feedback=search_horizontal_visible_release(page,source,goal)
            found=feedback["visible_region_found"]
            stable=feedback["visual_signal_retained"]
            lane=feedback["used_horizontal_lane"]
        else:raise ValueError("H152 invalid experiment arm")
        page.wait_for_timeout(140)
        actual=json.loads(page.evaluate("localStorage.getItem('tasks')"))
        changed=any(row["task"]==task["title"] and
                    row["status"]==task["destination_status"] for row in actual)
        page.reload(wait_until="commit",timeout=16000)
        saved=json.loads(page.evaluate("localStorage.getItem('tasks')"))
        completed=changed and any(
            row["task"]==task["title"] and
            row["status"]==task["destination_status"] for row in saved)
        return {"qualified":True,"persisted":bool(completed),
                "found":found,"stable":stable,"lane":lane}
    finally:ctx.close()

def run(owner_root,output,max_attempts=240):
    from playwright.sync_api import sync_playwright
    owner_root,output=Path(owner_root),Path(output)
    if (output.exists() or
        not any(p.name=="private-h152" for p in output.parents) or
        max_attempts<N):
        raise ValueError("H152 independent source-only pilot write-once")
    original=verify_owner_original(
        owner_root/"private-h140"/"upstream"/"react-native-board")
    episodes=[];excluded=defaultdict(int)
    with sync_playwright() as pw:
        cfg={"headless":True}
        binary=_chromium_binary()
        if binary:cfg["executable_path"]=binary
        browser=pw.chromium.launch(**cfg,timeout=22000)
        try:
            with serve(original) as url:
                for index in range(OFFSET,OFFSET+max_attempts):
                    if len(episodes)>=N:break
                    case=new_source(index)
                    group=(case["language"],case["destination_status"])
                    if sum((x["language"],x["destination_status"])==group
                           for x in episodes)>=4:
                        excluded["stratum_full"]+=1
                        continue
                    a=one_arm(url,browser,case,"prior_h150")
                    b=one_arm(url,browser,case,"horizontal_lanes")
                    if not a["qualified"] or not b["qualified"]:
                        excluded["unqualified_source_browser"]+=1
                        continue
                    episodes.append({
                        "attempt":index,
                        "language":case["language"],
                        "destination_status":case["destination_status"],
                        "layout":case["layout"],
                        "baseline_persisted":a["persisted"],
                        "new_persisted":b["persisted"],
                        "baseline_found":a["found"],
                        "new_found":b["found"],
                        "new_stable":b["stable"],
                        "new_lane":b["lane"],
                    })
                    if len(episodes)%4==0:
                        print("H152_TRUE_NATIVE_HORIZONTAL_SEARCH_AB",
                              len(episodes),N,flush=True)
        finally:browser.close()
    strata=Counter((e["language"],e["destination_status"]) for e in episodes)
    layouts=Counter(e["layout"] for e in episodes)
    if (len(episodes)!=N or
        strata!={(a,b):4 for a in ("ar","en") for b in ("doing","done")} or
        len(layouts)!=4):
        raise ValueError("H152 original task cohort incomplete")
    summary={
        "schema":SCHEMA,
        "status":"COMPLETE_FRESH_H152_SOURCE_CONTROLLER_ONLY",
        "new_original_tasks":N,"native_mouse_actions":2*N,
        "old_h150_persisted":sum(e["baseline_persisted"] for e in episodes),
        "new_horizontal_search_persisted":sum(e["new_persisted"] for e in episodes),
        "new_only":sum(e["new_persisted"] and not e["baseline_persisted"] for e in episodes),
        "old_only":sum(e["baseline_persisted"] and not e["new_persisted"] for e in episodes),
        "new_regions_detected":sum(e["new_found"] for e in episodes),
        "new_regions_stable":sum(e["new_stable"] for e in episodes),
        "horizontal_offset_px":HORIZONTAL_PERTURB,
        "lanes_px":list(LANES),
        "strata":{"/".join(k):v for k,v in strata.items()},
        "layouts":dict(layouts),"rejections":dict(excluded),
        "mouse_controller_read_hidden_drop_dom":False,
        "model_neural_calls":0,"optimizer_steps":0,
        "h140_sealed_blind_calls":0,"Hala_35B_changed":False,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf8") as handle:
        handle.write(json.dumps({
            "summary":summary,"owner_private_cases":episodes
        },sort_keys=True,indent=2)+"\n")
    return summary

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner-root",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    print(json.dumps(run(a.owner_root,a.output),sort_keys=True,indent=2))
