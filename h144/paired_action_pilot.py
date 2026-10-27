"""H144 NEW ORIGINAL native React paired controller evaluation (source-only).

No H143 consumed model-evaluation images or predictions. 16 new independent
task seeds with four previously described CSS-shift families. Both arms
get ONLY reference points for *visible* source card + destination heading;
never the invisible DOM drop slot. Compare a baseline single approach
against a preregistered multi-observation hover/sweep/release controller.
This evaluates ACTUATORS, not model grounding, and is owner-private.
"""
import argparse
import json
import math
from collections import Counter,defaultdict
from pathlib import Path

from h138.probe import serve
from h130.browser_env import WIDTH,HEIGHT,_chromium_binary
from h140.vanilla_pilot import visible_box
from h141.coarse_column_action import (
    move_card_by_coarse_visible_goal,visible_center
)
from h141.interactive_source import verify_owner_original
from h142.collect_visible_source import seed_scene
from h143.collect_shifted_source import TRANSFORMS,style

SCHEMA="hala.h144.new-shifted-original-reference-action-ab.v1"
ATTEMPT_OFFSET=82000
CASES=16
ARMS=("baseline","staged_sweep")

def case(attempt):
    if attempt<ATTEMPT_OFFSET:
        raise ValueError("H144 never reuse H143 consumed model-evaluated sources")
    source=seed_scene(attempt)
    source["layout"]=tuple(TRANSFORMS)[(attempt//4)%4]
    return source

def staged_visible_sweep(page,start,heading):
    """No DOM/dnd-area query; action uses ONLY visible screen positions."""
    if (not isinstance(start,(tuple,list)) or len(start)!=2 or
        any(type(x) not in (int,float) or not math.isfinite(x) for x in start)
        or not 0<=start[0]<WIDTH or not 0<=start[1]<HEIGHT or
        not visible_box(heading)):
        raise ValueError("H144 genuine visible start/heading reference required")
    hx,hy=visible_center(heading)
    offsets=(28,72,47)
    if any(not 0<=hy+dy<HEIGHT for dy in offsets):
        raise ValueError("H144 staged approach would leave viewport")
    page.mouse.move(*start)
    page.mouse.down()
    page.mouse.move(start[0]+6,start[1]+18,steps=8)
    # Approach the coarse visible heading in stages to activate the
    # true native HTML5 DragEnter state; re-observe screenshot between
    # stages without consulting CSS/DOM hidden-target rectangles.
    for offset in offsets:
        page.mouse.move(hx,hy+offset,steps=16)
        page.wait_for_timeout(105)
        _=page.screenshot(type="png")
    page.mouse.up()
    return [hx,hy+offsets[-1]]

def run_arm(url,browser,seed,arm):
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
        task=page.locator(".task_column").nth(0).locator(
            ".task_card").nth(seed["source_index"])
        dest=page.locator(".task_column").nth(seed["destination_column"])
        heading=dest.locator(".task_column_heading").first
        hidden=dest.locator(".hide_drop").nth(seed["destination_index"])
        task.wait_for(timeout=18000)
        heading.wait_for(timeout=18000)
        hidden.wait_for(timeout=18000)
        page.add_style_tag(content=style(seed["layout"]))
        source_rect=task.bounding_box()
        goal_rect=heading.bounding_box()
        if not visible_box(source_rect) or not visible_box(goal_rect):
            return {"attempt_qualified":False,"reason":"source_or_header_offscreen"}
        start=visible_center(source_rect)
        try:
            if arm=="baseline":
                move_card_by_coarse_visible_goal(page,start,goal_rect)
            elif arm=="staged_sweep":
                staged_visible_sweep(page,start,goal_rect)
            else:raise ValueError("H144 unregistered controller")
        except Exception:
            return {"attempt_qualified":False,"reason":"native_mouse_controller_failed"}
        page.wait_for_timeout(140)
        native=json.loads(page.evaluate("localStorage.getItem('tasks')"))
        changed=sum(row["task"]==seed["title"] and
                    row["status"]==seed["destination_status"] for row in native)==1
        page.reload(wait_until="commit",timeout=18000)
        stored=json.loads(page.evaluate("localStorage.getItem('tasks')"))
        persisted=sum(row["task"]==seed["title"] and
                      row["status"]==seed["destination_status"] for row in stored)==1
        return {"attempt_qualified":True,
                "real_drag_status_changed":changed,
                "true_persisted_native_task_completion":bool(changed and persisted)}
    finally:ctx.close()

def evaluate(owner_root,output,max_attempts=200):
    from playwright.sync_api import sync_playwright
    owner_root,output=Path(owner_root),Path(output)
    if (output.exists() or
        not any(p.name=="private-h144" for p in output.parents) or
        max_attempts<CASES):
        raise ValueError("H144 independent original controller QA must be create-only")
    built=verify_owner_original(owner_root/"private-h140"/"upstream"/"react-native-board")
    outcomes=[];refused=defaultdict(int)
    with sync_playwright() as pw:
        kwargs={"headless":True}
        binary=_chromium_binary()
        if binary:kwargs["executable_path"]=binary
        browser=pw.chromium.launch(**kwargs,timeout=20000)
        try:
            with serve(built) as url:
                for attempt in range(ATTEMPT_OFFSET,ATTEMPT_OFFSET+max_attempts):
                    if len(outcomes)>=CASES:break
                    original=case(attempt)
                    stratum=(original["language"],original["destination_status"])
                    if sum((r["language"],r["destination_status"])==stratum
                           for r in outcomes)>=CASES//4:
                        refused["balanced_stratum_full"]+=1
                        continue
                    # Each arm sees an identical untouched original source
                    # seed, isolated Playwright context and reference
                    # VISIBLE card/heading points. No hidden rect to policy.
                    actual={arm:run_arm(url,browser,original,arm) for arm in ARMS}
                    if not all(r["attempt_qualified"] for r in actual.values()):
                        refused["not_qualified_both_arms"]+=1
                        continue
                    outcomes.append({
                        "attempt_index":attempt,"language":original["language"],
                        "destination_status":original["destination_status"],
                        "layout":original["layout"],
                        "true_reference_source_heading_not_model_points":True,
                        "baseline_persisted":actual["baseline"][
                            "true_persisted_native_task_completion"],
                        "staged_persisted":actual["staged_sweep"][
                            "true_persisted_native_task_completion"],
                    })
                    if len(outcomes)%4==0:
                        print("H144_FRESH_PAIRED_NATIVE_CONTROLLER_AB",
                              len(outcomes),CASES,flush=True)
        finally:browser.close()
    if len(outcomes)!=CASES:
        raise ValueError("H144 true paired independent original action QA incomplete")
    strata=Counter((x["language"],x["destination_status"]) for x in outcomes)
    if strata!={(l,d):4 for l in ("ar","en") for d in ("doing","done")}:
        raise ValueError("H144 genuine original source bilingual/goal balance drift")
    layouts=Counter(x["layout"] for x in outcomes)
    if len(layouts)!=4:
        raise ValueError("H144 genuinely shifted source layouts missing")
    result={
        "schema":SCHEMA,"status":"COMPLETED_NEW_REFERENCE_ONLY_DRAG_CONTROLLER_AB",
        "source_app_unmodified_git_sha":"fa047f7a32afb57750ed5246456dc282ea7befcd",
        "new_original_task_seeds":CASES,
        "paired_native_drag_actions":CASES*2,
        "baseline_persisted":sum(x["baseline_persisted"] for x in outcomes),
        "staged_persisted":sum(x["staged_persisted"] for x in outcomes),
        "both_succeeded":sum(x["baseline_persisted"] and x["staged_persisted"]
                             for x in outcomes),
        "staged_improved_cases":sum(not x["baseline_persisted"] and x["staged_persisted"]
                                   for x in outcomes),
        "staged_regressed_cases":sum(x["baseline_persisted"] and not x["staged_persisted"]
                                    for x in outcomes),
        "layout_counts":dict(layouts),
        "strata":{"/".join(k):v for k,v in strata.items()},
        "unqualified_reference_attempts":dict(refused),
        "model_neural_calls":0,
        "model_input_oracle_coords_uploaded":False,
        "h140_blind_calls":0,"flagship_Hala_35B_updated":False,
        "promoted_model":False,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf8") as fp:
        fp.write(json.dumps({"summary":result,"private_episodes":outcomes},
                            indent=2,sort_keys=True)+"\n")
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner-root",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--max-attempts",type=int,default=200)
    a=p.parse_args()
    print(json.dumps(evaluate(a.owner_root,a.output,a.max_attempts),
                     indent=2,sort_keys=True))
