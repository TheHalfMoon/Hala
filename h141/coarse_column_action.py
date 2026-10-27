"""H141 original React coarse-column approach controller source-reference QA.

The ACTUATOR uses only a source point and a VISIBLE column-heading
rectangle, never invisible DOM drop rectangles. During this owner-only
pilot, the source/heading reference boxes come from DOM ground truth,
not a neural model. It does NOT establish independent model accuracy.
"""
import argparse
import json
import math
from collections import Counter,defaultdict
from pathlib import Path

from h138.probe import serve
from h130.browser_env import WIDTH,HEIGHT,_chromium_binary
from h140.vanilla_pilot import visible_box
from h141.interactive_source import original_case,verify_owner_original

SCHEMA="hala.h141.coarse-visible-column-action-reference-qa.v1"

def visible_center(rect):
    if not visible_box(rect):
        raise ValueError("H141 needs a visible predicted point/column")
    return (rect["x"]+rect["width"]/2,
            rect["y"]+rect["height"]/2)

def approach_point_from_visible_column_heading(rect):
    """Approach first slot using ONLY visibly identifiable heading bounds.

    This deterministic offset is an owner original-source action prototype,
    not a universal learned computer-use navigation policy.
    """
    x,y=visible_center(rect)
    result=(x,rect["y"]+rect["height"]+12)
    if not 0<=result[0]<WIDTH or not 0<=result[1]<HEIGHT:
        raise ValueError("H141 coarse approach proposed offscreen")
    return result

def move_card_by_coarse_visible_goal(page,start,visible_heading_box):
    """No DOM destination query, CSS state read or true drop bbox used here."""
    if (not isinstance(start,(tuple,list)) or len(start)!=2
        or any(type(v) not in (float,int) or not math.isfinite(v) for v in start)
        or not (0<=start[0]<WIDTH and 0<=start[1]<HEIGHT)):
        raise ValueError("H141 original source card point invalid")
    end=approach_point_from_visible_column_heading(visible_heading_box)
    page.mouse.move(*start)
    page.mouse.down()
    page.mouse.move(start[0]+6,start[1]+18,steps=7)
    page.mouse.move(*end,steps=24)
    page.wait_for_timeout(150)
    page.mouse.up()
    return end

def check_original_reference(source,count=8,attempt_offset=46000,max_attempts=120):
    from playwright.sync_api import sync_playwright
    built=verify_owner_original(source)
    if (count not in (8,16) or attempt_offset<46000 or attempt_offset%2
        or max_attempts<count):
        raise ValueError("H141 original-action source preregistration invalid")
    accepted=[];rejected=defaultdict(int)
    with sync_playwright() as pw:
        opts={"headless":True}
        path=_chromium_binary()
        if path:opts["executable_path"]=path
        browser=pw.chromium.launch(**opts,timeout=20000)
        try:
            with serve(built) as url:
                for attempt in range(attempt_offset,attempt_offset+max_attempts):
                    if len(accepted)>=count:break
                    case=original_case(attempt)
                    lang=case["language"]
                    if sum(x["language"]==lang for x in accepted)>=count//2:
                        rejected["bilingual_cap"]+=1
                        continue
                    ctx=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                                            device_scale_factor=1,service_workers="block")
                    ctx.add_init_script(
                        "if(!localStorage.getItem('tasks'))"+
                        "localStorage.setItem('tasks',"+
                        json.dumps(json.dumps(case["rows"]))+");")
                    ctx.route("https://**/*",lambda route:route.abort())
                    try:
                        page=ctx.new_page()
                        page.goto(url,wait_until="commit",timeout=18000)
                        src=page.locator(".task_column").nth(0).locator(
                            ".task_card").nth(case["source_index"])
                        heading=page.locator(".task_column").nth(1).locator(
                            ".task_column_heading").first
                        src.wait_for(timeout=16000)
                        heading.wait_for(timeout=16000)
                        start_rect,goal_rect=src.bounding_box(),heading.bounding_box()
                        if not visible_box(start_rect) or not visible_box(goal_rect):
                            rejected["visible_reference_source_offscreen"]+=1
                            continue
                        pt=visible_center(start_rect)
                        # ACTUATOR sees only the visible start/heading geometry.
                        action=move_card_by_coarse_visible_goal(page,pt,goal_rect)
                        page.wait_for_timeout(120)
                        state=json.loads(page.evaluate("localStorage.getItem('tasks')"))
                        changed=sum(x["task"]==case["title"] and
                                    x["status"]=="doing" for x in state)==1
                        page.reload(wait_until="commit",timeout=18000)
                        new=json.loads(page.evaluate("localStorage.getItem('tasks')"))
                        persisted=sum(x["task"]==case["title"] and
                                      x["status"]=="doing" for x in new)==1
                        if not (changed and persisted):
                            rejected["actual_native_action_not_persisted"]+=1
                            continue
                        accepted.append({
                            "attempt_index":attempt,
                            "language":lang,
                            "source_proposal":"owner_source_reference_not_model_prediction",
                            "goal_proposal":"owner_VISIBLE_column_heading_not_hidden_drop",
                            "coarse_release_x":action[0],
                            "coarse_release_y":action[1],
                            "true_native_persisted_task_completion":True,
                        })
                        if len(accepted)%2==0:
                            print("H141_OWNER_COARSE_COLUMN_NO_HIDDEN_BBOX_ACTION_QA",
                                  len(accepted),count,flush=True)
                    finally:ctx.close()
        finally:browser.close()
    if len(accepted)!=count or Counter(x["language"] for x in accepted)!={
        "ar":count//2,"en":count//2}:
        raise ValueError("H141 original coarse action QA count or language balance failed")
    return {
        "schema":SCHEMA,
        "status":"OWNER_VISIBLE_COLUMN_REFERENCE_ACTION_PASS_NOT_NEURAL",
        "source_cases":count,
        "real_persisted_task_completions":count,
        "hidden_drop_dom_box_used_by_controller":False,
        "visible_start_heading_proposals_from_owner_reference":True,
        "neural_model_predictions_used":0,
        "h140_blind_model_calls":0,
        "languages":dict(Counter(x["language"] for x in accepted)),
        "rejected_source_attempts":dict(rejected),
        "main_hala_35B_updated":False,
    }

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--count",type=int,default=8)
    p.add_argument("--attempt-offset",type=int,default=46000)
    p.add_argument("--max-attempts",type=int,default=120)
    a=p.parse_args()
    print(json.dumps(check_original_reference(
        a.source,count=a.count,attempt_offset=a.attempt_offset,
        max_attempts=a.max_attempts),indent=2,sort_keys=True))
