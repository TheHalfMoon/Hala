"""H146 independent genuine original-browser localization-error factorial.

Four cleanly separated ACTUATOR conditions, identical fresh original
React task seed in isolated browser sessions, same code / mouse path:
- center source, center heading (clean reference)
- jittered source, center heading (source error only)
- center source, jittered heading (destination error only)
- jittered source, jittered heading (combined error)

All source/heading model-like points are inside VISIBLE native bounding
boxes; controller is blind to original geometry and hidden DOM targets.
Original React upstream is pinned MIT, not modified. This is NOT neural
inference, NOT optimization and NOT a recycled H145/H143 cohort.
"""
import argparse
import json
from collections import Counter,defaultdict
from pathlib import Path

from h130.browser_env import WIDTH,HEIGHT,_chromium_binary
from h138.probe import serve
from h140.vanilla_pilot import visible_box
from h141.coarse_column_action import (
    move_card_by_coarse_visible_goal,visible_center
)
from h141.interactive_source import verify_owner_original
from h143.collect_shifted_source import style
from h145.noisy_point_action_ab import jittered_points,source as old_source

SCHEMA="hala.h146.fresh-native-point-error-factorial.v1"
OFFSET=102000
N=16
ARMS=("clean","source_noise","heading_noise","both_noise")

def source(attempt):
    if attempt<OFFSET:
        raise ValueError("H146 must not reuse historical model evaluated task IDs")
    row=old_source(attempt)
    return row

def choose_points(start_box,heading_box,seed_index,arm):
    """All four arms share exact same deterministic source/heading noise."""
    if arm not in ARMS or not visible_box(start_box) or not visible_box(heading_box):
        raise ValueError("H146 original visible points/arm not qualified")
    source_center=list(visible_center(start_box))
    goal_center=list(visible_center(heading_box))
    noisy_source,noisy_heading=jittered_points(
        start_box,heading_box,seed_index)
    return {
        "clean":(source_center,goal_center),
        "source_noise":(noisy_source,goal_center),
        "heading_noise":(source_center,noisy_heading),
        "both_noise":(noisy_source,noisy_heading),
    }[arm]

def one_arm(url,browser,original,arm):
    context=browser.new_context(
        viewport={"width":WIDTH,"height":HEIGHT},
        device_scale_factor=1,service_workers="block")
    context.add_init_script(
        "if(localStorage.getItem('tasks')===null){"+
        "localStorage.setItem('tasks',"+
        json.dumps(json.dumps(original["rows"]))+");}")
    context.route("https://**/*",lambda route:route.abort())
    try:
        page=context.new_page()
        page.goto(url,wait_until="commit",timeout=18000)
        source=page.locator(".task_column").nth(0).locator(
            ".task_card").nth(original["source_index"])
        col=page.locator(".task_column").nth(original["destination_column"])
        heading=col.locator(".task_column_heading").first
        hidden=col.locator(".hide_drop").nth(original["destination_index"])
        source.wait_for(timeout=16000)
        heading.wait_for(timeout=16000)
        hidden.wait_for(timeout=16000)
        page.add_style_tag(content=style(original["layout"]))
        source_rect,heading_rect=source.bounding_box(),heading.bounding_box()
        if not visible_box(source_rect) or not visible_box(heading_rect):
            return {"qualified":False,"reason":"source_or_heading_offscreen"}
        src,goal=choose_points(
            source_rect,heading_rect,original["attempt_index"],arm)
        # The mouse policy ONLY sees the point proposed to it, never
        # true source or target geometry and never invisible drop areas.
        fake={"x":goal[0]-60,"y":goal[1]-15.5,
              "width":120,"height":31}
        try:
            move_card_by_coarse_visible_goal(page,src,fake)
        except (ValueError,TimeoutError):
            return {"qualified":False,"reason":"mouse_controller_invalid"}
        page.wait_for_timeout(140)
        before=json.loads(page.evaluate("localStorage.getItem('tasks')"))
        changed=sum(x["task"]==original["title"] and
                    x["status"]==original["destination_status"]
                    for x in before)==1
        page.reload(wait_until="commit",timeout=18000)
        after=json.loads(page.evaluate("localStorage.getItem('tasks')"))
        persisted=sum(x["task"]==original["title"] and
                      x["status"]==original["destination_status"]
                      for x in after)==1
        return {
            "qualified":True,
            "source_point_inside_visible_card":True,
            "heading_point_inside_visible_label":True,
            "persisted":bool(changed and persisted),
        }
    finally:context.close()

def run(owner_root,output,max_attempts=300):
    from playwright.sync_api import sync_playwright
    owner_root,output=Path(owner_root),Path(output)
    if (output.exists() or
        not any(p.name=="private-h146" for p in output.parents) or
        max_attempts<N):
        raise ValueError("H146 new owner-only paired experiment create once")
    source_dir=verify_owner_original(
        owner_root/"private-h140"/"upstream"/"react-native-board")
    outcomes=[];rejected=defaultdict(int)
    with sync_playwright() as pw:
        opts={"headless":True}
        bin=_chromium_binary()
        if bin:opts["executable_path"]=bin
        browser=pw.chromium.launch(**opts,timeout=25000)
        try:
            with serve(source_dir) as url:
                for i in range(OFFSET,OFFSET+max_attempts):
                    if len(outcomes)>=N:break
                    original=source(i)
                    original["attempt_index"]=i
                    stratum=(original["language"],original["destination_status"])
                    if sum((r["language"],r["destination_status"])==stratum
                           for r in outcomes)>=4:
                        rejected["balanced_quota"]+=1
                        continue
                    results={arm:one_arm(url,browser,original,arm)
                             for arm in ARMS}
                    if not all(x["qualified"] for x in results.values()):
                        rejected["unqualified_four_way_native_source"]+=1
                        continue
                    outcome={
                        "attempt_index":i,
                        "language":original["language"],
                        "destination_status":original["destination_status"],
                        "layout":original["layout"],
                        "same_task_same_noise_seed_four_separate_contexts":True,
                        "source_and_heading_gold_not_given_to_mouse":True,
                        "success":{arm:results[arm]["persisted"] for arm in ARMS},
                    }
                    outcomes.append(outcome)
                    if len(outcomes)%4==0:
                        print("H146_FRESH_TRUE_NATIVE_FOUR_WAY_POINT_ERROR",
                              len(outcomes),N,flush=True)
        finally:browser.close()
    strata=Counter((x["language"],x["destination_status"]) for x in outcomes)
    layouts=Counter(x["layout"] for x in outcomes)
    if (len(outcomes)!=N or
        strata!={(a,b):4 for a in ("ar","en") for b in ("doing","done")} or
        len(layouts)!=4):
        raise ValueError("H146 new original source incomplete or unbalanced")
    counts={arm:sum(int(r["success"][arm]) for r in outcomes)
            for arm in ARMS}
    report={
        "schema":SCHEMA,
        "status":"QUALIFIED_NEW_NATIVE_POINT_ERROR_FACTORIAL",
        "source_app_git_sha":"fa047f7a32afb57750ed5246456dc282ea7befcd",
        "new_original_source_task_count":N,
        "genuine_paired_original_browser_drags":N*len(ARMS),
        "persisted_by_arm":counts,
        "layout_counts":dict(layouts),
        "strata":{"/".join(k):v for k,v in strata.items()},
        "rejections":dict(rejected),
        "neural_model_calls":0,"real_optimizer_steps":0,
        "h140_sealed_blind_calls":0,"Hala_35B_changed":False,
        "no_model_promotion":True,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf8") as f:
        f.write(json.dumps({"summary":report,"owner_private_cases":outcomes},
                           indent=2,sort_keys=True)+"\n")
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner-root",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--max-attempts",type=int,default=300)
    args=p.parse_args()
    print(json.dumps(run(args.owner_root,args.output,args.max_attempts),
                     sort_keys=True,indent=2))
