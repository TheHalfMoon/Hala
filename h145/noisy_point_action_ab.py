"""H145 independent original React paired visual-point NOISE robustness.

H143 original neural test already consumed; these task IDs >92000 are new.
Both policy arms get the SAME independently seeded perturbed coordinates
strictly WITHIN the visible source+heading rectangles, never hidden drop
boxes. This is source-only action control, not neural evaluation or SFT.
"""
import argparse
import json
import math
import random
from collections import Counter,defaultdict
from pathlib import Path

from h138.probe import serve
from h130.browser_env import WIDTH,HEIGHT,_chromium_binary
from h140.vanilla_pilot import visible_box
from h141.coarse_column_action import move_card_by_coarse_visible_goal
from h141.interactive_source import verify_owner_original
from h142.collect_visible_source import seed_scene
from h143.collect_shifted_source import TRANSFORMS,style
from h144.paired_action_pilot import staged_visible_sweep

SCHEMA="hala.h145.new-original-noisy-visible-point-action-ab.v1"
OFFSET=92000
N=16
ARMS=("direct","staged_sweep")

def source(attempt):
    if attempt<OFFSET:
        raise ValueError("H145 never reuse consumed H143 or reference H144 source")
    row=seed_scene(attempt)
    row["layout"]=tuple(TRANSFORMS)[(attempt//4)%4]
    return row

def jittered_points(start,heading,attempt):
    """Predictor-like, bounded uncertainty, no invisible rectangle inputs.

    Perturb center at 20% source card width/height and 20% heading width,
    32% heading height. The same points feed BOTH controllers.
    """
    if not visible_box(start) or not visible_box(heading):
        raise ValueError("H145 requires native visible original target boxes")
    rng=random.Random(145000+attempt)
    sx=start["x"]+start["width"]*(.5+rng.choice((-1,1))*.20)
    sy=start["y"]+start["height"]*(.5+rng.choice((-1,1))*.20)
    gx=heading["x"]+heading["width"]*(.5+rng.choice((-1,1))*.20)
    gy=heading["y"]+heading["height"]*(.5+rng.choice((-1,1))*.32)
    for x,y,b in ((sx,sy,start),(gx,gy,heading)):
        if (not b["x"]<=x<=b["x"]+b["width"] or
            not b["y"]<=y<=b["y"]+b["height"] or
            not 0<=x<WIDTH or not 0<=y<HEIGHT):
            raise ValueError("H145 original noisy visible point out of bounding box")
    return [sx,sy],[gx,gy]

def attempt_arm(url,browser,seed,arm):
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
        dst=page.locator(".task_column").nth(seed["destination_column"])
        heading=dst.locator(".task_column_heading").first
        hidden=dst.locator(".hide_drop").nth(seed["destination_index"])
        src.wait_for(timeout=16000)
        heading.wait_for(timeout=16000)
        hidden.wait_for(timeout=16000)
        page.add_style_tag(content=style(seed["layout"]))
        source_box,heading_box=src.bounding_box(),heading.bounding_box()
        if not visible_box(source_box) or not visible_box(heading_box):
            return {"qualified":False,"reason":"source_or_heading_offscreen"}
        source_point,goal_point=jittered_points(
            source_box,heading_box,seed["attempt"])
        # The actuator receives ONLY noisy predictor-like points. The
        # owner DOM rectangles are never inputs to the mouse controller.
        fake_heading={
            "x":goal_point[0]-60,
            "y":goal_point[1]-15.5,
            "width":120,"height":31,
        }
        try:
            if arm=="direct":
                move_card_by_coarse_visible_goal(page,source_point,fake_heading)
            elif arm=="staged_sweep":
                staged_visible_sweep(page,source_point,fake_heading)
            else:raise ValueError("H145 unexpected unregistered arm")
        except Exception:
            return {"qualified":False,"reason":"mouse_control_not_actionable"}
        page.wait_for_timeout(100)
        now=json.loads(page.evaluate("localStorage.getItem('tasks')"))
        changed=sum(x["task"]==seed["title"] and
                    x["status"]==seed["destination_status"] for x in now)==1
        page.reload(wait_until="commit",timeout=16000)
        later=json.loads(page.evaluate("localStorage.getItem('tasks')"))
        persisted=sum(x["task"]==seed["title"] and
                      x["status"]==seed["destination_status"] for x in later)==1
        return {"qualified":True,"real_task_changed":changed,
                "persisted":bool(changed and persisted),
                "reference_heading_oracle_to_controller":False}
    finally:ctx.close()

def compare(owner_root,output,max_attempts=260):
    from playwright.sync_api import sync_playwright
    owner_root,output=Path(owner_root),Path(output)
    if (output.exists() or
        not any(x.name=="private-h145" for x in output.parents) or
        max_attempts<N):
        raise ValueError("H145 owner reference action QA create-only")
    original=verify_owner_original(
        owner_root/"private-h140"/"upstream"/"react-native-board")
    chosen=[];refusals=defaultdict(int)
    with sync_playwright() as pw:
        options={"headless":True}
        binary=_chromium_binary()
        if binary:options["executable_path"]=binary
        browser=pw.chromium.launch(**options,timeout=20000)
        try:
            with serve(original) as url:
                for attempt in range(OFFSET,OFFSET+max_attempts):
                    if len(chosen)>=N:break
                    case=source(attempt)
                    case["attempt"]=attempt
                    key=(case["language"],case["destination_status"])
                    if sum((x["language"],x["destination_status"])==key
                           for x in chosen)>=4:
                        refusals["stratum_full"]+=1
                        continue
                    paired={arm:attempt_arm(url,browser,case,arm)
                            for arm in ARMS}
                    if not all(x["qualified"] for x in paired.values()):
                        refusals["pair_not_qualified"]+=1
                        continue
                    chosen.append({
                        "attempt":attempt,
                        "language":case["language"],
                        "destination_status":case["destination_status"],
                        "layout":case["layout"],
                        "same_noisy_points_both_policies":True,
                        "direct_persisted":paired["direct"]["persisted"],
                        "sweep_persisted":paired["staged_sweep"]["persisted"],
                    })
                    if len(chosen)%4==0:
                        print("H145_ORIGINAL_NOISY_POINT_NATIVE_AB",
                              len(chosen),N,flush=True)
        finally:browser.close()
    if len(chosen)!=N:
        raise ValueError("H145 incomplete genuine original noisy-point A/B cohort")
    strata=Counter((r["language"],r["destination_status"]) for r in chosen)
    layouts=Counter(x["layout"] for x in chosen)
    if strata!={(l,d):4 for l in ("ar","en") for d in ("doing","done")}:
        raise ValueError("H145 bilingual/task goal source role not balanced")
    if len(layouts)!=4:
        raise ValueError("H145 original shifted layouts not represented")
    report={
        "schema":SCHEMA,"status":"ORIGINAL_OWNER_NOISY_VISIBLE_ACTION_AB_COMPLETE",
        "original_new_cases":N,"actual_paired_native_drags":N*2,
        "reference_points_disturbed_within_real_visible_bbox":True,
        "direct_persisted":sum(x["direct_persisted"] for x in chosen),
        "sweep_persisted":sum(x["sweep_persisted"] for x in chosen),
        "both_success":sum(x["direct_persisted"] and x["sweep_persisted"]
                           for x in chosen),
        "sweep_only_success":sum(not x["direct_persisted"] and x["sweep_persisted"]
                                 for x in chosen),
        "direct_only_success":sum(x["direct_persisted"] and not x["sweep_persisted"]
                                  for x in chosen),
        "strata":{"/".join(k):v for k,v in strata.items()},
        "layouts":dict(layouts),
        "refusals":dict(refusals),
        "model_calls":0,"optimizer_steps":0,
        "h140_blind_calls":0,"flagship_Hala_35B_updated":False,
        "model_promoted":False,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf8") as f:
        f.write(json.dumps({"summary":report,"private_episodes":chosen},
                           sort_keys=True,indent=2)+"\n")
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner-root",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--max-attempts",type=int,default=260)
    args=p.parse_args()
    print(json.dumps(compare(args.owner_root,args.output,args.max_attempts),
                     sort_keys=True,indent=2))
