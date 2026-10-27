"""H152 independent original GUI horizontal-error robustness A/B.

Fresh source IDs >=184000. The browser mouse receives ONLY visible
source-card reference point and deliberately off-center visible
destination-heading point. A new pixel-guided controller searches
laterally for a stable active drop region; baseline is frozen H150.
True hidden DOM drop rectangle is NEVER given to either controller.
This is source-only action QA, not Hala 35B or neural inference.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from h138.probe import serve
from h130.browser_env import WIDTH,HEIGHT,_chromium_binary
from h140.vanilla_pilot import visible_box
from h141.coarse_column_action import visible_center
from h141.interactive_source import verify_owner_original
from h145.noisy_point_action_ab import source as original_case
from h143.collect_shifted_source import style
from h147.visual_drag_entry_ab import changed_samples,MIN_VISIBLE_CHANGED_SAMPLES
from h150.visual_region_controller import visually_center_drop,locate_visible_change

SCHEMA="hala.h152.new-original-lateral-drag-search-ab.v1"
OFFSET=184000
N=16
X_OFFSETS=(0,-32,32,-64,64)
Y_SCAN=(-16,-6,6,18,26,36,50,64)

def seed_scene(i):
    if i<OFFSET:raise ValueError("H152 prior benchmark source cannot be reused")
    row=original_case(i)
    row["attempt_index"]=i
    return row

def offcenter_visible_goal(heading,attempt):
    """New preregistered challenge: ±38% horizontal, ±30% vertical."""
    if not visible_box(heading):
        raise ValueError("H152 visible header geometry invalid")
    rand=random.Random(152000+attempt)
    px=heading["x"]+heading["width"]*(.5+rand.choice((-1,1))*.38)
    py=heading["y"]+heading["height"]*(.5+rand.choice((-1,1))*.30)
    if not (0<=px<WIDTH and 0<=py<HEIGHT and
            heading["x"]<px<heading["x"]+heading["width"] and
            heading["y"]<py<heading["y"]+heading["height"]):
        raise ValueError("H152 target point must remain within visible header")
    return [px,py]

def lateral_visual_search(page,start,goal):
    """Only screenshot input and two visible point proposals."""
    sx,sy=start;gx,gy=goal
    if not all(0<=v<bound for v,bound in
               ((sx,WIDTH),(sy,HEIGHT),(gx,WIDTH),(gy,HEIGHT))):
        raise ValueError("H152 mouse input offscreen")
    base=page.screenshot(type="png")
    page.mouse.move(sx,sy)
    page.mouse.down()
    page.mouse.move(sx+6,sy+18,steps=8)
    found=None;visual_retained=False;attempts=0;dx_selected=None
    try:
        for dx in X_OFFSETS:
            x=gx+dx
            if not 0<=x<WIDTH:continue
            page.mouse.move(x,gy-40,steps=12)
            for dy in Y_SCAN:
                y=gy+dy
                if not 0<=y<HEIGHT:continue
                page.mouse.move(x,y,steps=6)
                page.wait_for_timeout(70)
                attempts+=1
                frame=page.screenshot(type="png")
                if (dy<18 or
                    changed_samples(base,frame,[x,gy])<MIN_VISIBLE_CHANGED_SAMPLES):
                    continue
                page.wait_for_timeout(210)
                proposal=locate_visible_change(base,page.screenshot(type="png"),[x,gy])
                if proposal is None:continue
                page.mouse.move(*proposal,steps=12)
                page.wait_for_timeout(100)
                check=page.screenshot(type="png")
                if changed_samples(base,check,[x,gy])<MIN_VISIBLE_CHANGED_SAMPLES:
                    continue
                found=proposal
                visual_retained=True
                dx_selected=dx
                break
            if visual_retained:break
    finally:
        page.mouse.up()
    return {"visible_release_found":found is not None,
            "visible_feedback_stable":visual_retained,
            "lateral_offset":dx_selected,
            "screenshots_scanned":attempts,
            "controller_read_hidden_dom":False}

def one_arm(url,browser,scene,arm):
    ctx=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                            device_scale_factor=1,service_workers="block")
    ctx.add_init_script(
        "if(localStorage.getItem('tasks')===null){localStorage.setItem('tasks',"+
        json.dumps(json.dumps(scene["rows"]))+");}")
    ctx.route("https://**/*",lambda route:route.abort())
    try:
        page=ctx.new_page()
        page.goto(url,wait_until="commit",timeout=18000)
        card=page.locator(".task_column").nth(0).locator(
            ".task_card").nth(scene["source_index"])
        dest=page.locator(".task_column").nth(scene["destination_column"])
        heading=dest.locator(".task_column_heading").first
        hidden=dest.locator(".hide_drop").nth(scene["destination_index"])
        card.wait_for(timeout=16000)
        heading.wait_for(timeout=16000)
        hidden.wait_for(timeout=16000)
        page.add_style_tag(content=style(scene["layout"]))
        sb,hb=card.bounding_box(),heading.bounding_box()
        if not visible_box(sb) or not visible_box(hb):
            return {"qualified":False}
        start=list(visible_center(sb))
        goal=offcenter_visible_goal(hb,scene["attempt_index"])
        feedback=(visually_center_drop(page,start,goal) if arm=="h150"
                  else lateral_visual_search(page,start,goal))
        page.wait_for_timeout(120)
        current=json.loads(page.evaluate("localStorage.getItem('tasks')"))
        changed=any(x["task"]==scene["title"] and
                    x["status"]==scene["destination_status"] for x in current)
        page.reload(wait_until="commit",timeout=18000)
        after=json.loads(page.evaluate("localStorage.getItem('tasks')"))
        saved=changed and any(
            x["task"]==scene["title"] and
            x["status"]==scene["destination_status"] for x in after)
        return {"qualified":True,"persisted":bool(saved),
                "visual_feedback":(
                    feedback.get("retained_visual_change_at_release")
                    if arm=="h150" else feedback["visible_feedback_stable"]),
                "lateral_dx":feedback.get("lateral_offset")}
    finally:ctx.close()

def run(owner_root,output,max_attempts=300):
    from playwright.sync_api import sync_playwright
    owner_root,output=Path(owner_root),Path(output)
    if (output.exists() or
        not any(p.name=="private-h152" for p in output.parents) or
        max_attempts<N):
        raise ValueError("H152 owner evaluation must be create-only")
    app=verify_owner_original(
        owner_root/"private-h140"/"upstream"/"react-native-board")
    outcomes=[]
    with sync_playwright() as pw:
        opts={"headless":True}
        browser_bin=_chromium_binary()
        if browser_bin:opts["executable_path"]=browser_bin
        browser=pw.chromium.launch(**opts,timeout=20000)
        try:
            with serve(app) as url:
                for i in range(OFFSET,OFFSET+max_attempts):
                    if len(outcomes)>=N:break
                    scene=seed_scene(i)
                    kind=(scene["language"],scene["destination_status"])
                    if sum((r["language"],r["destination_status"])==kind
                           for r in outcomes)>=4:
                        continue
                    baseline=one_arm(url,browser,scene,"h150")
                    lateral=one_arm(url,browser,scene,"lateral")
                    if not baseline["qualified"] or not lateral["qualified"]:
                        continue
                    outcomes.append({
                        "attempt":i,"language":scene["language"],
                        "destination_status":scene["destination_status"],
                        "layout":scene["layout"],
                        "same_new_offcenter_visible_heading_proposal":True,
                        "h150_persisted":baseline["persisted"],
                        "lateral_persisted":lateral["persisted"],
                        "lateral_stable_visual_feedback":lateral["visual_feedback"],
                        "lateral_dx":lateral["lateral_dx"],
                    })
                    if len(outcomes)%4==0:
                        print("H152_NEW_NATIVE_LATERAL_VISUAL_ACTION_AB",
                              len(outcomes),N,flush=True)
        finally:browser.close()
    strata=Counter((x["language"],x["destination_status"]) for x in outcomes)
    if (len(outcomes)!=N or
        strata!={(l,d):4 for l in ("ar","en") for d in ("doing","done")}
        or len({x["layout"] for x in outcomes})!=4):
        raise ValueError("H152 new original native source not sufficiently qualified")
    doc={
        "schema":SCHEMA,"status":"COMPLETED_FRESH_LATERAL_ORIGINAL_REFERENCE_AB",
        "new_original_tasks":N,"real_native_mouse_drags":2*N,
        "prior_h150_persisted":sum(x["h150_persisted"] for x in outcomes),
        "new_lateral_persisted":sum(x["lateral_persisted"] for x in outcomes),
        "new_only":sum(x["lateral_persisted"] and not x["h150_persisted"]
                       for x in outcomes),
        "old_only":sum(x["h150_persisted"] and not x["lateral_persisted"]
                       for x in outcomes),
        "stable_visual_feedback":sum(x["lateral_stable_visual_feedback"]
                                     for x in outcomes),
        "strata":{"/".join(k):v for k,v in strata.items()},
        "layouts":dict(Counter(x["layout"] for x in outcomes)),
        "source_reference_card_point":True,
        "visible_heading_lateral_error_within_bbox":True,
        "controller_dom_oracle_access":False,
        "model_neural_calls":0,"optimizer_steps":0,
        "sealed_h140_blind_calls":0,"Hala35B_unchanged":True,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf8") as f:
        f.write(json.dumps({"summary":doc,"private_cases":outcomes},
                           sort_keys=True,indent=2)+"\n")
    return doc

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner-root",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    print(json.dumps(run(a.owner_root,a.output),sort_keys=True,indent=2))
