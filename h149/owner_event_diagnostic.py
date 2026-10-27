"""Owner-only H149 original React drag-event diagnostics, NEW source IDs.

The screenshot-only controller cannot read the event log. Event probes
record browser signals AFTER release for research about 8/16 failures.
No model, no training or H140 sealed BLIND use.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from h130.browser_env import WIDTH,HEIGHT,_chromium_binary
from h138.probe import serve
from h140.vanilla_pilot import visible_box
from h141.coarse_column_action import visible_center
from h141.interactive_source import verify_owner_original
from h145.noisy_point_action_ab import jittered_points,source as original_case
from h147.visual_drag_entry_ab import visually_observed_drop
from h143.collect_shifted_source import style

SCHEMA="hala.h149.owner-private-original-event-diagnostic.v1"
OFFSET=134000
N=16
EVENT_HOOK=r"""
window.__halaEvents=[];
for(const name of ['dragstart','dragenter','dragleave','dragover','drop','dragend']){
  document.addEventListener(name,(e)=>{
    if(window.__halaEvents.length>=400)return;
    const node=e.target;
    const column=node?.closest?.('.task_column');
    const heading=column?.querySelector('.task_column_heading');
    window.__halaEvents.push({
      event:name,
      targetClass:String(node?.className||''),
      columnHeading:String(heading?.textContent||'').trim(),
      x:Math.round(e.clientX),y:Math.round(e.clientY)
    });
  },true);
}
"""

def case(i):
    if i<OFFSET:raise ValueError("H149 requires fresh original source ids")
    result=original_case(i)
    result["attempt_index"]=i
    return result

def inspect_one(url,browser,seed):
    ctx=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                            device_scale_factor=1,service_workers="block")
    ctx.add_init_script(
        "if(localStorage.getItem('tasks')===null){localStorage.setItem('tasks',"+
        json.dumps(json.dumps(seed["rows"]))+");}")
    ctx.route("https://**/*",lambda request:request.abort())
    try:
        page=ctx.new_page()
        page.goto(url,wait_until="commit",timeout=18000)
        card=page.locator(".task_column").nth(0).locator(
            ".task_card").nth(seed["source_index"])
        col=page.locator(".task_column").nth(seed["destination_column"])
        heading=col.locator(".task_column_heading").first
        # Only source harness reads native selectors. Controller does not.
        area=col.locator(".hide_drop").nth(seed["destination_index"])
        card.wait_for(timeout=16000)
        heading.wait_for(timeout=16000)
        area.wait_for(timeout=16000)
        page.add_style_tag(content=style(seed["layout"]))
        sb,hb=card.bounding_box(),heading.bounding_box()
        if not visible_box(sb) or not visible_box(hb):
            return None
        source_point=visible_center(sb)
        _,goal_point=jittered_points(sb,hb,seed["attempt_index"])
        page.evaluate(EVENT_HOOK)
        # Pure pixel-based/controller action gets NO event/DOM data.
        feedback=visually_observed_drop(page,source_point,goal_point)
        events=page.evaluate("window.__halaEvents")
        page.wait_for_timeout(120)
        state=json.loads(page.evaluate("localStorage.getItem('tasks')"))
        changed=any(r["task"]==seed["title"] and
                    r["status"]==seed["destination_status"] for r in state)
        page.reload(wait_until="commit",timeout=18000)
        persisted=json.loads(page.evaluate("localStorage.getItem('tasks')"))
        completed=changed and any(
            r["task"]==seed["title"] and
            r["status"]==seed["destination_status"] for r in persisted)
        target_label="Doing" if seed["destination_status"]=="doing" else "Done"
        belongs=lambda e: target_label.lower() in e["columnHeading"].lower()
        dest_enter=any(e["event"]=="dragenter" and belongs(e) for e in events)
        dest_drop=any(e["event"]=="drop" and belongs(e) for e in events)
        any_drop=any(e["event"]=="drop" for e in events)
        return {
            "persisted":bool(completed),
            "pixel_signal":feedback["image_feedback_activated"],
            "release_offset":feedback["release_offset"],
            "dragstart":any(e["event"]=="dragstart" for e in events),
            "target_dragenter":dest_enter,
            "target_drop":dest_drop,
            "any_drop":any_drop,
            "dragend":any(e["event"]=="dragend" for e in events),
            "owner_private_events":events,
        }
    finally:
        ctx.close()

def run(owner_root,output,max_attempts=250):
    from playwright.sync_api import sync_playwright
    owner_root,output=Path(owner_root),Path(output)
    if (output.exists() or
        not any(p.name=="private-h149" for p in output.parents) or
        max_attempts<N):
        raise ValueError("H149 is owner-private create-only")
    app=verify_owner_original(
        owner_root/"private-h140"/"upstream"/"react-native-board")
    examples=[]
    with sync_playwright() as pw:
        config={"headless":True}
        binary=_chromium_binary()
        if binary:config["executable_path"]=binary
        browser=pw.chromium.launch(**config,timeout=20000)
        try:
            with serve(app) as url:
                for i in range(OFFSET,OFFSET+max_attempts):
                    if len(examples)>=N:break
                    task=case(i)
                    key=(task["language"],task["destination_status"])
                    if sum((x["language"],x["destination_status"])==key
                           for x in examples)>=4:
                        continue
                    result=inspect_one(url,browser,task)
                    if result is None:continue
                    examples.append({
                        "attempt":i,
                        "language":task["language"],
                        "destination_status":task["destination_status"],
                        "layout":task["layout"],**result,
                    })
                    if len(examples)%4==0:
                        print("H149_NATIVE_EVENT_SOURCE_DIAGNOSTIC",len(examples),
                              N,flush=True)
        finally:browser.close()
    strata=Counter((r["language"],r["destination_status"]) for r in examples)
    if (len(examples)!=N or
        strata!={(l,d):4 for l in ("en","ar") for d in ("doing","done")}
        or len({r["layout"] for r in examples})!=4):
        raise ValueError("H149 new original source not qualified")
    fields=("persisted","pixel_signal","dragstart",
            "target_dragenter","target_drop","any_drop","dragend")
    summary={
        "schema":SCHEMA,"status":"NEW_NATIVE_OWNER_ONLY_EVENT_DIAGNOSTIC",
        "new_cases":N,"counts":{field:sum(bool(r[field]) for r in examples)
                               for field in fields},
        "cross_target_drop_vs_task":{
            f"drop={d},persisted={p}":sum(
                r["target_drop"]==d and r["persisted"]==p for r in examples)
            for d in (False,True) for p in (False,True)},
        "layouts":dict(Counter(r["layout"] for r in examples)),
        "controller_had_access_to_event_logs":False,
        "controller_had_access_to_hidden_drop_box":False,
        "model_calls":0,"optimizer_steps":0,"h140_blind_calls":0,
        "Hala_35B_changed":False,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf8") as handle:
        handle.write(json.dumps({
            "summary":summary,"owner_private_case_events":examples},
            ensure_ascii=False,sort_keys=True,indent=2)+"\n")
    return summary

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner-root",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    print(json.dumps(run(a.owner_root,a.output),sort_keys=True,indent=2))
