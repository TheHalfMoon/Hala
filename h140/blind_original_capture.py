"""Owner-private ORIGINAL unseen MIT app source capture — never train on this.

Both apps remain completely separate from H140 TRAIN/DEV. Native Chromium
mouse pointer action, actual localStorage status and post-reload persistence
are required before a screenshot can be admitted to private source QA.
This file does not train, score or publish a model.
"""
import argparse
import io
import json
from collections import Counter,defaultdict
from pathlib import Path

from PIL import Image
from h138.probe import serve,mouse_drag
from h130.browser_env import WIDTH,HEIGHT,_chromium_binary
from h140.qualification import digest,position_bin
from h140.vanilla_pilot import visible_box
from h140.blind_source_preflight import SOURCES,check_original_source,only_local_requests

SCHEMA="hala.h140.private-ood-original-app-geometry-candidate.v1"

def param(attempt):
    n=4+attempt%6
    src=(attempt*7+attempt//6)%n
    end=(attempt*3+attempt//5)%n
    lang="ar" if attempt%2 else "en"
    scroll=(attempt*47)%160
    return n,src,end,lang,scroll

def board_for(app,attempt,n,src):
    uid=f"h140-private-blind-qa-task-{attempt:05d}"
    if app=="ood-vanilla-complete":
        tasks=[]
        for i in range(n):
            tasks.append({"id":uid if i==src else f"todo_{attempt}_{i}",
                          "text":f"Unseen private card {attempt} {i}",
                          "status":"todo","priority":"medium","tag":"Dev",
                          "dueDate":None,"subtasks":[]})
            tasks.append({"id":f"progress_{attempt}_{i}",
                          "text":f"Existing progress {attempt} {i}",
                          "status":"in-progress","priority":"medium",
                          "tag":"Study","dueDate":None,"subtasks":[]})
        return {"storage":"kanbanTasks","seed":tasks,"id":uid,
                "source":"todo","destination":"in-progress"}
    if app=="ood-second-drag":
        date="2026-10-26T10:00:00.000Z"
        tasks=[]
        for i in range(n):
            tasks.append({"id":uid if i==src else f"todo_{attempt}_{i}",
                          "title":f"Unseen private card {attempt} {i}",
                          "description":"Unseen independent original source action",
                          "status":"todo","priority":"medium","dueDate":None,
                          "createdAt":date,"tags":[],"swimlane":None})
            tasks.append({"id":f"progress_{attempt}_{i}",
                          "title":f"Existing progress {attempt} {i}",
                          "description":"Original private benchmark source",
                          "status":"inprogress","priority":"medium",
                          "dueDate":None,"createdAt":date,"tags":[],"swimlane":None})
        board={"id":f"original_{attempt}","name":"Independent native blind source",
               "description":"","color":"#667eea","createdAt":date,
               "columns":[
                  {"id":"todo","name":"To Do","emoji":"A","color":"#3498db","order":0},
                  {"id":"inprogress","name":"In Progress","emoji":"B","color":"#f39c12","order":1},
                  {"id":"done","name":"Done","emoji":"C","color":"#2ecc71","order":2}],
               "swimlanes":[],"tasks":tasks}
        return {"storage":"kanbanBoards","seed":[board],"id":uid,
                "source":"todo","destination":"inprogress",
                "board_id":board["id"]}
    raise ValueError("H140 private OOD app is not registered")

def native_card(page,app,role,index,uid):
    if app=="ood-vanilla-complete":
        if role=="start":
            return page.locator(f"#{'todo'} .task-card").nth(index)
        return page.locator("#in-progress .task-card").nth(index)
    if app=="ood-second-drag":
        if role=="start":
            return page.locator(f'.task-card[data-task-id="{uid}"]').first
        return page.locator('.tasks-container[data-status="inprogress"] .task-card').nth(index)
    raise ValueError("Unknown immutable OOD source")

def moved(app,original,state):
    if app=="ood-vanilla-complete":
        return any(t.get("id")==original["id"] and
                   t.get("status")==original["destination"] for t in state)
    if app=="ood-second-drag":
        return any(t.get("id")==original["id"] and
                   t.get("status")==original["destination"]
                   for board in state for t in board.get("tasks",[]))
    return False

def center(box):
    return (box["x"]+box["width"]/2,box["y"]+box["height"]/2)

def collect(app,private_root,output,count=16,attempt_offset=30000,max_attempts=100):
    from playwright.sync_api import sync_playwright
    if app not in SOURCES or count!=16 or attempt_offset<0 or attempt_offset%2:
        raise ValueError("H140 exactly 16 balanced one-time original cases per OOD app")
    private_root=Path(private_root)
    output=Path(output)
    if (output.exists() or output.with_name(output.name+".incomplete").exists()
        or not any(p.name=="private-h140" for p in output.parents)):
        raise ValueError("H140 private original OOD write-once data only")
    policy=SOURCES[app]
    check_original_source(private_root/app,policy)
    work=output.with_name(output.name+".incomplete")
    work.mkdir(parents=True)
    records=[];seen=set();rejected=defaultdict(int)
    with sync_playwright() as pw:
        opts={"headless":True}
        exe=_chromium_binary()
        if exe:opts["executable_path"]=exe
        browser=pw.chromium.launch(**opts,timeout=20000)
        try:
            with serve(private_root/app) as url:
                for attempt in range(attempt_offset,attempt_offset+max_attempts):
                    if len(records)>=count:break
                    n,si,di,lang,scroll=param(attempt)
                    if sum(r["language"]==lang for r in records)>=count//2:
                        rejected["bilingual_cap"]+=1
                        continue
                    original=board_for(app,attempt,n,si)
                    ctx=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                                            device_scale_factor=1,service_workers="block")
                    key=original["storage"]
                    state=json.dumps(json.dumps(original["seed"]))
                    ctx.add_init_script(
                        "if(localStorage.getItem("+json.dumps(key)+")===null){"+
                        "localStorage.setItem("+json.dumps(key)+","+state+");"+
                        (("localStorage.setItem('currentBoardId',"+
                          json.dumps(original["board_id"])+");")
                         if app=="ood-second-drag" else "")+"}")
                    ctx.route("**/*",only_local_requests)
                    try:
                        page=ctx.new_page()
                        page.goto(url,wait_until="commit",timeout=18000)
                        src=native_card(page,app,"start",si,original["id"])
                        dest=native_card(page,app,"end",di,original["id"])
                        src.wait_for(timeout=18000);dest.wait_for(timeout=18000)
                        pre_start=src.bounding_box();pre_end=dest.bounding_box()
                        if pre_start is None or pre_end is None:
                            rejected["source_dom_missing"]+=1;continue
                        last=max(center(pre_start)[1],center(pre_end)[1])
                        page.evaluate("(y)=>window.scrollTo(0,y)",max(0,last-620)+scroll)
                        page.wait_for_timeout(120)
                        box1,box2=src.bounding_box(),dest.bounding_box()
                        if not visible_box(box1) or not visible_box(box2):
                            rejected["offscreen_original_geometry"]+=1;continue
                        snapshot=page.screenshot(type="png")
                        photo=Image.open(io.BytesIO(snapshot))
                        if photo.format!="PNG" or photo.size!=(WIDTH,HEIGHT):
                            rejected["invalid_source_pixels"]+=1;continue
                        sha=digest(snapshot)
                        if sha in seen:
                            rejected["duplicate_source_sha"]+=1;continue
                        mouse_drag(page,center(box1),center(box2))
                        page.wait_for_timeout(130)
                        state=json.loads(page.evaluate("(k)=>localStorage.getItem(k)",key))
                        if not moved(app,original,state):
                            rejected["actual_native_drag_failed"]+=1;continue
                        page.reload(wait_until="commit",timeout=18000)
                        after=json.loads(page.evaluate("(k)=>localStorage.getItem(k)",key))
                        if not moved(app,original,after):
                            rejected["original_reload_persistence_failed"]+=1;continue
                        filename=f"owner-private-original-{len(records):03d}.png"
                        (work/filename).write_bytes(snapshot)
                        seen.add(sha)
                        title=f"Unseen private card {attempt} {si}"
                        records.append({
                            "id":f"original-private-{app}-{len(records):03d}",
                            "app":app,"app_git_sha":policy["sha"],
                            "image":filename,"image_sha256":sha,
                            "source_role":"blind_candidate_not_model_evaluated",
                            "language":lang,
                            "instruction":(("اسحب البطاقة "+title+" إلى عمود قيد التنفيذ")
                                           if lang=="ar" else
                                           ("Move card "+title+" to In Progress")),
                            "start_bbox":box1,"end_bbox":box2,
                            "reference_drag_persisted":True,
                            "reference_survived_page_reload":True,
                            "model_input_contains_oracle_coordinates":False,
                            "attempt_index":attempt,
                        })
                        if len(records)%4==0:
                            print("H140_PRIVATE_OOD_SOURCE_REFERENCE",app,len(records),
                                  "start_bins",len({position_bin(x["start_bbox"]) for x in records}),
                                  "end_bins",len({position_bin(x["end_bbox"]) for x in records}),
                                  flush=True)
                    finally:
                        ctx.close()
        finally:
            browser.close()
    if len(records)!=count:
        raise ValueError("H140 original OOD reference count not qualified")
    sb=len({position_bin(x["start_bbox"]) for x in records})
    eb=len({position_bin(x["end_bbox"]) for x in records})
    report={"schema":SCHEMA,"private":True,
            "status":("PRIVATE_ORIGINAL_BLIND_SOURCE_GEOMETRY_READY"
                      if sb>=8 and eb>=8 else "PRIVATE_ORIGINAL_GEOMETRY_INSUFFICIENT"),
            "training_eligible":False,"blind_model_evaluated":False,
            "source_role":"source_only_blind_candidate",
            "app":app,"app_git_sha":policy["sha"],
            "source_repo":policy["repository"],"original_mit_verified":True,
            "case_count":len(records),
            "language_count":dict(Counter(r["language"] for r in records)),
            "start_bins":sb,"end_bins":eb,"refusal_counts":dict(rejected),
            "model_neural_calls":0,"main_Hala_35B_updated":False,
            "cases":records}
    (work/"manifest.json").write_text(
        json.dumps(report,sort_keys=True,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    work.rename(output)
    return {key:report[key] for key in
            ("status","case_count","start_bins","end_bins","language_count","refusal_counts")}

if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--app",choices=tuple(SOURCES),required=True)
    cli.add_argument("--sources",required=True)
    cli.add_argument("--output",required=True)
    cli.add_argument("--attempt-offset",type=int,default=30000)
    cli.add_argument("--max-attempts",type=int,default=100)
    args=cli.parse_args()
    print("H140_PRIVATE_BLIND_SOURCE_ONLY",json.dumps(
        collect(args.app,args.sources,args.output,
                attempt_offset=args.attempt_offset,max_attempts=args.max_attempts),
        sort_keys=True))
