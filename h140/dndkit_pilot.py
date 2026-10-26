"""Private ORIGINAL @dnd-kit source capture on owner Windows (NO training).

Uses browser-native mouse movements, exact PNG SHA, real persistent task
transition and independent page-reload state checking. Never export media.
"""
import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from h130.browser_env import _chromium_binary
from h138.probe import serve,mouse_drag
from h140.qualification import WIDTH,HEIGHT,digest,position_bin
from h140.vanilla_pilot import visible_box

SOURCE_SHA="f3e129998e3140a386f1884faff314345e0eb10b"
SOURCE_REPO="TomCani/kanban-dndkit-localstorage"
SCHEMA="hala.h140.owner-original-dndkit-reference-pilot.v1"

def qualified_build(root):
    root=Path(root)
    pinned=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True,timeout=10).strip()
    diff=subprocess.check_output(["git","-C",str(root),"diff","--name-only","HEAD"],text=True,timeout=10).strip()
    license=root/"LICENSE"
    built=root/"dist"/"index.html"
    if (pinned!=SOURCE_SHA or diff or not license.is_file() or
        not license.read_text("utf8").startswith("MIT License") or
        not built.is_file()):
        raise ValueError("H140 original pinned MIT dndkit application not qualified")
    return built.parent

def own_board(attempt,n,target_index):
    cards={}
    todo=[]
    doing=[]
    chosen=f"Real H140 dndkit owner card {attempt:04d}"
    target_id=f"h140_target_{attempt:04d}"
    for i in range(n):
        cid=target_id if i==target_index else f"todo_{attempt:04d}_{i:02d}"
        todo.append(cid)
        cards[cid]={"id":cid,"title":chosen if i==target_index else f"Todo {attempt:04d}-{i:02d}"}
        did=f"doing_{attempt:04d}_{i:02d}"
        doing.append(did)
        cards[did]={"id":did,"title":f"Doing {attempt:04d}-{i:02d}"}
    cards["owner_done"]={"id":"owner_done","title":"Existing done task"}
    board={"columns":[{"id":"col_todo","title":"To Do","cardIds":todo},
                      {"id":"col_doing","title":"In Progress","cardIds":doing},
                      {"id":"col_done","title":"Done","cardIds":["owner_done"]}],
           "cards":cards}
    return board,target_id,chosen

def middle(rect):
    return [rect["x"]+rect["width"]/2,rect["y"]+rect["height"]/2]

def native_geometry_proposal(attempt):
    """Original dnd-kit card density, indices and genuine browser scroll."""
    n=6+(attempt%6)
    source_index=(attempt*5+attempt//6)%n
    delta=(attempt//4)%3-1
    destination_index=max(0,min(n-1,source_index+delta))
    scroll_offset=(attempt*61)%190
    lang="ar" if attempt%2 else "en"
    return n,source_index,destination_index,scroll_offset,lang

def run(source,output,count=16,max_attempts=100,attempt_offset=0):
    from playwright.sync_api import sync_playwright
    source=Path(source);output=Path(output)
    built=qualified_build(source)
    if count<16 or count>128 or max_attempts<count or attempt_offset<0 or attempt_offset%2:
        raise ValueError("H140 DnD original source pilot target count unsupported")
    if output.exists() or not any(p.name=="private-h140" for p in output.parents):
        raise ValueError("H140 DnD source output must be owner private and create-only")
    partial=output.with_name(output.name+".incomplete")
    if partial.exists():
        raise ValueError("H140 unfinished original source collection requires owner review")
    partial.mkdir(parents=True,exist_ok=False)
    accepted=[];rejected=defaultdict(int);seen=set()
    with sync_playwright() as pw:
        opts={"headless":True}
        path=_chromium_binary()
        if path:opts["executable_path"]=path
        browser=pw.chromium.launch(**opts,timeout=15000)
        try:
            with serve(built) as url:
                for attempt in range(attempt_offset,attempt_offset+max_attempts):
                    if len(accepted)>=count:break
                    n,sidx,eidx,offset,lang=native_geometry_proposal(attempt)
                    board,card_id,title=own_board(attempt,n,sidx)
                    ctx=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                                            device_scale_factor=1,service_workers="block")
                    ctx.add_init_script("if(localStorage.getItem('kanban:v1')===null){"+
                                        "localStorage.setItem('kanban:v1',"+json.dumps(json.dumps(board))+");}")
                    ctx.route("https://**/*",lambda route:route.abort())
                    try:
                        page=ctx.new_page()
                        page.goto(url,wait_until="domcontentloaded",timeout=12000)
                        src=page.locator("#col_todo .card-draggable").nth(sidx)
                        dst=page.locator("#col_doing .card-draggable").nth(eidx)
                        src.wait_for(timeout=12000)
                        original_start,original_end=src.bounding_box(),dst.bounding_box()
                        if not original_start or not original_end:
                            rejected["missing_original_card"]+=1;continue
                        max_y=max(middle(original_start)[1],middle(original_end)[1])
                        desired=max(0,max_y-610)+offset
                        page.evaluate("(y)=>window.scrollTo(0,y)",desired)
                        page.wait_for_timeout(120)
                        start_box,end_box=src.bounding_box(),dst.bounding_box()
                        if not visible_box(start_box) or not visible_box(end_box):
                            rejected["offscreen_original_geometry"]+=1;continue
                        scroll=page.evaluate("window.scrollY")
                        photo=page.screenshot(type="png")
                        sha=digest(photo)
                        if sha in seen:
                            rejected["duplicate_frame"]+=1;continue
                        mouse_drag(page,middle(start_box),middle(end_box))
                        page.wait_for_timeout(180)
                        moved=json.loads(page.evaluate("localStorage.getItem('kanban:v1')"))
                        dest=next(c for c in moved["columns"] if c["id"]=="col_doing")
                        src_col=next(c for c in moved["columns"] if c["id"]=="col_todo")
                        if card_id not in dest["cardIds"] or card_id in src_col["cardIds"]:
                            rejected["original_pointer_drop_not_persisted"]+=1;continue
                        page.reload(wait_until="domcontentloaded",timeout=12000)
                        after=json.loads(page.evaluate("localStorage.getItem('kanban:v1')"))
                        dest_again=next(c for c in after["columns"] if c["id"]=="col_doing")
                        if card_id not in dest_again["cardIds"]:
                            rejected["original_task_lost_after_reload"]+=1;continue
                        filename=f"original-dndkit-pilot-{len(accepted):03d}.png"
                        (partial/filename).write_bytes(photo)
                        seen.add(sha)
                        accepted.append({
                            "id":f"dndkit-pilot-{len(accepted):03d}",
                            "app":"react-dndkit-board","app_git_sha":SOURCE_SHA,
                            "split":"pilot_only_not_training",
                            "image":filename,"image_sha256":sha,
                            "language":lang,
                            "instruction":("اسحب البطاقة "+title+" من المهام إلى قيد التنفيذ"
                                           if lang=="ar" else
                                           "Move task card "+title+" from To Do to In Progress"),
                            "start_bbox":start_box,"end_bbox":end_box,
                            "reference_drag_persisted":True,
                            "reference_survived_page_reload":True,
                            "model_input_contains_oracle_coordinates":False,
                            "attempt_index":attempt,"original_task_count":n,
                            "actual_scroll":scroll,
                        })
                        if len(accepted)%4==0:
                            print("H140_DNDKIT_ORIGINAL_REFERENCE_SUCCESS",len(accepted),
                                  "start_bins",len({position_bin(x["start_bbox"]) for x in accepted}),
                                  "end_bins",len({position_bin(x["end_bbox"]) for x in accepted}),
                                  flush=True)
                    finally:
                        ctx.close()
        finally:browser.close()
    if len(accepted)!=count:
        raise ValueError("H140 genuine DnD source reference task threshold not achieved")
    s=len({position_bin(x["start_bbox"]) for x in accepted})
    d=len({position_bin(x["end_bbox"]) for x in accepted})
    report={
        "schema":SCHEMA,"private":True,"training_eligible":False,
        "blind_eligible":False,
        "status":("POSITION_DIVERSE_DIAGNOSTIC_PILOT_ONLY"
                  if s>=8 and d>=8 else "DIAGNOSTIC_GEOMETRY_COLLAPSE_NOT_QUALIFIED"),
        "app_repo":SOURCE_REPO,"source_git_sha":SOURCE_SHA,
        "license":"MIT","tracked_source_modified":False,
        "cases":accepted,"case_count":len(accepted),
        "start_bins":s,"end_bins":d,"rejections":dict(rejected),
        "real_neural_calls":0,"main_Hala_35B_updated":False,
    }
    (partial/"manifest.json").write_text(
        json.dumps(report,sort_keys=True,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    partial.rename(output)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--count",type=int,default=16)
    p.add_argument("--max-attempts",type=int,default=100)
    p.add_argument("--attempt-offset",type=int,default=0)
    x=p.parse_args()
    r=run(x.source,x.output,x.count,x.max_attempts,x.attempt_offset)
    print("H140_DNDKIT_OWNER_DIAGNOSTIC_FINAL",
          json.dumps({"count":r["case_count"],"status":r["status"],
                      "start_bins":r["start_bins"],"end_bins":r["end_bins"],
                      "rejections":r["rejections"]},sort_keys=True))
