"""Owner-private source QA of an UNMODIFIED MIT React Kanban application.

Records only authentic native screenshots and real persisted drag references.
NOT training, evaluation, or a claim of H140 neural model improvement.
"""
import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from h130.browser_env import _chromium_binary
from h138.probe import serve
from h140.qualification import WIDTH,HEIGHT,digest,position_bin
from h140.vanilla_pilot import visible_box,proposal

SOURCE_SHA="fa047f7a32afb57750ed5246456dc282ea7befcd"
SOURCE_REPO="dandemslay/React-KanbanBoard"
SCHEMA="hala.h140.owner-private-unmodified-react-reference-pilot.v1"

def verify_app(source):
    source=Path(source)
    head=subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True,timeout=10).strip()
    diff=subprocess.check_output(["git","-C",str(source),"diff","--name-only","HEAD"],text=True,timeout=10).strip()
    license=source/"LICENSE"
    built=source/"dist"/"index.html"
    if (head!=SOURCE_SHA or diff or not license.is_file() or
        not license.read_text("utf8").startswith("MIT License") or
        not built.is_file()):
        raise ValueError("H140 owner React original MIT app source/build not qualified")
    return built.parent

def midpoint(rect):
    return [rect["x"]+rect["width"]/2,rect["y"]+rect["height"]/2]

def run(source,output,count=16,max_attempts=100,attempt_offset=0):
    from playwright.sync_api import sync_playwright
    source=Path(source)
    built=verify_app(source)
    output=Path(output)
    if count<16 or count>128 or max_attempts<count or attempt_offset<0 or attempt_offset%2:
        raise ValueError("H140 React source pilot size not supported")
    if output.exists() or not any(p.name=="private-h140" for p in output.parents):
        raise ValueError("H140 only create new owner-private source pilot")
    temp=output.with_name(output.name+".incomplete")
    if temp.exists():
        raise ValueError("H140 existing partial diagnostic requires owner review")
    temp.mkdir(parents=True,exist_ok=False)
    records=[];rejected=defaultdict(int);seen=set()
    with sync_playwright() as pw:
        opts={"headless":True}
        exe=_chromium_binary()
        if exe:opts["executable_path"]=exe
        browser=pw.chromium.launch(**opts,timeout=15000)
        try:
            with serve(built) as url:
                for attempt in range(attempt_offset,attempt_offset+max_attempts):
                    if len(records)>=count:break
                    n,src_idx,dst_idx,offset,language=proposal(attempt)
                    # Never let rejected/offscreen attempts skew per-app bilingual balance.
                    if sum(r["language"]==language for r in records)>=count//2:
                        rejected["balanced_language_cap"]+=1
                        continue
                    chosen=f"H140 owner React draggable {attempt:04d}"
                    seed=[
                        {"task": chosen if i==src_idx else f"Todo {attempt:04d}-{i:02d}",
                         "status":"todo","tags":["Original"]} for i in range(n)]
                    seed.extend(
                        {"task":f"Doing {attempt:04d}-{i:02d}",
                         "status":"doing","tags":["Original"]} for i in range(n))
                    seed.append({"task":f"Done {attempt:04d}","status":"done","tags":["Original"]})
                    ctx=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                                            device_scale_factor=1,service_workers="block")
                    ctx.add_init_script("if(localStorage.getItem('tasks')===null){"+
                                        "localStorage.setItem('tasks',"+json.dumps(json.dumps(seed))+");}")
                    ctx.route("https://**/*",lambda route:route.abort())
                    try:
                        page=ctx.new_page()
                        page.goto(url,wait_until="domcontentloaded",timeout=12000)
                        src=page.locator(".task_column").nth(0).locator(".task_card").nth(src_idx)
                        dst=page.locator(".task_column").nth(1).locator(".hide_drop").nth(dst_idx)
                        src.wait_for(timeout=12000)
                        sb=src.bounding_box();db=dst.bounding_box()
                        if not sb or not db:
                            rejected["not_visible_in_DOM"]+=1;continue
                        last=max(midpoint(sb)[1],midpoint(db)[1])
                        desired=max(0,last-605)+offset
                        page.evaluate("(y)=>window.scrollTo(0,y)",desired)
                        page.wait_for_timeout(120)
                        start_box,end_box=src.bounding_box(),dst.bounding_box()
                        if not visible_box(start_box) or not visible_box(end_box):
                            rejected["offscreen_geometry"]+=1;continue
                        observed_scroll=page.evaluate("window.scrollY")
                        photo=page.screenshot(type="png")
                        h=digest(photo)
                        if h in seen:
                            rejected["duplicate_image"]+=1;continue
                        src.drag_to(dst,force=True,timeout=15000)
                        page.wait_for_timeout(100)
                        state=json.loads(page.evaluate("localStorage.getItem('tasks')"))
                        if (sum(row.get("task")==chosen and row.get("status")=="doing"
                                for row in state)!=1):
                            rejected["real_reference_drag_failure"]+=1;continue
                        page.reload(wait_until="domcontentloaded",timeout=12000)
                        persisted=json.loads(page.evaluate("localStorage.getItem('tasks')"))
                        if (sum(row.get("task")==chosen and row.get("status")=="doing"
                                for row in persisted)!=1):
                            rejected["reloaded_state_failure"]+=1;continue
                        file=f"owner-react-original-pilot-{len(records):03d}.png"
                        (temp/file).write_bytes(photo)
                        seen.add(h)
                        records.append({
                            "id":f"original-react-pilot-{len(records):03d}",
                            "app":"react-native-board","app_git_sha":SOURCE_SHA,
                            "split":"pilot_only_not_training",
                            "image":file,"image_sha256":h,
                            "language":language,
                            "instruction":("اسحب البطاقة "+chosen+" من المهام إلى قيد التنفيذ"
                                            if language=="ar" else
                                            "Move task card "+chosen+" from To Do to Doing"),
                            "start_bbox":start_box,"end_bbox":end_box,
                            "reference_drag_persisted":True,
                            "reference_survived_page_reload":True,
                            "model_input_contains_oracle_coordinates":False,
                            "attempt_index":attempt,"original_task_count":n,
                            "actual_scroll":observed_scroll,
                        })
                        if len(records)%4==0:
                            print("H140_ORIGINAL_REACT_PILOT_SUCCESS",len(records),
                                  "start_bins",len({position_bin(x["start_bbox"]) for x in records}),
                                  "end_bins",len({position_bin(x["end_bbox"]) for x in records}),
                                  flush=True)
                    finally:
                        ctx.close()
        finally:
            browser.close()
    if len(records)!=count:
        raise ValueError("H140 insufficient honest real React browser persisted actions")
    start_bins=len({position_bin(x["start_bbox"]) for x in records})
    end_bins=len({position_bin(x["end_bbox"]) for x in records})
    result={
        "schema":SCHEMA,"private":True,"training_eligible":False,
        "blind_eligible":False,
        "status":("POSITION_DIVERSE_DIAGNOSTIC_PILOT_ONLY"
                  if start_bins>=8 and end_bins>=8 else
                  "DIAGNOSTIC_GEOMETRY_COLLAPSE_NOT_QUALIFIED"),
        "source_repo":SOURCE_REPO,"source_git_sha":SOURCE_SHA,
        "source_license":"MIT","source_tracked_files_modified":False,
        "cases":records,"count":len(records),
        "start_bins":start_bins,"end_bins":end_bins,
        "rejections":dict(rejected),
        "model_neural_calls":0,"flagship_35B_updated":False,
    }
    (temp/"manifest.json").write_text(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+"\n",encoding="utf8")
    temp.rename(output)
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--count",type=int,default=16)
    parser.add_argument("--max-attempts",type=int,default=100)
    parser.add_argument("--attempt-offset",type=int,default=0)
    a=parser.parse_args()
    x=run(a.source,a.output,a.count,a.max_attempts,a.attempt_offset)
    print("H140_REACT_PILOT_FINAL",json.dumps({
        "status":x["status"],"count":x["count"],"start_bins":x["start_bins"],
        "end_bins":x["end_bins"],"rejections":x["rejections"]},sort_keys=True))
