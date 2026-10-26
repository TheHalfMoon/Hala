"""Owner-Windows private pilot: authentic native HTML5 drag screenshot geometry.

Never use these pilot screens as model training data. Qualification of a
complete H140 corpus and historical hash exclusions is a separate gate.
"""
import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from h138.probe import serve,mouse_drag
from h130.browser_env import _chromium_binary
from h140.qualification import WIDTH,HEIGHT,digest,center,position_bin

SOURCE_SHA="47cfba359a71b70a6a23e511e554416aec006474"
SOURCE_REPO="anantagarwal1307/Drag-Drop-Kanban-Board"
SCHEMA="hala.h140.owner-private-original-mouse-drag-pilot.v1"

def check_source(source):
    source=Path(source)
    sha=subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],
                                text=True,timeout=10).strip()
    dirty=subprocess.check_output(["git","-C",str(source),"diff","--name-only","HEAD"],
                                  text=True,timeout=10).strip()
    license_path=source/"LICENSE"
    if (sha!=SOURCE_SHA or dirty or not license_path.is_file() or
        not license_path.read_text(encoding="utf8").startswith("MIT License") or
        not (source/"index.html").is_file() or not (source/"script.js").is_file()):
        raise ValueError("H140 source SHA, original MIT license or clean tracked code mismatch")
    return sha

def proposal(attempt):
    """Schedule legitimate task-list size, source/destination card and viewport scroll."""
    n=5+(attempt%4)
    source_index=(attempt*7+attempt//4)%n
    delta=(attempt//3)%3-1
    target_index=max(0,min(n-1,source_index+delta))
    scroll_offset=(attempt*47)%190
    lang="ar" if attempt%2 else "en"
    return n,source_index,target_index,scroll_offset,lang

def visible_box(rect):
    if not isinstance(rect,dict):
        return False
    try:
        x,y=center(rect)
    except ValueError:
        return False
    return (15<x<WIDTH-15 and 15<y<HEIGHT-15 and
            0<=rect["x"] and 0<=rect["y"] and
            rect["x"]+rect["width"]<=WIDTH and
            rect["y"]+rect["height"]<=HEIGHT)

def owner_pilot(source,output,count=32,max_attempts=160):
    from playwright.sync_api import sync_playwright
    source,output=Path(source),Path(output)
    check_source(source)
    if count<16 or count>64 or max_attempts<count:
        raise ValueError("H140 private pilot count/attempts out of bounds")
    if output.exists() or not any(parent.name=="private-h140" for parent in output.parents):
        raise ValueError("H140 capture create-only under owner private-h140")
    partial=output.with_name(output.name+".incomplete")
    if partial.exists():
        raise ValueError("H140 interrupted pilot needs explicit owner review")
    partial.mkdir(parents=True,exist_ok=False)
    records=[]; rejected=defaultdict(int); seen_sha=set()
    browser=None
    try:
        with sync_playwright() as pw:
            opts={"headless":True}
            binary=_chromium_binary()
            if binary:
                opts["executable_path"]=binary
            browser=pw.chromium.launch(**opts,timeout=15000)
            try:
                with serve(source) as url:
                    for attempt in range(max_attempts):
                        if len(records)>=count:
                            break
                        n,source_index,target_index,scroll_offset,lang=proposal(attempt)
                        title=f"H140 unique task {attempt:04d}"
                        seed={"todo":[{"title":title if j==source_index else
                                       f"Other todo {attempt:04d}-{j:02d}",
                                       "desc":"Native original owner reference"} for j in range(n)],
                              "progress":[{"title":f"Existing progress {attempt:04d}-{j:02d}",
                                            "desc":"Native original owner reference"} for j in range(n)],
                              "done":[]}
                        context=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                                                    device_scale_factor=1,service_workers="block")
                        # Seed only on FIRST page load. On reload, the native
                        # application must restore its own persisted drag state.
                        context.add_init_script("if(localStorage.getItem('tasks')===null){"+
                                                "localStorage.setItem('tasks',"+
                                                json.dumps(json.dumps(seed))+");}")
                        context.route("https://**/*",lambda route:route.abort())
                        try:
                            page=context.new_page()
                            page.goto(url,wait_until="domcontentloaded",timeout=12000)
                            first=page.locator("#todo .task").first
                            first.wait_for(timeout=10000)
                            src=page.locator("#todo .task").nth(source_index)
                            dst=page.locator("#progress .task").nth(target_index)
                            src_initial=src.bounding_box()
                            dst_initial=dst.bounding_box()
                            if not src_initial or not dst_initial:
                                rejected["missing_card"]+=1
                                continue
                            # Initial task rectangles may be BELOW the viewport.
                            # Scroll using their raw owner-only DOM geometry first;
                            # visible_box validates native points after scrolling.
                            last=max(b["y"]+b["height"]/2 for b in (src_initial,dst_initial))
                            desired=max(0,last-600)+scroll_offset
                            page.evaluate("(value)=>window.scrollTo(0,value)",desired)
                            page.wait_for_timeout(130)
                            start_box=src.bounding_box()
                            end_box=dst.bounding_box()
                            if not visible_box(start_box) or not visible_box(end_box):
                                rejected["offscreen_geometry"]+=1
                                continue
                            start,end=center(start_box),center(end_box)
                            source_image=page.screenshot(type="png")
                            observed_scroll=page.evaluate("window.scrollY")
                            image_sha=digest(source_image)
                            if image_sha in seen_sha:
                                rejected["duplicate_frame"]+=1
                                continue
                            mouse_drag(page,start,end)
                            state=json.loads(page.evaluate("localStorage.getItem('tasks')"))
                            passed=(any(t["title"]==title for t in state.get("progress",[]))
                                    and all(t["title"]!=title for t in state.get("todo",[])))
                            if not passed:
                                rejected["failed_reference_drag"]+=1
                                continue
                            page.reload(wait_until="domcontentloaded",timeout=12000)
                            persisted=json.loads(page.evaluate("localStorage.getItem('tasks')"))
                            if not any(t["title"]==title for t in persisted.get("progress",[])):
                                rejected["failed_reload_persistence"]+=1
                                continue
                            filename=f"owner-pilot-{len(records):03d}.png"
                            (partial/filename).write_bytes(source_image)
                            seen_sha.add(image_sha)
                            records.append({
                                "id":f"original-pilot-{len(records):03d}",
                                "app":"vanilla-native-board",
                                "app_git_sha":SOURCE_SHA,
                                "split":"pilot_only_not_training",
                                "image":filename,
                                "image_sha256":image_sha,
                                "language":lang,
                                "instruction":("اسحب البطاقة "+title+" من قائمة المهام إلى عمود قيد التنفيذ"
                                               if lang=="ar" else
                                               "Drag task card "+title+" from To Do to In Progress"),
                                "start_bbox":start_box,
                                "end_bbox":end_box,
                                "reference_drag_persisted":True,
                                "reference_survived_page_reload":True,
                                "model_input_contains_oracle_coordinates":False,
                                "true_start_point":start,
                                "true_end_point":end,
                                "attempt_index":attempt,
                                "seed_card_count":n,
                                "requested_scroll":desired,
                                "actual_scroll":observed_scroll,
                            })
                            if len(records)%4==0:
                                print("H140_PILOT_ACTUAL_SUCCESS",len(records),
                                      "start_bins",len({position_bin(r["start_bbox"]) for r in records}),
                                      "end_bins",len({position_bin(r["end_bbox"]) for r in records}),
                                      flush=True)
                        finally:
                            context.close()
            finally:
                browser.close()
                browser=None
        start_bins={position_bin(r["start_bbox"]) for r in records}
        end_bins={position_bin(r["end_bbox"]) for r in records}
        if len(records)!=count:
            raise ValueError("H140 insufficient original browser reference successes")
        report={
            "schema":SCHEMA,
            "status":("POSITION_DIVERSE_DIAGNOSTIC_PILOT_ONLY"
                      if len(start_bins)>=8 and len(end_bins)>=8 else
                      "DIAGNOSTIC_GEOMETRY_COLLAPSE_NOT_QUALIFIED"),
            "private":True,
            "training_eligible":False,
            "blind_eligible":False,
            "app_sources":{"vanilla-native-board":{
                "repository":SOURCE_REPO,"upstream_sha":SOURCE_SHA,
                "license":"MIT","license_file_verified":True,
                "tracked_source_modified":False}},
            "source_count":len(records),
            "start_geometry_bins":len(start_bins),
            "end_geometry_bins":len(end_bins),
            "rejected_attempts":dict(rejected),
            "cases":records,
            "model_neural_calls":0,
            "Hala_35B_updated":False,
        }
        (partial/"manifest.json").write_text(
            json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False)+"\n",
            encoding="utf8")
        partial.rename(output)
        return report
    finally:
        if browser:
            browser.close()

if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--source",required=True)
    cli.add_argument("--output",required=True)
    cli.add_argument("--cases",type=int,default=32)
    cli.add_argument("--max-attempts",type=int,default=160)
    args=cli.parse_args()
    result=owner_pilot(args.source,args.output,args.cases,args.max_attempts)
    print("H140_OWNER_PILOT_FINAL",json.dumps({
        "status":result["status"],"count":result["source_count"],
        "start_bins":result["start_geometry_bins"],"end_bins":result["end_geometry_bins"],
        "rejections":result["rejected_attempts"]},sort_keys=True))
