"""H141 owner-only ORIGINAL React interactive drag-reveal source QA.

This module collects original staged screenshots BEFORE and DURING a real
HTML5 drag. It never trains or tests a neural model, and the highlighted
DURING screenshot may contain an ORACLE-GUIDED cursor/target: diagnostic
ONLY; it must NEVER be promoted to independent model input.
"""
import argparse
import io
import json
import subprocess
from collections import Counter,defaultdict
from pathlib import Path
from PIL import Image

from h130.browser_env import WIDTH,HEIGHT,_chromium_binary
from h138.probe import serve
from h140.qualification import digest
from h140.vanilla_pilot import visible_box

SCHEMA="hala.h141.original-react-drag-reveal-private-source-qa.v1"
APP_SHA="fa047f7a32afb57750ed5246456dc282ea7befcd"
APP="react-native-board"

def language(attempt):
    return "ar" if attempt%2 else "en"

def original_case(attempt):
    if attempt<40000:
        raise ValueError("H141 interactive source QA only new post-H140 cases")
    n=3+attempt%4
    src=(attempt*7+attempt//3)%n
    dest=(attempt*3+attempt//7)%n
    title=f"H141 original draggable source {attempt}"
    rows=[{"task":title if j==src else f"Todo fresh {attempt}-{j}",
           "status":"todo","tags":["SourceQA"]} for j in range(n)]
    rows += [{"task":f"Doing fresh {attempt}-{j}",
              "status":"doing","tags":["SourceQA"]} for j in range(n)]
    rows.append({"task":f"Done fresh {attempt}","status":"done","tags":["SourceQA"]})
    task=("اسحب البطاقة "+title+" إلى عمود قيد التنفيذ"
          if language(attempt)=="ar" else
          "Drag the task "+title+" to the Doing column")
    return {"title":title,"rows":rows,"source_index":src,
            "destination_index":dest,"language":language(attempt),
            "instruction":task}

def verify_owner_original(source):
    source=Path(source)
    sha=subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],
                                text=True,timeout=35).strip()
    tracked=subprocess.check_output(["git","-C",str(source),"diff","--name-only","HEAD"],
                                    text=True,timeout=35).strip()
    license=source/"LICENSE"
    dist=source/"dist"/"index.html"
    css=source/"src"/"components"/"DropArea.css"
    drop=source/"src"/"components"/"DropArea.jsx"
    if (sha!=APP_SHA or tracked or not license.is_file() or
        not license.read_text("utf8").startswith("MIT License") or
        not dist.is_file() or not css.is_file() or not drop.is_file()):
        raise ValueError("H141 original MIT source provenance/built app changed")
    if (".hide_drop" not in css.read_text("utf8")
        or "opacity: 0;" not in css.read_text("utf8")
        or 'showDrop ? "drop_area" : "hide_drop"' not in drop.read_text("utf8")):
        raise ValueError("H141 original source no longer hides target before drag")
    return source/"dist"

def visible_state(target):
    return target.evaluate(
        "(node)=>({klass:node.className,"
        "opacity:Number(getComputedStyle(node).opacity)})")

def point(rect):
    return [rect["x"]+rect["width"]/2,rect["y"]+rect["height"]/2]

def reference_gesture(page,case):
    src=page.locator(".task_column").nth(0).locator(".task_card").nth(
        case["source_index"])
    target=page.locator(".task_column").nth(1).locator(".hide_drop").nth(
        case["destination_index"])
    src.wait_for(timeout=16000)
    target.wait_for(timeout=16000)
    before=visible_state(target)
    if before["klass"]!="hide_drop" or before["opacity"]!=0:
        raise ValueError("H141 reference source target MUST be invisible initially")
    sb,db=src.bounding_box(),target.bounding_box()
    if sb is None or db is None:
        raise ValueError("H141 source reference lacked task/drop rectangles")
    last=max(point(sb)[1],point(db)[1])
    page.evaluate("(y)=>window.scrollTo(0,y)",max(0,last-570))
    page.wait_for_timeout(130)
    sb,db=src.bounding_box(),target.bounding_box()
    if not visible_box(sb) or not visible_box(db):
        return None
    frozen_rest=page.screenshot(type="png")
    x,y=point(sb)
    dx,dy=point(db)
    page.mouse.move(x,y)
    page.mouse.down()
    page.mouse.move(x+8,y+15,steps=8)
    page.mouse.move(dx,dy,steps=20)
    page.wait_for_timeout(230)
    active=page.locator(".task_column").nth(1).locator(".drop_area")
    if active.count()==0:
        page.mouse.up()
        return None
    actual=visible_state(active.first)
    if actual["klass"]!="drop_area" or actual["opacity"]<0.7:
        page.mouse.up()
        return None
    visible=active.first.bounding_box()
    if not visible_box(visible):
        page.mouse.up()
        return None
    frozen_active=page.screenshot(type="png")
    page.mouse.up()
    page.wait_for_timeout(150)
    rows=json.loads(page.evaluate("localStorage.getItem('tasks')"))
    moved=sum(x["task"]==case["title"] and x["status"]=="doing" for x in rows)==1
    page.reload(wait_until="commit",timeout=16000)
    latest=json.loads(page.evaluate("localStorage.getItem('tasks')"))
    persisted=sum(x["task"]==case["title"] and x["status"]=="doing" for x in latest)==1
    if not moved or not persisted:
        return None
    for png in (frozen_rest,frozen_active):
        with Image.open(io.BytesIO(png)) as im:
            if im.format!="PNG" or im.size!=(WIDTH,HEIGHT):
                raise ValueError("H141 expected genuine 960x720 screenshot")
            im.load()
    if frozen_rest==frozen_active:
        raise ValueError("H141 no visual transition between invisible and active target")
    return {"at_rest":frozen_rest,"during_drag":frozen_active,
            "hidden_css_opacity":before["opacity"],
            "active_css_opacity":actual["opacity"],
            "active_drop_area_visible":True,
            "reference_drag_persisted":True,
            "reference_survived_page_reload":True,
            "at_rest_source_bbox":sb,
            "at_rest_hidden_target_bbox":db,
            "drag_active_target_bbox":visible}

def collect(original_source,private_folder,count=8,attempt_offset=41000,max_attempts=80):
    from playwright.sync_api import sync_playwright
    private_folder=Path(private_folder)
    if (count<4 or count>16 or count%2 or
        max_attempts<count or attempt_offset<40000 or attempt_offset%2):
        raise ValueError("H141 preregistered even-count diagnostic new cases required")
    if (private_folder.exists() or
        private_folder.with_name(private_folder.name+".incomplete").exists() or
        not any(p.name=="private-h141" for p in private_folder.parents)):
        raise ValueError("H141 owner-private diagnostic create-only")
    built=verify_owner_original(original_source)
    temp=private_folder.with_name(private_folder.name+".incomplete")
    temp.mkdir(parents=True)
    successes=[]
    failures=defaultdict(int)
    hashes=set()
    with sync_playwright() as pw:
        config={"headless":True}
        binary=_chromium_binary()
        if binary:config["executable_path"]=binary
        browser=pw.chromium.launch(**config,timeout=20000)
        try:
            with serve(built) as url:
                for attempt in range(attempt_offset,attempt_offset+max_attempts):
                    if len(successes)>=count:break
                    seed=original_case(attempt)
                    if sum(x["language"]==seed["language"] for x in successes)>=count//2:
                        failures["bilingual_cap"]+=1
                        continue
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
                        try: result=reference_gesture(page,seed)
                        except Exception:
                            failures["browser_reference_exception"]+=1
                            continue
                        if result is None:
                            failures["genuine_reference_or_visibility_failed"]+=1
                            continue
                        shas=[digest(result["at_rest"]),digest(result["during_drag"])]
                        if len(set(shas))!=2 or any(s in hashes for s in shas):
                            failures["duplicate_original_frame"]+=1
                            continue
                        idx=len(successes)
                        first=f"original-rest-{idx:03d}.png"
                        second=f"original-active-{idx:03d}.png"
                        (temp/first).write_bytes(result.pop("at_rest"))
                        (temp/second).write_bytes(result.pop("during_drag"))
                        hashes.update(shas)
                        successes.append({
                            "id":f"original-react-interactive-{idx:03d}",
                            "attempt_index":attempt,
                            "app":APP,
                            "app_git_sha":APP_SHA,
                            "language":seed["language"],
                            "instruction":seed["instruction"],
                            "at_rest_image":first,"at_rest_sha256":shas[0],
                            "during_drag_image":second,"during_drag_sha256":shas[1],
                            "source_role":"owner_oracle_guided_source_diagnostic_only",
                            "training_eligible":False,"blind_eligible":False,
                            "model_evaluation_eligible":False,
                            "gold_cursor_target_highlight_present":True,
                            "model_neural_calls":0,
                            **result,
                        })
                        if len(successes)%2==0:
                            print("H141_ORIGINAL_HIDDEN_TO_VISIBLE_REFERENCE_CAPTURE",
                                  len(successes),"of",count,flush=True)
                    finally:
                        ctx.close()
        finally:browser.close()
    if len(successes)!=count:
        raise ValueError("H141 source-only pilot incomplete; preserve incomplete evidence")
    if Counter(row["language"] for row in successes)!={"en":count//2,"ar":count//2}:
        raise ValueError("H141 source language imbalance")
    doc={
        "schema":SCHEMA,
        "status":"QUALIFIED_SOURCE_QA_NOT_MODEL_TRAINING",
        "source_app":APP,
        "source_git_sha":APP_SHA,
        "private":True,
        "training_eligible":False,
        "blind_eligible":False,
        "model_evaluation_eligible":False,
        "source_original_mit_unmodified":True,
        "oracle_guided_drag_reference":True,
        "warning":"DURING-DRAG highlight reveals cursor/oracle. Do not train or score model on it.",
        "cases":successes,
        "model_neural_calls":0,
        "flagship_Hala_35B_updated":False,
    }
    (temp/"manifest.json").write_text(
        json.dumps(doc,indent=2,sort_keys=True,ensure_ascii=False)+"\n",
        encoding="utf8")
    temp.rename(private_folder)
    return {"status":doc["status"],"cases":count,
            "languages":dict(Counter(row["language"] for row in successes)),
            "unique_new_original_screens":len(hashes),
            "hidden_before_drag":count,
            "visible_during_drag":count,
            "actual_persisted_mouse_drags":count,
            "training_eligible":False,
            "blind_eligible":False,
            "model_neural_calls":0}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--cases",type=int,default=8)
    parser.add_argument("--attempt-offset",type=int,default=41000)
    parser.add_argument("--max-attempts",type=int,default=80)
    a=parser.parse_args()
    print(json.dumps(collect(a.source,a.output,
                             count=a.cases,attempt_offset=a.attempt_offset,
                             max_attempts=a.max_attempts),
                     indent=2,sort_keys=True))
