"""Reproducible ORIGINAL one-shot OOD app source QA (not model evaluation).

This preflight NEVER saves Blind images, DOM gold boxes, model responses,
or trainable source samples. It checks that two codebase-independent MIT
apps support real native mouse task transitions and local persistence.
"""
import argparse
import json
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

from h130.browser_env import WIDTH,HEIGHT,_chromium_binary
from h138.probe import serve,center,mouse_drag

SOURCES={
    "ood-vanilla-complete":{
        "repository":"AishaChunawala/interactive-kanban-board-or-study-kanban-board",
        "sha":"f89d065e2cf16f91918bf1a72ae9ac132150197e",
        "files":("index.html","script.js","style.css"),
        "engine":"independent-html5-native-drag"},
    "ood-second-drag":{
        "repository":"michaelgermini/Kanban-Board",
        "sha":"a68c1d9d6281a793835ceaa5e4f620068d95200a",
        "files":("index.html","app.js","style.css"),
        "engine":"independent-vanilla-multiboard-drag"},
}
SCHEMA="hala.h140.two-independent-original-ood-source-preflight.v1"

def check_original_source(folder,policy):
    folder=Path(folder)
    sha=subprocess.check_output(
        ["git","-C",str(folder),"rev-parse","HEAD"],text=True,timeout=35).strip()
    dirty=subprocess.check_output(
        ["git","-C",str(folder),"diff","--name-only","HEAD"],text=True,timeout=35).strip()
    lic=folder/"LICENSE"
    if (sha!=policy["sha"] or dirty or not lic.is_file() or
        not lic.read_text("utf8").startswith("MIT License") or
        any(not (folder/f).is_file() for f in policy["files"])):
        raise ValueError("H140 original independent MIT source not frozen/complete/clean")
    return sha

def only_local_requests(route):
    host=urlsplit(route.request.url).hostname
    if host in ("localhost","127.0.0.1"):
        route.continue_()
    else:
        route.abort()

def original_seed_for(name):
    if name=="ood-vanilla-complete":
        tasks=[
            {"id":"h140_source_qa","text":"H140 original app source preflight",
             "status":"todo","priority":"medium","tag":"Dev",
             "dueDate":None,"subtasks":[]},
            {"id":"h140_existing","text":"Existing In Progress",
             "status":"in-progress","priority":"low","tag":"Study",
             "dueDate":None,"subtasks":[]}]
        setup="if(localStorage.getItem('kanbanTasks')===null){"+(
            "localStorage.setItem('kanbanTasks',"+json.dumps(json.dumps(tasks))+");}")
        return setup
    if name=="ood-second-drag":
        stamp="2026-10-26T10:00:00.000Z"
        def task(i,title,status):
            return {"id":i,"title":title,"description":"Original offline source-only QA",
                    "priority":"medium","status":status,
                    "createdAt":stamp,"dueDate":None,"tags":[],"swimlane":None}
        board={"id":"h140_source_qa_board","name":"Original Offline Source QA",
               "description":"","color":"#667eea","createdAt":stamp,
               "columns":[
                 {"id":"todo","name":"To Do","emoji":"A","color":"#3498db","order":0},
                 {"id":"inprogress","name":"In Progress","emoji":"B","color":"#f39c12","order":1},
                 {"id":"done","name":"Done","emoji":"C","color":"#2ecc71","order":2}],
               "swimlanes":[],"tasks":[
                 task("h140_source_qa","H140 source-only mouse reference","todo"),
                 task("h140_existing","Existing In Progress","inprogress")]}
        return ("if(localStorage.getItem('kanbanBoards')===null){"+
                "localStorage.setItem('kanbanBoards',"+json.dumps(json.dumps([board]))+");"+
                "localStorage.setItem('currentBoardId','h140_source_qa_board');}")
    raise ValueError("H140 no such authorized owner-only source QA")

def is_moved(name,raw):
    if name=="ood-vanilla-complete":
        state=json.loads(raw)
        return any(t.get("id")=="h140_source_qa" and
                   t.get("status")=="in-progress" for t in state)
    if name=="ood-second-drag":
        boards=json.loads(raw)
        return any(t.get("id")=="h140_source_qa" and
                   t.get("status")=="inprogress" for board in boards
                   for t in board.get("tasks",[]))
    raise ValueError("H140 source not in predeclared OOD source list")

def original_drag(page,name):
    if name=="ood-vanilla-complete":
        start=page.locator("#todo .task-card").first
        end=page.locator("#in-progress .task-card").first
        key="kanbanTasks"
    elif name=="ood-second-drag":
        start=page.locator('.task-card[data-task-id="h140_source_qa"]').first
        end=page.locator('.tasks-container[data-status="inprogress"]').first
        key="kanbanBoards"
    else:
        raise ValueError("Unknown original OOD source")
    start.wait_for(timeout=16000)
    end.wait_for(timeout=16000)
    p1,p2=center(start),center(end)
    if any(not (0<=a<bound) for p in (p1,p2)
           for a,bound in zip(p,(WIDTH,HEIGHT))):
        raise ValueError("H140 source-only original mouse coordinates offscreen")
    mouse_drag(page,p1,p2)
    page.wait_for_timeout(130)
    ok=is_moved(name,page.evaluate("(key)=>localStorage.getItem(key)",key))
    page.reload(wait_until="commit",timeout=16000)
    restored=is_moved(name,page.evaluate("(key)=>localStorage.getItem(key)",key))
    return ok,restored

def run_source_preflight(private_root):
    from playwright.sync_api import sync_playwright
    private_root=Path(private_root)
    results={}
    with sync_playwright() as pw:
        opts={"headless":True}
        exe=_chromium_binary()
        if exe:opts["executable_path"]=exe
        browser=pw.chromium.launch(**opts,timeout=20000)
        try:
            for name,policy in SOURCES.items():
                path=private_root/name
                revision=check_original_source(path,policy)
                with serve(path) as url:
                    ctx=browser.new_context(
                        viewport={"width":WIDTH,"height":HEIGHT},
                        device_scale_factor=1,service_workers="block")
                    ctx.add_init_script(original_seed_for(name))
                    ctx.route("**/*",only_local_requests)
                    try:
                        page=ctx.new_page()
                        page.goto(url,wait_until="commit",timeout=16000)
                        success,persisted=original_drag(page,name)
                        if not success or not persisted:
                            raise ValueError("H140 original external-source preflight drag failed")
                    finally:
                        ctx.close()
                results[name]={"source_repo":policy["repository"],
                               "git_sha":revision,"engine":policy["engine"],
                               "offline_native_mouse_drag":True,
                               "local_storage_moved":True,
                               "survived_page_reload":True,
                               "real_model_neural_calls":0,
                               "blind_screenshots_captured":0,
                               "training_eligible":False}
        finally:browser.close()
    return {"schema":SCHEMA,"status":"SOURCE_ONLY_OOD_RUNTIME_QUALIFIED",
            "qualified_independent_apps":len(results),
            "model_accuracy_measured":False,
            "blind_gold_published":False,
            "trained_Hala_35B":False,
            "results":results}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--sources",required=True)
    a=p.parse_args()
    print(json.dumps(run_source_preflight(a.sources),sort_keys=True,indent=2))
