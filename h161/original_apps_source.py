"""H161 original TWO-app GUI source TRAIN/DEV with private native reference QA.

Application #1: untouched MIT tastejs/todomvc ES6 (native click).
Application #2: untouched MIT react-dnd-kit-board (native drag).
No H158/H159/H160 holdout reuse, no model calls and no cloud costs.
The independent third-app cross-app BLIND must be separately locked later.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from h130.browser_env import _chromium_binary
from h138.probe import APPS, center, mouse_drag, serve
from h138.source import inspect_repo
from h158.original_todomvc_source import source_url, tasks_for
from h159.stable_source_qa import stable_png

SCHEMA = "hala.h161.original-two-app-gui-train-dev-source.v1"
APPS_USED = ("todomvc-es6", "react-dnd-kit-board")
PIN_BOARD = APPS["react-dnd-kit-board"]
TO_DO_TYPES = ("complete_item", "filter_active", "clear_completed")
LANGS = ("en", "ar")
QUOTAS = {"TRAIN": {"todomvc-es6": 24, "react-dnd-kit-board": 24},
          "DEV": {"todomvc-es6": 12, "react-dnd-kit-board": 12}}
WIDTH = 960
HEIGHT = 720
FOLDER = "private-h161/two-real-app-source-v1"


def digest(binary):
    return hashlib.sha256(binary).hexdigest()


def all_specs():
    """Frozen split: 48 TRAIN + 24 DEV, 36 per real upstream app."""
    specs = []
    for split in ("TRAIN", "DEV"):
        for app in APPS_USED:
            n = QUOTAS[split][app]
            if app == "todomvc-es6":
                if n % 6:
                    raise ValueError("H161 bilingual click-family quotas invalid")
                matrix = [(task,lang) for task in TO_DO_TYPES for lang in LANGS]
                for i in range(n):
                    task, lang = matrix[i % len(matrix)]
                    specs.append({"split":split,"app":app,"task":task,
                                  "language":lang,"index":161000 +
                                  (0 if split=="TRAIN" else 2000) + i})
            else:
                for i in range(n):
                    lang = LANGS[i % 2]
                    specs.append({"split":split,"app":app,"task":"drag_todo_progress",
                                  "language":lang,"index":181000 +
                                  (0 if split=="TRAIN" else 2000) + i})
    if len(specs)!=72 or len({(s["app"],s["index"]) for s in specs})!=72:
        raise ValueError("H161 split indices not isolated")
    return specs


def prior_shas(owner):
    owner=Path(owner)
    folders=(
        owner/"private-h158"/"todomvc-source-once-v1"/"images",
        owner/"private-h159"/"source-once-v1"/"images",
        owner/"private-h138"/"real-untouched-app8",
    )
    prior=set()
    for folder in folders:
        if not folder.is_dir():
            raise ValueError("H161 missing earlier original app source exclusion")
        for path in folder.glob("*.png"):
            prior.add(digest(path.read_bytes()))
    if len(prior)<32:
        raise ValueError("H161 prior cross-app image exclusion incomplete")
    return prior


def board_setup(case):
    i=case["index"]
    card=f"Review model integration {i}"
    priority=("Low","Medium","High")[i % 3]
    values={
        "todo":[{"id":i,"text":card,"priority":priority}],
        "progress":[{"id":9001,"text":"Verify test suite","priority":"Low"}],
        "done":[{"id":9002,"text":"Archive old logs","priority":"Medium"}],
    }
    # Original app localStorage API is the original UI app's own state bootstrap.
    return card,values


def prompt(case, title):
    lang=case["language"]
    task=case["task"]
    if task=="complete_item":
        return (f"Click the completion checkbox for '{title}'."
                if lang=="en" else f"اضغط على مربع إكمال المهمة «{title}».")
    if task=="filter_active":
        return ("Click the Active filter." if lang=="en"
                else "اضغط على فلتر المهام النشطة.")
    if task=="clear_completed":
        return ("Click the Clear completed button." if lang=="en"
                else "اضغط على زر مسح المهام المكتملة.")
    if task=="drag_todo_progress":
        return (f"Drag the '{title}' task card from To Do to In Progress."
                if lang=="en"
                else f"اسحب بطاقة المهمة «{title}» من عمود To Do إلى عمود In Progress.")
    raise ValueError("H161 unregistered native action task")


def native_todo(browser, url, case, *, execute):
    context=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                                device_scale_factor=1,locale="en-US")
    try:
        page=context.new_page()
        page.goto(url,wait_until="load",timeout=20000)
        items=tasks_for(case["language"],case["index"])
        for value in items:
            entry=page.locator("input.new-todo")
            entry.fill(value)
            entry.press("Enter")
        if page.locator("ul.todo-list li").count()!=3:
            raise ValueError("H161 original TodoMVC bootstrap failed")
        if case["task"]!="complete_item":
            page.locator("ul.todo-list li").filter(has_text=items[0]).locator(
                "input.toggle").click()
            page.wait_for_function(
                "() => document.querySelectorAll('ul.todo-list li.completed').length === 1",
                timeout=5000,
            )
        image=stable_png(page)
        if not execute:
            return image,None,None
        if case["task"]=="complete_item":
            target=page.locator("ul.todo-list li").filter(
                has_text=items[1]).locator("input.toggle")
        elif case["task"]=="filter_active":
            target=page.locator("ul.filters a[href='#/active']")
        else:
            target=page.locator("button.clear-completed")
        point=center(target)
        page.mouse.click(*point)
        page.wait_for_timeout(75)
        if case["task"]=="complete_item":
            passed=page.locator("ul.todo-list li.completed").filter(
                has_text=items[1]).count()==1
        elif case["task"]=="filter_active":
            passed=(page.url.endswith("#/active") and
                    page.locator("ul.todo-list li.completed").count()==0 and
                    page.locator("ul.todo-list li").count()==2)
        else:
            passed=(page.locator("ul.todo-list li").filter(
                has_text=items[0]).count()==0 and
                    page.locator("ul.todo-list li").count()==2)
        if not passed:
            raise ValueError("H161 source native real original TodoMVC click failed")
        return image,{"actions":[{"kind":"click","point":point}]},prompt(
            case,items[1] if case["task"]=="complete_item" else "")
    finally:
        context.close()


def native_board(browser, url, case, *, execute):
    card, values=board_setup(case)
    context=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                                device_scale_factor=1,service_workers="block",
                                locale="en-US")
    context.add_init_script(
        "localStorage.setItem('kanbanTasks'," +
        json.dumps(json.dumps(values,ensure_ascii=False)) + ");"
    )
    try:
        page=context.new_page()
        page.goto(url,wait_until="networkidle",timeout=20000)
        origin=page.locator(".todo [role=button]").first
        destination=page.locator(".progress .task-list")
        origin.wait_for(timeout=8000)
        if card not in page.locator(".todo").inner_text(timeout=3000):
            raise ValueError("H161 native board task was not rendered from original app state")
        image=stable_png(page)
        if not execute:
            return image,None,None
        a=center(origin)
        b=center(destination)
        mouse_drag(page,a,b)
        state=json.loads(page.evaluate(
            "(key) => localStorage.getItem(key)","kanbanTasks"))
        passed=any(x.get("id")==case["index"] for x in state["progress"])
        if not passed:
            raise ValueError("H161 source native real MIT Kanban drag failed")
        return image,{"actions":[{"kind":"drag","start":a,"end":b}]},prompt(
            case,card)
    finally:
        context.close()


def check_inputs(todo_root, board_root):
    todo=source_url(todo_root)
    pin=inspect_repo(Path(board_root),PIN_BOARD)
    if pin!=PIN_BOARD:
        raise ValueError("H161 react native pinned vendor unexpected")
    return todo, pin


def episode(browser, urls, case, *, execute):
    if case["app"]=="todomvc-es6":
        return native_todo(browser,urls["todomvc-es6"],case,execute=execute)
    if case["app"]=="react-dnd-kit-board":
        return native_board(browser,urls["react-dnd-kit-board"],case,execute=execute)
    raise ValueError("H161 app not preregistered")


def run(todo_root, board_root, owner, mode="capture"):
    owner=Path(owner)
    todo,_=check_inputs(todo_root,board_root)
    root=owner/FOLDER
    if mode=="capture" and root.exists():
        raise FileExistsError("H161 original source is create-only")
    if mode=="verify" and not (root/"manifest.json").is_file():
        raise FileNotFoundError("H161 exact original source manifest missing")
    old=prior_shas(owner)
    from playwright.sync_api import sync_playwright
    rows=[]
    with sync_playwright() as playwright:
        options={"headless":True}
        binary=_chromium_binary()
        if binary:
            options["executable_path"]=binary
        browser=playwright.chromium.launch(**options,timeout=20000)
        try:
            with serve(Path(board_root)/"dist") as board_url:
                urls={"todomvc-es6":todo,"react-dnd-kit-board":board_url}
                if mode=="smoke":
                    samples=(all_specs()[0],all_specs()[24])
                elif mode=="capture":
                    root.mkdir(parents=True,exist_ok=False)
                    (root/"images").mkdir()
                    samples=all_specs()
                elif mode=="verify":
                    current=json.loads((root/"manifest.json").read_text("utf8"))
                    if current.get("schema")!=SCHEMA:
                        raise ValueError("H161 prior frozen source schema not matched")
                    samples=current["private_cases"]
                else:
                    raise ValueError("Unknown H161 source action")
                for n,case in enumerate(samples,1):
                    png,supervision,instruction=episode(
                        browser,urls,case,execute=mode!="verify")
                    image_sha=digest(png)
                    if image_sha in old:
                        raise ValueError("H161 historical source/eval image reused")
                    if mode=="verify":
                        expected=(root/"images"/case["image"]).read_bytes()
                        if digest(expected)!=case["image_sha256"] or png!=expected:
                            raise ValueError(
                                f"H161 new native app screenshot mismatch at case {n}"
                            )
                    elif mode!="smoke":
                        if image_sha in {row["image_sha256"] for row in rows}:
                            raise ValueError("H161 source images duplicated across splits")
                        name=f'{case["app"]}-{case["task"]}-{case["index"]}.png'
                        (root/"images"/name).write_bytes(png)
                        rows.append({
                            **case, "image":name,"image_sha256":image_sha,
                            "instruction":instruction,
                            "training_supervision":supervision,
                            "original_native_reference_success":True,
                            "model_calls":0,
                        })
                    print(f"H161_{mode.upper()}_ORIGINAL_APP",n,len(samples),
                          case["app"],case["task"],flush=True)
        finally:
            browser.close()
    if mode=="smoke":
        return {"status":"H161_TWO_REAL_MIT_APPS_SMOKE_PASS",
                "source_attempts":len(samples),"model_calls":0}
    if mode=="verify":
        if len(samples)!=72:
            raise ValueError("H161 frozen source incomplete")
        qa=owner/"private-h161"/"H161_INDEPENDENT_PIXEL_QA.json"
        if qa.exists():
            raise FileExistsError("H161 independent replay already recorded")
        proof={"status":"H161_72_OF_72_CROSS_CONTEXT_EXACT_NATIVE_SCREENSHOT_MATCH",
               "source_cases":72,"train_images":48,"dev_images":24,
               "additional_model_calls":0,"h140_blind_calls":0,"paid_gpu":False}
        qa.write_text(json.dumps(proof,indent=2,sort_keys=True)+"\n","utf8")
        return proof
    counts=Counter((row["split"],row["app"]) for row in rows)
    expected={(split,app):count for split,group in QUOTAS.items()
              for app,count in group.items()}
    if (counts!=expected or len(rows)!=72 or
        len({row["image_sha256"] for row in rows})!=72):
        raise ValueError("H161 native browser source quota failed")
    metadata={"schema":SCHEMA, "training_source_only":True,
              "source_apps":{"todomvc-es6":"ff43b02e59dfa604386bb382034b2cd07c2bcd8a",
                             "react-dnd-kit-board":PIN_BOARD},
              "source_reference_success":72,
              "train":48,"dev":24,"quotas":QUOTAS,
              "h140_blind_calls":0,"paid_gpu":False,
              "dev_is_not_training":True,
              "private_cases":rows}
    data=(json.dumps(metadata,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode()
    (root/"manifest.json").write_bytes(data)
    return {"status":"H161_TWO_REAL_UPSTREAM_MIT_APPS_SOURCE_72_PASS",
            "source_manifest_sha256":digest(data),
            "train":48,"dev":24,"apps":list(APPS_USED),"model_calls":0}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("smoke","capture","verify"))
    p.add_argument("--todo-root",required=True)
    p.add_argument("--board-root",required=True)
    p.add_argument("--owner",required=True)
    args=p.parse_args()
    print(json.dumps(run(args.todo_root,args.board_root,args.owner,args.mode),
                     ensure_ascii=False,sort_keys=True,indent=2))
