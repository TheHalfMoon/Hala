"""H158 original official TodoMVC cross-app native source qualification.

Different original, unmodified MIT app from H157 MiniWoB. Real native
click actions scored from the source application's live state.
This ES6 demo stores tasks in memory, NOT across page reloads.
"""
import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from h130.browser_env import _chromium_binary

PIN="ff43b02e59dfa604386bb382034b2cd07c2bcd8a"
SCHEMA="hala.h158.unmodified-todomvc-crossapp-click-source.v1"
COUNTS={"complete_item":{"en":4,"ar":4},
        "filter_active":{"en":2,"ar":2},
        "clear_completed":{"en":2,"ar":2}}
TOTAL=16


def source_url(path):
    root=Path(path)
    if "MIT" not in (root/"license.md").read_text("utf8"):
        raise ValueError("TodoMVC original license absent")
    head=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],
                                  text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(root),"status","--porcelain"],
                                   text=True).strip()
    if head!=PIN or dirty:
        raise ValueError("H158 requires pristine pinned original upstream")
    folder=root/"examples"/"javascript-es6"/"dist"
    for name in ("index.html","app.bundle.js","app.css","base.js"):
        if not (folder/name).is_file():
            raise ValueError("Original upstream prebuilt ES6 app missing")
    return (folder/"index.html").resolve().as_uri()


def point(locator):
    r=locator.bounding_box(timeout=3000)
    if not r or r["width"]<1 or r["height"]<1:
        raise ValueError("Original clickable element not visible")
    return r["x"]+r["width"]/2,r["y"]+r["height"]/2


def tasks_for(language,index):
    if language=="en":
        return [f"Review report {index:03d}",f"Check deployment {index:03d}",
                f"Update notes {index:03d}"]
    if language=="ar":
        return [f"مراجعة التقرير {index:03d}",f"فحص النظام {index:03d}",
                f"تحديث الملاحظات {index:03d}"]
    raise ValueError("Unsupported instruction language")


def episode(browser,url,task,language,index,images):
    ctx=browser.new_context(viewport={"width":960,"height":720},
                            device_scale_factor=1,locale="en-US")
    try:
        page=ctx.new_page()
        page.goto(url,wait_until="load",timeout=20000)
        items=tasks_for(language,index)
        for item in items:
            input=page.locator("input.new-todo")
            input.fill(item)
            input.press("Enter")
        if page.locator("ul.todo-list li").count()!=3:
            raise ValueError("Original TodoMVC native setup failed")
        if task!="complete_item":
            page.locator("ul.todo-list li").filter(
                has_text=items[0]).locator("input.toggle").click()
        if task=="complete_item":
            target=page.locator("ul.todo-list li").filter(
                has_text=items[1]).locator("input.toggle")
            prompt=(f"اضغط مربع إكمال المهمة «{items[1]}»." if language=="ar"
                    else f"Click the completion checkbox for '{items[1]}'.")
        elif task=="filter_active":
            target=page.locator("ul.filters a[href='#/active']")
            prompt=("اضغط عامل تصفية المهام النشطة." if language=="ar"
                    else "Click the Active filter.")
        elif task=="clear_completed":
            target=page.locator("button.clear-completed")
            prompt=("اضغط زر مسح المهام المكتملة." if language=="ar"
                    else "Click the Clear completed button.")
        else:
            raise ValueError("Original independent task not preregistered")
        x,y=point(target)
        png=page.screenshot(type="png")
        page.mouse.click(x,y)
        if task=="complete_item":
            success=(page.locator("ul.todo-list li.completed").filter(
                has_text=items[1]).count()==1)
        elif task=="filter_active":
            page.wait_for_function("() => location.hash === '#/active'",
                                   timeout=2000)
            success=(page.locator("ul.todo-list li").count()==2 and
                     page.locator("ul.todo-list li.completed").count()==0)
        else:
            success=(page.locator("ul.todo-list li").filter(
                has_text=items[0]).count()==0 and
                     page.locator("ul.todo-list li").count()==2)
        if not success:
            raise ValueError("Original app source action had no correct state")
        filename=f"{task}-{language}-{index:03d}.png"
        (images/filename).write_bytes(png)
        return {"task":task,"language":language,"index":index,
                "image":filename,
                "image_sha256":hashlib.sha256(png).hexdigest(),
                "instruction":prompt,"reference_point":[round(x,4),round(y,4)],
                "original_app_state_correct":True,
                "state_persistent_after_reload":False,
                "model_inference_calls":0}
    finally:
        ctx.close()


def run(upstream,owner):
    url=source_url(upstream)
    folder=Path(owner)/"private-h158"/"todomvc-source-once-v1"
    if folder.exists():
        raise FileExistsError("H158 external app source frozen once")
    folder.mkdir(parents=True,exist_ok=False)
    (folder/"SOURCE_ONCE.lock").write_text(SCHEMA,encoding="utf8")
    media=folder/"images"
    media.mkdir()
    from playwright.sync_api import sync_playwright
    cases=[]
    with sync_playwright() as pw:
        options={"headless":True}
        chrome=_chromium_binary()
        if chrome:
            options["executable_path"]=chrome
        browser=pw.chromium.launch(**options,timeout=20000)
        try:
            for task,groups in COUNTS.items():
                for lang,count in groups.items():
                    for _ in range(count):
                        case=episode(browser,url,task,lang,len(cases)+1,media)
                        cases.append(case)
                        print("H158_REAL_EXTERNAL_APP_REFERENCE",
                              len(cases),TOTAL,task,lang,flush=True)
        finally:
            browser.close()
    actual=Counter((c["task"],c["language"]) for c in cases)
    expected={(task,language):n for task,groups in COUNTS.items()
              for language,n in groups.items()}
    if (actual!=expected or len(cases)!=TOTAL or
        len({c["image_sha256"] for c in cases})!=TOTAL):
        raise ValueError("H158 real second-app source qualification failed")
    summary={"schema":SCHEMA,
             "status":"H158_EXTERNAL_UNMODIFIED_TODOMVC_SOURCE_QUALIFIED",
             "original_upstream":"tastejs/todomvc",
             "original_upstream_sha":PIN,"license":"MIT",
             "new_native_reference_tasks":TOTAL,
             "new_unique_original_images":TOTAL,
             "per_task_language_counts":COUNTS,
             "real_app_state_correct":TOTAL,
             "app_state_persists_after_reload":False,
             "model_accuracy_measured":False,
             "model_inference_calls":0,"optimizer_steps":0,
             "h140_blind_calls":0,"paid_compute":False,
             "private_cases":cases}
    (folder/"manifest.json").write_text(
        json.dumps(summary,indent=2,sort_keys=True,ensure_ascii=False)+"\n",
        encoding="utf8")
    return {k:v for k,v in summary.items() if k!="private_cases"}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--upstream",required=True)
    p.add_argument("--owner-root",required=True)
    args=p.parse_args()
    print(json.dumps(run(args.upstream,args.owner_root),indent=2,sort_keys=True))
