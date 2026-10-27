"""H164 TRAIN-only pilot: genuine native scenes from four original UI apps.

No old H138/H158-H163 media is trained or re-used. All source and gold actions
stay on owner Windows. This is *not* a completed H164 TRAIN/DEV/OOD dataset.
"""
import argparse
import hashlib
import json
import subprocess
from contextlib import ExitStack
from pathlib import Path

from h130.browser_env import _chromium_binary
from h138.probe import center, serve
from h159.stable_source_qa import stable_png
from h161.original_apps_source import native_todo, native_board
from h164.protocol import locked_prior_shas

SCHEMA="hala.h164.four-original-app-train-pilot.v1"
APPS=("todomvc-es6","react-dnd-kit-board","simple-table","calculator")
PRIVATE="private-h164/four-app-train-pilot-v1"
N_PER_APP=24
START=164001
PINS={
    "simple-table":"7580270d18fb764bbf437a5e5b88ea66d63fc449",
    "calculator":"8f27a92072dde9658d8d94db8918a184ed7bd416"
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prereq(owner):
    owner=Path(owner)
    root=owner/"private-h164"/"upstream"
    for name,sha in PINS.items():
        app=root/name
        head=subprocess.check_output(
            ["git","-C",str(app),"rev-parse","HEAD"],text=True).strip()
        dirty=subprocess.check_output(
            ["git","-C",str(app),"status","--porcelain"],text=True).strip()
        readme=(app/("readme.md" if name=="simple-table" else "README.md")).read_text(
            "utf8",errors="replace")
        if head!=sha or dirty or "MIT" not in readme:
            raise ValueError("H164 original MIT-claimed upstream changed")
    return root


def specs():
    items=[]
    for app in APPS:
        for n in range(N_PER_APP):
            task=("complete_item","filter_active","clear_completed")[n%3] if app=="todomvc-es6" else (
                "drag_todo_progress" if app=="react-dnd-kit-board" else (
                ("clear_selection","filter_name","sort_age")[n%3]
                if app=="simple-table" else
                ("press_plus","clear","calculate")[n%3]))
            items.append({"split":"TRAIN","app":app,"task":task,
                          "language":"en" if n%2==0 else "ar",
                          "index":START+APPS.index(app)*100+n})
    if len(items)!=96:
        raise ValueError("H164 pilot app/task registry changed")
    return items


def native_table(browser,url,case,execute):
    context=browser.new_context(viewport={"width":960,"height":720},
                                device_scale_factor=1,locale="en-US")
    try:
        page=context.new_page()
        response=page.goto(url,wait_until="load",timeout=20000)
        if response.status!=200:
            raise ValueError("H164 original table demo unavailable")
        page.locator("#table-container tbody tr").first.wait_for(timeout=8000)
        index=case["index"]-START-200
        # Use the original SimpleTable pageSize API to produce three
        # genuinely distinct page layouts; original source remains untouched.
        size=(8,12,16)[index//8]
        page.evaluate("(n) => {table.pageSize=n; table.currentPage=1; table.render();}",size)
        # Vary native table sorting and selected original employee rows.
        col=(0,4,5,6)[index%4]
        page.locator(".simple-table thead tr").first.locator("th").nth(col).click()
        page.locator(".simple-table tbody tr").nth(index%8).click()
        selected=page.locator("#selection-info").inner_text()
        if "No row selected" in selected:
            raise ValueError("H164 real original table row selection not set")
        png=stable_png(page)
        if not execute:
            return png,None,None
        task=case["task"]
        if task=="clear_selection":
            target=page.locator("#btn-clear")
            point=center(target)
            page.mouse.click(*point)
            success=page.locator("#selection-info").inner_text()=="No row selected"
            gold={"actions":[{"kind":"click","point":point}]}
            en="Click Clear selection to deselect the highlighted employee."
            ar="اضغط على زر Clear selection لإلغاء تحديد الموظف."
        elif task=="sort_age":
            target=page.locator(".simple-table thead tr").first.locator("th").nth(5)
            before=target.locator(".sort-indicator").inner_text()
            point=center(target)
            page.mouse.click(*point)
            after=target.locator(".sort-indicator").inner_text()
            success=before!=after
            gold={"actions":[{"kind":"click","point":point}]}
            en="Click the Age column heading to change the employee sort."
            ar="اضغط على عنوان عمود Age لتغيير ترتيب الموظفين."
        elif task=="filter_name":
            target=page.locator('.simple-table input[data-field="name"]')
            point=center(target)
            page.mouse.click(*point)
            page.keyboard.type("Ana",delay=10)
            page.wait_for_timeout(650)
            filtered=page.locator(".simple-table tbody tr")
            success=(filtered.count()==1 and
                     "Ana Garcia" in filtered.first.inner_text())
            gold={"actions":[{"kind":"type","point":point,"text":"Ana"}]}
            en="Type Ana in the Name column filter."
            ar="اكتب Ana في حقل تصفية عمود الاسم Name."
        else:
            raise ValueError("H164 unregistered original table task")
        if not success:
            raise ValueError(f"H164 actual table UI action failed: {case['task']} index={case['index']} selected={page.locator('#selection-info').inner_text()} rows={page.locator('.simple-table tbody tr').count()}")
        return png,(en if case["language"]=="en" else ar),gold
    finally:
        context.close()


def native_calculator(browser,url,case,execute):
    context=browser.new_context(viewport={"width":960,"height":720},
                                device_scale_factor=1,locale="en-US")
    try:
        page=context.new_page()
        response=page.goto(url,wait_until="load",timeout=20000)
        if response.status!=200:
            raise ValueError("H164 original calculator demo unavailable")
        i=case["index"]-START-300
        a=11+(i*3)
        for digit in str(a):
            page.locator(f'.keypad button[data-value="{digit}"]').click()
        if case["task"]=="calculate":
            page.locator('.keypad button[data-value="+"]').click()
            page.locator('.keypad button[data-value="4"]').click()
        before=page.locator("#expression").inner_text()
        if str(a) not in before:
            raise ValueError("H164 original calculator native seed mismatch")
        png=stable_png(page)
        if not execute:
            return png,None,None
        task=case["task"]
        if task=="press_plus":
            target=page.locator('.keypad button[data-value="+"]')
            point=center(target)
            page.mouse.click(*point)
            success=page.locator("#expression").inner_text().endswith("+")
            en="Click the addition operator (+)."
            ar="اضغط على زر عملية الجمع (+)."
        elif task=="clear":
            target=page.locator('.keypad button[data-action="clear"]')
            point=center(target)
            page.mouse.click(*point)
            success=page.locator("#expression").inner_text().strip()=="0"
            en="Click AC to clear the calculator expression."
            ar="اضغط على زر AC لمسح العملية الحسابية."
        elif task=="calculate":
            target=page.locator('.keypad button[data-action="calculate"]')
            point=center(target)
            page.mouse.click(*point)
            expected=a+4
            actual=page.locator("#result").inner_text().replace(",","")
            success=actual==str(expected)
            en="Click the equals (=) button to calculate the expression."
            ar="اضغط على زر المساواة (=) لحساب النتيجة."
        else:
            raise ValueError("H164 original calculator task unregistered")
        if not success:
            raise ValueError("H164 real original calculator native click failed")
        gold={"actions":[{"kind":"click","point":point}]}
        return png,(en if case["language"]=="en" else ar),gold
    finally:
        context.close()


def run(owner,mode):
    owner=Path(owner)
    paths=prereq(owner)
    historical=locked_prior_shas(owner)
    private=owner/PRIVATE
    if mode=="capture" and private.exists():
        raise FileExistsError("H164 owner source TRAIN pilot create-once")
    if mode=="verify" and not (private/"manifest.json").exists():
        raise FileNotFoundError("H164 owner pilot not yet captured")
    from playwright.sync_api import sync_playwright
    rows=[]
    with sync_playwright() as pw:
        opts={"headless":True}
        chrome=_chromium_binary()
        if chrome:
            opts["executable_path"]=chrome
        browser=pw.chromium.launch(**opts,timeout=20000)
        try:
            with ExitStack() as stack:
                todo=owner/"private-h158"/"upstream"/"todomvc"
                board=owner/"private-h138"/"vendor-apps"/"react-dnd-kit-board"
                urls={
                    "todomvc-es6":todo,
                    "react-dnd-kit-board":stack.enter_context(serve(board/"dist")),
                    "simple-table":stack.enter_context(serve(paths/"simple-table"))+"/examples/index.html",
                    "calculator":stack.enter_context(serve(paths/"calculator"))+"/index.html",
                }
                from h158.original_todomvc_source import source_url
                urls["todomvc-es6"]=source_url(todo)
                if mode=="capture":
                    private.mkdir(parents=True,exist_ok=False)
                    (private/"images").mkdir()
                frozen=(json.loads((private/"manifest.json").read_text("utf8"))
                        if mode=="verify" else None)
                if frozen and (frozen.get("schema")!=SCHEMA or
                               len(frozen.get("private_cases",[]))!=96):
                    raise ValueError("H164 frozen 4 original source app cohort changed")
                all_cases=specs()
                if mode=="smoke":
                    all_cases=([all_cases[0],all_cases[24]]+
                               all_cases[48:51]+all_cases[72:75])
                for n,case in enumerate(all_cases,1):
                    if case["app"]=="todomvc-es6":
                        img,gold,inst=native_todo(
                            browser,urls[case["app"]],case,
                            execute=mode!="verify")
                    elif case["app"]=="react-dnd-kit-board":
                        img,gold,inst=native_board(
                            browser,urls[case["app"]],case,
                            execute=mode!="verify")
                    elif case["app"]=="simple-table":
                        img,inst,gold=native_table(
                            browser,urls[case["app"]],case,
                            execute=mode!="verify")
                    else:
                        img,inst,gold=native_calculator(
                            browser,urls[case["app"]],case,
                            execute=mode!="verify")
                    h=digest(img)
                    if h in historical:
                        raise ValueError("H164 previously consumed original image reused")
                    if mode=="capture":
                        if h in {r["image_sha256"] for r in rows}:
                            raise ValueError("H164 pilot duplicate source screenshot")
                        name=f'{case["app"]}-{case["index"]}.png'
                        (private/"images"/name).write_bytes(img)
                        action=gold["actions"][0]
                        rows.append({**case,"image":name,"image_sha256":h,
                            "instruction":inst,
                            "action_type":action["kind"],
                            "training_supervision":gold,
                            "native_reference_success":True})
                    elif mode=="verify":
                        prior=frozen["private_cases"][n-1]
                        if (prior["image_sha256"]!=h or
                            prior["index"]!=case["index"] or
                            (private/"images"/prior["image"]).read_bytes()!=img):
                            raise ValueError(f"H164 pixel mismatch at position {n}")
                    if n%8==0:
                        print("H164_ORIGINAL_TRAIN_"+mode.upper(),
                              n,len(all_cases),case["app"],flush=True)
        finally:
            browser.close()
    if mode=="smoke":
        return {"status":"H164_FOUR_ORIGINAL_APPS_NATIVE_SMOKE_PASS",
                "tasks":8,"model_calls":0}
    if mode=="verify":
        receipt={"status":"H164_96_TRAIN_ONLY_EXACT_BROWSER_SCREENSHOT_REPLAY",
                 "train":96,"native_reference_success":96,
                 "train_apps":4,"optimizer_steps":0,
                 "new_DEV_or_OOD_completed":False,"h140_blind_calls":0,
                 "paid_GPU":False}
        proof=owner/"private-h164"/"H164_TRAIN_PILOT_PIXEL_QA.json"
        if proof.exists():
            raise FileExistsError("H164 train pilot browser QA already recorded")
        proof.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n","utf8")
        return receipt
    if len(rows)!=96 or len({r["image_sha256"] for r in rows})!=96:
        raise ValueError("H164 original UI training pilot incomplete")
    doc={"schema":SCHEMA,"train_only":True,
         "pilot_not_final_H164_qualification":True,
         "applications":list(APPS),
         "app_upstream_git":PINS,
         "train_count":96,"native_reference_success":96,
         "optimizer_steps":0,"h140_blind_calls":0,
         "paid_GPU":False,"private_cases":rows}
    raw=(json.dumps(doc,indent=2,ensure_ascii=False,sort_keys=True)+"\n").encode()
    (private/"manifest.json").write_bytes(raw)
    return {"status":"H164_96_REAL_FOUR_APP_TRAIN_ONLY_SOURCES",
            "manifest_sha256":digest(raw),"train_cases":96,
            "train_apps":4,"actual_native_source_successes":96,
            "model_training_started":False}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("smoke","capture","verify"))
    p.add_argument("--owner",required=True)
    args=p.parse_args()
    print(json.dumps(run(args.owner,args.mode),indent=2,sort_keys=True))
