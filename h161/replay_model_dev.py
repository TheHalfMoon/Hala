"""H161 owner-native paired DEV task reward, genuine model-generated actions.

All DEV screenshots are verified pixel-exact before actions. Owner references
are used ONLY for scoring successful original app state, not model/controller.
"""
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from h130.browser_env import _chromium_binary
from h138.probe import center, mouse_drag, serve, APPS
from h138.source import inspect_repo
from h158.original_todomvc_source import source_url, tasks_for
from h159.stable_source_qa import stable_png
from h161.original_apps_source import board_setup
from h161.train_gpu import actions,DEV_SHA,TRAIN_SHA,OLD_LORA_SHA

PRED_BEFORE="h161-private-dev-before.json"
PRED_AFTER="h161-private-dev-after.json"
RECEIPT="h161-genuine-learning-receipt.json"
LOCK="H161_DEV_REAL_NEURAL_ONESHOT_CONSUMED.lock"
OUTPUT="H161_DEV_REAL_NEURAL_NATIVE_REWARD.json"


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def preflight(owner,predict_root,todo_root,board_root):
    owner=Path(owner)
    source=json.loads((owner/"private-h161"/"two-real-app-source-v1"/
                       "manifest.json").read_text("utf8"))
    gate=json.loads((owner/"private-h161"/"H161_INDEPENDENT_PIXEL_QA.json")
                    .read_text("utf8"))
    if (len(source.get("private_cases",[]))!=72 or
        gate.get("status")!=
        "H161_72_OF_72_CROSS_CONTEXT_EXACT_NATIVE_SCREENSHOT_MATCH"):
        raise ValueError("H161 original browser source qualification missing")
    dev=[c for c in source["private_cases"] if c["split"]=="DEV"]
    if len(dev)!=24 or len({c["image_sha256"] for c in dev})!=24:
        raise ValueError("H161 source DEV split drift")
    root=Path(predict_root)
    learned=json.loads((root/RECEIPT).read_text("utf8"))
    if (learned.get("status")!=
        "H161_GENUINE_48_T4_TWO_REAL_APP_TRAIN_STEPS_COMPLETE" or
        learned.get("optimizer_steps")!=48 or
        learned.get("unique_train")!=48 or
        learned.get("dev_image_count")!=24 or
        learned.get("dev_labels_seen_by_GPU") is not False or
        learned.get("source_train_zip_sha256")!=TRAIN_SHA or
        learned.get("source_dev_zip_sha256")!=DEV_SHA or
        learned.get("starting_H157_adapter_sha256")!=OLD_LORA_SHA or
        learned.get("new_mixed_app_adapter_sha256")==OLD_LORA_SHA or
        learned.get("h140_blind_calls")!=0 or
        learned.get("paid_GPU") is not False):
        raise ValueError("H161 genuine trained LoRA receipt incomplete")
    present={(c["image_sha256"],role) for c in dev
             for role in ("before","after")}
    actual={}
    for role,file in (("before",PRED_BEFORE),("after",PRED_AFTER)):
        rows=json.loads((root/file).read_text("utf8"))
        if not isinstance(rows,list) or len(rows)!=24:
            raise ValueError("H161 GPU predictions missing DEV action")
        for row in rows:
            if (not isinstance(row,dict) or
                set(row)!={"app","task","language","image_sha","raw","parsed","role"}
                or row["role"]!=role or
                (row["image_sha"],role) not in present or
                (row["image_sha"],role) in actual or
                row["parsed"]!=actions(row["raw"])):
                raise ValueError("H161 native action did not match GPU raw prediction")
            actual[(row["image_sha"],role)]=row["parsed"]
    if set(actual)!=present:
        raise ValueError("H161 unpaired neural DEV predictions")
    for case in dev:
        for role in ("before","after"):
            match=(case["image_sha256"],role)
            row=json.loads((root/(PRED_BEFORE if role=="before"
                        else PRED_AFTER)).read_text("utf8"))
            # Validate matching instructions/data without using owner's reference point.
            x=[r for r in row if r["image_sha"]==case["image_sha256"]]
            if (len(x)!=1 or x[0]["app"]!=case["app"] or
                x[0]["task"]!=case["task"] or
                x[0]["language"]!=case["language"]):
                raise ValueError("H161 external model image role inconsistent")
    url=source_url(todo_root)
    if inspect_repo(Path(board_root),APPS["react-dnd-kit-board"])!=APPS["react-dnd-kit-board"]:
        raise ValueError("H161 original board source SHA mismatched")
    return dev,actual,url


def todo_trial(browser,url,case,proposal):
    context=browser.new_context(viewport={"width":960,"height":720},
                                device_scale_factor=1,locale="en-US")
    try:
        page=context.new_page()
        page.goto(url,wait_until="load",timeout=20000)
        items=tasks_for(case["language"],case["index"])
        for item in items:
            field=page.locator("input.new-todo")
            field.fill(item)
            field.press("Enter")
        if page.locator("ul.todo-list li").count()!=3:
            raise ValueError("H161 owner original todo scene setup failed")
        if case["task"]!="complete_item":
            page.locator("ul.todo-list li").filter(
                has_text=items[0]).locator("input.toggle").click()
            page.wait_for_function(
                "() => document.querySelectorAll('ul.todo-list li.completed').length===1",
                timeout=5000)
        if digest(stable_png(page))!=case["image_sha256"]:
            raise ValueError("H161 model action would use changed original screenshot")
        sent=proposal["valid"] and proposal["kind"]=="click"
        if sent:
            page.mouse.click(*proposal["points"][0])
            page.wait_for_timeout(75)
        if case["task"]=="complete_item":
            success=page.locator("ul.todo-list li.completed").filter(
                has_text=items[1]).count()==1
        elif case["task"]=="filter_active":
            success=(page.url.endswith("#/active") and
                     page.locator("ul.todo-list li.completed").count()==0 and
                     page.locator("ul.todo-list li").count()==2)
        else:
            success=(page.locator("ul.todo-list li").filter(
                has_text=items[0]).count()==0 and
                     page.locator("ul.todo-list li").count()==2)
        return bool(sent and success),bool(sent)
    finally:
        context.close()


def board_trial(browser,url,case,proposal):
    task,setup=board_setup(case)
    context=browser.new_context(viewport={"width":960,"height":720},
                                device_scale_factor=1,locale="en-US",
                                service_workers="block")
    context.add_init_script(
        "localStorage.setItem('kanbanTasks',"+
        json.dumps(json.dumps(setup,ensure_ascii=False))+");")
    try:
        page=context.new_page()
        page.goto(url,wait_until="networkidle",timeout=20000)
        if task not in page.locator(".todo").inner_text():
            raise ValueError("H161 original app card not visible")
        if digest(stable_png(page))!=case["image_sha256"]:
            raise ValueError("H161 board original screenshot not pixel identical")
        sent=proposal["valid"] and proposal["kind"]=="drag"
        if sent:
            a,b=proposal["points"]
            mouse_drag(page,a,b)
        state=json.loads(page.evaluate(
            "(key) => localStorage.getItem(key)","kanbanTasks"))
        success=any(r.get("id")==case["index"] for r in state["progress"])
        return bool(sent and success),bool(sent)
    finally:
        context.close()


def run(owner,todo_root,board_root,model_root):
    owner=Path(owner)
    cases,proposals,todo_url=preflight(
        owner,model_root,todo_root,board_root)
    private=owner/"private-h161"
    consumed=private/LOCK
    output=private/OUTPUT
    if consumed.exists() or output.exists():
        raise FileExistsError("H161 DEV real browser one-shot already used")
    with consumed.open("x",encoding="utf8") as f:
        f.write(json.dumps({
            "status":"H161_DEV_NEURAL_48_NATIVE_ATTEMPTS_CONSUMED",
            "case_count":24,"h140_blind_calls":0,
            "source_manifest_sha256":
             "0710797074d8407640fa8c0e65bbfe8672175a644483701ce156aaab02621997"
        },sort_keys=True)+"\n")
    from playwright.sync_api import sync_playwright
    rows=[]
    with sync_playwright() as pw:
        opts={"headless":True}
        binary=_chromium_binary()
        if binary:
            opts["executable_path"]=binary
        browser=pw.chromium.launch(**opts,timeout=20000)
        try:
            with serve(Path(board_root)/"dist") as board_url:
                for case in cases:
                    for role in ("before","after"):
                        action=proposals[(case["image_sha256"],role)]
                        if case["app"]=="todomvc-es6":
                            success,sent=todo_trial(
                                browser,todo_url,case,action)
                        else:
                            success,sent=board_trial(
                                browser,board_url,case,action)
                        rows.append({
                            "app":case["app"],"language":case["language"],
                            "task":case["task"],"image_sha":case["image_sha256"],
                            "role":role,"native_action_sent":sent,
                            "actual_original_app_success":success,
                            "model_output_valid":action["valid"],
                            "model_action_kind":action["kind"],
                        })
                    if len(rows)%8==0:
                        print("H161_REAL_MIXED_APP_NEURAL_NATIVE",
                              len(rows),48,flush=True)
        finally:
            browser.close()
    by={(r["image_sha"],r["role"]):r for r in rows}
    counts=Counter()
    strata=defaultdict(Counter)
    for c in cases:
        prior=by[(c["image_sha256"],"before")]["actual_original_app_success"]
        after=by[(c["image_sha256"],"after")]["actual_original_app_success"]
        counts["before"]+=int(prior)
        counts["after"]+=int(after)
        counts["gains"]+=int(after and not prior)
        counts["regressions"]+=int(prior and not after)
        group=strata[(c["app"],c["language"])]
        group["n"]+=1
        group["before"]+=int(prior)
        group["after"]+=int(after)
    d=counts["gains"]+counts["regressions"]
    small=min(counts["gains"],counts["regressions"])
    p=min(1.,2*sum(math.comb(d,k) for k in range(small+1))/(2**d)) if d else 1.
    result={
        "status":"H161_COMPLETED_48_GENUINE_NATIVE_TRAINED_MODEL_DEV_TASKS",
        "before_app_success":counts["before"],
        "after_app_success":counts["after"],
        "paired_gains":counts["gains"],
        "paired_regressions":counts["regressions"],
        "paired_sign_p":p,
        "actual_native_browser_trials":48,"model_dev_cases":24,
        "per_app_language":{f"{app}/{lang}":dict(v)
            for (app,lang),v in sorted(strata.items())},
        "same_apps_as_training_not_OOD_BLIND":True,
        "h140_blind_calls":0,"paid_GPU":False,
        "Hala35B_unchanged":True,
        "private_rows":rows,
    }
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n","utf8")
    return {k:v for k,v in result.items() if k!="private_rows"}


if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--owner",required=True)
    a.add_argument("--todo-root",required=True)
    a.add_argument("--board-root",required=True)
    a.add_argument("--model-root",required=True)
    args=a.parse_args()
    print(json.dumps(run(args.owner,args.todo_root,args.board_root,args.model_root),
                     indent=2,sort_keys=True))
