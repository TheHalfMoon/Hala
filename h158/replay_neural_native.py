"""H158 one-shot paired neural transfer on original untouched TodoMVC app.

The source-app state is intentionally IN MEMORY, not durable across reload.
No reference point is supplied to either model or the native action executor.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from h130.browser_env import _chromium_binary
from h158.audit_source import audit as audit_source
from h158.gpu_infer import validated_model_action,MODEL_SHAS
from h158.original_todomvc_source import source_url,tasks_for,COUNTS,TOTAL

LOCK="H158_DIFFERENT_APP_NEURAL_ONESHOT_CONSUMED.lock"
OUT="H158_DIFFERENT_APP_REAL_NEURAL_BROWSER_TASKS.json"
PRED="h158-private-external-ood-predictions.json"


def preflight(owner,upstream,model_root):
    owner=Path(owner)
    original=source_url(upstream)
    receipt=audit_source(owner)
    model_root=Path(model_root)
    gpu=json.loads((model_root/PRED).read_text("utf8"))
    if (gpu.get("schema")!="hala.h158.private-external-todomvc-gpu-inference.v1" or
        gpu.get("status")!="H158_COMPLETE_REAL_32_GPU_NEW_APP_INFERENCES" or
        gpu.get("total_gpu_calls")!=32 or
        gpu.get("optimizer_steps")!=0 or
        gpu.get("h140_blind_calls")!=0 or
        gpu.get("source_manifest_sha256")!=receipt["manifest_sha256"] or
        gpu.get("older_research_lora_sha")!=MODEL_SHAS["before"] or
        gpu.get("newer_research_lora_sha")!=MODEL_SHAS["after"]):
        raise ValueError("H158 exact paired no-training GPU receipt not proven")
    src=json.loads((owner/"private-h158"/"todomvc-source-once-v1"/
                    "manifest.json").read_text("utf8"))
    cases=src["private_cases"]
    expected={(c["image_sha256"],role) for c in cases
              for role in ("before","after")}
    entries=gpu["private_predictions"]
    observed={}
    if not isinstance(entries,list) or len(entries)!=32 or len(expected)!=32:
        raise ValueError("H158 private cross-app inference predictions incomplete")
    for r in entries:
        if (not isinstance(r,dict) or
            set(r)!={"role","task","language","image_sha","raw","model_action"} or
            not isinstance(r["raw"],str) or
            (r["image_sha"],r["role"]) not in expected or
            (r["image_sha"],r["role"]) in observed):
            raise ValueError("H158 model proposal missing/duplicate original task")
        parsed=validated_model_action(r["raw"])
        if parsed!=r["model_action"]:
            raise ValueError("H158 private model action parsed metadata mismatch")
        observed[(r["image_sha"],r["role"])]=r
    if set(observed)!=expected:
        raise ValueError("H158 external paired task images not matched")
    for case in cases:
        for role in ("before","after"):
            entry=observed[(case["image_sha256"],role)]
            if (entry["task"]!=case["task"] or
                entry["language"]!=case["language"]):
                raise ValueError("H158 neural task/Arabic strata drift")
    return original,receipt,cases,observed


def native(browser,url,case,action):
    ctx=browser.new_context(viewport={"width":960,"height":720},
                            device_scale_factor=1,locale="en-US")
    try:
        page=ctx.new_page()
        page.goto(url,wait_until="load",timeout=20000)
        items=tasks_for(case["language"],case["index"])
        for name in items:
            input=page.locator("input.new-todo")
            input.fill(name)
            input.press("Enter")
        if page.locator("ul.todo-list li").count()!=3:
            raise ValueError("H158 official TodoMVC original scenario setup failed")
        if case["task"]!="complete_item":
            page.locator("ul.todo-list li").filter(has_text=items[0]).locator(
                "input.toggle").click()
        picture=page.screenshot(type="png")
        if hashlib.sha256(picture).hexdigest()!=case["image_sha256"]:
            raise ValueError("H158 new original app screenshot mismatch")
        sent=False
        click=action["click"]
        if action["valid"] and click is not None and not action["abstain"]:
            page.mouse.click(*click)
            sent=True
        if case["task"]=="complete_item":
            correct=(page.locator("ul.todo-list li.completed")
                     .filter(has_text=items[1]).count()==1)
        elif case["task"]=="filter_active":
            correct=(page.url.endswith("#/active") and
                     page.locator("ul.todo-list li.completed").count()==0 and
                     page.locator("ul.todo-list li").count()==2)
        elif case["task"]=="clear_completed":
            correct=(page.locator("ul.todo-list li").filter(
                has_text=items[0]).count()==0 and
                     page.locator("ul.todo-list li").count()==2)
        else:
            raise ValueError("H158 unregistered OOD task")
        return bool(sent and correct),bool(sent)
    finally:
        ctx.close()


def run(owner,upstream,model_root):
    owner=Path(owner)
    url,source,cases,pred=preflight(owner,upstream,model_root)
    root=owner/"private-h158"
    consumed=root/LOCK
    result=root/OUT
    if consumed.exists() or result.exists():
        raise FileExistsError("H158 independent OOD evaluation already consumed")
    with consumed.open("x",encoding="utf8") as f:
        f.write(json.dumps({
            "status":"H158_INDEPENDENT_ORIGINAL_APP_ONESHOT_CONSUMED",
            "source_manifest_sha256":source["manifest_sha256"],
            "model_sha":MODEL_SHAS,"new_cases":16,
            "h140_blind_calls":0},sort_keys=True)+"\n")
    from playwright.sync_api import sync_playwright
    rows=[]
    with sync_playwright() as pw:
        opts={"headless":True}
        exe=_chromium_binary()
        if exe:
            opts["executable_path"]=exe
        browser=pw.chromium.launch(**opts,timeout=20000)
        try:
            for case in cases:
                for role in ("before","after"):
                    model=pred[(case["image_sha256"],role)]
                    action=model["model_action"]
                    passed,sent=native(browser,url,case,action)
                    rows.append({
                        "role":role,"task":case["task"],
                        "language":case["language"],
                        "image_sha":case["image_sha256"],
                        "valid":action["valid"],
                        "abstained":action["abstain"],
                        "click_sent":sent,"original_app_task_success":passed,
                    })
                if len(rows)%8==0:
                    print("H158_REAL_TWO_MODEL_SECOND_APP_BROWSER",
                          len(rows),32,flush=True)
        finally:
            browser.close()
    scores=Counter()
    per={}
    for row in rows:
        scores[row["role"]+"_passed"]+=int(row["original_app_task_success"])
        scores[row["role"]+"_valid"]+=int(row["valid"])
        per[(row["role"],row["task"],row["language"])]=(
            per.get((row["role"],row["task"],row["language"]),0)+
            int(row["original_app_task_success"]))
    summary={
        "schema":"hala.h158.private-genuine-second-app-native-reward.v1",
        "status":"H158_COMPLETED_32_REAL_UNTRAINED_NEW_APP_MODEL_ACTIONS",
        "upstream_original":"tastejs/todomvc",
        "upstream_sha":source["upstream_sha"],
        "frozen_source_manifest_sha256":source["manifest_sha256"],
        "baseline_adapter_sha":MODEL_SHAS["before"],
        "trained_adapter_sha":MODEL_SHAS["after"],
        "native_browser_trials":32,"new_original_task_images":16,
        "before_actual_task_success":scores["before_passed"],
        "after_actual_task_success":scores["after_passed"],
        "before_valid_model_output":scores["before_valid"],
        "after_valid_model_output":scores["after_valid"],
        "per_original_task_language":{
            f"{role}/{task}/{lang}":score
            for (role,task,lang),score in sorted(per.items())},
        "app_state_persistent_after_reload":False,
        "new_app_inference_only":True,
        "optimizer_steps":0,"h140_blind_calls":0,
        "paid_gpu":False,"Hala35B_unchanged":True,
        "model_published":False,"release_qualified":False,
    }
    result.write_text(json.dumps({"summary":summary,"private_cases":rows},
                     sort_keys=True,indent=2)+"\n",encoding="utf8")
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--owner-root",required=True)
    parser.add_argument("--upstream",required=True)
    parser.add_argument("--model-root",required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.owner_root,args.upstream,args.model_root),
                     sort_keys=True,indent=2))
