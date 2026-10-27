"""H160 one-shot native browser execution of true frozen model-generated clicks.

No H158 consumed screenshots/predictions. No DOM oracle target or gold point is
passed to the controller. All 32 original-app success checks are on live UI.
"""
import argparse
import json
from pathlib import Path

from h130.browser_env import _chromium_binary
from h158.audit_neural_transfer import paired
from h158.gpu_infer import validated_model_action
from h158.original_todomvc_source import source_url, tasks_for
from h159.stable_source_qa import stable_png, digest
from h160.gpu_infer import validate, MANIFEST_SHA, SOURCE_ZIP_SHA
from h160.pack_private_eval import SHA_H156, SHA_H157, SOURCE_ZIP

PRED = "h160-private-predictions.json"
LOCK = "H160_NEW_IMAGE_STABLE_MODEL_ONESHOT_CONSUMED.lock"
RESULT = "H160_32_ORIGINAL_APP_MODEL_NATIVE_TASKS.json"


def preflight(owner, upstream, model_root):
    owner = Path(owner)
    source_zip = owner / "private-h160" / "kaggle-eval-v1" / SOURCE_ZIP
    cases = validate(source_zip)
    snapshot = json.loads((
        owner / "private-h159" / "H159_SOURCE_REPLAY_QA.json"
    ).read_text("utf8"))
    if (snapshot.get("status") !=
        "H159_16_OF_16_INDEPENDENT_SCREENSHOT_SHA_MATCH"):
        raise ValueError("H160 cannot score without pre-inference 16/16 pixel replay")
    path = Path(model_root) / PRED
    doc = json.loads(path.read_text("utf8"))
    if (doc.get("schema") !=
        "hala.h160.private-stable-exploratory-gpu-inference.v1"
        or doc.get("status") != "H160_COMPLETED_32_REAL_GPU_CALLS"
        or doc.get("gpu_calls") != 32
        or doc.get("source_manifest_sha256") != MANIFEST_SHA
        or doc.get("source_zip_sha256") != SOURCE_ZIP_SHA
        or doc.get("frozen_old_lora_sha256") != SHA_H156
        or doc.get("frozen_new_lora_sha256") != SHA_H157
        or doc.get("optimizer_steps") != 0
        or doc.get("h140_blind_calls") != 0
        or doc.get("paid_gpu") is not False):
        raise ValueError("H160 GPU/data/model provenance incomplete")
    expected = {
        (c["image_sha256"], role) for c in cases
        for role in ("before", "after")
    }
    predictions = {}
    rows = doc.get("private_predictions")
    if not isinstance(rows, list) or len(rows) != 32:
        raise ValueError("H160 missing one of 32 paired GPU predictions")
    for row in rows:
        if (not isinstance(row, dict) or
            set(row) != {"role","task","language","image_sha","raw","model_action"} or
            (row["image_sha"],row["role"]) not in expected or
            (row["image_sha"],row["role"]) in predictions or
            validated_model_action(row["raw"]) != row["model_action"]):
            raise ValueError("H160 GPU model action tampering/duplication")
        predictions[(row["image_sha"],row["role"])] = row
    if set(predictions) != expected:
        raise ValueError("H160 frozen 16x2 image pair incomplete")
    for case in cases:
        for role in ("before", "after"):
            p = predictions[(case["image_sha256"],role)]
            if p["task"] != case["task"] or p["language"] != case["language"]:
                raise ValueError("H160 task/language/image pairing drift")
    return source_url(upstream), cases, predictions


def episode(browser, url, case, action):
    ctx = browser.new_context(
        viewport={"width":960,"height":720},
        device_scale_factor=1, locale="en-US",
    )
    try:
        page = ctx.new_page()
        page.goto(url, wait_until="load", timeout=20000)
        items = tasks_for(case["language"], case["index"])
        for item in items:
            field = page.locator("input.new-todo")
            field.fill(item)
            field.press("Enter")
        if page.locator("ul.todo-list li").count() != 3:
            raise ValueError("H160 original app seed invalid")
        if case["task"] != "complete_item":
            page.locator("ul.todo-list li").filter(
                has_text=items[0]
            ).locator("input.toggle").click()
            page.wait_for_function(
                "() => document.querySelectorAll('ul.todo-list li.completed').length === 1",
                timeout=5000
            )
        before = digest(stable_png(page))
        if before != case["image_sha256"]:
            raise ValueError("H160 model would act on a different original screenshot")
        sent = False
        click = action["click"]
        if action["valid"] and click is not None and not action["abstain"]:
            # Only the model's click, never owner source/oracle coordinates.
            page.mouse.click(*click)
            sent = True
            page.wait_for_timeout(75)
        if case["task"] == "complete_item":
            correct = (page.locator("ul.todo-list li.completed").filter(
                has_text=items[1]).count() == 1)
        elif case["task"] == "filter_active":
            correct = (
                page.url.endswith("#/active") and
                page.locator("ul.todo-list li.completed").count() == 0 and
                page.locator("ul.todo-list li").count() == 2
            )
        elif case["task"] == "clear_completed":
            correct = (
                page.locator("ul.todo-list li").filter(
                    has_text=items[0]).count() == 0 and
                page.locator("ul.todo-list li").count() == 2
            )
        else:
            raise ValueError("H160 unsupported original UI action")
        return bool(correct and sent), sent
    finally:
        ctx.close()


def run(owner, upstream, model_root):
    owner = Path(owner)
    url, cases, predictions = preflight(owner,upstream,model_root)
    root = owner / "private-h160"
    consumed = root / LOCK
    result = root / RESULT
    if consumed.exists() or result.exists():
        raise FileExistsError("H160 16-case native evaluation is one-shot consumed")
    with consumed.open("x", encoding="utf8") as f:
        f.write(json.dumps({
            "status":"H160_REAL_NATIVE_MODEL_SCORING_CONSUMED",
            "source_manifest_sha256":MANIFEST_SHA,
            "source_zip_sha256":SOURCE_ZIP_SHA,
            "adapter_shas":{"before":SHA_H156,"after":SHA_H157},
            "h140_blind_calls":0
        },sort_keys=True)+"\n")
    from playwright.sync_api import sync_playwright
    rows = []
    with sync_playwright() as pw:
        opts = {"headless":True}
        chrome = _chromium_binary()
        if chrome:
            opts["executable_path"] = chrome
        browser = pw.chromium.launch(**opts,timeout=20000)
        try:
            for case in cases:
                for role in ("before","after"):
                    proposal = predictions[(case["image_sha256"],role)]["model_action"]
                    passed,sent = episode(browser,url,case,proposal)
                    rows.append({
                        "role":role,"task":case["task"],
                        "language":case["language"],"image_sha":case["image_sha256"],
                        "valid":proposal["valid"],"abstained":proposal["abstain"],
                        "click_sent":sent,"original_app_task_success":passed,
                    })
                if len(rows) % 8 == 0:
                    print("H160_REAL_NATIVE_MODEL_TASK",len(rows),32,flush=True)
        finally:
            browser.close()
    score = paired(rows,cases)
    doc = {
        "status":"H160_COMPLETED_32_REAL_ORIGINAL_APP_MODEL_TASKS",
        "source_manifest_sha256":MANIFEST_SHA,
        "native_browser_trials":32,"model_gpu_calls":32,
        "optimizer_steps":0,"h140_blind_calls":0,"paid_gpu":False,
        "original_app_memory_only":True,
        "cross_app_exploratory_not_blind":True,
        "Hala35B_unchanged":True,"paired":score,
        "private_cases":rows,
    }
    result.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n","utf8")
    return {k:v for k,v in doc.items() if k!="private_cases"}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner",required=True)
    p.add_argument("--upstream",required=True)
    p.add_argument("--model-root",required=True)
    args=p.parse_args()
    print(json.dumps(run(args.owner,args.upstream,args.model_root),
                     indent=2,sort_keys=True))
