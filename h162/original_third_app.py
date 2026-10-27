"""H162 fresh original MIT HTML5 Kanban unseen-at-training OOD source.

This third app was evaluated historically (H138), so it is **NOT** a
fully sealed fresh-app BLIND. 16 newly seeded untouched original app tasks
for one frozen exploratory H161 cross-app transfer gate only.
"""
import argparse
import hashlib
import json
from pathlib import Path

from h130.browser_env import _chromium_binary
from h138.probe import APPS, center, mouse_drag, serve
from h138.source import inspect_repo
from h159.stable_source_qa import stable_png

APP="react-html5-board"
PIN=APPS[APP]
SCHEMA="hala.h162.third-original-html5-board-exploratory-ood.v1"
PRIVATE="private-h162/original-html5-board-source-once-v1"
CASE_COUNT=16
FIRST_ID=192001


def sha(data):
    return hashlib.sha256(data).hexdigest()


def old_shas(owner):
    owner=Path(owner)
    old=set()
    for f in (owner/"private-h138"/"real-untouched-app8").glob("*.png"):
        old.add(sha(f.read_bytes()))
    h161=json.loads((owner/"private-h161"/"two-real-app-source-v1"/
                     "manifest.json").read_text("utf8"))
    old.update(r["image_sha256"] for r in h161["private_cases"])
    if len(old)<80:
        raise ValueError("H162 incomplete prior application image exclusion")
    return old


def spec(n):
    index=FIRST_ID+n
    return {"index":index,"task":"drag_backlog_to_todo",
            "language":"en" if n%2==0 else "ar",
            "app":APP}


def state_for(case):
    index=case["index"]
    title=f"Validate integration issue {index}"
    return title, [
        {"id":"backlog","title":"Backlog","cards":[
            {"id":"h162-"+str(index),"title":title,
             "description":"Review and finalize","priority":"medium"}]},
        {"id":"todo","title":"To Do","cards":[
            {"id":"todo-other","title":"QA daily report",
             "description":"","priority":"high"}]},
        {"id":"in-progress","title":"In Progress","cards":[]},
        {"id":"done","title":"Done","cards":[]},
    ]


def prompt(case,title):
    if case["language"]=="ar":
        return f"اسحب بطاقة «{title}» من عمود Backlog إلى عمود To Do."
    return f"Drag the '{title}' card from Backlog to To Do."


def episode(browser,url,case,*,execute):
    title,state=state_for(case)
    ctx=browser.new_context(viewport={"width":960,"height":720},
                            device_scale_factor=1,service_workers="block",
                            locale="en-US")
    ctx.add_init_script(
        "localStorage.setItem('kanban',"+
        json.dumps(json.dumps(state,ensure_ascii=False))+");")
    try:
        page=ctx.new_page()
        page.goto(url,wait_until="networkidle",timeout=20000)
        source=page.locator('[draggable="true"]').filter(has_text=title).first
        source.wait_for(timeout=8000)
        target=page.locator(".grid.grid-cols-4 > div").nth(1)
        picture=stable_png(page)
        if not execute:
            return picture,None,None
        start=center(source)
        end=center(target)
        mouse_drag(page,start,end)
        updated=json.loads(page.evaluate(
            "(k)=>localStorage.getItem(k)","kanban"))
        if not any(c.get("id")=="h162-"+str(case["index"])
                   for c in updated[1]["cards"]):
            raise ValueError("H162 genuine original HTML5 native drag failed")
        return picture,prompt(case,title),{
            "actions":[{"kind":"drag","start":start,"end":end}]}
    finally:
        ctx.close()


def run(owner,original,mode):
    owner=Path(owner)
    original=Path(original)
    if inspect_repo(original,PIN)!=PIN:
        raise ValueError("H162 original independent app upstream not pinned")
    root=owner/PRIVATE
    if mode=="capture" and root.exists():
        raise FileExistsError("H162 original new-app OOD source only once")
    if mode=="verify" and not (root/"manifest.json").is_file():
        raise FileNotFoundError("H162 original frozen source absent")
    excluded=old_shas(owner)
    from playwright.sync_api import sync_playwright
    rows=[]
    with sync_playwright() as playwright:
        opts={"headless":True}
        binary=_chromium_binary()
        if binary:
            opts["executable_path"]=binary
        browser=playwright.chromium.launch(**opts,timeout=20000)
        try:
            with serve(original/"dist") as url:
                if mode=="capture":
                    root.mkdir(parents=True,exist_ok=False)
                    (root/"images").mkdir()
                samples=[spec(i) for i in range(2 if mode=="smoke" else CASE_COUNT)]
                frozen=None
                if mode=="verify":
                    frozen=json.loads((root/"manifest.json").read_text("utf8"))
                    if frozen["schema"]!=SCHEMA or len(frozen["private_cases"])!=16:
                        raise ValueError("H162 unseen-app source registry changed")
                for n,case in enumerate(samples,1):
                    media,instruction,gold=episode(
                        browser,url,case,execute=mode!="verify")
                    digest=sha(media)
                    if digest in excluded:
                        raise ValueError("H162 prior source application image was reused")
                    if mode=="verify":
                        item=frozen["private_cases"][n-1]
                        if (item["image_sha256"]!=digest or
                            item["image"]!=f"case-{case['index']}.png" or
                            (root/"images"/item["image"]).read_bytes()!=media):
                            raise ValueError("H162 original external-app PNG not replay exact")
                    elif mode=="capture":
                        if digest in {r["image_sha256"] for r in rows}:
                            raise ValueError("H162 original app duplicate screenshot")
                        name=f"case-{case['index']}.png"
                        (root/"images"/name).write_bytes(media)
                        rows.append({
                            **case,"image":name,"image_sha256":digest,
                            "instruction":instruction,
                            "owner_reference_action":gold,
                            "native_original_app_success":True,
                        })
                    print("H162_"+mode.upper()+"_ORIGINAL_HTML5_APP",n,len(samples),
                          flush=True)
        finally:
            browser.close()
    if mode=="smoke":
        return {"status":"H162_THIRD_APP_NATIVE_SMOKE_PASS","cases":2}
    if mode=="verify":
        receipt={"status":"H162_16_EXACT_ORIGINAL_THIRD_APP_REPLAY_PASS",
                 "model_calls":0,"app_not_used_in_H161_TRAIN":True,
                 "fully_unseen_app_BLIND":False,
                 "h140_blind_calls":0,"paid_GPU":False}
        filename=owner/"private-h162"/"H162_SOURCE_PIXEL_QA.json"
        if filename.exists():
            raise FileExistsError("H162 owner one-shot source QA already recorded")
        filename.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n","utf8")
        return receipt
    if len(rows)!=16 or len({r["image_sha256"] for r in rows})!=16:
        raise ValueError("H162 new independent app source generation incomplete")
    source_doc={
        "schema":SCHEMA,"upstream_app":APP,"upstream_sha":PIN,
        "source_task_count":16,"native_reference_success":16,
        "training_data":False,"model_calls":0,"h140_blind_calls":0,
        "fully_sealed_BLIND":False,"historically_seen_app":True,
        "private_cases":rows}
    raw=(json.dumps(source_doc,sort_keys=True,indent=2,ensure_ascii=False)+"\n").encode()
    (root/"manifest.json").write_bytes(raw)
    return {"status":"H162_THIRD_ORIGINAL_HTML5_OOD_SOURCE_16_PASS",
            "source_manifest_sha256":sha(raw),
            "native_reference_success":16,"model_calls":0,
            "historically_seen_app_not_BLIND":True}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("smoke","capture","verify"))
    p.add_argument("--owner",required=True)
    p.add_argument("--original",required=True)
    args=p.parse_args()
    print(json.dumps(run(args.owner,args.original,args.mode),
                     ensure_ascii=False,indent=2,sort_keys=True))
