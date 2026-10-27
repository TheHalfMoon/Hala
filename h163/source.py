"""H163: genuinely new original MIT GridStack dashboard, never used in Hala.

Owner-only native screenshot capture/independent pixel replay on two task types,
eight pre-action layouts. The official untouched demo/column.html and npm v14.0.0
dist are pinned and remain outside GitHub. NO training or model calls here.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from h130.browser_env import _chromium_binary
from h138.probe import serve, center
from h159.stable_source_qa import stable_png

SCHEMA="hala.h163.original-gridstack-v14-new-application-ood.v1"
GIT_SHA="b05b79dcbca21caca046c1e8c4da6e59546ec806"
NPM_VERSION="14.0.0"
NPM_ARCHIVE_SHA="c34298b66f84418a1c4e5130ad2fb6185ac1dea05c54e4223d64cefb2d524c2e"
PRIVATE="private-h163/source-once-v1"
FIRST=163001
N=16
LAYOUTS=("list","case 1")
BEFORE_COLUMN={"set_column":(2,4,6,8),"add_widget":(1,3,10,12)}
AFTER_COLUMN=(10,12,3,1)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def specs():
    result=[]
    for task in ("set_column","add_widget"):
        for layout in LAYOUTS:
            for i,pre in enumerate(BEFORE_COLUMN[task]):
                result.append({
                    "index":FIRST+len(result), "app":"gridstack-original-column-demo",
                    "task":task,"language":"en" if i%2==0 else "ar",
                    "layout":layout,"pre_column":pre,
                    "target_column":AFTER_COLUMN[i] if task=="set_column" else None,
                })
    if len(result)!=N or len({s["index"] for s in result})!=N:
        raise ValueError("H163 original app source split drift")
    return result


def admit(root, owner):
    root=Path(root); owner=Path(owner)
    git=subprocess.check_output(
        ["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(
        ["git","-C",str(root),"status","--porcelain"],text=True).strip()
    if git!=GIT_SHA or dirty:
        raise ValueError("H163 original open-source demo git SHA/source modified")
    if "MIT License" not in (root/"LICENSE").read_text("utf8"):
        raise ValueError("H163 original LICENSE is not MIT")
    archive=owner/"private-h163"/"gridstack-14.0.0.tgz"
    if sha(archive.read_bytes())!=NPM_ARCHIVE_SHA:
        raise ValueError("H163 published original npm v14 dependency mutated")
    if not (root/"dist"/"gridstack-all.js").is_file():
        raise ValueError("H163 original MIT demo missing its pinned official runtime")
    if "onClick" not in (root/"demo"/"column.html").read_text("utf8"):
        raise ValueError("H163 original demo controls changed")
    return root


def exclusions(owner):
    owner=Path(owner)
    prior=set()
    for group in ("private-h158/todomvc-source-once-v1",
                  "private-h159/source-once-v1",
                  "private-h161/two-real-app-source-v1",
                  "private-h162/original-html5-board-source-once-v1"):
        p=owner/group/"manifest.json"
        if not p.is_file():
            raise ValueError("H163 cannot check earlier frozen original-app work")
        doc=json.loads(p.read_text("utf8"))
        prior.update(x["image_sha256"] for x in doc["private_cases"])
    if len(prior)<100:
        raise ValueError("H163 historic original screenshots provenance incomplete")
    return prior


def instruction(case):
    if case["task"]=="set_column":
        n=case["target_column"]
        return (f"Click the {n} column layout button."
                if case["language"]=="en"
                else f"اضغط على زر تخطيط الأعمدة رقم {n}.")
    return ("Click Add Widget to create another dashboard tile."
            if case["language"]=="en"
            else "اضغط على زر إضافة عنصر Add Widget إلى لوحة المعلومات.")


def episode(browser,url,case,*,execute):
    context=browser.new_context(
        viewport={"width":960,"height":720},device_scale_factor=1,
        locale="en-US",service_workers="block")
    try:
        page=context.new_page()
        errors=[]
        page.on("pageerror",lambda error:errors.append(str(error)))
        page.route("https://**/*",lambda route:route.abort())
        page.goto(url,wait_until="load",timeout=20000)
        page.wait_for_function(
            "() => typeof GridStack === 'function' && "
            "typeof grid !== 'undefined' && grid.opts.column===12",
            timeout=10000)
        setup=('"grid.removeAll().load(list)"'
               if case["layout"]=="list" else
               '"grid.removeAll().load(test1)"')
        page.locator("a[onclick="+setup+"]").click()
        page.locator(f'a[onclick="column({case["pre_column"]})"]').click()
        page.wait_for_function(
            "(number) => grid.opts.column===number",arg=case["pre_column"],timeout=5000)
        if errors:
            raise ValueError("H163 untouched MIT app JavaScript error: "+errors[0][:150])
        png=stable_png(page)
        if not execute:
            return png,None,None
        if case["task"]=="set_column":
            target=f'a[onclick="column({case["target_column"]})"]'
            locator=page.locator(target)
            point=center(locator)
            page.mouse.click(*point)
            correct=(page.evaluate("() => grid.opts.column")==case["target_column"]
                     and page.locator("#column-text").inner_text().strip()
                     ==str(case["target_column"]))
        else:
            prev=int(page.evaluate("() => grid.save().length"))
            locator=page.locator('a[onclick="addWidget()"]')
            point=center(locator)
            page.mouse.click(*point)
            correct=(int(page.evaluate("() => grid.save().length"))==prev+1)
        if not correct:
            raise ValueError("H163 original GridStack real native mouse click unsuccessful")
        return png,instruction(case),{
            "actions":[{"kind":"click","point":point}]}
    finally:
        context.close()


def run(owner,original,mode):
    owner=Path(owner)
    original=admit(original,owner)
    old=exclusions(owner)
    root=owner/PRIVATE
    if mode=="capture" and root.exists():
        raise FileExistsError("H163 original new application source create-once")
    if mode=="verify" and not (root/"manifest.json").is_file():
        raise FileNotFoundError("H163 no previously qualified frozen app source")
    from playwright.sync_api import sync_playwright
    rows=[]
    with sync_playwright() as pw:
        opts={"headless":True}
        exe=_chromium_binary()
        if exe:
            opts["executable_path"]=exe
        browser=pw.chromium.launch(**opts,timeout=20000)
        try:
            with serve(original) as base:
                url=base+"/demo/column.html"
                cases=specs()
                if mode=="smoke":
                    cases=[cases[0],cases[8]]
                if mode=="capture":
                    root.mkdir(parents=True,exist_ok=False)
                    (root/"images").mkdir()
                if mode=="verify":
                    frozen=json.loads((root/"manifest.json").read_text("utf8"))
                    if frozen.get("schema")!=SCHEMA or len(frozen["private_cases"])!=N:
                        raise ValueError("H163 frozen original app manifest changed")
                for n,case in enumerate(cases,1):
                    image,text,gold=episode(browser,url,case,execute=mode!="verify")
                    digest=sha(image)
                    if digest in old:
                        raise ValueError("H163 original screenshot collides with old app")
                    if mode=="capture":
                        if digest in {x["image_sha256"] for x in rows}:
                            raise ValueError("H163 new application scene duplicated")
                        filename=f"gridstack-{case['index']}.png"
                        (root/"images"/filename).write_bytes(image)
                        rows.append({
                            **case,"image":filename,"image_sha256":digest,
                            "instruction":text,"owner_gold_click":gold,
                            "original_native_reference_success":True
                        })
                    elif mode=="verify":
                        item=frozen["private_cases"][n-1]
                        filename=root/"images"/item["image"]
                        if (item["image_sha256"]!=digest or
                            item["index"]!=case["index"] or
                            filename.read_bytes()!=image):
                            raise ValueError(
                                f"H163 frozen original app image mismatch case {n}")
                    print("H163_"+mode.upper()+"_NEW_APP",n,len(cases),
                          case["task"],case["language"],flush=True)
        finally:
            browser.close()
    if mode=="smoke":
        return {"status":"H163_ORIGINAL_MIT_GRIDSTACK_TWO_NATIVE_SMOKE_PASS",
                "source_reference_actions":2,"model_calls":0}
    if mode=="verify":
        proof={"status":"H163_16_OF_16_ORIGINAL_NEW_APP_EXACT_PIXEL_REPLAY_PASS",
               "new_application":"gridstack.js official original column demo",
               "original_demo_unmodified":True,
               "model_calls":0,"h140_blind_calls":0,"paid_GPU":False}
        filename=owner/"private-h163"/"H163_SOURCE_PIXEL_QA.json"
        if filename.exists():
            raise FileExistsError("H163 original independent replay one-shot consumed")
        filename.write_text(json.dumps(proof,indent=2,sort_keys=True)+"\n","utf8")
        return proof
    if len(rows)!=N or len({x["image_sha256"] for x in rows})!=N:
        raise ValueError("H163 actual native original app source count invalid")
    doc={"schema":SCHEMA,"upstream_git_sha":GIT_SHA,
         "upstream_npm_version":NPM_VERSION,
         "upstream_npm_zip_sha256":NPM_ARCHIVE_SHA,
         "source_screenshot_count":N,"native_reference_success":16,
         "original_app_not_ever_used_as_Hala_training":True,
         "heldout_evaluation_only":True,"model_calls":0,
         "h140_blind_calls":0,"paid_GPU":False,
         "private_cases":rows}
    manifest=(json.dumps(doc,sort_keys=True,indent=2,ensure_ascii=False)+"\n").encode()
    (root/"manifest.json").write_bytes(manifest)
    return {"status":"H163_NEW_REAL_UPSTREAM_GRIDSTACK_SOURCE_16_PASS",
            "source_manifest_sha256":sha(manifest),
            "native_reference_success":16,"model_calls":0}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("smoke","capture","verify"))
    p.add_argument("--owner",required=True)
    p.add_argument("--original",required=True)
    a=p.parse_args()
    print(json.dumps(run(a.owner,a.original,a.mode),indent=2,sort_keys=True))
