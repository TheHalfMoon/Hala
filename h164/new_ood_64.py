"""H164 untouched two original MIT app families reserved for independent OOD.

Pell editor and Pikaday date picker. Source-reference mouse clicks only;
gold stored OWNER-PRIVATE in separate file and omitted from model input.
"""
import argparse,json,subprocess
from pathlib import Path
from contextlib import ExitStack
from datetime import datetime,timezone
from h130.browser_env import _chromium_binary
from h138.probe import center,serve
from h159.stable_source_qa import stable_png
from h164.expand_288_train import digest,prior

SCHEMA="hala.h164.new-two-mit-ood-original-apps.v1"
PRIVATE="private-h164/new-two-app-ood-64-v1"
APPS=("pell","pikaday")
PINS={"pell":"d5556baa2e0bfc1411621ee4d780752200b367fd",
      "pikaday":"9a90eea5cdbc48784d3a909ddb36e052c16bf25d"}
TAGS={"Bold":"b","Italic":"i","Underline":"u","Strike-through":"strike"}

def sources(owner):
    root=Path(owner)/"private-h164/upstream-ood"
    for app in APPS:
        dir=root/app
        head=subprocess.check_output(
            ["git","-C",str(dir),"rev-parse","HEAD"],text=True).strip()
        dirty=subprocess.check_output(
            ["git","-C",str(dir),"status","--porcelain"],text=True).strip()
        license=(dir/("LICENSE.md" if app=="pell" else "LICENSE")).read_text("utf8")
        if head!=PINS[app] or dirty or "MIT" not in license:
            raise ValueError("H164 new OOD app license or original source mutated")
    return root

def cases():
    rows=[]
    for app in APPS:
        for n in range(32):
            rows.append({
                "split":"OOD","app":app,
                "task":"format_text" if app=="pell" else "select_calendar_day",
                "language":"en" if n%2==0 else "ar",
                "index":174001+(APPS.index(app)*1000)+n,
                "local_index":n
            })
    return rows

def native_pell(browser,url,case,execute):
    ctx=browser.new_context(viewport={"width":960,"height":720},device_scale_factor=1,
                            locale="en-US")
    try:
        page=ctx.new_page()
        page.goto(url,wait_until="domcontentloaded",timeout=15000)
        editor=page.locator(".pell-content")
        editor.wait_for(timeout=8000)
        message=f"Review OOD case {case['index']} now"
        editor.fill(message)
        editor.press("ControlOrMeta+A")
        title=list(TAGS)[case["local_index"]%4]
        target=page.locator(f'.pell-actionbar button[title="{title}"]')
        if target.count()!=1:raise ValueError("H164 real Pell original toolbar changed")
        screenshot=stable_png(page)
        if not execute:return screenshot,None,None
        point=center(target)
        page.mouse.click(*point)
        after=editor.inner_html().lower()
        tag=TAGS[title]
        # The original Pell browser uses <strike> or <s> based on document.execCommand.
        success=message.lower() in after and (
            f"<{tag}>" in after or
            (tag=="strike" and "<s>" in after))
        if not success:raise ValueError(f"H164 native Pell formatting failed {title}")
        en=f"Apply {title} formatting to the selected text."
        ar=f"طبّق تنسيق {title} على النص المحدد في المحرر."
        return screenshot,(en if case["language"]=="en" else ar),{
            "actions":[{"kind":"click","point":point}]}
    finally:ctx.close()

def native_pikaday(browser,url,case,execute):
    ctx=browser.new_context(viewport={"width":960,"height":720},device_scale_factor=1,
                            locale="en-US")
    try:
        page=ctx.new_page()
        page.clock.install(time=datetime(2019,1,1,tzinfo=timezone.utc))
        page.goto(url,wait_until="domcontentloaded",timeout=15000)
        page.wait_for_function("() => typeof picker === 'object'",timeout=8000)
        field=page.locator("#datepicker")
        month=case["local_index"]//4
        day=8+(case["local_index"]%4)*3
        page.evaluate("(date)=>picker.setDate(new Date(2019,date.month,date.day))",
                      {"month":month,"day":day})
        seed=field.input_value()
        field.click()
        target_day=day+1
        target=page.locator(
            f'.pika-single button.pika-button[data-pika-day="{target_day}"]')
        target.first.wait_for(timeout=8000)
        if target.count()!=1:
            raise ValueError("H164 original Pikaday selected month's day not unique")
        screenshot=stable_png(page)
        if not execute:return screenshot,None,None
        point=center(target)
        page.mouse.click(*point)
        after=field.input_value()
        result=page.evaluate("picker.getDate().getDate()")
        if after==seed or result!=target_day:
            raise ValueError("H164 actual original Pikaday calendar selection failed")
        en=f"Select day {target_day} on the open date picker."
        ar=f"اختر اليوم {target_day} من التقويم المفتوح."
        return screenshot,(en if case["language"]=="en" else ar),{
            "actions":[{"kind":"click","point":point}]}
    finally:ctx.close()

def old_hashes(owner):
    owner=Path(owner)
    old=prior(owner)
    proof=json.loads((owner/"private-h164/H164_NEW_DEV_PIXEL_QA.json").read_text("utf8"))
    if proof.get("status")!="H164_DEV_96_ORIGINAL_APP_EXACT_PIXEL_REPLAY":
        raise ValueError("H164 no OOD before independently qualified 96 DEV")
    for path in (
        "private-h164/additional-three-app-288-train-v1",
        "private-h164/new-dev-96-v1"
    ):
        p=owner/path/"manifest.json"
        if not p.exists():
            raise ValueError("H164 OOD can only freeze after TRAIN/DEV sources")
        old.update(r["image_sha256"] for r in json.loads(p.read_text("utf8"))["private_cases"])
    return old

def run(owner,mode):
    owner=Path(owner);root=sources(owner)
    older=old_hashes(owner) if mode!="smoke" else prior(owner)
    dest=owner/PRIVATE
    if mode=="capture" and dest.exists():
        raise FileExistsError("H164 never reuse OOD source capture")
    frozen=json.loads((dest/"manifest.json").read_text("utf8")) if mode=="verify" else None
    if frozen and (frozen.get("schema")!=SCHEMA or len(frozen.get("private_cases",[]))!=64
                   or frozen.get("gold_model_inputs") is not False):
        raise ValueError("H164 sealed original OOD source changed")
    entries=cases()
    if mode=="smoke":
        entries=[entries[i] for i in (0,1,2,3,32,33)]
    from playwright.sync_api import sync_playwright
    collected=[];gold=[]
    with sync_playwright() as pw:
        config={"headless":True}
        chrome=_chromium_binary()
        if chrome:config["executable_path"]=chrome
        browser=pw.chromium.launch(**config,timeout=20000)
        try:
            with ExitStack() as scope:
                urls={app:scope.enter_context(serve(root/app)) for app in APPS}
                if mode=="capture":
                    dest.mkdir(parents=True,exist_ok=False)
                    (dest/"images").mkdir()
                for n,c in enumerate(entries,1):
                    if c["app"]=="pell":
                        png,task,reference=native_pell(
                            browser,urls["pell"]+"/demo.html",c,mode!="verify")
                    else:
                        png,task,reference=native_pikaday(
                            browser,urls["pikaday"]+"/index.html",c,mode!="verify")
                    h=digest(png)
                    if h in older:raise ValueError("H164 consumed original benchmark leaked")
                    if mode=="capture":
                        if any(row["image_sha256"]==h for row in collected):
                            raise ValueError("H164 OOD duplicate scene")
                        name=f'ood-{c["app"]}-{c["index"]}.png'
                        (dest/"images"/name).write_bytes(png)
                        collected.append({**c,"image":name,
                                          "image_sha256":h,
                                          "instruction":task,
                                          "action_type":reference["actions"][0]["kind"]})
                        gold.append({"image_sha256":h,
                                     "native_reference_success":True,
                                     "owner_supervision":reference})
                    elif mode=="verify":
                        expected=frozen["private_cases"][n-1]
                        if (any(expected.get(k)!=v for k,v in c.items()) or
                            expected["image_sha256"]!=h or
                            (dest/"images"/expected["image"]).read_bytes()!=png):
                            raise ValueError("H164 original OOD source browser image drift")
                    if n%8==0 or n==len(entries):
                        print("H164_OOD_"+mode.upper(),n,len(entries),c["app"],flush=True)
        finally:browser.close()
    if mode=="smoke":
        return {"status":"H164_TWO_NEW_OOD_ORIGINAL_APPS_NATIVE_SMOKE_PASS",
                "source_reference_tests":6,"model_calls":0}
    if mode=="verify":
        result={"status":"H164_OOD_64_NEW_APP_EXACT_BROWSER_PIXEL_REPLAY",
                "new_apps":2,"cases":64,"owner_gold_uploaded":False,
                "model_calls":0,"optimizer_steps":0,"paid_GPU":False}
        proof=owner/"private-h164/H164_OOD_64_PIXEL_QA.json"
        if proof.exists():raise FileExistsError("H164 new OOD browser replay already used")
        proof.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n","utf8")
        return result
    if len(collected)!=64 or len({x["image_sha256"] for x in collected})!=64:
        raise ValueError("H164 frozen true-original OOD incomplete")
    manifest={"schema":SCHEMA,"ood_only":True,"gold_model_inputs":False,
              "cases":64,"apps":list(APPS),"git_source_shas":PINS,
              "optimizer_steps":0,"model_calls":0,
              "private_cases":collected}
    raw=(json.dumps(manifest,indent=2,sort_keys=True,ensure_ascii=False)+"\n").encode()
    (dest/"manifest.json").write_bytes(raw)
    (dest/"owner-gold-not-for-model.json").write_text(
        json.dumps({"source_manifest_sha256":digest(raw),
                    "reference_actions":gold},
                   indent=2,sort_keys=True,ensure_ascii=False)+"\n","utf8")
    return {"status":"H164_64_GENUINE_NEW_ORIGINAL_OOD_CASES_CAPTURED",
            "manifest_sha256":digest(raw),"references":64,
            "gpu_model_calls":0,"gold_model_inputs":False}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("smoke","capture","verify"))
    parser.add_argument("--owner",required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.owner,args.mode),indent=2,sort_keys=True))
