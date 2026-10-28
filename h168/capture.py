"""Actual Chromium Playwright capture/replay on unchanged, licensed original demos.

Private TRAIN_PILOT ONLY. No model, no DEV/OOD, no sealed cases.
"""
import argparse
import json
from pathlib import Path

from h130.browser_env import _chromium_binary
from h138.probe import center,mouse_drag,serve
from h159.stable_source_qa import stable_png
from h168.source import (
    PINS,SCHEMA,APPS_FOR_PILOT,cases,historical_image_shas,
    sha,verify_sources,validate_pilot)

def navigate(browser,url,case,execute=True):
    ctx=browser.new_context(viewport={"width":960,"height":720},
                            device_scale_factor=1,locale="en-US",
                            service_workers="block")
    try:
        page=ctx.new_page()
        app=case["app"]
        if app=="choices":
            target=url+"choices/public/test/text/index.html"
        elif app=="nouislider":
            target=url+"nouislider/tests/origin.html"
        else:
            raise ValueError("H168 app not preregistered TRAIN PILOT")
        response=page.goto(target,wait_until="load",timeout=25000)
        if response is None or response.status!=200:
            raise ValueError("H168 original upstream demo HTTP response invalid")
        n=case["variant"]
        if app=="choices":
            hook=("basic","edit-items","remove-button","unique-values")[n%4]
            host=page.locator('[data-test-hook="'+hook+'"]')
            widget=host.locator(".choices__input--cloned").first
            widget.wait_for(timeout=10000)
            host.scroll_into_view_if_needed()
            seed="halaseed"+str(168000+n)
            widget.click()
            widget.fill(seed)
            widget.press("Enter")
            if host.locator(".choices__item[data-item]").filter(
                    has_text=seed).count()!=1:
                raise ValueError("H168 upstream Choices native tag seed did not persist")
            screenshot=stable_png(page)
            point=center(widget)
            text="halanew"+str(168000+n)
            instruction=(
              "Type "+text+" as a new tag in the "+hook+" field and press Enter."
              if case["language"]=="en" else
              "اكتب "+text+" كوسم جديد في حقل "+hook+" ثم اضغط Enter."
            )
            gold=dict(kind="type",point=point,text=text)
            if execute:
                page.mouse.click(*point)
                page.keyboard.type(text)
                page.keyboard.press("Enter")
                good=host.locator(".choices__item[data-item]").filter(
                    has_text=text).count()==1
                if not good:
                    raise ValueError("H168 original Choices tag action failed")
            else:
                good=None
        else:
            slider1=2+(n%5)
            slider2=2+(n//5)
            page.locator("#slider1 .noUi-handle").wait_for(timeout=10000)
            page.evaluate("([a,b]) => { slider1.noUiSlider.set(a); slider2.noUiSlider.set(b); }",
                          [slider1,slider2])
            orientation="horizontal" if n%2==0 else "vertical"
            selector="#slider1" if orientation=="horizontal" else "#slider2"
            handle=page.locator(selector+" .noUi-handle").first
            screenshot=stable_png(page)
            start=center(handle)
            end=[start[0]+48,start[1]] if orientation=="horizontal" else [
                 start[0],start[1]+48]
            if not all(0<=v<960 for v in (start[0],end[0])) or not all(
                    0<=v<720 for v in (start[1],end[1])):
                raise ValueError("H168 original slider drag not on-screen")
            instruction=(
               "Drag the "+orientation+" slider handle to increase its value."
               if case["language"]=="en" else
               "اسحب مقبض شريط التمرير "+(
                    "الأفقي" if orientation=="horizontal" else "الرأسي")+
               " لزيادة قيمته."
            )
            gold=dict(kind="drag",start=start,end=end)
            if execute:
                initial=float(page.evaluate(
                    "(sel) => document.querySelector(sel).noUiSlider.get()",selector))
                mouse_drag(page,start,end)
                after=float(page.evaluate(
                    "(sel) => document.querySelector(sel).noUiSlider.get()",selector))
                good=after>initial+0.5
                if not good:
                    raise ValueError("H168 original noUiSlider real drag did not increase")
            else:
                good=None
        return screenshot,instruction,gold,good
    finally:
        ctx.close()

def run(owner,upstream,destination,mode):
    from playwright.sync_api import sync_playwright
    owner=Path(owner);upstream=Path(upstream);destination=Path(destination)
    qualified=verify_sources(upstream)
    historical=historical_image_shas(owner)
    schedule=cases()
    if mode=="smoke":
        schedule=[schedule[0],schedule[1],schedule[16],schedule[17]]
    if mode=="capture" and destination.exists():
        raise FileExistsError("H168 owner-private one-shot TRAIN PILOT already exists")
    manifest={"schema":SCHEMA,"split":"TRAIN_PILOT","count":len(schedule),
              "input_gold_separated":True,"upstreams":qualified,
              "private_cases":[]}
    private={"schema":"hala.h168.owner-private-train-actions.v1",
             "private_training_actions":[]}
    screenshots={}
    with serve(upstream) as url:
        with sync_playwright() as pw:
            options={"headless":True}
            exe=_chromium_binary()
            if exe:
                options["executable_path"]=exe
            browser=pw.chromium.launch(**options)
            try:
                for case in schedule:
                    if mode=="verify":
                        existing=json.loads((destination/"manifest.json").read_text("utf8"))
                        source=next((x for x in existing["private_cases"]
                                    if x["index"]==case["index"]),None)
                        if source is None:
                            raise ValueError("H168 verified source case not found")
                    png,instruction,gold,good=navigate(browser,url,case,execute=mode!="verify")
                    image_sha=sha(png)
                    if image_sha in historical or image_sha in (
                        v["image_sha256"] for v in manifest["private_cases"]):
                        raise ValueError("H168 duplicate or old consumed image")
                    row=dict(case,image="images/"+str(case["index"])+".png",
                             image_sha256=image_sha,
                             instruction=instruction,
                             native_browser_state_verified=good is True if mode!="verify" else True)
                    if mode=="verify":
                        if source["image_sha256"]!=image_sha or source["instruction"]!=instruction:
                            raise ValueError("H168 exact independent original-browser image replay mismatch")
                        if (destination/row["image"]).read_bytes()!=png:
                            raise ValueError("H168 owner source media altered since capture")
                    else:
                        manifest["private_cases"].append(row)
                        private["private_training_actions"].append(
                            dict(index=case["index"],native_success=True,**gold))
                        screenshots[case["index"]]=png
                    print("H168_ORIGINAL_BROWSER_"+mode.upper(),case["app"],
                          case["index"],image_sha[:12],flush=True)
            finally:
                browser.close()
    if mode=="verify":
        stored=json.loads((destination/"manifest.json").read_text("utf8"))
        refs=json.loads((destination/"train-supervision-private.json").read_text("utf8"))
        result=validate_pilot(stored,refs,historical)
        if len(schedule)!=result["cases"]:
            raise ValueError("H168 source replay schedule changed")
        return dict(result,status="H168_ORIGINAL_BROWSER_PIXEL_EXACT_TRAIN_PILOT_VERIFIED")
    result=validate_pilot(manifest,private,historical)
    if mode=="smoke":
        return dict(result,status="H168_ORIGINAL_BROWSER_SMOKE_PASS")
    destination.mkdir(parents=True,exist_ok=False)
    (destination/"images").mkdir()
    for key,blob in screenshots.items():
        (destination/"images"/(str(key)+".png")).write_bytes(blob)
    (destination/"manifest.json").write_text(json.dumps(
        manifest,indent=2,sort_keys=True,ensure_ascii=False)+"\n","utf8")
    (destination/"train-supervision-private.json").write_text(json.dumps(
        private,indent=2,sort_keys=True,ensure_ascii=False)+"\n","utf8")
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner",required=True)
    p.add_argument("--upstream",required=True)
    p.add_argument("--destination",required=True)
    p.add_argument("--mode",choices=("smoke","capture","verify"),required=True)
    a=p.parse_args()
    print(json.dumps(run(a.owner,a.upstream,a.destination,a.mode),
                     indent=2,sort_keys=True,ensure_ascii=False))
