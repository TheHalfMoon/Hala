"""H171 private unmodified original-browser native click TRAIN only."""
import argparse,json
from pathlib import Path
from h130.browser_env import _chromium_binary
from h138.probe import serve,center
from h159.stable_source_qa import stable_png
from h171.image_integrity import same_exact_png_pixels
from h171.source import (
    SCHEMA,PREFIXES,cases,sha,verify_sources,prior_hashes,validate)

def page_case(browser,base,row,execute):
    ctx=browser.new_context(viewport={"width":960,"height":720},
            device_scale_factor=1,locale="en-US",service_workers="block")
    try:
        page=ctx.new_page()
        page.route("**/*",lambda route:route.continue_()
                    if route.request.url.startswith(base) else route.abort())
        app=row["app"];n=row["variant"]
        path=("flatpickr-stage/package/index.template.html" if app=="flatpickr"
              else "awesomplete-upstream/index.html")
        response=page.goto(base+path,wait_until="load",timeout=25000)
        if response is None or response.status!=200:
            raise ValueError("H171 original upstream app not served")
        if app=="flatpickr":
            page.wait_for_function(
                "()=>typeof fp!=='undefined' && !!fp.setDate",timeout=10000)
            seed_day=5+n
            select_day=seed_day+1
            page.evaluate("(d)=>fp.setDate(new Date(2026,9,d),true)",seed_day)
            input=page.locator("input.date")
            input.click()
            target=page.locator(
                f'.flatpickr-day[aria-label="October {select_day}, 2026"]')
            target.wait_for(timeout=8000)
            target.scroll_into_view_if_needed()
            if input.input_value()!=f"2026-10-{seed_day:02}":
                raise ValueError("H171 original date seed misconfigured")
            instruction=(f"Select October {select_day}, 2026 in the date picker."
                if row["language"]=="en" else
                f"اختر يوم {select_day} أكتوبر 2026 من التقويم.")
            expected=f"2026-10-{select_day:02}"
        else:
            input=page.locator("input.awesomplete").first
            input.wait_for(timeout=9000)
            input.scroll_into_view_if_needed()
            input.fill(PREFIXES[n])
            target=page.locator("div.awesomplete li").filter(
                has_text="JavaScript").first
            target.wait_for(timeout=8000)
            target.scroll_into_view_if_needed()
            instruction=("Select JavaScript from the autocomplete suggestions."
                if row["language"]=="en" else
                "اختر JavaScript من اقتراحات الإكمال التلقائي.")
            expected="JavaScript"
        screenshot=stable_png(page)
        point=center(target)
        if (not 0<=point[0]<960 or not 0<=point[1]<720):
            raise ValueError("H171 native reference point offscreen")
        success=None
        if execute:
            page.mouse.click(*point)
            success=input.input_value()==expected
            if success is not True:
                raise ValueError("H171 actual original app native click failed")
        return screenshot,instruction,point,success
    finally:
        ctx.close()

def run(owner,upstream_flatpickr,upstream_awesomplete,stage,
        package,h168_train,h169_dev,dest,mode):
    from playwright.sync_api import sync_playwright
    dest=Path(dest)
    proof=verify_sources(upstream_flatpickr,upstream_awesomplete,stage,package)
    history=prior_hashes(owner,h168_train,h169_dev)
    rows=cases()
    if mode=="smoke":
        rows=[rows[i] for i in (0,1,16,17)]
    if mode=="capture" and dest.exists():
        raise FileExistsError("H171 original TRAIN source is one-shot, already frozen")
    old=None
    if mode=="verify":
        old=json.loads((dest/"manifest.json").read_text("utf8"))
        if old.get("schema")!=SCHEMA or old.get("count")!=32:
            raise ValueError("H171 frozen source not qualified")
    manifest=dict(schema=SCHEMA,split="TRAIN_PILOT",count=len(rows),
                  no_prior_dev_or_ood_images=True,source_proof=proof,
                  private_cases=[])
    private=dict(schema="hala.h171.owner-only-click-train-actions.v1",
                 owner_train_actions=[])
    imgs={}
    with serve(Path(stage)) as url:
        with sync_playwright() as pw:
            opts={"headless":True}
            exe=_chromium_binary()
            if exe:opts["executable_path"]=exe
            browser=pw.chromium.launch(**opts)
            try:
                for item in rows:
                    png,text,point,succeeded=page_case(
                        browser,url,item,execute=mode!="verify")
                    hashed=sha(png)
                    if hashed in history or hashed in (
                         x["image_sha256"] for x in manifest["private_cases"]):
                        raise ValueError("H171 fresh original TRAIN image overlaps old")
                    record=dict(item,
                        image="images/"+str(item["index"])+".png",
                        image_sha256=hashed,instruction=text,
                        reference_native_success=mode!="verify" and succeeded is True)
                    if mode=="verify":
                        original=next((r for r in old["private_cases"]
                                      if r["index"]==item["index"]),None)
                        frozen=(dest/record["image"]).read_bytes()
                        if (not original or original["instruction"]!=text
                            or sha(frozen)!=original["image_sha256"]
                            or not same_exact_png_pixels(frozen,png)):
                            raise ValueError("H171 original app exact RGBA pixel replay mismatch")
                    else:
                        manifest["private_cases"].append(record)
                        private["owner_train_actions"].append(
                            {"index":item["index"],"kind":"click",
                             "point":point,"native_success":True})
                    imgs[item["index"]]=png
                    print("H171_SOURCE_"+mode.upper(),item["app"],
                          item["index"],hashed[:12],flush=True)
            finally:
                browser.close()
    if mode=="smoke":
        return dict(status="H171_NATIVE_TWO_NEW_CLICK_APPS_SMOKE_PASS",
                    cases=len(rows),model_calls=0)
    if mode=="verify":
        stored=json.loads((dest/"manifest.json").read_text("utf8"))
        refs=json.loads((dest/"train-actions-private.json").read_text("utf8"))
        qualified=validate(stored,refs,history)
        if len(imgs)!=32:
            raise ValueError("H171 original byte replay not 32 scenes")
        return dict(qualified,status="H171_NEW_CLICK_TRAIN_32_PIXEL_EXACT_REPLAY_PASS")
    qualified=validate(manifest,private,history)
    dest.mkdir(parents=True,exist_ok=False)
    (dest/"images").mkdir()
    for ix,blob in imgs.items():
        (dest/"images"/(str(ix)+".png")).write_bytes(blob)
    (dest/"manifest.json").write_text(json.dumps(
        manifest,ensure_ascii=False,indent=2,sort_keys=True)+"\n","utf8")
    (dest/"train-actions-private.json").write_text(json.dumps(
        private,ensure_ascii=False,indent=2,sort_keys=True)+"\n","utf8")
    return qualified

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for arg in ("owner","upstream-flatpickr","upstream-awesomplete",
                "stage","package","h168-train","h169-dev","dest"):
        parser.add_argument("--"+arg,required=True)
    parser.add_argument("--mode",choices=("smoke","capture","verify"),required=True)
    a=parser.parse_args()
    print(json.dumps(run(a.owner,a.upstream_flatpickr,a.upstream_awesomplete,
        a.stage,a.package,a.h168_train,a.h169_dev,a.dest,a.mode),
        indent=2,sort_keys=True))
