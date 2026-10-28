"""H172 preregistered fresh same-family original GUI DEV, before GPU selection.

New original datepicker November scenes and original autocomplete Node.js
scenes, disjoint from H171 October/JavaScript TRAIN. No H164/H169 OOD reuse.
All native reference actions stay owner-only and separate from GPU images.
"""
import argparse,hashlib,json
from pathlib import Path
from h130.browser_env import _chromium_binary
from h138.probe import serve,center
from h159.stable_source_qa import stable_png
from h171.source import verify_sources,prior_hashes,sha
from h171.image_integrity import same_exact_png_pixels

SCHEMA="hala.h172.new-four-app-candidate-dev32-owner-only.v1"
N=32
NODE_PREFIXES=("No","no","NO","nO","Nod","nod","NOD","nOd",
               "Node","node","NODE","nOde","NoDe","Node.","node.","Node.j")

def cases():
    results=[]
    for app in ("flatpickr","awesomplete"):
        for n in range(16):
            results.append(dict(
                index=172000+(0 if app=="flatpickr" else 100)+n,
                split="DEV",app=app,variant=n,action_type="click",
                task="choose_november_next_day" if app=="flatpickr" else
                     "choose_node_autocomplete",
                language="en" if n%2==0 else "ar"))
    return results

def historical(owner,h168_train,h169_dev,h171_train):
    prior=prior_hashes(owner,h168_train,h169_dev)
    source=Path(h171_train)/"manifest.json"
    data=json.loads(source.read_text("utf8"))
    if (data.get("schema")!="hala.h171.fresh-two-original-click-app-train-only.v1"
        or data.get("count")!=32 or data.get("split")!="TRAIN_PILOT"):
        raise ValueError("H172 new original DEV cannot see unfrozen H171 TRAIN")
    for item in data.get("private_cases",[]):
        identity=item.get("image_sha256")
        if not isinstance(identity,str) or len(identity)!=64:
            raise ValueError("H172 previous image SHA malformed")
        prior.add(identity)
    lock=Path(owner)/"private-h169"/"H169_NEW_ORIGINAL_DEV_ONE_SHOT_CONSUMED.lock"
    if not lock.is_file():
        raise ValueError("H172 previous DEV must be marked irrevocably consumed")
    return prior

def execute_original(browser,url,row,run_native):
    ctx=browser.new_context(viewport={"width":960,"height":720},
                device_scale_factor=1,locale="en-US",service_workers="block")
    try:
        page=ctx.new_page()
        page.route("**/*",lambda route:route.continue_()
                   if route.request.url.startswith(url) else route.abort())
        app=row["app"];n=row["variant"]
        path=("flatpickr-stage/package/index.template.html"
              if app=="flatpickr" else "awesomplete-upstream/index.html")
        response=page.goto(url+path,wait_until="load",timeout=25000)
        if not response or response.status!=200:
            raise ValueError("H172 original upsteam page HTTP mismatch")
        if app=="flatpickr":
            page.wait_for_function("()=>typeof fp!=='undefined' && !!fp.setDate",
                                    timeout=8000)
            seed=5+n; target_day=seed+1
            page.evaluate("(d)=>fp.setDate(new Date(2026,10,d),true)",seed)
            input=page.locator("input.date")
            input.click()
            target=page.locator(
                f'.flatpickr-day[aria-label="November {target_day}, 2026"]')
            target.wait_for(timeout=9000)
            target.scroll_into_view_if_needed()
            if input.input_value()!=f"2026-11-{seed:02}":
                raise ValueError("H172 November original datepicker seed failed")
            expected=f"2026-11-{target_day:02}"
            instruction=(f"Select November {target_day}, 2026 in the date picker."
                if row["language"]=="en" else
                f"اختر يوم {target_day} نوفمبر 2026 من التقويم.")
        else:
            input=page.locator("input.awesomplete").first
            input.wait_for(timeout=9000)
            input.scroll_into_view_if_needed()
            input.fill(NODE_PREFIXES[n])
            target=page.locator("div.awesomplete li").filter(
                has_text="Node.js").first
            target.wait_for(timeout=9000)
            target.scroll_into_view_if_needed()
            expected="Node.js"
            instruction=("Select Node.js from autocomplete suggestions."
                if row["language"]=="en" else
                "اختر Node.js من اقتراحات الإكمال التلقائي.")
        picture=stable_png(page)
        point=center(target)
        if not 0<=point[0]<960 or not 0<=point[1]<720:
            raise ValueError("H172 original native click offscreen")
        success=None
        if run_native:
            page.mouse.click(*point)
            success=input.input_value()==expected
            if not success:
                raise ValueError("H172 original upsteam native click had no effect")
        return picture,instruction,point,success
    finally:
        ctx.close()

def run(owner,flat,awesome,stage,tarball,h168_train,h169_dev,h171_train,
        dest,mode):
    from playwright.sync_api import sync_playwright
    dest=Path(dest);plan=cases()
    if mode=="smoke":
        plan=[plan[0],plan[1],plan[16],plan[17]]
    if mode=="capture" and dest.exists():
        raise FileExistsError("H172 fresh DEV source may only be captured once")
    proof=verify_sources(flat,awesome,stage,tarball)
    previous=historical(owner,h168_train,h169_dev,h171_train)
    old=None
    if mode=="verify":
        old=json.loads((dest/"manifest.json").read_text("utf8"))
        if old.get("schema")!=SCHEMA or old.get("count")!=32:
            raise ValueError("H172 previous DEV screenshot source unknown")
    doc=dict(schema=SCHEMA,split="DEV",count=len(plan),git_proof=proof,
             owner_gold_uploaded=False,previous_consumed_sources_excluded=True,
             private_cases=[])
    gold=dict(schema="hala.h172.owner-private-november-node-actions.v1",
              owner_native_actions=[])
    pictures={}
    with serve(Path(stage)) as url:
        with sync_playwright() as pw:
            opts={"headless":True}
            exe=_chromium_binary()
            if exe:opts["executable_path"]=exe
            browser=pw.chromium.launch(**opts)
            try:
                for spec in plan:
                    png,task,point,success=execute_original(
                        browser,url,spec,run_native=mode!="verify")
                    digest=sha(png)
                    if digest in previous or digest in [
                            x["image_sha256"] for x in doc["private_cases"]]:
                        raise ValueError("H172 new DEV overlaps consumed source images")
                    record=dict(spec,image="images/"+str(spec["index"])+".png",
                        image_sha256=digest,instruction=task,
                        owner_native_reference_passed=success is True if mode!="verify" else True)
                    if mode=="verify":
                        frozen=next((r for r in old["private_cases"]
                                    if r["index"]==spec["index"]),None)
                        image_bytes=(dest/record["image"]).read_bytes()
                        if (not frozen or frozen["instruction"]!=task
                            or sha(image_bytes)!=frozen["image_sha256"]
                            or not same_exact_png_pixels(image_bytes,png)):
                            raise ValueError("H172 fresh DEV exact original RGBA pixel replay failed")
                    else:
                        doc["private_cases"].append(record)
                        gold["owner_native_actions"].append(dict(
                            index=spec["index"],kind="click",point=point,
                            native_success=True))
                    pictures[spec["index"]]=png
                    print("H172_ORIGINAL_DEV_"+mode.upper(),spec["app"],
                          spec["index"],digest[:12],flush=True)
            finally:browser.close()
    if mode=="smoke":
        return dict(status="H172_NOVEMBER_NODE_NATIVE_SMOKE_PASS",
                    cases=len(plan),model_calls=0)
    if len(pictures)!=32:
        raise ValueError("H172 complete original DEV only")
    if mode=="verify":
        return dict(status="H172_NEW_DEV_32_EXACT_RGBA_PIXEL_REPLAY_PASS",
                    cases=32,model_calls=0)
    dest.mkdir(exist_ok=False,parents=True)
    (dest/"images").mkdir()
    for ix,png in pictures.items():
        (dest/"images"/(str(ix)+".png")).write_bytes(png)
    (dest/"manifest.json").write_text(json.dumps(
        doc,ensure_ascii=False,indent=2,sort_keys=True)+"\n","utf8")
    (dest/"owner-reference-private.json").write_text(json.dumps(
        gold,ensure_ascii=False,indent=2,sort_keys=True)+"\n","utf8")
    return dict(status="H172_NEW_DEV_32_SOURCE_FROZEN",
        cases=32,arabic=16,english=16,optimizer_steps=0,
        owner_gold_uploaded=False,prior_consumed_images_used=False)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("owner","flat","awesome","stage","tarball",
        "h168-train","h169-dev","h171-train","dest"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--mode",choices=("smoke","capture","verify"),required=True)
    a=parser.parse_args()
    print(json.dumps(run(a.owner,a.flat,a.awesome,a.stage,a.tarball,
        a.h168_train,a.h169_dev,a.h171_train,a.dest,a.mode),
        indent=2,sort_keys=True))
