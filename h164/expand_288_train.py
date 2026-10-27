"""H164 288 additional TRAIN-only screenshot/native actions from real MIT app engines."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from contextlib import ExitStack
from h130.browser_env import _chromium_binary
from h138.probe import center,mouse_drag,serve
from h159.stable_source_qa import stable_png
from h164.four_app_train_pilot import native_calculator,START
from h164.protocol import locked_prior_shas

SCHEMA="hala.h164.additional-288-real-app-train.v1"
PRIVATE="private-h164/additional-three-app-288-train-v1"
APPS=("simple-table","calculator","sortablejs")
PINS={"simple-table":"7580270d18fb764bbf437a5e5b88ea66d63fc449",
      "calculator":"8f27a92072dde9658d8d94db8918a184ed7bd416",
      "sortablejs":"48b626bbc61afbb0f179730c1dd3cca3dcc8b788"}
LIC={"simple-table":"readme.md","calculator":"README.md","sortablejs":"LICENSE"}

def digest(x): return hashlib.sha256(x).hexdigest()

def prior(owner):
    owner=Path(owner)
    pilot=owner/"private-h164/four-app-train-pilot-v1/manifest.json"
    if digest(pilot.read_bytes())!="887eddd886d7f0505fcca4afd79ee96ce0dadf2443c72c0e5932f418e751d577":
        raise ValueError("H164 first pilot SHA changed")
    qa=json.loads((owner/"private-h164/H164_TRAIN_PILOT_PIXEL_QA.json").read_text("utf8"))
    if qa.get("status")!="H164_96_TRAIN_ONLY_EXACT_BROWSER_SCREENSHOT_REPLAY":
        raise ValueError("H164 original first pilot not exact browser qualified")
    old=locked_prior_shas(owner)
    old.update(c["image_sha256"] for c in json.loads(pilot.read_text("utf8"))["private_cases"])
    quarantine=owner/"private-h164/four-app-train-pilot-v1-quarantined/images"
    if quarantine.exists():
        old.update(digest(p.read_bytes()) for p in quarantine.glob("*.png"))
    return old

def admitted_apps(owner):
    source=Path(owner)/"private-h164/upstream"
    for app in APPS:
        root=source/app
        head=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
        dirty=subprocess.check_output(["git","-C",str(root),"status","--porcelain"],text=True).strip()
        if head!=PINS[app] or dirty or "MIT" not in (root/LIC[app]).read_text("utf8",errors="replace"):
            raise ValueError("H164 original app upstream/claimed license changed")
    return source

def specs():
    tasks={"simple-table":("clear_selection","filter_name","sort_age"),
           "calculator":("press_plus","clear","calculate"),
           "sortablejs":("reorder_list",)}
    return [{"split":"TRAIN","app":app,"task":tasks[app][n%len(tasks[app])],
             "language":"en" if n%2==0 else "ar","index":170001+ai*1000+n,
             "local_index":n}
            for ai,app in enumerate(APPS) for n in range(96)]

def native_table(browser,url,spec,execute):
    n=spec["local_index"]
    ctx=browser.new_context(viewport={"width":960,"height":720},device_scale_factor=1,locale="en-US")
    try:
        page=ctx.new_page()
        page.goto(url,wait_until="load",timeout=20000)
        page.locator(".simple-table tbody tr").first.wait_for(timeout=8000)
        size=(8,12,16,20)[n%4]
        page.evaluate("(x)=>{table.pageSize=x; table.currentPage=1; table.render()}",size)
        col=(0,4,5,6)[(n//4)%4]
        page.locator(".simple-table thead tr").first.locator("th").nth(col).click()
        value=50001+n*7
        page.locator('.simple-table input[data-field="salary"][placeholder="Min"]').fill(str(value))
        page.wait_for_timeout(650)
        page.locator(".simple-table tbody tr").nth((n//16)%8).click()
        if "No row selected" in page.locator("#selection-info").inner_text():
            raise ValueError("H164 scaled real table row not selected")
        img=stable_png(page)
        if not execute:return img,None,None
        task=spec["task"]
        if task=="clear_selection":
            target=page.locator("#btn-clear")
            point=center(target);page.mouse.click(*point)
            success=page.locator("#selection-info").inner_text()=="No row selected"
            action={"kind":"click","point":point}
            en="Click Clear selection to deselect the employee."
            ar="اضغط على Clear selection لإلغاء تحديد الموظف."
        elif task=="sort_age":
            target=page.locator(".simple-table thead tr").first.locator("th").nth(5)
            old=target.locator(".sort-indicator").inner_text()
            point=center(target);page.mouse.click(*point)
            success=target.locator(".sort-indicator").inner_text()!=old
            action={"kind":"click","point":point}
            en="Click the Age heading to change employee sorting."
            ar="اضغط على عنوان Age لتغيير فرز الموظفين."
        else:
            target=page.locator('.simple-table input[data-field="name"]')
            point=center(target);page.mouse.click(*point)
            page.keyboard.type("Ana",delay=3);page.wait_for_timeout(650)
            rows=page.locator(".simple-table tbody tr")
            success=rows.count()==1 and "Ana Garcia" in rows.first.inner_text()
            action={"kind":"type","point":point,"text":"Ana"}
            en="Type Ana into the Name column filter."
            ar="اكتب Ana في مرشح عمود Name."
        if not success:
            raise ValueError(f"H164 scaled native table failed {n} {task}")
        return img,(en if spec["language"]=="en" else ar),{"actions":[action]}
    finally:ctx.close()

def native_sortable(browser,url,spec,execute):
    n=spec["local_index"]
    ctx=browser.new_context(viewport={"width":960,"height":720},device_scale_factor=1,
                            locale="en-US",service_workers="block")
    try:
        page=ctx.new_page()
        page.route("**/*",lambda route:route.continue_()
                   if route.request.url.startswith(url) else route.abort())
        page.goto(url+"/index.html",wait_until="domcontentloaded",timeout=15000)
        items=page.locator("#example1 .list-group-item")
        items.first.wait_for(timeout=10000)
        if items.count()!=6:raise ValueError("H164 untouched SortableJS demo missing list")
        page.evaluate("""(seed)=>{Array.from(document.querySelectorAll('#example1 .list-group-item'))
            .forEach((el,i)=>el.textContent='Issue '+seed+' - step '+(i+1));}""",spec["index"])
        start_i=n%3;end_i=3+(n//3)%3
        items.nth(start_i).scroll_into_view_if_needed()
        page.wait_for_timeout(120)
        values=items.all_inner_texts()
        label=values[start_i]
        img=stable_png(page)
        if not execute:return img,None,None
        start=center(items.nth(start_i));end=center(items.nth(end_i))
        mouse_drag(page,start,end)
        page.wait_for_timeout(120)
        newer=items.all_inner_texts()
        if newer==values or newer.index(label)==start_i:
            raise ValueError(f"H164 original SortableJS actual drag failed {n}")
        en=f"Drag {label} down the sortable list."
        ar=f"اسحب البطاقة {label} إلى أسفل القائمة."
        return img,(en if spec["language"]=="en" else ar),{
            "actions":[{"kind":"drag","start":start,"end":end}]}
    finally:ctx.close()

def run(owner,mode):
    owner=Path(owner);source=admitted_apps(owner);excluded=prior(owner)
    root=owner/PRIVATE
    if mode=="capture" and root.exists():raise FileExistsError("H164 scaled TRAIN create-once")
    frozen=json.loads((root/"manifest.json").read_text("utf8")) if mode=="verify" else None
    if mode=="verify" and (frozen.get("schema")!=SCHEMA or
                            len(frozen.get("private_cases",[]))!=288):
        raise ValueError("H164 scaled original source frozen structure changed")
    cohort=specs()
    if mode=="smoke":
        cohort=[cohort[i] for i in (0,1,2,96,97,98,192,193,194)]
    from playwright.sync_api import sync_playwright
    rows=[]
    with sync_playwright() as pw:
        opts={"headless":True}
        exe=_chromium_binary()
        if exe:opts["executable_path"]=exe
        browser=pw.chromium.launch(**opts,timeout=20000)
        try:
            with ExitStack() as stack:
                urls={app:stack.enter_context(serve(source/app)) for app in APPS}
                if mode=="capture":
                    root.mkdir(parents=True,exist_ok=False);(root/"images").mkdir()
                for j,case in enumerate(cohort,1):
                    if case["app"]=="simple-table":
                        image,text,gold=native_table(
                            browser,urls["simple-table"]+"/examples/index.html",case,
                            execute=mode!="verify")
                    elif case["app"]=="calculator":
                        temporary={**case,"index":START+300+24+case["local_index"]}
                        image,text,gold=native_calculator(
                            browser,urls["calculator"]+"/index.html",temporary,
                            execute=mode!="verify",rest_mouse=True)
                    else:
                        image,text,gold=native_sortable(
                            browser,urls["sortablejs"],case,execute=mode!="verify")
                    h=digest(image)
                    if h in excluded:raise ValueError("H164 consumed/earlier original app data reused")
                    if mode=="capture":
                        if any(r["image_sha256"]==h for r in rows):
                            raise ValueError(f"H164 duplicate new TRAIN frame {j}")
                        name=f'{case["app"]}-{case["index"]}.png'
                        (root/"images"/name).write_bytes(image)
                        rows.append({**case,"image":name,"image_sha256":h,
                                     "instruction":text,"action_type":gold["actions"][0]["kind"],
                                     "training_supervision":gold,
                                     "native_reference_success":True})
                    elif mode=="verify":
                        expected=frozen["private_cases"][j-1]
                        if (expected["image_sha256"]!=h
                            or any(expected.get(k)!=v for k,v in case.items())
                            or (root/"images"/expected["image"]).read_bytes()!=image):
                            raise ValueError(f"H164 browser image exact replay mismatched {j}")
                    if j%12==0 or j==len(cohort):
                        print("H164_"+mode.upper()+"_NATIVE",j,len(cohort),case["app"],flush=True)
        finally:browser.close()
    if mode=="smoke":return {"status":"H164_REAL_SCALED_THREE_APP_SMOKE_PASS",
                             "native_source_tests":9,"model_calls":0}
    if mode=="verify":
        receipt={"status":"H164_288_NEW_TRAIN_EXACT_CHROMIUM_REPLAY",
                 "frames":288,"original_app_engines":3,
                 "optimizer_steps":0,"paid_GPU":False,"h140_blind_calls":0}
        qa=owner/"private-h164/H164_EXPANDED_288_PIXEL_QA.json"
        if qa.exists():raise FileExistsError("H164 expanded replay already recorded")
        qa.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n","utf8")
        return receipt
    if len(rows)!=288 or len({r["image_sha256"] for r in rows})!=288:
        raise ValueError("H164 scaled TRAIN source failed full quotas")
    doc={"schema":SCHEMA,"train_only":True,"train_count":288,
         "native_reference_success":288,
         "applications":list(APPS),"source_git_shas":PINS,
         "previous_source_images_excluded":len(excluded),
         "optimizer_steps":0,"paid_GPU":False,"h140_blind_calls":0,
         "private_cases":rows}
    raw=(json.dumps(doc,indent=2,sort_keys=True,ensure_ascii=False)+"\n").encode()
    (root/"manifest.json").write_bytes(raw)
    return {"status":"H164_288_REAL_ORIGINAL_APP_TRAIN_CASES_CAPTURED",
            "source_manifest_sha256":digest(raw),"train":288,"model_calls":0}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("smoke","capture","verify"))
    p.add_argument("--owner",required=True)
    args=p.parse_args()
    print(json.dumps(run(args.owner,args.mode),indent=2,sort_keys=True))
