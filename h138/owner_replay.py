"""H138 owner independent 8-case untouched-upstream React app model replay."""
import json
from pathlib import Path
from h138.source import validate_source,seed_columns,digest,TITLES
from h138.probe import APPS,serve,mouse_drag
from h138.gpu_eval import SCHEMA_OUT,SOURCE_SHA,ADAPTER_SHA
from h132.grounder_contract import native_point
from h130.browser_env import WIDTH,HEIGHT,_chromium_binary
from h133.paired_stats import exact_discordant_p

ARMS=("base_pretrained","h136_adapter")
def verify(source,report,raw):
    if (report.get("schema")!=SCHEMA_OUT or
        report.get("status")!="H138_PRIVATE_2_FULL_UPSTREAM_APPS_32_NEURAL_COMPLETE" or
        report.get("source_sha256")!=SOURCE_SHA or
        report.get("adapter_sha256")!=ADAPTER_SHA or
        report.get("source_app_revisions")!=APPS or
        report.get("actual_neural_calls")!=32 or
        report.get("actual_optimizer_steps")!=0 or
        report.get("paid_compute") is not False or
        report.get("main_Hala_flagship_weights_changed") is not False):
        raise ValueError("H138 independent full-app frozen model provenance failed")
    if not isinstance(raw,list) or len(raw)!=32:
        raise ValueError("H138 real model raw result count incomplete")
    indexed={};i=0
    for arm in ARMS:
        for case in source["cases"]:
            for role in ("start","end"):
                item=raw[i];i+=1
                if (item.get("arm")!=arm or item.get("case_id")!=case["case_id"]
                    or item.get("image_sha256")!=case["image_sha256"]
                    or item.get("role")!=role):
                    raise ValueError("H138 exact model scene/role pairing invalid")
                point=native_point(item.get("raw"),WIDTH,HEIGHT)
                if item.get("point")!=point:
                    raise ValueError("H138 native neural response corrupted")
                b=case[role+"_bbox"]
                hit=(point is not None and b["x"]<=point[0]<=b["x"]+b["width"]
                     and b["y"]<=point[1]<=b["y"]+b["height"])
                if item.get("hit") is not bool(hit):
                    raise ValueError("H138 neural reported bbox score incorrect")
                indexed[(arm,case["case_id"],role)]=point
    return indexed

def replay(source_dir,model_dir,upstream_root,outfile):
    from playwright.sync_api import sync_playwright
    source_dir,model_dir,upstream_root,outfile=map(
        Path,(source_dir,model_dir,upstream_root,outfile))
    src=validate_source(source_dir)
    if digest((source_dir/"manifest.json").read_bytes())!=SOURCE_SHA:
        raise ValueError("H138 heldout original source SHA changed")
    report=json.loads((model_dir/"private-h138-report.json").read_text("utf8"))
    raw=json.loads((model_dir/"private-h138-neural.json").read_text("utf8"))
    points=verify(src,report,raw)
    for app,pin in APPS.items():
        from h138.source import inspect_repo
        inspect_repo(upstream_root/app,pin)
    actual_drags=0;rows=[]
    with sync_playwright() as pw:
        opts={"headless":True}
        exe=_chromium_binary()
        if exe:opts["executable_path"]=exe
        browser=pw.chromium.launch(**opts)
        try:
            for arm in ARMS:
                for app in APPS:
                    with serve(upstream_root/app/"dist") as url:
                        cases=[x for x in src["cases"] if x["app"]==app]
                        for case in cases:
                            index=case["case_index"]
                            ctx=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                                                    device_scale_factor=1,service_workers="block")
                            key="kanbanTasks" if app=="react-dnd-kit-board" else "kanban"
                            seed=seed_columns(app,index)
                            init="localStorage.setItem("+json.dumps(key)+","+json.dumps(json.dumps(seed))+");"
                            ctx.add_init_script(init)
                            page=ctx.new_page()
                            page.goto(url,wait_until="networkidle")
                            if digest(page.screenshot(type="png"))!=case["image_sha256"]:
                                raise ValueError("H138 actual upstream full-app screenshot drift")
                            start=points[(arm,case["case_id"],"start")]
                            end=points[(arm,case["case_id"],"end")]
                            if start is not None and end is not None:
                                actual_drags+=1
                                mouse_drag(page,start,end)
                                state=json.loads(page.evaluate("(k)=>localStorage.getItem(k)",key))
                                if app=="react-dnd-kit-board":
                                    success=any(t["id"]==8101+index for t in state["progress"])
                                else:
                                    success=any(t["id"]=="a"+str(index) for t in state[1]["cards"])
                            else:success=False
                            rows.append({"arm":arm,"case_id":case["case_id"],
                                         "app":app,"lang":case["lang"],
                                         "full_task_success":success})
                            ctx.close()
        finally:browser.close()
    base={x["case_id"]:x["full_task_success"] for x in rows if x["arm"]==ARMS[0]}
    adapter={x["case_id"]:x["full_task_success"] for x in rows if x["arm"]==ARMS[1]}
    aonly=sum(adapter[k] and not base[k] for k in base)
    bonly=sum(base[k] and not adapter[k] for k in base)
    per={}
    for case in src["cases"]:
        label=case["app"]+"__"+case["lang"]
        seg=per.setdefault(label,{"n":0,"pretrained":0,"adapter":0})
        seg["n"]+=1;seg["pretrained"]+=int(base[case["case_id"]])
        seg["adapter"]+=int(adapter[case["case_id"]])
    result={
      "status":"H138_OWNER_TWO_UNMODIFIED_UPSTREAM_REAL_APP_REPLAY_PASS",
      "manifest_sha256":SOURCE_SHA,"adapter_sha256":ADAPTER_SHA,
      "neural_inference_calls":32,"evaluated_app_scene_pairs":16,
      "actual_model_proposed_drag_attempts":actual_drags,
      "pretrained_persisted_task_success":sum(base.values()),
      "adapter_persisted_task_success":sum(adapter.values()),
      "paired_adapter_only":aonly,"paired_pretrained_only":bonly,
      "exact_two_sided_p":exact_discordant_p(aonly,bonly),
      "per_app_and_language":per,
      "full_upstream_app_source_modified":False,
      "model_35B_flagship_improvement_proven":False,
      "generalized_enterprise_app_ability_proven":False}
    if outfile.exists():raise ValueError("H138 once-only owner evidence create-only")
    outfile.parent.mkdir(parents=True,exist_ok=True)
    outfile.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n","utf8")
    return result
