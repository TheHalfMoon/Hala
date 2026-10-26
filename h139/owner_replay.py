"""Independent Windows-only H139 model-pointer replay in the untouched React app.

The frozen private OOD screenshots and real per-task browser state, not
training loss or oracle geometry, decide whether model proposals worked.
"""
import json
from pathlib import Path
from h139.registry import SCHEMA,UPSTREAM,check_split_manifest,check_upstream,sha
from h138.probe import serve,mouse_drag
from h130.browser_env import WIDTH,HEIGHT,_chromium_binary
from h132.grounder_contract import native_point
from h133.paired_stats import exact_discordant_p

ARMS=("base_pretrained","h139_adapter")
NEURAL_SCHEMA="hala.h139.redacted-blind8-frozen-native-neural.v1"
FOUR=("Audit component","Review deployment","Verify model","Prepare backlog")


def load_private_source(directory):
    directory=Path(directory)
    raw=(directory/"manifest.json").read_bytes()
    data=json.loads(raw)
    cases=check_split_manifest(data)
    if data.get("split")!="blind_test" or len(cases)!=8 or any(
            r.get("app")!="ood-board" or r.get("role")!="blind_test"
            for r in cases):
        raise ValueError("H139 immutable React OOD case split invalid")
    if len({r["image_sha256"] for r in cases})!=8:
        raise ValueError("H139 OOD same image reuse")
    for row in cases:
        if sha((directory/row["image"]).read_bytes())!=row["image_sha256"]:
            raise ValueError("H139 OOD owner image bytes drift")
    return data,sha(raw)


def prediction_pairs(source,receipt,items,owner_sha,expected_adapter_sha):
    if (receipt.get("schema")!=NEURAL_SCHEMA or
        receipt.get("status")!="H139_COMPLETE_32_REAL_BLIND_NEURAL_CALLS" or
        receipt.get("original_owner_manifest_sha256")!=owner_sha or
        receipt.get("model_calls")!=32 or
        receipt.get("optimizer_steps")!=0 or
        receipt.get("source_role")!="blind_test" or
        receipt.get("paid_compute_used") is not False or
        receipt.get("main_35B_updated") is not False or
        receipt.get("adapter_sha256")!=expected_adapter_sha or
        not isinstance(expected_adapter_sha,str) or
        len(expected_adapter_sha)!=64):
        raise ValueError("H139 frozen model/owner source admission failed")
    if not isinstance(items,list) or len(items)!=32:
        raise ValueError("H139 actual paired neural calls missing")
    by_key={}; n=0
    for arm in ARMS:
        for row in source["cases"]:
            for role in ("start","end"):
                item=items[n];n+=1
                if (item.get("arm")!=arm or item.get("id")!=row["id"] or
                    item.get("image_sha256")!=row["image_sha256"] or
                    item.get("role")!=role):
                    raise ValueError("H139 neural item order or image SHA corrupted")
                p=native_point(item.get("raw"),WIDTH,HEIGHT)
                if item.get("point")!=p:
                    raise ValueError("H139 native raw response coordinate mismatch")
                by_key[arm,row["id"],role]=p
    return by_key


def owner_browser_replay(source_dir,upstream_root,results_dir,output,expected_adapter_sha):
    from playwright.sync_api import sync_playwright
    source_dir,upstream_root,results_dir,output=map(
        Path,(source_dir,upstream_root,results_dir,output))
    if output.exists():
        raise ValueError("H139 independent replay receipt create-only")
    source,owner_sha=load_private_source(source_dir)
    check_upstream(upstream_root/"ood-board","ood-board")
    result_doc=json.loads((results_dir/"private-h139-blind-report.json").read_text("utf8"))
    model_items=json.loads((results_dir/"private-h139-blind-neural.json").read_text("utf8"))
    coords=prediction_pairs(source,result_doc,model_items,owner_sha,expected_adapter_sha)
    outcomes=[]; actual_drags=0
    with sync_playwright() as pw:
        options={"headless":True}
        executable=_chromium_binary()
        if executable:options["executable_path"]=executable
        browser=pw.chromium.launch(**options)
        try:
            with serve(upstream_root/"ood-board"/UPSTREAM["ood-board"]["built_dir"]) as url:
                for arm in ARMS:
                    for row in source["cases"]:
                        i=row["seed_index"]
                        title=FOUR[i%4]+" "+str(i+1)
                        task_id=f"h139-ood-task-{i}"
                        seed={
                            "Todo":[{"id":task_id,"column":"Todo","title":title,"color":"blue.300"}],
                            "In Progress":[{"id":"other-1","column":"In Progress",
                                           "title":"Current sprint","color":"yellow.300"}],
                            "Blocked":[{"id":"other-2","column":"Blocked",
                                        "title":"Waiting review","color":"red.300"}],
                            "Completed":[],
                        }
                        ctx=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                                                device_scale_factor=1,service_workers="block")
                        ctx.add_init_script("localStorage.setItem('tasks',"+
                                            json.dumps(json.dumps(seed))+");")
                        ctx.route("https://**/*",lambda route:route.abort())
                        page=ctx.new_page()
                        try:
                            # React cards animate their initial opacity after DOM readiness.
                            # Wait for the EXACT frozen PNG bytes, never relax the SHA gate.
                            page.goto(url,wait_until="domcontentloaded",timeout=12000)
                            page.locator("textarea").first.wait_for(timeout=12000)
                            if (not page.locator('[role="group"]').first.bounding_box() or
                                not page.locator('.chakra-stack').nth(1).bounding_box()):
                                raise ValueError("H139 original app card/column not visible")
                            stable=False
                            for attempt in range(16):
                                if sha(page.screenshot(type="png"))==row["image_sha256"]:
                                    stable=True
                                    break
                                page.wait_for_timeout(100)
                            if not stable:
                                raise ValueError("H139 original React screenshot never stabilized to frozen SHA")
                            start=coords[arm,row["id"],"start"]
                            end=coords[arm,row["id"],"end"]
                            success=False
                            if start is not None and end is not None:
                                actual_drags+=1
                                mouse_drag(page,start,end)
                                state=json.loads(page.evaluate("localStorage.getItem('tasks')"))
                                success=any(item.get("id")==task_id
                                            for item in state["In Progress"])
                            outcomes.append({"arm":arm,"id":row["id"],"lang":row["lang"],
                                             "full_persisted_task_success":success})
                        finally:
                            ctx.close()
        finally:
            browser.close()
    base={x["id"]:x["full_persisted_task_success"] for x in outcomes if x["arm"]==ARMS[0]}
    adapter={x["id"]:x["full_persisted_task_success"] for x in outcomes if x["arm"]==ARMS[1]}
    aonly=sum(adapter[k] and not base[k] for k in base)
    bonly=sum(base[k] and not adapter[k] for k in base)
    report={"status":"H139_OWNER_INDEPENDENT_UNMODIFIED_REACT_OOD_REPLAY_COMPLETE",
            "model_calls":32,"owner_manifest_sha256":owner_sha,
            "adapter_sha256":result_doc["adapter_sha256"],
            "model_proposed_browser_drags":actual_drags,
            "base_success":sum(base.values()),"adapter_success":sum(adapter.values()),
            "adapter_only":aonly,"pretrained_only":bonly,
            "paired_exact_two_sided_p":exact_discordant_p(aonly,bonly),
            "real_app_scenes":8,"main_35B_improvement_proven":False,
            "unmodified_upstream_app":True,"rows":outcomes}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf8")
    return report
