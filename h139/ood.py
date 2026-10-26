"""H139 separate, unseen React-DnD app OOD source, never TRAIN eligible."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
from h139.registry import SCHEMA,check_upstream,check_split_manifest,sha,UPSTREAM
from h138.probe import serve,mouse_drag,center
from h130.browser_env import WIDTH,HEIGHT,_chromium_binary

def qualify(root,output,num=8):
    root,output=Path(root),Path(output)
    if output.exists():raise ValueError("H139 blind app source create-only")
    source=check_upstream(root/"ood-board","ood-board")
    output.mkdir(parents=True)
    cases=[]
    with sync_playwright() as pw:
        opts={"headless":True}
        binary=_chromium_binary()
        if binary:opts["executable_path"]=binary
        browser=pw.chromium.launch(**opts)
        try:
            with serve(root/"ood-board"/UPSTREAM["ood-board"]["built_dir"]) as url:
                for i in range(num):
                    lang="ar" if i%2 else "en"
                    title=("Audit component","Review deployment","Verify model","Prepare backlog")[i%4]+" "+str(i+1)
                    item_id=f"h139-ood-task-{i}"
                    seed={
                        "Todo":[{"id":item_id,"column":"Todo","title":title,"color":"blue.300"}],
                        "In Progress":[{"id":"other-1","column":"In Progress","title":"Current sprint","color":"yellow.300"}],
                        "Blocked":[{"id":"other-2","column":"Blocked","title":"Waiting review","color":"red.300"}],
                        "Completed":[],
                    }
                    ctx=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                                            device_scale_factor=1,service_workers="block")
                    ctx.add_init_script("localStorage.setItem('tasks',"+json.dumps(json.dumps(seed))+");")
                    ctx.route("https://**/*",lambda route:route.abort())
                    page=ctx.new_page()
                    page.goto(url,wait_until="load")
                    page.locator('textarea').first.wait_for()
                    start=page.locator('[role="group"]').first
                    end=page.locator('.chakra-stack').nth(1)
                    if start.count()!=1 or end.count()!=1:
                        raise ValueError("H139 true React DnD control/target DOM absent")
                    start_box=start.bounding_box();end_box=end.bounding_box()
                    if not start_box or not end_box:raise ValueError("H139 OOD bbox absent")
                    start_px,end_px=center(start),center(end)
                    picture=page.screenshot(type="png")
                    mouse_drag(page,start_px,end_px)
                    state=json.loads(page.evaluate("localStorage.getItem('tasks')"))
                    if not any(x["id"]==item_id for x in state["In Progress"]):
                        raise ValueError("H139 independent React DnD source gold drag did not persist")
                    filename=f"h139-independent-ood-{i:02d}.png"
                    (output/filename).write_bytes(picture)
                    cases.append({
                        "id":f"h139-blind-{i}","app":"ood-board",
                        "app_git_sha":source["git_sha"],"role":"blind_test",
                        "lang":lang,"instruction":(
                            "اسحب بطاقة "+title+" إلى عمود In Progress."
                            if lang=="ar" else
                            "Move the "+title+" task to the In Progress column."),
                        "image":filename,"image_sha256":sha(picture),
                        "start_bbox":start_box,"end_bbox":end_box,
                        "seed_index":i,"real_browser_reference_state_persisted":True,
                    })
                    ctx.close()
        finally:browser.close()
    manifest={
       "schema":SCHEMA,"private":True,"split":"blind_test",
       "training_eligible":False,"cases":cases,"app_sources":{"ood-board":source}}
    check_split_manifest(manifest)
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n","utf8")
    return manifest

if __name__=="__main__":
    import sys
    count=int(sys.argv[3]) if len(sys.argv)>3 else 8
    x=qualify(sys.argv[1],sys.argv[2],count)
    print("H139_NEW_UPSTREAM_OOD_BLIND_GOLD_PASS",len(x["cases"]))
