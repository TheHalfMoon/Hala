"""H139 train-source admission from unseen *real* MIT Kanban apps.

These training images are new and application-disjoint from all exposed
H136/H137/H138 tests and from the H139 frozen React blind-test app.
"""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
from h138.probe import serve,mouse_drag,center
from h130.browser_env import WIDTH,HEIGHT,_chromium_binary
from h139.registry import UPSTREAM,SCHEMA,check_upstream,check_split_manifest,sha

TITLES=(
 "Audit server response","Review design tokens","Fix navigation overlay",
 "Prepare localization","Resolve permission warning","Check upload errors",
 "Validate accessibility","Summarize progress","Inspect failed workflow",
 "Document API contract","Verify analytics report","Confirm browser layout",
 "Repair timeline","Review dashboard","Export project notes","Organize tasks",
)
COUNTS={"train":32,"dev":8}
def app_seed(name,seed):
    title=TITLES[(seed*7+3)%len(TITLES)]
    level=("low","medium","high")[seed%3]
    task_id=f"h139-{name}-{seed}"
    if name=="js-board":
        return "kanban-tasks",[
         {"id":task_id,"content":title,"status":"todo","priority":level},
         {"id":f"other-{seed}","content":"Check logs","status":"progress","priority":"low"},
         {"id":f"also-{seed}","content":"Review task queue","status":"todo","priority":"medium"},
        ],task_id,title
    if name=="alt-static-board":
        return "kanban-todo.tasks",[
         {"id":task_id,"text":title,"status":"todo","priority":level,"due":None,"createdAt":1},
         {"id":f"other-{seed}","text":"Check logs","status":"progress","priority":"low","due":None,"createdAt":2},
         {"id":f"also-{seed}","text":"Review task queue","status":"todo","priority":"medium","due":None,"createdAt":3},
        ],task_id,title
    raise ValueError("H139 forbidden OOD app training")

def instructions(title,lang):
    if lang=="ar":
        return "اسحب بطاقة "+title+" من قائمة المهام إلى عمود قيد التنفيذ."
    return "Drag the "+title+" task card into the In Progress column."

def qualify(root,output,split,n_override=None):
    root,output=Path(root),Path(output)
    if split not in COUNTS:raise ValueError("H139 TRAIN/DEV only")
    if output.exists():raise ValueError("H139 source create-only")
    output.mkdir(parents=True)
    sources={n:check_upstream(root/n,n) for n in ("js-board","alt-static-board")}
    rows=[]
    with sync_playwright() as pw:
        opts={"headless":True}
        exe=_chromium_binary()
        if exe:opts["executable_path"]=exe
        browser=pw.chromium.launch(**opts)
        try:
            for name in sources:
                app_dir=root/name/UPSTREAM[name]["built_dir"]
                with serve(app_dir) as url:
                    count=n_override if n_override is not None else COUNTS[split]
                    for i in range(count):
                        seed=(1000+i) if split=="train" else (2000+i)
                        key,initial,task_id,title=app_seed(name,seed)
                        ctx=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                                                device_scale_factor=1,service_workers="block")
                        init="localStorage.setItem("+json.dumps(key)+","+json.dumps(json.dumps(initial))+");"
                        ctx.add_init_script(init)
                        ctx.route("https://**/*",lambda req:req.abort())
                        page=ctx.new_page()
                        page.goto(url,wait_until="load")
                        if name=="js-board":
                            start_locator=page.locator('.task-card[data-id="'+task_id+'"]')
                            end_locator=page.locator("#progress-container")
                        else:
                            start_locator=page.locator('.task[data-id="'+task_id+'"]')
                            end_locator=page.locator('[data-dropzone="progress"]')
                        start_locator.wait_for()
                        start_locator.scroll_into_view_if_needed()
                        page.wait_for_timeout(50)
                        start_box=start_locator.bounding_box()
                        end_box=end_locator.bounding_box()
                        if start_box is None or end_box is None:
                            raise ValueError("H139 owner real app handle/target bbox missing")
                        x,y=center(start_locator),center(end_locator)
                        if any(v<0 for v in (*x,*y)) or x[0]>=WIDTH or x[1]>=HEIGHT or y[0]>=WIDTH or y[1]>=HEIGHT:
                            raise ValueError("H139 original real app target not visible in screenshot")
                        screenshot=page.screenshot(type="png")
                        filename=f"h139-{split}-{name}-{i:03d}.png"
                        (output/filename).write_bytes(screenshot)
                        mouse_drag(page,x,y)
                        state=json.loads(page.evaluate("(k)=>localStorage.getItem(k)",key))
                        matched=[t for t in state if t["id"]==task_id]
                        if len(matched)!=1 or matched[0]["status"]!="progress":
                            raise ValueError(f"H139 real-app reference mouse failed: {name} {split} {i}")
                        rows.append({
                            "id":f"h139-{split}-{name}-{i:03d}",
                            "app":name,"app_git_sha":sources[name]["git_sha"],
                            "role":"train","split":split,"lang":"ar" if i%2 else "en",
                            "instruction":instructions(title,"ar" if i%2 else "en"),
                            "image":filename,"image_sha256":sha(screenshot),
                            "start_bbox":start_box,"end_bbox":end_box,
                            "source_task_id":task_id,
                            "real_browser_reference_state_persisted":True,
                        })
                        ctx.close()
        finally:
            browser.close()

    manifest={
       "schema":SCHEMA,"private":True,"split":split,
       "evaluation_heldouts_reused":False,
       "blind_ood_app_mounted":False,"app_sources":sources,
       "case_count":len(rows),"cases":rows}
    check_split_manifest(manifest)
    if len(rows)!=2*(n_override if n_override is not None else COUNTS[split]):
        raise ValueError("H139 exact number of qualified app cases missing")
    if len({x["image_sha256"] for x in rows})!=len(rows):
        raise ValueError("H139 duplicate original screenshot pixels")
    (output/"manifest.json").write_text(
        json.dumps(manifest,sort_keys=True,indent=2)+"\n","utf8")
    return manifest

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--split",choices=("train","dev"),required=True)
    parser.add_argument("--probe",action="store_true")
    args=parser.parse_args()
    value=qualify(args.root,args.output,args.split,1 if args.probe else None)
    print("H139_REAL_MULTIREPO_GOLD_SUCCESS",value["case_count"])

def check(folder,split):
    folder=Path(folder)
    if split not in COUNTS:raise ValueError("H139 training split invalid")
    data=json.loads((folder/"manifest.json").read_text("utf8"))
    rows=check_split_manifest(data)
    if data.get("split")!=split or len(rows)!=COUNTS[split]*2:
        raise ValueError("H139 exact original real app TRAIN/DEV split missing")
    for row in rows:
        if not (folder/row["image"]).is_file() or sha((folder/row["image"]).read_bytes())!=row["image_sha256"]:
            raise ValueError("H139 original image sha mismatch")
        if row.get("real_browser_reference_state_persisted") is not True:
            raise ValueError("H139 real source pointer action not certified")
    return data

# Strict counts for private ZIP admission; blindness is separate and excluded.
SPLITS={"train":(1000,64),"dev":(2000,16)}
