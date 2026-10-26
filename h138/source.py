"""H138 immutable evaluation-only source from TWO unchanged complete MIT apps."""
import hashlib
import json
import subprocess
from pathlib import Path

from h138.probe import APPS,serve,center,mouse_drag
from h130.browser_env import WIDTH,HEIGHT,_chromium_binary

SCHEMA="hala.h138.untouched-two-complete-upstream-kanban-apps.v1"
TITLES=("Release summary","Verify API gateway","Resolve merge conflict","Document data model")
def digest(b):return hashlib.sha256(b).hexdigest()

def seed_columns(app,index):
    title=TITLES[index]
    if app=="react-dnd-kit-board":
        return {"todo":[{"id":8101+index,"text":title,"priority":"High"}],
                "progress":[{"id":9101,"text":"Verify dashboard","priority":"Low"}],
                "done":[{"id":9102,"text":"Close ticket","priority":"Medium"}]}
    return [
      {"id":"backlog","title":"Backlog","cards":[
        {"id":"a"+str(index),"title":title,"description":"Review and finalize","priority":"medium"}]},
      {"id":"todo","title":"To Do","cards":[
        {"id":"todo-x","title":"Weekly report","description":"","priority":"high"}]},
      {"id":"in-progress","title":"In Progress","cards":[]},
      {"id":"done","title":"Done","cards":[]},
    ]

def inspect_repo(repo,expected):
    current=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],
                                    text=True).strip()
    if current!=expected:raise ValueError("H138 upstream source revision changed")
    if "MIT License" not in (repo/"LICENSE").read_text("utf8"):
        raise ValueError("H138 complete app does not carry MIT license")
    if not (repo/"dist"/"index.html").is_file():
        raise ValueError("H138 actual upstream Vite build missing")
    changed=subprocess.check_output(["git","-C",str(repo),"diff","--name-only","HEAD"],
                                    text=True).strip()
    if changed:raise ValueError("H138 upstream tracked source modified")
    return current

def task_instruction(app,title,lang):
    if lang=="ar":return "انقل بطاقة "+title+" من العمود الأول إلى عمود المهام التالية."
    return "Drag the "+title+" task card into the next task column."
def capture(root,out):
    from playwright.sync_api import sync_playwright
    root,out=Path(root),Path(out)
    if out.exists():raise ValueError("H138 pristine first test source create-only")
    pins={app:inspect_repo(root/app,sha) for app,sha in APPS.items()}
    out.mkdir(parents=True)
    cases=[]
    with sync_playwright() as pw:
        opts={"headless":True}
        binary=_chromium_binary()
        if binary:opts["executable_path"]=binary
        browser=pw.chromium.launch(**opts)
        try:
            for app in APPS:
                with serve(root/app/"dist") as url:
                    for i in range(4):
                        lang="ar" if i%2 else "en"
                        title=TITLES[i]
                        ctx=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                                                device_scale_factor=1,service_workers="block")
                        key="kanbanTasks" if app=="react-dnd-kit-board" else "kanban"
                        doc=seed_columns(app,i)
                        script="localStorage.setItem("+json.dumps(key)+","+json.dumps(json.dumps(doc))+");"
                        ctx.add_init_script(script)
                        page=ctx.new_page()
                        page.goto(url,wait_until="networkidle")
                        if app=="react-dnd-kit-board":
                            start_loc=page.locator(".todo [role=button]").first
                            end_loc=page.locator(".progress .task-list")
                        else:
                            start_loc=page.locator('[draggable="true"]').filter(has_text=title).first
                            end_loc=page.locator(".grid.grid-cols-4 > div").nth(1)
                        start_box=start_loc.bounding_box()
                        end_box=end_loc.bounding_box()
                        start,end=center(start_loc),center(end_loc)
                        png=page.screenshot(type="png")
                        if not png.startswith(b"\x89PNG\r\n\x1a\n"):
                            raise ValueError("H138 original Chromium screenshot invalid")
                        name="h138-"+app+"-"+str(i)+".png"
                        (out/name).write_bytes(png)
                        mouse_drag(page,start,end)
                        after=json.loads(page.evaluate("(k)=>localStorage.getItem(k)",key))
                        if app=="react-dnd-kit-board":
                            succeeded=any(t["id"]==8101+i for t in after["progress"])
                        else:
                            succeeded=any(t["id"]=="a"+str(i) for t in after[1]["cards"])
                        if not succeeded:
                            raise ValueError("H138 genuine upstream app real drag not persisted")
                        cases.append({
                          "case_id":"h138-"+app+"-"+str(i),"app":app,"lang":lang,
                          "image":name,"image_sha256":digest(png),
                          "instruction":task_instruction(app,title,lang),
                          "start_bbox":start_box,"end_bbox":end_box,
                          "reference_browser_drag_persisted":True,
                          "app_git_sha":pins[app],
                          "case_index":i})
                        ctx.close()
        finally:browser.close()
    receipt={
      "schema":SCHEMA,"evaluation_only":True,"scene_count":len(cases),
      "training_images_allowed":False,"real_upstream_apps_unchanged":True,
      "upstream_git_shas":pins,"viewport":[WIDTH,HEIGHT],
      "cases":cases}
    (out/"manifest.json").write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n","utf8")
    if len(cases)!=8 or len({c["image_sha256"] for c in cases})!=8:
        raise ValueError("H138 first 8 truly distinct app screenshots required")
    return receipt

if __name__=="__main__":
    import sys
    doc=capture(sys.argv[1],sys.argv[2])
    print("H138 GOLD PERSISTENCE SUCCESS",len(doc["cases"]))

def validate_source(path):
    path=Path(path)
    m=json.loads((path/"manifest.json").read_text("utf8"))
    if (m.get("schema")!=SCHEMA or m.get("evaluation_only") is not True or
        m.get("training_images_allowed") is not False or
        m.get("real_upstream_apps_unchanged") is not True or
        m.get("upstream_git_shas")!=APPS or
        m.get("scene_count")!=8 or m.get("viewport")!=[WIDTH,HEIGHT]):
        raise ValueError("H138 exact untouched-upstream eval-only source required")
    cases=m.get("cases")
    if not isinstance(cases,list) or len(cases)!=8:
        raise ValueError("H138 source scenes are incomplete")
    seen=set()
    for app in APPS:
        app_cases=[x for x in cases if x.get("app")==app]
        if len(app_cases)!=4:raise ValueError("H138 balanced app source missing")
        for i,case in enumerate(app_cases):
            if (case.get("case_id")!="h138-"+app+"-"+str(i) or
                case.get("lang")!=("ar" if i%2 else "en") or
                case.get("app_git_sha")!=APPS[app] or
                case.get("reference_browser_drag_persisted") is not True or
                case.get("instruction")!=task_instruction(app,TITLES[i],case["lang"])):
                raise ValueError("H138 true source app/version/task identity mismatch")
            image=path/case["image"]
            if not image.is_file() or image.is_symlink():
                raise ValueError("H138 original application screenshot missing")
            sha=digest(image.read_bytes())
            if sha!=case["image_sha256"] or sha in seen:
                raise ValueError("H138 screenshot SHA duplicate or mismatch")
            seen.add(sha)
    return m
