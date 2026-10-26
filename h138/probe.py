"""H138 eval-only verification of untouched MIT production-style Kanban apps."""
import json
import subprocess
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path

APPS={
  "react-dnd-kit-board":"553d4e92be3e86e0deb7045f4968152868399eae",
  "react-html5-board":"83413e1f5fab4690bb46b23201a3438a427ef265",
}
@contextmanager
def serve(root):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*a,**k):super().__init__(*a,directory=str(root),**k)
        def log_message(self,*a):pass
    server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:yield f"http://127.0.0.1:{server.server_port}/"
    finally:server.shutdown();server.server_close();thread.join(timeout=5)

def center(selector):
    box=selector.bounding_box()
    if box is None:raise ValueError("H138 upstream real element bbox missing")
    return [round(box["x"]+box["width"]/2),round(box["y"]+box["height"]/2)]

def mouse_drag(page,a,b):
    page.mouse.move(*a);page.mouse.down()
    page.mouse.move(*b,steps=30);page.mouse.up()
    page.wait_for_timeout(250)
def probe(root):
    from playwright.sync_api import sync_playwright
    from h130.browser_env import _chromium_binary
    with sync_playwright() as playwright:
        opts={"headless":True}
        exe=_chromium_binary()
        if exe:opts["executable_path"]=exe
        browser=playwright.chromium.launch(**opts)
        try:
            for name in APPS:
                with serve(Path(root)/name/"dist") as url:
                    ctx=browser.new_context(viewport={"width":960,"height":720},
                                            device_scale_factor=1,service_workers="block")
                    if name=="react-dnd-kit-board":
                        payload={"todo":[{"id":101,"text":"Review release report","priority":"High"}],
                                 "progress":[{"id":102,"text":"Prepare documentation","priority":"Medium"}],
                                 "done":[{"id":103,"text":"Approve design","priority":"Low"}]}
                        script="localStorage.setItem('kanbanTasks',"+json.dumps(json.dumps(payload))+");"
                        ctx.add_init_script(script)
                    page=ctx.new_page()
                    page.goto(url,wait_until="networkidle")
                    if name=="react-dnd-kit-board":
                        a=center(page.locator(".todo [role=button]").first)
                        b=center(page.locator(".progress .task-list"))
                        key="kanbanTasks"
                    else:
                        a=center(page.locator('[draggable="true"]').filter(has_text="Design system setup").first)
                        b=center(page.locator(".grid.grid-cols-4 > div").nth(1))
                        key="kanban"
                    print("GOLD_DRAG_ATTEMPT",name,"FROM",a,"TO",b,flush=True)
                    mouse_drag(page,a,b)
                    print("POST_DRAG_STORAGE",name,
                          page.evaluate("(k)=>localStorage.getItem(k)",key),flush=True)
                    ctx.close()
        finally:browser.close()

if __name__=="__main__":
    import sys
    probe(sys.argv[1])
