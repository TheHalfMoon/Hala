"""H151 owner independent original shifted-layout source-only SHA replay.

All scene pixels must match before any H142 LoRA outputs are evaluated.
No model predictions and no gold boxes are exported by this tool.
"""
import argparse
import json
from pathlib import Path
from h138.probe import serve
from h130.browser_env import WIDTH,HEIGHT,_chromium_binary
from h140.qualification import digest
from h141.interactive_source import verify_owner_original
from h151.collect_shifted_source import source_case,style
from h151.freeze_eval import audit,FOLDER

def reconstruct(page,row):
    seed=source_case(row["attempt_index"])
    src=page.locator(".task_column").nth(0).locator(
        ".task_card").nth(seed["source_index"])
    col=page.locator(".task_column").nth(seed["destination_column"])
    heading=col.locator(".task_column_heading").first
    hidden=col.locator(".hide_drop").nth(seed["destination_index"])
    src.wait_for(timeout=16000)
    heading.wait_for(timeout=16000)
    hidden.wait_for(timeout=16000)
    page.add_style_tag(content=style(seed["layout"]))
    native=page.screenshot(type="png")
    if digest(native)!=row["image_sha256"]:
        raise ValueError("H151 exact shifted native browser pixel replay failed: "+row["id"])
    return seed

def replay(root):
    from playwright.sync_api import sync_playwright
    root=Path(root)
    approved=audit(root)
    source=json.loads((root/"private-h151"/FOLDER/"manifest.json").read_text("utf8"))
    original=verify_owner_original(root/"private-h140"/"upstream"/"react-native-board")
    count=0
    with sync_playwright() as pw:
        options={"headless":True}
        binary=_chromium_binary()
        if binary:options["executable_path"]=binary
        browser=pw.chromium.launch(**options,timeout=20000)
        try:
            with serve(original) as url:
                for row in source["cases"]:
                    seed=source_case(row["attempt_index"])
                    ctx=browser.new_context(viewport={"width":WIDTH,"height":HEIGHT},
                                            device_scale_factor=1,service_workers="block")
                    ctx.add_init_script(
                        "if(localStorage.getItem('tasks')===null){"+
                        "localStorage.setItem('tasks',"+
                        json.dumps(json.dumps(seed["rows"]))+");}")
                    ctx.route("https://**/*",lambda route:route.abort())
                    try:
                        page=ctx.new_page()
                        page.goto(url,wait_until="commit",timeout=18000)
                        reconstruct(page,row)
                        count+=1
                    finally:ctx.close()
        finally:browser.close()
    if count!=16:raise ValueError("H151 independent shifted cohort incomplete")
    return {"status":"H151_SOURCE_ONLY_PIXEL_EXACT_16_REPLAY_QUALIFIED",
            "manifest_sha256":approved["manifest_sha256"],
            "cases":count,"neural_calls":0,"reference_gold_uploaded":False,
            "sealed_h140_blind_calls":0}
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner-root",required=True)
    a=p.parse_args()
    print(json.dumps(replay(a.owner_root),sort_keys=True,indent=2))
