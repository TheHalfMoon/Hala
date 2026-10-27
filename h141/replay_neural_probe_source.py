"""Owner-only reproduce new H141 model-evaluation screenshots exactly, no neural calls."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
from h130.browser_env import WIDTH,HEIGHT,_chromium_binary
from h138.probe import serve
from h140.qualification import digest
from h141.interactive_source import original_case,verify_owner_original
from h141.qualify_neural_probe import audit,PRIVATE_FOLDER

def replay(owner_root):
    root=Path(owner_root)
    qualified=audit(root)
    source=root/"private-h141"/PRIVATE_FOLDER
    manifest=json.loads((source/"manifest.json").read_text("utf8"))
    built=verify_owner_original(root/"private-h140"/"upstream"/"react-native-board")
    count=0
    with sync_playwright() as pw:
        params={"headless":True}
        browser_binary=_chromium_binary()
        if browser_binary:
            params["executable_path"]=browser_binary
        browser=pw.chromium.launch(**params,timeout=20000)
        try:
            with serve(built) as url:
                for row in manifest["cases"]:
                    seed=original_case(row["attempt_index"])
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
                        # Reproduce the exact source collector's React
                        # hydration/visibility synchronization BEFORE pixels.
                        page.locator(".task_column").nth(0).locator(
                            ".task_card").nth(seed["source_index"]).wait_for(timeout=18000)
                        page.locator(".task_column").nth(1).locator(
                            ".task_column_heading").first.wait_for(timeout=18000)
                        page.locator(".task_column").nth(1).locator(
                            ".hide_drop").nth(seed["destination_index"]).wait_for(timeout=18000)
                        if digest(page.screenshot(type="png"))!=row["image_sha256"]:
                            raise ValueError("H141 model actually saw different original native screenshot")
                        count+=1
                    finally:
                        ctx.close()
        finally:
            browser.close()
    if count!=16:
        raise ValueError("H141 original neural probe not exactly replayable")
    return {"status":"H141_ORIGINAL_NEURAL_PROBE_16_SOURCE_SHA_REPLAY_PASS",
            "original_screenshots":count,
            "neural_calls_executed":0,
            "gold_boxes_uploaded":False,
            "sealed_h140_blind_neural_calls":0,
            "source_sha":qualified["source_manifest_sha256"]}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--owner-root",required=True)
    args=parser.parse_args()
    print(json.dumps(replay(args.owner_root),sort_keys=True,indent=2))
