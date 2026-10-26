"""Capture private ACTUAL local Chromium screenshots, no DOM-coordinate leakage."""
import json
from pathlib import Path
from h130.browser_env import BrowserDragEnv, TASKS, WIDTH, HEIGHT
from h131.contract import SCHEMA, sha

def capture_initial(path):
    path=Path(path)
    if path.exists():
        raise ValueError("H131 create-only private screenshot output required")
    path.mkdir(parents=True)
    rows=[]
    for task in TASKS:
        with BrowserDragEnv(task) as env:
            o=env.observe()
            if o.width!=WIDTH or o.height!=HEIGHT:
                raise ValueError("H131 browser viewport drift")
            (path/(task+".png")).write_bytes(o.screenshot_png)
            rows.append({"task":task,"image":task+".png",
                         "image_sha256":sha(o.screenshot_png),
                         "instruction":o.instruction})
    manifest={"schema":SCHEMA,"phase":"initial","width":WIDTH,"height":HEIGHT,"cases":rows,
              "real_browser_snapshots":True,"oracle_browser_geometry_exposed":False,
              "model_inference_calls":0}
    (path/"manifest.json").write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n","utf8")
    return manifest

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    print(json.dumps(capture_initial(args.output),indent=2))
