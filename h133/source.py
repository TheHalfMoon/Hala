"""Immutable private H133 screenshot source contract and capture.

Eight NEW offline synthetic Chromium screenshots, no H130 visual reuse.
Only images/intent may enter free T4; DOM is owner-scoring-only.
"""
import hashlib
import json
from pathlib import Path
from h133.browser_env import CASES, SOURCE_SCHEMA, UnseenBrowserEnv
from h130.browser_env import WIDTH,HEIGHT

def sha(raw): return hashlib.sha256(raw).hexdigest()

def validate(folder, *, forbid_shas=()):
    folder=Path(folder)
    doc=json.loads((folder/"manifest.json").read_text("utf8"))
    if (doc.get("schema")!=SOURCE_SCHEMA or doc.get("cases_count")!=8
        or doc.get("width")!=WIDTH or doc.get("height")!=HEIGHT
        or doc.get("gold_in_model_inputs") is not False):
        raise ValueError("H133 frozen private manifest invalid")
    rows=doc.get("cases")
    if not isinstance(rows,list) or len(rows)!=8 or tuple(x.get("task") for x in rows)!=CASES:
        raise ValueError("H133 exact preregistered eight cases required")
    used=set()
    for row in rows:
        if (row.get("image")!=row["task"]+".png" or
            not isinstance(row.get("instruction"),str) or
            not 12<=len(row["instruction"])<=500):
            raise ValueError("H133 invalid source filename/instruction")
        file=folder/row["image"]
        if file.is_symlink() or not file.is_file():
            raise ValueError("H133 original PNG missing/symlink")
        raw=file.read_bytes()
        if (not raw.startswith(b"\x89PNG\r\n\x1a\n") or
            sha(raw)!=row.get("image_sha256") or
            sha(raw) in used or sha(raw) in forbid_shas):
            raise ValueError("H133 nonunique/previous or corrupted screenshot")
        used.add(sha(raw))
    return doc

def capture(path,old_sha=()):
    path=Path(path)
    if path.exists():
        raise ValueError("H133 private source output create-only")
    path.mkdir(parents=True)
    cases=[]
    for task in CASES:
        with UnseenBrowserEnv(task) as env:
            initial=env.observe()
            file=path/(task+".png")
            file.write_bytes(initial.screenshot_png)
            cases.append({"task":task,"image":file.name,
                "image_sha256":sha(initial.screenshot_png),
                "instruction":initial.instruction})
    doc={"schema":SOURCE_SCHEMA,"cases_count":8,"width":WIDTH,"height":HEIGHT,
         "gold_in_model_inputs":False,"cases":cases,
         "real_browser_screenshots":True,
         "all_previous_H130_screenshots_excluded":True}
    (path/"manifest.json").write_text(json.dumps(doc,sort_keys=True,indent=2)+"\n","utf8")
    return validate(path,forbid_shas=old_sha)

if __name__=="__main__":
    import argparse
    cli=argparse.ArgumentParser()
    cli.add_argument("--output",required=True)
    args=cli.parse_args()
    receipt=capture(args.output)
    print(json.dumps({"status":"H133_NEW_PRIVATE_CHROMIUM8_FROZEN",
        "scene_count":len(receipt["cases"])}))
