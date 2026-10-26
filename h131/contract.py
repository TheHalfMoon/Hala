"""H131 private screenshot-to-neural-action contract, zero oracle geometry."""
import ast
import hashlib
import json
import math
from pathlib import Path
from h130.browser_env import TASKS, WIDTH, HEIGHT

SCHEMA="hala.h131.private-browser-screenshots.v1"
MODEL_SCHEMA="hala.h131.private-showui-native-points.v1"
TASK_COUNT=4

def sha(data):
    return hashlib.sha256(data).hexdigest()

def parse_showui(raw, width=WIDTH, height=HEIGHT):
    """ShowUI native [0..1,0..1] parser, explicit viewport up to 960x720."""
    if (type(raw) is not str or not 1<=len(raw)<=128 or
        type(width) is not int or type(height) is not int or
        not 64<=width<=1920 or not 64<=height<=1080):
        return None
    try:
        point=ast.literal_eval(raw.strip())
    except (ValueError,SyntaxError,TypeError,MemoryError,RecursionError):
        return None
    if (type(point) is not list or len(point)!=2 or
        any(type(x) not in (float,int) or not math.isfinite(x) or not 0<=x<=1 for x in point)):
        return None
    return (min(width-1,round(point[0]*width)),
            min(height-1,round(point[1]*height)))

def verify_manifest(directory, *, expect_phase=None):
    directory=Path(directory)
    manifest=json.loads((directory/"manifest.json").read_text("utf8"))
    if (manifest.get("schema")!=SCHEMA or
        manifest.get("phase") not in ("initial","feedback") or
        (expect_phase is not None and manifest["phase"]!=expect_phase) or
        manifest.get("width")!=WIDTH or manifest.get("height")!=HEIGHT):
        raise ValueError("H131 private screenshot manifest version/geometry mismatch")
    cases=manifest.get("cases")
    if not isinstance(cases,list) or len(cases)!=TASK_COUNT:
        raise ValueError("H131 exactly 4 fixture screenshots required")
    seen=set()
    for row in cases:
        task=row.get("task")
        if task not in TASKS or task in seen:
            raise ValueError("H131 duplicate/unexpected task")
        seen.add(task)
        filename=row.get("image")
        if filename != task+".png" or not isinstance(row.get("instruction"),str):
            raise ValueError("H131 unsafe/private image filename or absent instruction")
        if len(row["instruction"])<12 or len(row["instruction"])>500:
            raise ValueError("H131 invalid screenshot task instruction")
        media=(directory/filename)
        if not media.is_file() or media.is_symlink() or sha(media.read_bytes())!=row.get("image_sha256"):
            raise ValueError("H131 owner private screenshot SHA mismatch")
        if not media.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("H131 actual PNG screenshot required")
    return manifest

def verify_model_predictions(document, manifest):
    if (document.get("schema")!=MODEL_SCHEMA or
        document.get("phase")!=manifest["phase"] or
        document.get("input_manifest_sha256")!=sha(
            json.dumps(manifest,sort_keys=True,separators=(",",":")).encode("utf8")) or
        document.get("model_id")!="showlab/ShowUI-2B" or
        document.get("model_revision")!="cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60"):
        raise ValueError("H131 frozen model/manifest/phase mismatch")
    outputs=document.get("cases")
    if not isinstance(outputs,list) or len(outputs)!=TASK_COUNT:
        raise ValueError("H131 require all four model results")
    source={x["task"]:x for x in manifest["cases"]}
    got=set()
    for item in outputs:
        task=item.get("task")
        if task not in source or task in got or item.get("image_sha256")!=source[task]["image_sha256"]:
            raise ValueError("H131 prediction source identity mismatch")
        got.add(task)
        for role in ("start","end"):
            parsed=parse_showui(item.get("raw_"+role))
            declared=item.get(role)
            if parsed is None or type(declared) not in (list,tuple) or list(parsed)!=list(declared):
                raise ValueError("H131 model-native coordinate mismatch")
    return {x["task"]:x for x in outputs}
