"""H132 exact pinned specialist 2B GUI point-grounding comparison.

A NEW pretrained arm (not an Hala fine-tune) on same H131 private four
synthetic Chromium screenshots; no training, original H71 source, DOM input.
"""
import hashlib
import json
import math
from pathlib import Path

from h130.browser_env import WIDTH, HEIGHT, TASKS
from h131.contract import sha, verify_manifest

MODEL_ID="patrickballou/gui-assist-grounder"
MODEL_SHA="31632edaa20749c7f532868a42635f7240033106"
FORMAT="hala.h132.owner-private-specialist-2b-inference.v1"
PROMPT_TEMPLATE=(
    'This is a screenshot. Where should I click to accomplish this '
    'instruction: "{instruction}"?\n'
    'Output the click point in JSON format as {{"point_2d": [x, y]}}.'
)

def instruction_for(intent, role):
    if role not in ("start","end") or not isinstance(intent,str) or not 8<=len(intent)<=500:
        raise ValueError("H132 native role + original GUI task instruction required")
    if role=="start":
        desc="Locate the draggable blue HANDLE to START: "+intent
    else:
        desc="Locate the green destination/target for the END of the drag: "+intent
    return PROMPT_TEMPLATE.format(instruction=desc)

def native_point(raw, width=WIDTH,height=HEIGHT):
    if type(raw) is not str or not 1<=len(raw)<=300:
        return None
    try:
        doc=json.loads(raw)
    except (ValueError,TypeError):
        return None
    if type(doc) is not dict or set(doc)!={"point_2d"}:
        return None
    point=doc["point_2d"]
    if type(point) is not list or len(point)!=2:
        return None
    if (any(type(v) not in (int,float) or not math.isfinite(v)
            or not 0<=v<=1000 for v in point)):
        return None
    if width<=0 or height<=0:
        return None
    return [min(width-1,round(point[0]/1000*width)),
            min(height-1,round(point[1]/1000*height))]

def verify_predictions(input_directory, response_document):
    manifest=verify_manifest(input_directory,expect_phase="initial")
    if (response_document.get("schema")!=FORMAT or
        response_document.get("model_id")!=MODEL_ID or
        response_document.get("model_sha")!=MODEL_SHA or
        response_document.get("source_manifest_sha256") != sha(
          json.dumps(manifest,sort_keys=True,separators=(",",":")).encode("utf8")) or
        response_document.get("training_steps") != 0 or
        response_document.get("browser_oracle_used") is not False):
        raise ValueError("H132 frozen specialist output provenance not qualified")
    seen=set()
    source={x["task"]:x for x in manifest["cases"]}
    candidates=response_document.get("cases")
    if not isinstance(candidates,list) or len(candidates)!=4:
        raise ValueError("H132 exact four pretrained scene outputs required")
    by_task={}
    for c in candidates:
        name=c.get("task")
        if name not in TASKS or name in seen or c.get("image_sha256")!=source[name]["image_sha256"]:
            raise ValueError("H132 model-native prediction/source mismatch")
        seen.add(name)
        for role in ("start","end"):
            parsed=native_point(c.get("raw_"+role))
            declared=c.get(role)
            if parsed is None:
                if declared is not None:
                    raise ValueError("H132 invalid model raw point must be null, no fallback")
            elif type(declared) is not list or parsed!=declared:
                raise ValueError("H132 specialist model-native 1000-grid mismatch")
        by_task[name]=c
    return by_task
