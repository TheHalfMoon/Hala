"""Independent H142 original-source audit, SHA-seal TRAIN and DEV privately.

Reject any overlap with the frozen H140 and all H141 diagnostics/model
evaluation. Never touch the H140 independent 32-image BLIND input.
"""
import argparse
import io
import json
from collections import Counter
from pathlib import Path

from PIL import Image
from h140.qualification import digest
from h140.vanilla_pilot import visible_box
from h141.audit import excluded_originals
from h141.interactive_source import APP_SHA,APP
from h142.collect_visible_source import SCHEMA,COUNTS,OFFSETS,DESTINATIONS

FOLDERS={"train":"train64-original-visible-column",
         "dev":"dev16-original-visible-column"}
STRATA={"ar/doing","en/doing","ar/done","en/done"}

def history(root):
    root=Path(root)
    hashes,_,_=excluded_originals(root)
    # Refused H142 first-attempt browser frames are NOT independent new
    # training/eval screenshots; exclude their real native pixel SHA too.
    failed_root=root/"private-h142"
    if failed_root.is_dir():
        for failed in failed_root.glob("*failed*"):
            if failed.is_dir():
                for picture in failed.glob("*.png"):
                    hashes.add(digest(picture.read_bytes()))
    for name in ("original-react-interactive-diagnostic8",
                 "react-original-visible-goal16",
                 "react-original-neural-probe16-v1"):
        directory=root/"private-h141"/name
        manifest=json.loads((directory/"manifest.json").read_text("utf8"))
        if (manifest.get("source_original_mit_sha") not in (None,APP_SHA) or
            manifest.get("source_git_sha") not in (None,APP_SHA)):
            raise ValueError("H142 H141 source historical SHA drift")
        for row in manifest["cases"]:
            for field in ("image_sha256","at_rest_sha256","during_drag_sha256"):
                if field in row:hashes.add(row[field])
    return hashes

def verify(root):
    root=Path(root)
    directory=root/"private-h142"
    excluded=history(root)
    source_hashes=set()
    attempts=set()
    manifests={}
    result={}
    for split in ("train","dev"):
        src=directory/FOLDERS[split]
        binary=(src/"manifest.json").read_bytes()
        doc=json.loads(binary)
        rows=doc.get("cases")
        if (doc.get("schema")!=SCHEMA or
            doc.get("status")!="H142_ORIGINAL_PERSISTED_UI_SOURCE_READY" or
            doc.get("split")!=split or doc.get("private") is not True or
            doc.get("blind_eligible") is not False or
            doc.get("training_eligible") is not (split=="train") or
            doc.get("evaluation_eligible") is not (split=="dev") or
            doc.get("real_neural_calls")!=0 or
            doc.get("target_role")!="visible_destination_column_heading" or
            doc.get("original_screenshot_dimensions")!=[960,720] or
            doc.get("app_sources")!={APP:{"sha":APP_SHA,"license":"MIT"}} or
            not isinstance(rows,list) or len(rows)!=COUNTS[split]):
            raise ValueError("H142 original source manifest schema/role/app changed")
        strata=Counter();ids=set()
        for row in rows:
            if (row.get("id") in ids or
                row.get("split")!=split or
                row.get("app")!=APP or row.get("app_git_sha")!=APP_SHA or
                row.get("language") not in ("ar","en") or
                row.get("destination_status") not in DESTINATIONS or
                row.get("training_eligible") is not (split=="train") or
                row.get("evaluation_eligible") is not (split=="dev") or
                row.get("blind_eligible") is not False or
                row.get("reference_drag_persisted") is not True or
                row.get("reference_survived_page_reload") is not True or
                row.get("hidden_exact_drop_point_is_not_output_target") is not True or
                row.get("target_is_visible_at_rest") is not True or
                row.get("model_neural_calls")!=0 or
                not isinstance(row.get("attempt_index"),int) or
                row["attempt_index"]<OFFSETS[split] or
                row["attempt_index"] in attempts or
                not isinstance(row.get("instruction"),str) or
                not visible_box(row.get("start_bbox")) or
                not visible_box(row.get("visible_destination_column_heading_bbox"))):
                raise ValueError("H142 native owner source/gold or split integrity failed")
            image=row.get("image")
            sha=row.get("image_sha256")
            if (not isinstance(image,str) or not image.endswith(".png") or
                "/" in image or "\\" in image or ".." in image or
                not isinstance(sha,str) or len(sha)!=64 or
                sha in excluded or sha in source_hashes):
                raise ValueError("H142 old/historical or train-dev original screenshot leaked")
            pixels=(src/image).read_bytes()
            if digest(pixels)!=sha:
                raise ValueError("H142 original source screenshot SHA mismatch")
            with Image.open(io.BytesIO(pixels)) as picture:
                if picture.format!="PNG" or picture.size!=(960,720):
                    raise ValueError("H142 native 960x720 screenshot required")
                picture.load()
            attempts.add(row["attempt_index"])
            source_hashes.add(sha)
            ids.add(row["id"])
            strata[(row["language"],row["destination_status"])]+=1
        if strata!={(l,d):COUNTS[split]//4 for l in ("ar","en")
                         for d in DESTINATIONS}:
            raise ValueError("H142 language and two visible goal columns not balanced")
        manifests[split]=doc
        result[split]={
            "original_cases":len(rows),
            "manifest_sha256":digest(binary),
            "source_licenses":{"React-KanbanBoard":"MIT"},
            "balanced_strata":{"/".join(key):v for key,v in sorted(strata.items())},
            "true_native_persisted_reference_drag":len(rows)}
    return {
        "schema":"hala.h142.private-fresh-visible-source-qualification.v1",
        "status":"FROZEN_ORIGINAL_SOURCE_64_TRAIN_16_DEV_SHA_DISJOINT",
        "history_excluded_sha":len(excluded),
        "new_unique_original_screenshots":len(source_hashes),
        "train":result["train"],"dev":result["dev"],
        "real_neural_calls":0,
        "h140_blind_neural_calls":0,
        "flagship_Hala_35B_updated":False,
    }

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner-root",required=True)
    args=p.parse_args()
    print(json.dumps(verify(args.owner_root),indent=2,sort_keys=True))
