"""H141 genuine visible-at-rest semantic target pilot owner-private audit.

Audits only NEW original images and source DOM provenance. NO hidden
release bbox is a visible model target, and NO H140 DEV or OOD media may
be promoted into H141 training.
"""
import argparse
import io
import json
from collections import Counter
from pathlib import Path

from PIL import Image
from h140.qualification import WIDTH,HEIGHT,digest
from h140.vanilla_pilot import visible_box
from h141.audit import excluded_originals,PILOT_NAME
from h141.visible_goal_source import SCHEMA
from h141.interactive_source import APP,APP_SHA

FOLDER_NAME="react-original-visible-goal16"

def audit(owner_root):
    root=Path(owner_root)
    private=root/"private-h141"
    path=private/FOLDER_NAME
    doc=json.loads((path/"manifest.json").read_text("utf8"))
    cases=doc.get("cases")
    if (doc.get("schema")!=SCHEMA or
        doc.get("status")!="VISIBLE_ORIGINAL_COLUMN_DIAGNOSTIC_ONLY" or
        doc.get("private") is not True or
        doc.get("training_eligible") is not False or
        doc.get("blind_eligible") is not False or
        doc.get("model_evaluation_eligible") is not False or
        doc.get("real_neural_calls")!=0 or
        doc.get("target_role")!="visible_destination_column_approach_not_release" or
        doc.get("source_original_mit_sha")!=APP_SHA or
        not isinstance(cases,list) or len(cases)!=16):
        raise ValueError("H141 visible semantic source is unqualified or oracle-leaking")
    old,stages,pilots=excluded_originals(root)
    prior=private/PILOT_NAME
    pilot=json.loads((prior/"manifest.json").read_text("utf8"))
    if pilot.get("training_eligible") is not False:
        raise ValueError("H141 dynamic oracle reference pilot unexpectedly trainable")
    previous=set()
    for row in pilot.get("cases",[]):
        previous.add(row["at_rest_sha256"])
        previous.add(row["during_drag_sha256"])
    if len(previous)!=16:
        raise ValueError("H141 previous oracle-guided original pilot missing")
    old.update(previous)
    new=set();ids=set();langs=Counter()
    for row in cases:
        if (row.get("id") in ids or row.get("app")!=APP or
            row.get("app_git_sha")!=APP_SHA or
            row.get("target_is_visible_at_rest") is not True or
            row.get("hidden_exact_drop_point_is_not_output_target") is not True or
            row.get("model_input_contains_oracle_coordinates") is not False or
            row.get("reference_drag_persisted") is not True or
            row.get("reference_survived_page_reload") is not True or
            row.get("training_eligible") is not False or
            row.get("model_evaluation_eligible") is not False or
            row.get("blind_eligible") is not False or
            not isinstance(row.get("attempt_index"),int) or row["attempt_index"]<44000 or
            row.get("language") not in ("en","ar")):
            raise ValueError("H141 visually identifiable source role/provenance invalid")
        goal=row.get("visible_destination_column_heading_bbox")
        start=row.get("start_bbox")
        if not visible_box(goal) or not visible_box(start):
            raise ValueError("H141 at-rest original source or semantic heading not visible")
        photo=row.get("image");h=row.get("image_sha256")
        if (not isinstance(photo,str) or not photo.endswith(".png") or
            "/" in photo or "\\" in photo or ".." in photo or
            not isinstance(h,str) or len(h)!=64 or h in new or h in old):
            raise ValueError("H141 original visible semantic image reused historic source")
        content=(path/photo).read_bytes()
        if digest(content)!=h:
            raise ValueError("H141 at-rest visible image SHA failed")
        with Image.open(io.BytesIO(content)) as image:
            if image.format!="PNG" or image.size!=(WIDTH,HEIGHT):
                raise ValueError("H141 original source screenshot not 960x720 PNG")
            image.load()
        new.add(h)
        ids.add(row["id"])
        langs[row["language"]]+=1
    if len(new)!=16 or langs!={"en":8,"ar":8}:
        raise ValueError("H141 new semantic target source lacks required balance")
    return {
        "schema":"hala.h141.original-visible-at-rest-semantic-column-audit.v1",
        "status":"REAL_VISIBLE_COLUMN_SOURCE_DIAGNOSTIC_QUALIFIED",
        "new_original_at_rest_png":len(new),
        "prior_original_image_sha_excluded":len(old),
        "genuine_original_persisted_drag_count":len(cases),
        "previous_h141_oracle_highlight_frames_excluded":len(previous),
        "source_original_mit_sha":APP_SHA,
        "ar_en_counts":dict(langs),
        "train_eligible":False,"blind_eligible":False,
        "valid_inference_output_role":"visible_column_approach_not_direct_release",
        "real_neural_calls":0,
        "hala_flagship_35B_updated":False,
    }

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--owner-root",required=True)
    a=p.parse_args()
    print(json.dumps(audit(a.owner_root),sort_keys=True,indent=2))
