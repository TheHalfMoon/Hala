"""Fail-closed ORIGINAL H141 interactive visibility screenshot source audit.

Historical screenshot SHA only. Never copy/learn from old DEV or BLIND gold,
and never pretend oracle-revealed drag-time screenshot is model input.
"""
import argparse
import io
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from h140.freeze_train_dev import image_sha_exclusions
from h140.qualification import WIDTH,HEIGHT,digest
from h141.interactive_source import APP_SHA,APP,SCHEMA

PRIVATE_ROOT_NAME="private-h141"
PILOT_NAME="original-react-interactive-diagnostic8"

def excluded_originals(owner_root):
    owner_root=Path(owner_root)
    old=owner_root/"private-h140"
    hashes,stages,pilots=image_sha_exclusions(old,owner_root)
    frozen=old/"frozen-source-v1"
    for split,expected in (("train",256),("dev",64),("blind_test",32)):
        rows=json.loads((frozen/split/"manifest.json").read_text("utf8"))["cases"]
        if len(rows)!=expected:
            raise ValueError("H141 prior H140 source train/dev/blind role missing")
        hashes.update(row["image_sha256"] for row in rows)
    for folder in old.glob("blind-candidate-*"):
        if folder.is_dir():
            hashes.update(digest(file.read_bytes()) for file in folder.glob("*.png"))
    return hashes,stages,pilots

def audit(owner_root):
    root=Path(owner_root)
    private=root/PRIVATE_ROOT_NAME
    folder=private/PILOT_NAME
    if not folder.is_dir() or not private.is_dir():
        raise ValueError("H141 owner original interactive source not found")
    doc=json.loads((folder/"manifest.json").read_text("utf8"))
    cases=doc.get("cases")
    if (doc.get("schema")!=SCHEMA or doc.get("private") is not True or
        doc.get("status")!="QUALIFIED_SOURCE_QA_NOT_MODEL_TRAINING" or
        doc.get("training_eligible") is not False or
        doc.get("blind_eligible") is not False or
        doc.get("model_evaluation_eligible") is not False or
        doc.get("oracle_guided_drag_reference") is not True or
        doc.get("model_neural_calls")!=0 or
        doc.get("source_app")!=APP or
        doc.get("source_git_sha")!=APP_SHA or
        not isinstance(cases,list) or len(cases)!=8):
        raise ValueError("H141 oracle-guided private source cannot be model eligible")
    old,stages,pilots=excluded_originals(root)
    new=set()
    langs=Counter()
    ids=set()
    attempts=set()
    for row in cases:
        if (row.get("id") in ids or row.get("attempt_index") in attempts or
            row.get("app")!=APP or row.get("app_git_sha")!=APP_SHA or
            row.get("source_role")!="owner_oracle_guided_source_diagnostic_only" or
            row.get("training_eligible") is not False or
            row.get("blind_eligible") is not False or
            row.get("model_evaluation_eligible") is not False or
            row.get("gold_cursor_target_highlight_present") is not True or
            row.get("active_drop_area_visible") is not True or
            row.get("reference_drag_persisted") is not True or
            row.get("reference_survived_page_reload") is not True or
            row.get("hidden_css_opacity")!=0 or
            not 0.7<=row.get("active_css_opacity",0)<=1 or
            row.get("model_neural_calls")!=0 or
            not isinstance(row.get("attempt_index"),int) or
            row["attempt_index"]<40000):
            raise ValueError("H141 original source gesture, hidden/visible state or privacy invalid")
        if row.get("language") not in ("en","ar"):
            raise ValueError("H141 original bilingual instruction missing")
        if not isinstance(row.get("instruction"),str) or len(row["instruction"])<12:
            raise ValueError("H141 original task text missing")
        langs[row["language"]]+=1
        ids.add(row["id"])
        attempts.add(row["attempt_index"])
        for name,hash_field in (
            ("at_rest_image","at_rest_sha256"),
            ("during_drag_image","during_drag_sha256"),
        ):
            filename=row.get(name)
            reference_sha=row.get(hash_field)
            if (not isinstance(filename,str) or not filename.endswith(".png") or
                "/" in filename or "\\" in filename or ".." in filename or
                not isinstance(reference_sha,str) or len(reference_sha)!=64 or
                reference_sha in new or reference_sha in old):
                raise ValueError("H141 new original source image duplicated or path invalid")
            data=(folder/filename).read_bytes()
            if digest(data)!=reference_sha:
                raise ValueError("H141 altered owner source screenshot SHA mismatch")
            with Image.open(io.BytesIO(data)) as image:
                if image.format!="PNG" or image.size!=(WIDTH,HEIGHT):
                    raise ValueError("H141 original native 960x720 screenshot required")
                image.load()
            new.add(reference_sha)
    if langs!={"en":4,"ar":4} or len(new)!=16:
        raise ValueError("H141 owner original new two-stage image count/balance failed")
    return {
        "schema":"hala.h141.original-visible-target-owner-audit.v1",
        "status":"OWNER_REAL_UI_DYNAMIC_VISIBILITY_DIAGNOSTIC_PASS",
        "real_browser_persisted_drag_count":8,
        "new_unique_original_png":16,
        "historical_and_prior_excluded_sha":len(old),
        "source_app_original_mit_commit":APP_SHA,
        "historical_source_manifests":stages,
        "previous_h140_diagnostic_pilots":pilots,
        "languages":dict(langs),
        "model_trainable":False,
        "model_evaluation_eligible":False,
        "independent_h140_blind_neural_calls":0,
        "hala_flagship_35B_updated":False,
    }

if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--owner-root",required=True)
    args=cli.parse_args()
    print(json.dumps(audit(args.owner_root),sort_keys=True,indent=2))
