"""Qualified separate original visual-goal model PROBE; no optimizer and no H140 blind.

Owner-private golden rectangles, never uploaded or copied into model ZIP.
"""
import argparse
import io
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from PIL import Image
from h140.qualification import WIDTH,HEIGHT,digest
from h140.vanilla_pilot import visible_box
from h141.audit_visible import audit as validate_prior_visible,excluded_originals
from h141.visible_goal_source import EVAL_SCHEMA
from h141.interactive_source import APP,APP_SHA

PRIVATE_FOLDER="react-original-neural-probe16-v1"
PUBLIC_SCHEMA="hala.h141.private-original-visible-heading-model-probe.v1"
MODEL_FIELDS={"id","app","app_git_sha","language","instruction","image","image_sha256"}

def audit(root):
    owner=Path(root)
    folder=owner/"private-h141"/PRIVATE_FOLDER
    doc=json.loads((folder/"manifest.json").read_text("utf8"))
    rows=doc.get("cases")
    if (doc.get("schema")!=EVAL_SCHEMA or
        doc.get("status")!="PREDECLARED_ORIGINAL_NEURAL_PROBE_NO_TRAIN" or
        doc.get("role")!="owner_original_neural_probe" or
        doc.get("eval_scope")!="visible_start_and_heading_only_not_hidden_release" or
        doc.get("private") is not True or
        doc.get("training_eligible") is not False or
        doc.get("blind_eligible") is not False or
        doc.get("model_evaluation_eligible") is not True or
        doc.get("source_original_mit_sha")!=APP_SHA or
        doc.get("real_neural_calls")!=0 or
        not isinstance(rows,list) or len(rows)!=16):
        raise ValueError("H141 fresh original model PROBE is not qualified")
    old,stages,pilots=excluded_originals(owner)
    for prev in ("original-react-interactive-diagnostic8","react-original-visible-goal16"):
        prior=owner/"private-h141"/prev
        pdoc=json.loads((prior/"manifest.json").read_text("utf8"))
        if (pdoc.get("training_eligible") is not False or
            pdoc.get("model_evaluation_eligible") is not False):
            raise ValueError("H141 prior diagnostic changed role")
        for r in pdoc["cases"]:
            for k in ("image_sha256","at_rest_sha256","during_drag_sha256"):
                if k in r:old.add(r[k])
    seen=set();ids=set();counts=Counter()
    for row in rows:
        if (row.get("id") in ids or row.get("app")!=APP or
            row.get("app_git_sha")!=APP_SHA or row.get("language") not in ("en","ar") or
            not isinstance(row.get("attempt_index"),int) or
            row["attempt_index"]<48000 or
            row.get("model_evaluation_eligible") is not True or
            row.get("training_eligible") is not False or
            row.get("blind_eligible") is not False or
            row.get("reference_drag_persisted") is not True or
            row.get("reference_survived_page_reload") is not True or
            row.get("target_is_visible_at_rest") is not True or
            row.get("hidden_exact_drop_point_is_not_output_target") is not True or
            row.get("model_input_contains_oracle_coordinates") is not False or
            not visible_box(row.get("start_bbox")) or
            not visible_box(row.get("visible_destination_column_heading_bbox"))):
            raise ValueError("H141 original visible source/heading gold or role invalid")
        filename=row.get("image");sha=row.get("image_sha256")
        if (not isinstance(filename,str) or not filename.endswith(".png") or
            "/" in filename or "\\" in filename or ".." in filename or
            not isinstance(sha,str) or len(sha)!=64 or sha in seen or sha in old):
            raise ValueError("H141 new original evaluation photo reused source/history")
        binary=(folder/filename).read_bytes()
        if digest(binary)!=sha:
            raise ValueError("H141 model probe original pixel SHA mismatch")
        with Image.open(io.BytesIO(binary)) as image:
            if image.format!="PNG" or image.size!=(WIDTH,HEIGHT):
                raise ValueError("H141 original GUI only native PNGs")
            image.load()
        seen.add(sha);ids.add(row["id"]);counts[row["language"]]+=1
    if counts!={"ar":8,"en":8} or len(seen)!=16:
        raise ValueError("H141 qualified original probe bilingual count invalid")
    return {
        "schema":"hala.h141.original-visual-goal-private-neural-probe-audit.v1",
        "status":"FRESH_VISIBLE_ORIGINAL_16_MODEL_INPUT_ELIGIBLE_NOT_TRAINING",
        "source_manifest_sha256":digest((folder/"manifest.json").read_bytes()),
        "source_qualified_cases":len(rows),
        "unique_native_png":len(seen),
        "prior_disjoint_historical_original_sha":len(old),
        "languages":dict(counts),
        "model_evaluation_eligible":True,
        "training_eligible":False,
        "sealed_h140_blind_model_calls":0,
        "adapter_training_steps":0,
    }

def export_model_inputs(owner_root,output):
    owner=Path(owner_root)
    output=Path(output)
    if (output.exists() or output.with_suffix(".incomplete").exists()
        or not any(x.name=="private-h141" for x in output.parents)):
        raise ValueError("H141 only new owner-private exported input ZIP")
    approved=audit(owner)
    directory=owner/"private-h141"/PRIVATE_FOLDER
    source=json.loads((directory/"manifest.json").read_text("utf8"))
    clean=[{name:row[name] for name in MODEL_FIELDS} for row in source["cases"]]
    if any(set(row)!=MODEL_FIELDS for row in clean):
        raise ValueError("H141 model input cannot contain owner original DOM gold")
    doc={
        "schema":PUBLIC_SCHEMA,"status":"EVAL_ONLY",
        "original_source_manifest_sha256":approved["source_manifest_sha256"],
        "private":True,
        "training_eligible":False,"model_evaluation_eligible":True,
        "model_action_roles":["start","visible_column_heading"],
        "source_app_git_sha":APP_SHA,
        "cases":clean,
    }
    payload=(json.dumps(doc,sort_keys=True,indent=2,ensure_ascii=False)+"\n").encode("utf8")
    if any(word in payload for word in (
        b"start_bbox",b"end_bbox",b"true_start_point",
        b"visible_destination_column_heading_bbox",
        b"localStorage",b"reference_drag_persisted",
        b"hidden_exact_drop_point")):
        raise ValueError("H141 gold labels reached model-facing ZIP")
    tmp=output.with_suffix(".incomplete")
    with ZipFile(tmp,"w",ZIP_DEFLATED) as z:
        z.writestr("manifest.json",payload)
        for row in clean:
            binary=(directory/row["image"]).read_bytes()
            if digest(binary)!=row["image_sha256"]:
                raise ValueError("H141 owner image mutated during export")
            z.writestr(row["image"],binary)
    with ZipFile(tmp) as z:
        if (z.testzip() is not None or len(z.namelist())!=17 or
            digest(z.read("manifest.json"))!=digest(payload)):
            raise ValueError("H141 private original model ZIP integrity failed")
    tmp.rename(output)
    return {"status":"PRIVATE_NEW_SOURCE_MODEL_PROBE16_REDACTED",
            "case_count":16,"languages":approved["languages"],
            "redacted_manifest_sha256":digest(payload),
            "owner_zip_sha256":digest(output.read_bytes()),
            "source_manifest_sha256":approved["source_manifest_sha256"],
            "owner_gold_coordinates_exported":False,
            "training_eligible":False,"sealed_h140_blind_model_calls":0}

if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--owner-root",required=True)
    cli.add_argument("--output")
    a=cli.parse_args()
    print(json.dumps(export_model_inputs(a.owner_root,a.output) if a.output
                     else audit(a.owner_root),sort_keys=True,indent=2))
