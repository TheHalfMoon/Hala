"""Owner-private train/dev source freeze, never a model training command.

Staging is create-only. Every example must come from an actual original
browser drag and persisted reload; previous pilot and H136-H139 images are
excluded. Requires newly gathered source runs, not earlier diagnostics.
"""
import argparse
import io
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from PIL import Image

from h140.qualification import WIDTH,HEIGHT,SCHEMA,load_split,digest,position_bin
from h140.private_pilot_audit import historical_exclusions

SOURCE_APPS={
    "vanilla-native-board":{
        "sha":"47cfba359a71b70a6a23e511e554416aec006474",
        "repo":"anantagarwal1307/Drag-Drop-Kanban-Board",
        "schema":"hala.h140.owner-private-original-mouse-drag-pilot.v1"},
    "react-native-board":{
        "sha":"fa047f7a32afb57750ed5246456dc282ea7befcd",
        "repo":"dandemslay/React-KanbanBoard",
        "schema":"hala.h140.owner-private-unmodified-react-reference-pilot.v1"},
    "react-dndkit-board":{
        "sha":"f3e129998e3140a386f1884faff314345e0eb10b",
        "repo":"TomCani/kanban-dndkit-localstorage",
        "schema":"hala.h140.owner-original-dndkit-reference-pilot.v1"},
}
SOURCE_RUNS={
    "train":{"vanilla-native-board":("train-candidate-vanilla86",86),
             "react-native-board":("train-candidate-react86",86),
             "react-dndkit-board":("train-candidate-dndkit84",84)},
    "dev":{"vanilla-native-board":("dev-candidate-vanilla22",22),
           "react-native-board":("dev-candidate-react22",22),
           "react-dndkit-board":("dev-candidate-dndkit20",20)},
}

def source_checkout(private_root,app):
    policy=SOURCE_APPS[app]
    source=private_root/"upstream"/app
    revision=subprocess.check_output(
        ["git","-C",str(source),"rev-parse","HEAD"],text=True,timeout=35).strip()
    dirty=subprocess.check_output(
        ["git","-C",str(source),"diff","--name-only","HEAD"],text=True,timeout=35).strip()
    license_path=source/"LICENSE"
    if (revision!=policy["sha"] or dirty or not license_path.is_file()
        or not license_path.read_text("utf8").startswith("MIT License")):
        raise ValueError("H140 immutable original MIT training source not qualified")
    return {"repository":policy["repo"],"upstream_sha":policy["sha"],
            "license":"MIT","license_file_verified":True,
            "tracked_source_modified":False,"application_is_complete":True}

def image_sha_exclusions(private_root,experiments_root):
    prior,prior_count=historical_exclusions(experiments_root)
    pilot_count=0
    for pilot in sorted(private_root.iterdir()):
        if not pilot.is_dir() or "pilot" not in pilot.name:
            continue
        path=pilot/"manifest.json"
        if not path.is_file():
            raise ValueError("H140 old diagnostic pilot unsealed: "+pilot.name)
        doc=json.loads(path.read_text("utf8"))
        if doc.get("training_eligible") is not False or doc.get("blind_eligible") is not False:
            raise ValueError("H140 old diagnostic role not explicitly excluded")
        rows=doc.get("cases")
        if not isinstance(rows,list) or not rows:
            raise ValueError("H140 old diagnostic examples missing")
        for row in rows:
            h=row.get("image_sha256")
            if not isinstance(h,str) or len(h)!=64:
                raise ValueError("H140 old diagnostic image SHA invalid")
            prior.add(h)
        pilot_count+=1
    if pilot_count<3:
        raise ValueError("H140 previous originals must be excluded before new training")
    return prior,prior_count,pilot_count

def validate_run(folder,app,expected_count,exclusions,seen):
    policy=SOURCE_APPS[app]
    doc=json.loads((folder/"manifest.json").read_text("utf8"))
    cases=doc.get("cases")
    if (doc.get("schema")!=policy["schema"] or
        doc.get("status")!="POSITION_DIVERSE_DIAGNOSTIC_PILOT_ONLY" or
        doc.get("private") is not True or
        doc.get("training_eligible") is not False or
        doc.get("blind_eligible") is not False or
        not isinstance(cases,list) or len(cases)!=expected_count):
        raise ValueError("H140 unqualified new original source candidate: "+app)
    languages=Counter()
    starts=set()
    ends=set()
    ids=set()
    for row in cases:
        h=row.get("image_sha256")
        filename=row.get("image")
        if (not isinstance(h,str) or len(h)!=64 or h in exclusions or h in seen
            or not isinstance(filename,str) or "/" in filename or "\\" in filename
            or ".." in filename or not filename.endswith(".png")
            or row.get("id") in ids or row.get("app")!=app
            or row.get("app_git_sha")!=policy["sha"]
            or row.get("split")!="pilot_only_not_training"
            or row.get("reference_drag_persisted") is not True
            or row.get("reference_survived_page_reload") is not True
            or row.get("model_input_contains_oracle_coordinates") is not False):
            raise ValueError("H140 duplicate, leaked or non-native new source")
        png=(folder/filename).read_bytes()
        if digest(png)!=h:
            raise ValueError("H140 screenshot source hash changed")
        with Image.open(io.BytesIO(png)) as image:
            if image.format!="PNG" or image.size!=(WIDTH,HEIGHT):
                raise ValueError("H140 wrong native screenshot pixels/geometry")
            image.load()
        if row.get("language") not in ("ar","en"):
            raise ValueError("H140 bilingual language missing")
        languages[row["language"]]+=1
        starts.add(position_bin(row["start_bbox"]))
        ends.add(position_bin(row["end_bbox"]))
        ids.add(row["id"])
        seen.add(h)
    if languages!={"ar":expected_count//2,"en":expected_count//2} or len(starts)<8 or len(ends)<8:
        raise ValueError("H140 new source lacks per-app bilingual or geometry diversity")
    return cases,{"count":len(cases),"start_bins":len(starts),
                  "end_bins":len(ends),"language_counts":dict(languages)}

def freeze(private_root,experiments_root=None,out_name="frozen-source-v1"):
    private_root=Path(private_root)
    experiments_root=Path(experiments_root or private_root.parent)
    dest=private_root/out_name
    temp=private_root/(out_name+".incomplete")
    if dest.exists() or temp.exists() or out_name!="frozen-source-v1":
        raise ValueError("H140 strict create-only frozen source destination")
    excluded,n_historic,n_pilots=image_sha_exclusions(private_root,experiments_root)
    seen=set()
    specs={app:source_checkout(private_root,app) for app in SOURCE_APPS}
    source_rows={}
    counts={}
    # Stage ALL input validation before creating even a temporary output.
    for split,apps in SOURCE_RUNS.items():
        source_rows[split]={}
        counts[split]={}
        for app,(folder,n) in apps.items():
            cases,metrics=validate_run(private_root/folder,app,n,excluded,seen)
            source_rows[split][app]=(private_root/folder,cases)
            counts[split][app]=metrics
    if len(seen)!=320:
        raise ValueError("H140 TRAIN256 and DEV64 must have 320 genuinely new cases")
    temp.mkdir(parents=True,exist_ok=False)
    frozen_sha={}
    for split,apps in source_rows.items():
        folder=temp/split
        folder.mkdir()
        rows=[]
        for app,(origin,cases) in apps.items():
            for i,item in enumerate(cases):
                filename=f"{app}-{i:03d}.png"
                binary=(origin/item["image"]).read_bytes()
                if digest(binary)!=item["image_sha256"]:
                    raise ValueError("H140 source mutation during frozen train/dev copy")
                (folder/filename).write_bytes(binary)
                rows.append({
                    "id":f"{split}-{app}-{i:03d}","app":app,
                    "app_git_sha":specs[app]["upstream_sha"],
                    "split":split,"language":item["language"],
                    "instruction":item["instruction"],"image":filename,
                    "image_sha256":item["image_sha256"],
                    "start_bbox":item["start_bbox"],
                    "end_bbox":item["end_bbox"],
                    "reference_drag_persisted":True,
                    "reference_survived_page_reload":True,
                    "model_input_contains_oracle_coordinates":False})
        doc={"schema":SCHEMA,"private":True,"split":split,
             "training_eligible":True,"app_sources":specs,
             "cases":rows}
        payload=(json.dumps(doc,sort_keys=True,ensure_ascii=False,indent=2)+"\n").encode()
        (folder/"manifest.json").write_bytes(payload)
        loaded=load_split(folder,split)
        if len(loaded["cases"])!=len(rows):
            raise ValueError("H140 independent native source split qualification failed")
        frozen_sha[split]=digest(payload)
    receipt={"schema":"hala.h140.owner-original-freeze-train-dev.v1",
             "status":"FROZEN_TRAIN_DEV_SOURCE_ONLY_BLIND_AND_NEURAL_PENDING",
             "training_case_count":256,"development_case_count":64,
             "independent_train_source_apps":len(specs),
             "unique_new_train_dev_png_sha":len(seen),
             "prior_excluded_sha":len(excluded),
             "previous_stage_manifests":n_historic,
             "previous_diagnostic_pilot_manifests":n_pilots,
             "prior_image_overlap":0,"cross_split_image_overlap":0,
             "per_split_per_app":counts,"manifest_sha256":frozen_sha,
             "gpu_optimizer_steps":0,"model_improvement_proven":False,
             "blind_test_evaluated":False,"main_35B_updated":False}
    (temp/"FREEZE_RECEIPT.json").write_text(
        json.dumps(receipt,sort_keys=True,indent=2)+"\n",encoding="utf8")
    temp.rename(dest)
    return receipt

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--private-root",required=True)
    p.add_argument("--experiments-root")
    a=p.parse_args()
    print(json.dumps(freeze(a.private_root,a.experiments_root),indent=2,sort_keys=True))
