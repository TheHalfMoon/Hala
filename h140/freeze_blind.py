"""Create-only PRIVATE H140 one-time original-app BLIND source seal.

Never uploads to cloud, trains a model, selects a checkpoint or tests
neural responses. The gold bbox stays owner-only in a separate directory.
"""
import argparse
import io
import json
import subprocess
from collections import Counter
from pathlib import Path

from PIL import Image
from h140.qualification import WIDTH,HEIGHT,SCHEMA,digest,position_bin,load_split,admit
from h140.freeze_train_dev import image_sha_exclusions
from h140.blind_source_preflight import SOURCES
from h140.blind_original_capture import SCHEMA as PRIVATE_SOURCE_SCHEMA

CANDIDATES={
    "ood-vanilla-complete":"blind-candidate-aisha16-v1",
    "ood-second-drag":"blind-candidate-michael16-v4",
}

def check_blind_source(root,app):
    source=root/"upstream"/app
    entry=SOURCES[app]
    sha=subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],
                                text=True,timeout=35).strip()
    dirty=subprocess.check_output(["git","-C",str(source),"diff","--name-only","HEAD"],
                                  text=True,timeout=35).strip()
    license_path=source/"LICENSE"
    if (sha!=entry["sha"] or dirty or not license_path.is_file() or
        not license_path.read_text("utf8").startswith("MIT License") or
        any(not (source/f).is_file() for f in entry["files"])):
        raise ValueError("H140 unseen original app is not MIT/Git SOURCE qualified")
    return {"repository":entry["repository"],"upstream_sha":sha,
            "license":"MIT","license_file_verified":True,
            "tracked_source_modified":False,"application_is_complete":True}

def old_evaluated_manifests(root):
    """Keep every old IMAGE-bearing manifest; metadata-only files have no SHA.

    Cross-check image-hash equality with the owner historical auditor so
    skipping non-image metadata cannot weaken H136-H139 exclusions.
    """
    from h140.private_pilot_audit import historical_exclusions
    found=[]
    collected=set()
    for name in ("private-h136","private-h137","private-h138","private-h139"):
        directory=root/name
        paths=sorted(directory.rglob("manifest.json")) if directory.is_dir() else []
        valid=0
        for path in paths:
            doc=json.loads(path.read_text("utf8"))
            rows=doc.get("cases",[])
            if not isinstance(rows,list):
                continue
            shas={r.get("image_sha256") for r in rows
                  if isinstance(r,dict) and
                  isinstance(r.get("image_sha256"),str) and
                  len(r["image_sha256"])==64}
            if shas:
                found.append(path)
                collected.update(shas)
                valid+=1
        if valid==0:
            raise ValueError("H140 historical stage lacks any IMAGE-bearing manifest: "+name)
    audited,_=historical_exclusions(root)
    if collected!=audited:
        raise ValueError("H140 filtered old manifests fail independent historic SHA reconciliation")
    return found

def admit_source(root,app,excluded,seen):
    folder=root/CANDIDATES[app]
    doc=json.loads((folder/"manifest.json").read_text("utf8"))
    cases=doc.get("cases")
    if (doc.get("schema")!=PRIVATE_SOURCE_SCHEMA
        or doc.get("app")!=app or doc.get("app_git_sha")!=SOURCES[app]["sha"]
        or doc.get("source_role")!="source_only_blind_candidate"
        or doc.get("status")!="PRIVATE_ORIGINAL_BLIND_SOURCE_GEOMETRY_READY"
        or doc.get("training_eligible") is not False
        or doc.get("blind_model_evaluated") is not False
        or doc.get("model_neural_calls")!=0
        or doc.get("private") is not True
        or not isinstance(cases,list) or len(cases)!=16):
        raise ValueError("H140 truly independent owner-blind source failed admission")
    langs=Counter()
    starts=set();ends=set();ids=set()
    for row in cases:
        photo=row.get("image")
        h=row.get("image_sha256")
        if (not isinstance(photo,str) or not photo.endswith(".png")
            or "/" in photo or "\\" in photo or ".." in photo
            or not isinstance(h,str) or len(h)!=64 or h in seen or h in excluded
            or row.get("id") in ids or row.get("app")!=app
            or row.get("app_git_sha")!=SOURCES[app]["sha"]
            or row.get("source_role")!="blind_candidate_not_model_evaluated"
            or row.get("reference_drag_persisted") is not True
            or row.get("reference_survived_page_reload") is not True
            or row.get("model_input_contains_oracle_coordinates") is not False):
            raise ValueError("H140 original independent case has data leakage or source drift")
        png=(folder/photo).read_bytes()
        if digest(png)!=h:
            raise ValueError("H140 private OOD image SHA mismatch")
        with Image.open(io.BytesIO(png)) as image:
            if image.format!="PNG" or image.size!=(WIDTH,HEIGHT):
                raise ValueError("H140 original OOD screenshot must be native PNG")
            image.load()
        lang=row.get("language")
        if lang not in ("en","ar"):
            raise ValueError("H140 bilingual task language missing")
        langs[lang]+=1
        starts.add(position_bin(row["start_bbox"]))
        ends.add(position_bin(row["end_bbox"]))
        seen.add(h);ids.add(row["id"])
    if langs!={"en":8,"ar":8} or len(starts)<8 or len(ends)<8:
        raise ValueError("H140 OOD original app geometry and EN/AR diversity insufficient")
    return folder,cases,{"start_bins":len(starts),"end_bins":len(ends),
                         "original_reference_success":16,"languages":dict(langs)}

def seal(private_root,experiments_root=None):
    root=Path(private_root)
    experiment_root=Path(experiments_root or root.parent)
    frozen=root/"frozen-source-v1"
    train=frozen/"train"
    dev=frozen/"dev"
    target=frozen/"blind_test"
    partial=frozen/"blind_test.incomplete"
    if not frozen.is_dir() or target.exists() or partial.exists():
        raise ValueError("H140 BLIND source is create-only after genuine TRAIN/DEV")
    receipt=json.loads((frozen/"FREEZE_RECEIPT.json").read_text("utf8"))
    if receipt.get("status")!="FROZEN_TRAIN_DEV_SOURCE_ONLY_BLIND_AND_NEURAL_PENDING":
        raise ValueError("H140 TRAIN/DEV not already immutable and qualified")
    for role in ("train","dev"):
        if digest((frozen/role/"manifest.json").read_bytes())!=receipt["manifest_sha256"][role]:
            raise ValueError("H140 original TRAIN/DEV manifest drift")
        load_split(frozen/role,role)
    excluded,n_history,n_pilots=image_sha_exclusions(root,experiment_root)
    # A previously rejected incomplete source capture may have real images;
    # NEVER silently recycle any of those 6 observed originals in BLIND v2.
    # Quarantine every previously tried OOD source frame, even completed
    # source-only runs that failed native point-diversity requirements.
    for old in root.glob("blind-candidate-*"):
        if old.name in set(CANDIDATES.values()) or not old.is_dir():
            continue
        for png in old.glob("*.png"):
            excluded.add(digest(png.read_bytes()))
    source_apps={app:check_blind_source(root,app) for app in CANDIDATES}
    if set(source_apps)&set(load_split(train,"train")["app_sources"]):
        raise ValueError("H140 BLIND app must be source-independent from TRAIN")
    existing=set()
    for role in ("train","dev"):
        existing.update(r["image_sha256"] for r in load_split(frozen/role,role)["cases"])
    excluded.update(existing)
    seen=set()
    sources={}
    metrics={}
    for app in CANDIDATES:
        folder,rows,counts=admit_source(root,app,excluded,seen)
        sources[app]=(folder,rows)
        metrics[app]=counts
    if len(seen)!=32 or len(seen&excluded)!=0:
        raise ValueError("H140 one-time BLIND must have 32 never-used native frames")
    partial.mkdir(parents=True,exist_ok=False)
    combined=[]
    for app,(source_dir,rows) in sources.items():
        for i,row in enumerate(rows):
            filename=f"{app}-{i:03d}.png"
            image=(source_dir/row["image"]).read_bytes()
            if digest(image)!=row["image_sha256"]:
                raise ValueError("H140 original OOD image mutation during freeze")
            (partial/filename).write_bytes(image)
            combined.append({
                "id":f"blind-{app}-{i:03d}",
                "app":app,"app_git_sha":source_apps[app]["upstream_sha"],
                "split":"blind_test","language":row["language"],
                "instruction":row["instruction"],
                "image":filename,"image_sha256":row["image_sha256"],
                "start_bbox":row["start_bbox"],
                "end_bbox":row["end_bbox"],
                "reference_drag_persisted":True,
                "reference_survived_page_reload":True,
                "model_input_contains_oracle_coordinates":False})
    doc={"schema":SCHEMA,"private":True,"split":"blind_test",
         "training_eligible":False,"app_sources":source_apps,
         "cases":combined}
    payload=(json.dumps(doc,indent=2,sort_keys=True,ensure_ascii=False)+"\n").encode()
    (partial/"manifest.json").write_bytes(payload)
    load_split(partial,"blind_test")
    past_manifests=old_evaluated_manifests(experiment_root)
    assert admit(train,dev,partial,past_manifests)["cases"]=={
        "train":256,"dev":64,"blind_test":32} # fail-closed extra independent full-set check
    result={"schema":"hala.h140.private-frozen-independent-original-ood.v1",
            "status":"SEALED_OOD_SOURCE_ONLY_NEURAL_NOT_YET_RUN",
            "owner_manifest_sha256":digest(payload),
            "source_qualified_original_apps":2,
            "new_original_blind_cases":32,
            "english_cases":16,"arabic_cases":16,
            "train_dev_cases":320,
            "historical_plus_old_pilots_excluded":len(excluded)-len(existing),
            "previous_source_manifests":n_history,
            "previous_diagnostic_pilot_manifests":n_pilots,
            "source_app_metrics":metrics,
            "cross_split_sha_overlap":0,
            "real_neural_calls":0,"optimizer_steps":0,
            "main_35B_updated":False,"model_transfer_proven":False,
            "blind_gold_ever_uploaded":False}
    (partial/"OWNER_BLIND_SEAL_RECEIPT.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf8")
    partial.rename(target)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--private-root",required=True)
    p.add_argument("--experiments-root")
    a=p.parse_args()
    print(json.dumps(seal(a.private_root,a.experiments_root),sort_keys=True,indent=2))
