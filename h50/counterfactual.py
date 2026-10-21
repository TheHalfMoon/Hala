"""H50 counterfactual training: six goals per original H49 TRAIN screenshot.

384 click labels from only 64 real first-party synthetic PNGs. Not 384 images,
not a trained model, and not a benchmark. Public H49 DEV stays out of TRAIN.
"""
from __future__ import annotations
import json
from pathlib import Path
import shutil
from h49.audit import inspect as inspect_h49
from h49.factory import FAMILIES, RIGHTS, canonical, scenes, sha
from h49.paired_eval import public_prompt
from training.artifacts import validate_action_completion

SCHEMA="hala.h50.counterfactual-six-way-train-from-h49.v1"
TRAIN=("justice_execution","traffic_services","fhir_resources","fhir_bundles")
DEV=("municipal_portal","research_lab")

def source(h49_root):
    h49_root=Path(h49_root)
    _,h49_manifest=inspect_h49(h49_root)
    capture=json.loads((h49_root/"capture.json").read_text("utf-8"))
    controls={record["id"]:record for record in capture["records"]}
    options={name:choices for name,split,lang,title,layout,choices in FAMILIES}
    rows=[];media=[];cases=[]
    for scene in scenes():
        if scene["split"]!="train":
            continue
        sid=scene["id"]
        proof=controls[sid]
        buttons=proof["controls"]
        digest=proof["png_sha256"]
        if sha((h49_root/"png"/(sid+".png")).read_bytes())!=digest:
            raise ValueError("Original screenshot SHA differs")
        if len(buttons)!=6 or {b["key"] for b in buttons}!=set(range(6)):
            raise ValueError("Six original browser controls required")
        path="png/"+sid+".png"
        media.append(dict(path=path,sha256=digest,family=scene["family"],provenance=RIGHTS))
        cases.append(dict(source_id=sid,png_sha256=digest,html_sha256=proof["html_sha256"]))
        for key in range(6):
            control=next(b for b in buttons if b["key"]==key)
            x,y,wide,tall=control["bounds"]
            point=[round(x+wide/2),round(y+tall/2)]
            matching=[b["key"] for b in buttons if b["bounds"][0]<=point[0]<
              b["bounds"][0]+b["bounds"][2] and b["bounds"][1]<=point[1]<
              b["bounds"][1]+b["bounds"][3]]
            if matching!=[key]:
                raise ValueError("Ambiguous target center")
            completion=json.dumps(dict(type="click",point=point,coordinate_space="window",
                  image_scale=1,element_ref_hint="visual_target"),sort_keys=True,ensure_ascii=False)
            validate_action_completion(completion)
            rows.append(dict(id="h50-"+sid+"-k"+str(key),split="train",
                family=scene["family"],language=scene["language"],
                prompt=public_prompt(options[scene["family"]][key][1]),
                completion=completion,image_path=path,image_sha256=digest))
    if (len(rows)!=384 or len(media)!=64 or len(cases)!=64 or
        len({r["id"] for r in rows})!=384 or
        len({r["image_path"] for r in rows})!=64 or
        {r["family"] for r in rows}!=set(TRAIN) or
        sum(r["language"]=="ar" for r in rows)!=192 or
        sum(r["language"]=="en" for r in rows)!=192):
        raise ValueError("Source-derived H50 count/split mismatch")
    data=("".join(json.dumps(r,sort_keys=True,ensure_ascii=False)+"\n" for r in rows)).encode("utf-8")
    manifest=dict(schema="hala.training.manifest.v1",source_schema=SCHEMA,
        split="train",source_kind="synthetic",rights_cleared=True,
        token_budget=4096,families=list(TRAIN),held_out_families=list(DEV),
        dataset_sha256=sha(data),source_sha256=h49_manifest["source_sha256"],
        capture_sha256=h49_manifest["capture_sha256"],
        h49_sft_sha256=h49_manifest["dataset_sha256"],
        media=media,source_screenshots=64,training_rows=384,
        dev_used_as_training=0,licensed_external_images=0,
        independent_blind_test=False,training_executed=False,adapter_qualified=False)
    proof=dict(schema=SCHEMA,train_screenshots=64,counterfactual_rows=384,
               arabic=192,english=192,source_sha256=manifest["source_sha256"],
               capture_sha256=manifest["capture_sha256"],
               dataset_sha256=manifest["dataset_sha256"],cases=cases)
    return data,manifest,proof

def verify_replay(value,source_manifest,source_proof):
    if (not isinstance(value,dict) or
        set(value)!={"schema","source_sha256","capture_sha256",
                      "case_count","click_count","real_service_access",
                      "model_inference","cases"} or
        value["schema"]!="hala.h50.original-browser-six-clicks.v1" or
        value["source_sha256"]!=source_manifest["source_sha256"] or
        value["capture_sha256"]!=source_manifest["capture_sha256"] or
        value["case_count"]!=64 or value["click_count"]!=384 or
        value["real_service_access"] is not False or value["model_inference"] is not False or
        len(value["cases"])!=64):
        raise ValueError("Missing original browser click replay")
    for expected,record in zip(source_proof["cases"],value["cases"]):
        if record!={**expected,"clicked_keys":list(range(6)),
                         "selected_keys":list(range(6)),
                         "click_counts":list(range(1,7))}:
            raise ValueError("Fake/missing/out-of-order browser click postcondition")

def audit(root,*,override_replay=None):
    root=Path(root)
    original=root.parent.parent/"h49/assets"
    data,manifest,proof=source(original)
    for name,expected in (("authoring_sft.jsonl",data),
                          ("sft_manifest.json",canonical(manifest)),
                          ("source_provenance.json",canonical(proof))):
        if (root/name).read_bytes()!=expected:
            raise ValueError("H50 source-bound data was altered: "+name)
    replay=(json.loads((root/"click_replay.json").read_text("utf-8"))
            if override_replay is None else override_replay)
    verify_replay(replay,manifest,proof)
    for r in manifest["media"]:
        image=root/r["path"]
        if (image.is_symlink() or image.read_bytes()!=(original/r["path"]).read_bytes()
            or sha(image.read_bytes())!=r["sha256"]):
            raise ValueError("H50 media bytes not same original H49")
    return [json.loads(line) for line in data.decode("utf-8").splitlines()],manifest

def load_export(path,manifest):
    path=Path(path)
    if path.name!="authoring_sft.jsonl" or path.is_symlink():
        raise ValueError("Only source-verified H50 export is eligible for training")
    rows,expected=audit(path.parent)
    if manifest!=expected:
        raise ValueError("H50 training manifest not original")
    return rows

def export(root):
    root=Path(root)
    if root.exists():
        raise FileExistsError("H50 assets create-only")
    original=root.parent.parent/"h49/assets"
    data,manifest,proof=source(original)
    root.mkdir(parents=True)
    try:
        (root/"png").mkdir()
        for r in manifest["media"]:
            with (root/r["path"]).open("xb") as f:
                f.write((original/r["path"]).read_bytes())
        for name,content in (("authoring_sft.jsonl",data),
                             ("sft_manifest.json",canonical(manifest)),
                             ("source_provenance.json",canonical(proof))):
            with (root/name).open("xb") as f:
                f.write(content)
    except Exception:
        shutil.rmtree(root)
        raise
    return dict(distinct_source_images=64,grounded_labels=384,arabic=192,english=192,
                sha256=manifest["dataset_sha256"],requires_actual_six_click_browser_replay=True,
                actual_model_training=False)

if __name__=="__main__":
    import argparse
    a=argparse.ArgumentParser()
    a.add_argument("--root",default="h50/assets")
    print(json.dumps(export(Path(a.parse_args().root))))
