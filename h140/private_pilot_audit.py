"""Independent CPU-only original H140 source-pilot auditor; no training.

Audit complete native PNG decodability, immutable app source revision,
reference/reload receipts, per-language balance and between-app SHA overlap.
Historical excluded model-evaluation screenshots remain OFF LIMITS.
"""
import argparse
import io
import json
import subprocess
from collections import Counter
from itertools import combinations
from pathlib import Path

from PIL import Image
from h140.qualification import WIDTH,HEIGHT,digest,position_bin,center

PILOTS={
    "vanilla-pilot32-independent":{
        "sha":"47cfba359a71b70a6a23e511e554416aec006474",
        "schema":"hala.h140.owner-private-original-mouse-drag-pilot.v1",
        "source":"vanilla-native-board",
        "count":32,"start":19,"end":15},
    "react-pilot16-firstpass":{
        "sha":"fa047f7a32afb57750ed5246456dc282ea7befcd",
        "schema":"hala.h140.owner-private-unmodified-react-reference-pilot.v1",
        "source":"react-native-board",
        "count":16,"start":13,"end":11},
    "dndkit-pilot32-expanded":{
        "sha":"f3e129998e3140a386f1884faff314345e0eb10b",
        "schema":"hala.h140.owner-original-dndkit-reference-pilot.v1",
        "source":"react-dndkit-board",
        "count":32,"start":12,"end":12},
}

def read_pilot(private_root,name,policy):
    folder=Path(private_root)/name
    data=json.loads((folder/"manifest.json").read_text("utf8"))
    rows=data.get("cases")
    if (data.get("schema")!=policy["schema"] or
        data.get("status")!="POSITION_DIVERSE_DIAGNOSTIC_PILOT_ONLY" or
        data.get("private") is not True or
        data.get("training_eligible") is not False or
        data.get("blind_eligible") is not False or
        not isinstance(rows,list) or len(rows)!=policy["count"]):
        raise ValueError("H140 private pilot identity/grain/status not qualified "+name)
    if data.get("model_neural_calls",data.get("real_neural_calls",0))!=0:
        raise ValueError("H140 pilot MUST NOT carry neural model outputs")
    by_image=set();by_id=set();langs=Counter();start=Counter();end=Counter()
    for row in rows:
        if (row.get("app_git_sha")!=policy["sha"] or
            row.get("app")!=policy["source"] or
            row.get("split")!="pilot_only_not_training" or
            row.get("reference_drag_persisted") is not True or
            row.get("reference_survived_page_reload") is not True or
            row.get("model_input_contains_oracle_coordinates") is not False):
            raise ValueError("H140 originally persisted owner reference receipt invalid")
        filename=row.get("image")
        if (not isinstance(filename,str) or not filename.endswith(".png") or
            "/" in filename or "\\" in filename or ".." in filename):
            raise ValueError("H140 original image filename must remain private basename")
        picture=(folder/filename).read_bytes()
        sha=digest(picture)
        if sha!=row.get("image_sha256"):
            raise ValueError("H140 tampered original native screenshot")
        with Image.open(io.BytesIO(picture)) as im:
            if im.format!="PNG" or im.size!=(WIDTH,HEIGHT):
                raise ValueError("H140 native screenshot must be decodable 960x720 PNG")
            im.load() # catches invalid/partially fabricated IHDR-only pseudo-PNGs
        if sha in by_image or row.get("id") in by_id:
            raise ValueError("H140 image or case identity duplicated")
        by_image.add(sha);by_id.add(row["id"])
        lang=row.get("language")
        if lang not in ("en","ar"):
            raise ValueError("H140 native prompt language absent")
        langs[lang]+=1
        for role,counts in (("start",start),("end",end)):
            box=row.get(role+"_bbox")
            if not isinstance(box,dict):
                raise ValueError("H140 actual original reference bbox required")
            bin_value=position_bin(box)
            if any(p<0 or p>=size for p,size in zip(center(box),(WIDTH,HEIGHT))):
                raise ValueError("H140 original point offscreen")
            counts[bin_value]+=1
    if (langs["en"]!=langs["ar"] or
        len(start)<8 or len(end)<8 or
        len(start)!=policy["start"] or len(end)!=policy["end"]):
        raise ValueError("H140 reference geometry or bilingual distribution not qualified")
    return {"case_count":len(rows),"app":policy["source"],
            "source_sha":policy["sha"],"languages":dict(langs),
            "unique_png_sha":len(by_image),
            "start_bins":len(start),"end_bins":len(end),
            "start_mode_share":round(max(start.values())/len(rows),3),
            "end_mode_share":round(max(end.values())/len(rows),3),
            "hashes":by_image}

def historical_exclusions(root):
    """Read all H136-H139 original manifest.json under owner PRIVATE dirs.

    Fails closed if any old stage lacks source manifests; no blank history
    argument can silently disable the exclusion check.
    """
    root=Path(root)
    previous=set();path_count=0
    for stage in ("private-h136","private-h137","private-h138","private-h139"):
        folder=root/stage
        files=list(folder.rglob("manifest.json")) if folder.is_dir() else []
        if not files:
            raise ValueError("H140 historical evaluated stage source missing: "+stage)
        stage_count=0
        for f in files:
            data=json.loads(f.read_text("utf8"))
            cases=data.get("cases",[])
            for row in cases:
                h=row.get("image_sha256")
                if isinstance(h,str) and len(h)==64:
                    previous.add(h);stage_count+=1
            path_count+=1
        if stage_count==0:
            raise ValueError("H140 old-stage historical source SHA absent: "+stage)
    return previous,path_count

def audit(private_root,experiment_root=None,also_check_source=True):
    private_root=Path(private_root)
    experiments=Path(experiment_root or private_root.parent)
    old,old_files=historical_exclusions(experiments)
    result={};all_hashes=set()
    for name,policy in PILOTS.items():
        item=read_pilot(private_root,name,policy)
        if (item["hashes"]&all_hashes or item["hashes"]&old):
            raise ValueError("H140 current original PILOT is overlapping previous/cross-app source")
        all_hashes.update(item["hashes"])
        if also_check_source:
            src=private_root/"upstream"/policy["source"]
            repo=subprocess.check_output(["git","-C",str(src),"rev-parse","HEAD"],
                                         text=True,timeout=35).strip()
            dirty=subprocess.check_output(["git","-C",str(src),"diff","--name-only","HEAD"],
                                          text=True,timeout=35).strip()
            license_file=src/"LICENSE"
            if (repo!=policy["sha"] or dirty or
                not license_file.is_file() or
                not license_file.read_text("utf8").startswith("MIT License")):
                raise ValueError("H140 original MIT source checkout/revision drift: "+name)
        del item["hashes"]
        result[name]=item
    return {"schema":"hala.h140.private-original-source-pilot-quality-receipt.v1",
            "status":"PASS_PRIVATE_DIAGNOSTIC_ONLY_NOT_TRAINING",
            "three_app_original_pilot_cases":sum(r["case_count"] for r in result.values()),
            "pairwise_image_sha_overlap":0,
            "overlap_with_h136_h139_original_images":0,
            "historical_source_manifests":old_files,
            "historical_excluded_unique_images":len(old),
            "data_training_eligible":False,
            "data_blind_eligible":False,
            "new_optimizer_steps":0,"flagship_35B_updated":False,
            "apps":result}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--private-root",required=True)
    p.add_argument("--experiments-root")
    args=p.parse_args()
    print(json.dumps(audit(args.private_root,args.experiments_root),
                     sort_keys=True,indent=2))
