"""H140 position-diverse ORIGINAL GUI corpus and OOD admission gates.

CPU-only validation; no neural inference, training, cloud spend, or release.
Blind applications are disjoint from the training/development applications.
"""
import hashlib
import json
import math
import struct
from collections import Counter,defaultdict
from pathlib import Path

SCHEMA="hala.h140.original-app-multi-geometry-bilingual-source.v1"
COUNTS={"train":256,"dev":64,"blind_test":32}
MIN_APPS={"train":3,"dev":3,"blind_test":2}
MIN_UNIQUE_BINS={"train":8,"dev":4,"blind_test":8}
WIDTH,HEIGHT=960,720
GRID_PIXELS=24
PNG_HEADER=b"\x89PNG\r\n\x1a\n"

def digest(data):
    return hashlib.sha256(data).hexdigest()

def center(box):
    if not isinstance(box,dict) or any(
            type(box.get(k)) not in (int,float) or
            not math.isfinite(box[k]) for k in ("x","y","width","height")):
        raise ValueError("H140 genuine numeric DOM bounding rectangle required")
    x,y,w,h=(box[k] for k in ("x","y","width","height"))
    if w<=0 or h<=0 or x<0 or y<0 or x+w>WIDTH or y+h>HEIGHT:
        raise ValueError("H140 target must be fully visible in original native screenshot")
    return (x+w/2,y+h/2)

def position_bin(box):
    x,y=center(box)
    return (int(x)//GRID_PIXELS,int(y)//GRID_PIXELS)

def actual_png_bytes(folder,row):
    name=row.get("image")
    if (not isinstance(name,str) or not name.endswith(".png") or
            "/" in name or "\\" in name or ".." in name):
        raise ValueError("H140 safe original PNG basename required")
    binary=(folder/name).read_bytes()
    if (digest(binary)!=row.get("image_sha256") or
            not binary.startswith(PNG_HEADER) or len(binary)<24):
        raise ValueError("H140 SHA-qualified original Chromium PNG required")
    width,height=struct.unpack(">II",binary[16:24])
    if (width,height)!=(WIDTH,HEIGHT):
        raise ValueError("H140 original 960x720 frame required; no synthetic resizing")
    return binary

def load_split(folder,split):
    folder=Path(folder)
    if split not in COUNTS:
        raise ValueError("H140 unknown research split")
    doc=json.loads((folder/"manifest.json").read_text(encoding="utf8"))
    rows=doc.get("cases")
    if (doc.get("schema")!=SCHEMA or doc.get("private") is not True or
        doc.get("split")!=split or doc.get("training_eligible") is not (split!="blind_test") or
        not isinstance(rows,list) or len(rows)!=COUNTS[split]):
        raise ValueError("H140 frozen source split/count/private role rejected")
    sources=doc.get("app_sources")
    if not isinstance(sources,dict) or len(sources)<MIN_APPS[split]:
        raise ValueError("H140 independent full-app source diversity insufficient")
    for name,entry in sources.items():
        if (not isinstance(entry,dict) or
            entry.get("license")!="MIT" or entry.get("license_file_verified") is not True or
            entry.get("tracked_source_modified") is not False or
            not isinstance(entry.get("upstream_sha"),str) or
            len(entry["upstream_sha"])!=40 or
            entry.get("application_is_complete") is not True):
            raise ValueError("H140 exact original MIT application provenance missing")
    seen=set(); by_app=defaultdict(list)
    for row in rows:
        app=row.get("app")
        if app not in sources or row.get("app_git_sha")!=sources[app]["upstream_sha"]:
            raise ValueError("H140 original immutable app Git revision mismatch")
        if row.get("split")!=split or row.get("language") not in ("en","ar"):
            raise ValueError("H140 split/language must be preregistered")
        if (row.get("reference_drag_persisted") is not True or
            row.get("model_input_contains_oracle_coordinates") is not False):
            raise ValueError("H140 real native Chromium reference action required, oracle never shown to model")
        instruction=row.get("instruction")
        if (not isinstance(instruction,str) or len(instruction)<8 or
            any(needle in instruction.lower() for needle in ("point_2d","bbox","pixel coordinate"))):
            raise ValueError("H140 external task text must omit oracle geometry")
        image_hash=row.get("image_sha256")
        if (not isinstance(image_hash,str) or len(image_hash)!=64 or
            image_hash in seen):
            raise ValueError("H140 source image SHA missing or duplicated")
        actual_png_bytes(folder,row)
        seen.add(image_hash)
        position_bin(row["start_bbox"])
        position_bin(row["end_bbox"])
        by_app[app].append(row)
    if set(by_app)!=set(sources):
        raise ValueError("H140 registered real apps cannot be unused")
    for app,cases in by_app.items():
        languages=Counter(r["language"] for r in cases)
        if languages["ar"]!=languages["en"]:
            raise ValueError("H140 each app must have equal real EN/AR cases")
        for role in ("start","end"):
            unique={position_bin(row[role+"_bbox"]) for row in cases}
            if len(unique)<MIN_UNIQUE_BINS[split]:
                raise ValueError("H140 fixed-coordinate collapse at original data source: "+app+" "+role)
    return doc

def admit(train,dev,blind,old_manifests):
    if not old_manifests:
        raise ValueError("H140 past experiments' image hashes must be excluded")
    old=set()
    for path in old_manifests:
        data=json.loads(Path(path).read_text(encoding="utf8"))
        cases=data.get("cases",[])
        if not isinstance(cases,list) or not cases:
            raise ValueError("H140 invalid original historic exclusion manifest")
        for row in cases:
            if isinstance(row.get("image_sha256"),str):
                old.add(row["image_sha256"])
    splits={s:load_split(folder,s) for s,folder in
            (("train",train),("dev",dev),("blind_test",blind))}
    training_apps=set(splits["train"]["app_sources"])
    if (set(splits["dev"]["app_sources"])!=training_apps or
        set(splits["blind_test"]["app_sources"])&training_apps):
        raise ValueError("H140 unseen BLIND app implementations must be wholly independent")
    seen=set()
    for split,doc in splits.items():
        for row in doc["cases"]:
            h=row["image_sha256"]
            if h in seen or h in old:
                raise ValueError("H140 cross-split or historic original image reused")
            seen.add(h)
    return {
        "schema":"hala.h140.source-geometry-admission-only.v1",
        "status":"QUALIFIED_SOURCE_ONLY_NOT_TRAINED",
        "cases":{s:len(d["cases"]) for s,d in splits.items()},
        "distinct_images":len(seen),
        "training_apps":sorted(training_apps),
        "blind_apps":sorted(splits["blind_test"]["app_sources"]),
        "real_neural_calls":0,
        "model_improvement_proven":False,
        "main_Hala_35B_updated":False,
    }

def collapse_gate(points):
    """Guard a dynamic 16-case native-point slice against coordinate copying."""
    if not isinstance(points,list) or len(points)!=16:
        raise ValueError("H140 exactly 16 original model outputs required")
    valid=[]
    for point in points:
        if (not isinstance(point,(list,tuple)) or len(point)!=2 or
            any(type(v) not in (int,float) or not math.isfinite(v) for v in point) or
            point[0]<0 or point[0]>=WIDTH or point[1]<0 or point[1]>=HEIGHT):
            raise ValueError("H140 invalid neural point cannot become an oracle fallback")
        valid.append((round(point[0],2),round(point[1],2)))
    counts=Counter(valid)
    unique=len(counts)
    mode=max(counts.values())
    return {"valid":16,"unique_points":unique,
            "modal_share":mode/16,
            "passed":unique>=8 and mode<=4}
