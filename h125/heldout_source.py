"""H125 true new original TRAIN64 image and dual-box per-arm admission."""
from pathlib import Path
from PIL import Image
from h113.dual_point import derive_two_independent_points,valid_box
from h125.source import Schedule,TOTAL
from h125.eval_plan import get_plan
from h125.model_contracts import MODELS,prompt_for
from h122.source_bound import admit_original_source_map,original_for_sft
from training.artifacts import artifact_path,sha256_file

def admit_original_train64(rows,root,plan:Schedule,model:str):
    get_plan(plan)
    if model not in MODELS:
        raise ValueError("H125 only two preregistered specialized 2B GUI models")
    root=Path(root).resolve()
    if len(rows)!=10000 or not root.is_dir() or root.is_symlink():
        raise ValueError("H125 original H71 full TRAIN source is required")
    upstream=admit_original_source_map(root)
    ids=set();image_shas=set();cases=[]
    for idx,entry in zip(plan.indices,plan.cases):
        row=rows[idx]
        original=original_for_sft(row,upstream)
        points=derive_two_independent_points(row,original)
        if (entry["id"] in ids or entry["image_sha256"] in image_shas
                or original.get("split")!="train"
                or original["id"]!=entry["id"]
                or original["image_sha256"]!=entry["image_sha256"]
                or original["intent"]!=entry["intent"]
                or points[0]["original_source_id"]!=entry["id"]
                or points[1]["original_source_id"]!=entry["id"]
                or entry["split"]!="train"):
            raise ValueError("H125 new source dual endpoint original SHA drift")
        media=artifact_path(root,original["image_path"])
        if not media.is_file() or media.is_symlink() or sha256_file(media)!=entry["image_sha256"]:
            raise ValueError("H125 original source media bytes changed")
        with Image.open(media) as picture:
            if picture.size!=(original["image_width"],original["image_height"]) or picture.format not in ("PNG","JPEG"):
                raise ValueError("H125 original TRAIN image mime or dimensions differ")
        w,h=original["image_width"],original["image_height"]
        for side in ("start","end"):
            if not valid_box(original[side+"_bbox"],w,h):
                raise ValueError("H125 authentic START and END bbox missing")
        cases.append({
           "id":entry["id"],"split":"train",
           "source_split":"TRAIN_ONLY_FRESH_NEVER_TRAINED",
           "family":entry["family"],"language":entry["language"],
           "image_path":str(media),"image_sha256":entry["image_sha256"],
           "image_width":w,"image_height":h,
           "start_bbox":list(original["start_bbox"]),
           "end_bbox":list(original["end_bbox"]),
           "prompt_start":prompt_for(model,original["intent"],"START"),
           "prompt_end":prompt_for(model,original["intent"],"END")})
        ids.add(entry["id"]);image_shas.add(entry["image_sha256"])
    if len(cases)!=TOTAL or len(ids)!=TOTAL or len(image_shas)!=TOTAL:
        raise ValueError("H125 exactly 64 entirely new original TRAIN source media required")
    return cases
