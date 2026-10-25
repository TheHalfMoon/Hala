"""H124 owner-private new 128 original H71 TRAIN-only double-box admission."""
from pathlib import Path
from PIL import Image
from h113.dual_point import valid_box
from h124.schedule import Schedule,HOLDOUT,FAMILIES
from h124.training_plan import get_plan
from h124.objective import from_original,parse
from h122.source_bound import admit_original_source_map,original_for_sft
from training.artifacts import artifact_path,sha256_file

def admit_original_train128(rows,root,schedule:Schedule):
    get_plan(schedule)
    root=Path(root).resolve()
    if len(rows)!=10000 or not root.is_dir() or root.is_symlink():
        raise ValueError("H124 genuine original H71 10000 TRAIN source missing")
    originals=admit_original_source_map(root)
    ids=set();media_shas=set();cases=[]
    for idx,role in zip(schedule.heldout_indices,schedule.heldout):
        original=rows[idx]
        source=original_for_sft(original,originals)
        expected=from_original(original,source)
        if (source["id"] in ids or source["image_sha256"] in media_shas
            or source.get("split")!="train" or source["family"] not in FAMILIES
            or source.get("language")!="en"
            or expected["original_source_id"]!=role["original_source_id"]
            or expected["prompt"]!=role["prompt"]
            or expected["completion"]!=role["completion"]
            or expected["image_sha256"]!=role["image_sha256"]):
            raise ValueError("H124 original independent source joint geometry drift")
        image=artifact_path(root,source["image_path"])
        if not image.is_file() or image.is_symlink() or sha256_file(image)!=source["image_sha256"]:
            raise ValueError("H124 genuine heldout source image bytes SHA changed")
        with Image.open(image) as media:
            if media.size!=(source["image_width"],source["image_height"]) or media.format not in ("PNG","JPEG"):
                raise ValueError("H124 original TRAIN image dimensions/mime differ")
        w,h=source["image_width"],source["image_height"]
        both=parse(role["completion"],w,h)
        if both is None:
            raise ValueError("H124 genuine both original source target boxes absent")
        for j,box_name in enumerate(("start_bbox","end_bbox")):
            box=source[box_name]
            if not valid_box(box,w,h):
                raise ValueError("H124 original source START/END box invalid")
            x,y=both[j]
            if not (box[0]<=x<=box[2] and box[1]<=y<=box[3]):
                raise ValueError("H124 source normalized both points escape source target boxes")
        cases.append({"id":source["id"],"split":"train",
          "source_split":"TRAIN_ONLY_HELDOUT","family":source["family"],
          "language":"en","image_path":str(image),
          "image_sha256":source["image_sha256"],
          "image_width":w,"image_height":h,"prompt":role["prompt"],
          "start_bbox":list(source["start_bbox"]),
          "end_bbox":list(source["end_bbox"])})
        ids.add(source["id"]);media_shas.add(source["image_sha256"])
    if len(cases)!=HOLDOUT or len(ids)!=HOLDOUT or len(media_shas)!=HOLDOUT:
        raise ValueError("H124 exactly 128 new original source TRAIN-only holdout cases required")
    return cases
