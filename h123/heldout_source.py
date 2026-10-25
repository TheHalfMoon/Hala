"""H123 authentic original H71 TRAIN128 image/END-box qualification."""
from pathlib import Path
from PIL import Image
from h113.dual_point import valid_box
from h123.schedule import Schedule,HOLDOUT,FAMILIES
from h123.training_plan import get_plan
from h123.objective import from_original
from h122.source_bound import admit_original_source_map,original_for_sft
from training.artifacts import artifact_path,sha256_file

def admit_original_train128(rows,root,schedule:Schedule):
    plan=get_plan(schedule)
    root=Path(root).resolve()
    if len(rows)!=10000 or not root.is_dir() or root.is_symlink():
        raise ValueError("H123 source original TRAIN root invalid")
    upstream=admit_original_source_map(root)
    seen=set();images=set();cases=[]
    for index,role in zip(schedule.heldout_indices,schedule.heldout):
        original=rows[index]
        source=original_for_sft(original,upstream)
        expected=from_original(original,source)
        if (source["id"] in seen or source["image_sha256"] in images
                or source.get("split")!="train" or source["family"] not in FAMILIES
                or source.get("language")!="en"
                or expected["completion"]!=role["completion"]
                or expected["prompt"]!=role["prompt"]
                or expected["image_sha256"]!=role["image_sha256"]):
            raise ValueError("H123 true original source END target drift")
        media=artifact_path(root,source["image_path"])
        if not media.is_file() or media.is_symlink() or sha256_file(media)!=source["image_sha256"]:
            raise ValueError("H123 genuine TRAIN screenshot SHA mismatch")
        with Image.open(media) as image:
            if image.size!=(source["image_width"],source["image_height"]) or image.format not in ("PNG","JPEG"):
                raise ValueError("H123 original image dimensions/media drift")
        box=source["end_bbox"]
        if not valid_box(box,source["image_width"],source["image_height"]):
            raise ValueError("H123 original END target bbox missing")
        cases.append({"id":source["id"],"split":"train",
                      "source_split":"TRAIN_ONLY_HELDOUT",
                      "family":source["family"],"language":"en",
                      "image_path":str(media),"image_sha256":source["image_sha256"],
                      "image_width":source["image_width"],
                      "image_height":source["image_height"],
                      "prompt":role["prompt"],"end_bbox":list(box)})
        seen.add(source["id"]);images.add(source["image_sha256"])
    if len(cases)!=HOLDOUT or len(seen)!=HOLDOUT or len(images)!=HOLDOUT:
        raise ValueError("H123 exactly 128 unique original heldout screenshots required")
    return cases
