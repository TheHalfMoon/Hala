"""H114 freeze a 256-step paired-start/end LOCALIZATION SFT pilot schedule.

This is ORIGINAL source TRAIN-only admission. Actual model training MUST use
the full attached and rederived H71 source-media archive on private free T4.
No model quality, checkpoint or training performed by this module.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from h113.dual_point import COORDINATE_SPACE,ROLES,SCHEMA,relative_to_source_pixel
from h61.curriculum import strict_json

ORIGINAL_H113_DERIVED_SHA="9a1bebc03c22825a6e0abef54edd2778f29e8144685884b811d95a7d8a7b02c9"
ORIGINAL_H105_TRAIN_PLAN_SHA="ed69e0ea119c4e208ba6aa0a486cb29fb0c857e61fb278d6543152d022c40d0d"
SCHEDULE="hala.h114.independent-dual-point-128-image-paired-256-step-fresh-lora.v1"
SCREENSHOTS=128
STEPS=SCREENSHOTS*2
SOURCE_ROWS=1024


@dataclass(frozen=True)
class QualifiedSchedule:
    selected:tuple[dict,...]
    summary:dict


def _read_verified_rows(path:Path)->tuple[dict,...]:
    path=Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("exact owner-private original H113 source-qualified JSONL required")
    blob=path.read_bytes()
    if hashlib.sha256(blob).hexdigest()!=ORIGINAL_H113_DERIVED_SHA:
        raise ValueError("original owner-SHA-qualified H113 TRAIN point data changed")
    lines=blob.splitlines()
    if len(lines)!=SOURCE_ROWS:
        raise ValueError("H113 must provide exactly 1024 original point examples")
    rows=tuple(strict_json(line.decode("utf8")) for line in lines)
    return rows


def qualify(rows,expected_pairs:int=512,selected_pairs:int=SCREENSHOTS)->QualifiedSchedule:
    """Data validation only; no source IDs or target coords in returned summary."""
    if (type(expected_pairs) is not int or type(selected_pairs) is not int
            or not 1<=selected_pairs<=expected_pairs<=512
            or len(rows)!=expected_pairs*2):
        raise ValueError("H114 exact original TRAIN pair counts required")
    seen=set()
    selected=[]
    languages=Counter()
    families=Counter()
    seen_images=set()
    for i in range(expected_pairs):
        pair=rows[2*i:2*i+2]
        if len(pair)!=2:
            raise ValueError("incomplete original source pair")
        x,y=pair
        if (type(x) is not dict or type(y) is not dict
                or tuple(z.get("endpoint_role") for z in pair)!=ROLES):
            raise ValueError("each original TRAIN source must have START and END")
        for z,role in zip(pair,ROLES):
            orig=z.get("original_source_id")
            w,h=z.get("source_image_width"),z.get("source_image_height")
            label=strict_json(z.get("completion",""))
            if (z.get("schema")!=SCHEMA or z.get("split")!="train"
                    or z.get("original_source_split")!="train"
                    or z.get("source_bound_validation_passed") is not True
                    or z.get("no_eval_or_test_source") is not True
                    or not isinstance(orig,str) or not orig
                    or z.get("id")!=orig+"::"+role
                    or type(w) is not int or type(h) is not int
                    or not 64<=w<=1024 or not 64<=h<=640
                    or z.get("language") not in ("ar","en")
                    or not isinstance(z.get("family"),str) or not z["family"]
                    or not isinstance(z.get("image_path"),str)
                    or not isinstance(z.get("image_sha256"),str)
                    or len(z["image_sha256"])!=64
                    or not isinstance(z.get("prompt"),str)
                    or role.upper() not in z["prompt"]
                    or "bbox" in z["prompt"]
                    or "source_image_pixels" in z["prompt"]
                    or type(label) is not dict
                    or set(label)!={"point","role","coordinate_space"}
                    or label["role"]!=role
                    or label["coordinate_space"]!=COORDINATE_SPACE):
                raise ValueError("H114 point target not original source-bound independent role")
            relative_to_source_pixel(label["point"],w,h)
            if z["id"] in seen:
                raise ValueError("duplicate TRAIN point role ID")
            seen.add(z["id"])
        if any(x.get(k)!=y.get(k) for k in
               ("original_source_id","image_sha256","image_path","family",
                "language","source_image_width","source_image_height")):
            raise ValueError("original paired endpoint training image identity split")
        if x["image_sha256"] in seen_images:
            raise ValueError("original TRAIN screenshot appears in multiple pairs")
        seen_images.add(x["image_sha256"])
        if i<selected_pairs:
            languages[x["language"]]+=1
            families[x["family"]]+=1
            selected.extend(pair)
    if len(selected)!=2*selected_pairs or len(set(z["id"] for z in selected))!=len(selected):
        raise ValueError("H114 optimizer source rows not unique")
    signature=hashlib.sha256((
        SCHEDULE+":"+ORIGINAL_H113_DERIVED_SHA+":"+
        hashlib.sha256(json.dumps([z["id"] for z in selected],
          ensure_ascii=False,separators=(",",":")).encode("utf8")).hexdigest()
        ).encode("utf8")).hexdigest()
    summary={
      "schedule":SCHEDULE,
      "original_h113_point_jsonl_sha256":ORIGINAL_H113_DERIVED_SHA,
      "original_h105_frozen_plan_sha256":ORIGINAL_H105_TRAIN_PLAN_SHA,
      "original_train_screenshots_inspected":expected_pairs,
      "pilot_selected_original_screenshots":selected_pairs,
      "selected_dual_point_optimizer_steps":2*selected_pairs,
      "start_examples":selected_pairs,"end_examples":selected_pairs,
      "selected_languages":dict(sorted(languages.items())),
      "selected_families":dict(sorted(families.items())),
      "selected_identities_anonymized_sha256":signature,
      "source_split":"TRAIN_ONLY","raw_point_coordinates_in_receipt":False,
      "hidden_test_or_eval_in_training":False,"model_trained":False}
    return QualifiedSchedule(tuple(selected),summary)


def qualify_owner_private(path:Path)->QualifiedSchedule:
    return qualify(_read_verified_rows(path),512,SCREENSHOTS)



def trainer_row(original:dict,derived:dict)->dict:
    """Preserve original media/path/format, replace ONLY private point objective.

    Call only after strict paired TRAIN dataset SHA and original source indices
    are admitted. This must not mutate immutable H71 or H113 records.
    """
    if (type(original) is not dict or type(derived) is not dict
            or derived.get("original_source_id")!=original.get("id")
            or derived.get("image_sha256")!=original.get("image_sha256")
            or derived.get("image_path")!=original.get("image_path")
            or derived.get("language")!=original.get("language")
            or derived.get("family")!=original.get("family")
            or derived.get("split")!="train"):
        raise ValueError("H114 private dual point overlay does not match original H71 TRAIN source")
    result=dict(original)
    for field in ("id","prompt","completion"):
        result[field]=derived[field]
    return result
