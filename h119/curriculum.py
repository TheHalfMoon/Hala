"""H119: TRAIN-only 512-original-image START/END with robust input anchors.

No neural training or EVAL/TEST access. Real source must match H113 SHA.
"""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

from h114.schedule import QualifiedSchedule, ORIGINAL_H113_DERIVED_SHA, _read_verified_rows, qualify
from h116.delta import derive_pair, prompt_end_delta, SPACE
from h61.official_eval import strict_json

SCHEMA = "hala.h119.full512-source-only-anchor-robust-signed-end.v1"
NONCE = b"hala.h119.original-train-only-anchor-shift.v1"
PAIRS = 512
SCALES = (0, 24, 64, 128)


@dataclass(frozen=True)
class Schedule:
    selected: tuple[dict, ...]
    summary: dict


def shift_anchor(start, image_sha256):
    if (type(start) is not list or len(start) != 2 or
            any(type(v) is not int or not 0 <= v <= 1000 for v in start) or
            type(image_sha256) is not str or len(image_sha256) != 64):
        raise ValueError("verified TRAIN point and image SHA required")
    try:
        digest = hashlib.sha256(NONCE + bytes.fromhex(image_sha256)).digest()
    except ValueError as exc:
        raise ValueError("invalid TRAIN source SHA") from exc
    scale = SCALES[digest[0] % 4]
    if scale == 0:
        return list(start), scale
    out = []
    for i, value in enumerate(start):
        sign = -1 if digest[1+i] % 2 else 1
        candidate = value + sign * scale
        if not 0 <= candidate <= 1000:
            candidate = value - sign * scale
        out.append(candidate)
    if out == start or any(not 0 <= x <= 1000 for x in out):
        raise ValueError("TRAIN anchor perturbation invalid")
    return out, scale


def derive_schedule(parent: QualifiedSchedule) -> Schedule:
    if (not isinstance(parent, QualifiedSchedule) or
            parent.summary.get("source_split") != "TRAIN_ONLY" or
            parent.summary.get("original_h113_point_jsonl_sha256") != ORIGINAL_H113_DERIVED_SHA or
            parent.summary.get("pilot_selected_original_screenshots") != PAIRS or
            parent.summary.get("selected_dual_point_optimizer_steps") != 2*PAIRS or
            len(parent.selected) != 2*PAIRS):
        raise ValueError("H119 requires exact original 512 TRAIN pairs")
    output = []
    media = set()
    bands, langs, families = Counter(), Counter(), Counter()
    for i in range(PAIRS):
        first, end = derive_pair(*parent.selected[2*i:2*i+2])
        if first["image_sha256"] in media:
            raise ValueError("duplicate source screenshot")
        media.add(first["image_sha256"])
        start = strict_json(first["completion"])["point"]
        delta = strict_json(end["completion"])["delta"]
        true_end = [x+y for x,y in zip(start,delta)]
        anchor, scale = shift_anchor(start, first["image_sha256"])
        new_delta = [x-y for x,y in zip(true_end,anchor)]
        if ([a+b for a,b in zip(anchor,new_delta)] != true_end or
                any(type(v) is not int or not -1000 <= v <= 1000 for v in new_delta)):
            raise ValueError("original TRAIN END changed")
        source_intent = first["prompt"].rsplit("\nLocate ONLY the START endpoint",1)
        if len(source_intent) != 2:
            raise ValueError("invalid original source instruction")
        revised = dict(end)
        revised.update(schema=SCHEMA, prompt=prompt_end_delta(source_intent[0],anchor),
                       completion=json.dumps(
                           {"role":"end_delta","delta":new_delta,"coordinate_space":SPACE},
                           separators=(",",":"),sort_keys=True))
        if revised["split"] != "train" or revised["original_source_split"] != "train":
            raise ValueError("no EVAL/TEST in H119")
        output.extend((first,revised))
        bands[scale] += 1
        langs[first["language"]] += 1
        families[first["family"]] += 1
    if len(output) != 2*PAIRS or len(media) != PAIRS:
        raise ValueError("H119 pair/source counts incomplete")
    raw = b"".join(json.dumps(row,ensure_ascii=False,sort_keys=True,
              separators=(",",":")).encode("utf8")+b"\n" for row in output)
    return Schedule(tuple(output),{
        "schema":SCHEMA,"source_split":"TRAIN_ONLY",
        "original_h113_sha256":ORIGINAL_H113_DERIVED_SHA,
        "original_source_images":PAIRS,"start_roles":PAIRS,"end_delta_roles":PAIRS,
        "candidate_optimizer_steps":2*PAIRS,
        "anchor_magnitudes":{str(k):bands[k] for k in SCALES},
        "languages":dict(sorted(langs.items())),
        "families":dict(sorted(families.items())),
        "derived_private_train_sha256":hashlib.sha256(raw).hexdigest(),
        "no_eval_or_test_data":True,"raw_coordinates_exposed":False,
        "actual_model_training_or_improvement_claimed":False})


def qualify_owner_private(path: Path) -> Schedule:
    return derive_schedule(qualify(_read_verified_rows(Path(path)),512,512))
