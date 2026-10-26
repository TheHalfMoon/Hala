"""H127 owner-private blind GUI identifiability audit; no model inference."""
import hashlib
import json
import math
import secrets
import zipfile
from collections import Counter
from pathlib import Path

EXPECTED_SCHEDULE = "8a54ff0af4cfdeb9572ef741b205cbe377738eee0bbd630373ee95739b20ab2d"
FAMILIES = (
    "hcompany_dragon_h61_sheet_train",
    "hcompany_dragon_h61_slide_resize_train",
    "hcompany_dragon_h61_slider_train",
    "hcompany_dragon_h61_text_highlight_train",
)
COUNT = 64
SCHEMA = "hala.h127.blind-review.v1"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def make_bundle(cases, root, schedule, pack_file, truth_file):
    """Two physically separate private artifacts: blind media and hidden source truth.

    Does not verify reviewer identity or human independence. Does not qualify the model.
    """
    from PIL import Image
    from training.artifacts import artifact_path

    root = Path(root)
    pack_file, truth_file = Path(pack_file), Path(truth_file)
    if schedule != EXPECTED_SCHEDULE or len(cases) != COUNT:
        raise ValueError("H127 requires frozen H126 64 original TRAIN-only source cases")
    if pack_file.exists() or truth_file.exists():
        raise ValueError("H127 private artifacts are create-only")
    if pack_file.parent != truth_file.parent:
        raise ValueError("H127 blind bundle and withheld truth must share private owner location")
    counts = Counter(c["family"] for c in cases)
    if counts != Counter({family: 16 for family in FAMILIES}):
        raise ValueError("H127 exact original TRAIN64 four-family balance required")
    ids, shas, task_records, truth_records = set(), set(), [], []
    items = []
    for c in cases:
        if c["id"] in ids or c["image_sha256"] in shas or c.get("split") != "train":
            raise ValueError("H127 duplicate or non-TRAIN original source")
        ids.add(c["id"])
        shas.add(c["image_sha256"])
        media = artifact_path(root, c["image_path"])
        if not media.is_file() or media.is_symlink():
            raise ValueError("H127 source media absent or symlink")
        raw = media.read_bytes()
        if digest(raw) != c["image_sha256"]:
            raise ValueError("H127 source media SHA mismatch")
        with Image.open(media) as image:
            if image.format not in ("JPEG", "PNG") or image.size != (c["image_width"], c["image_height"]):
                raise ValueError("H127 media format or native dimensions mismatch")
            ext = "jpg" if image.format == "JPEG" else "png"
        if not isinstance(c.get("intent"), str) or not c["intent"].strip():
            raise ValueError("H127 missing authentic task instruction")
        for name in ("start_bbox", "end_bbox"):
            b = c[name]
            if len(b) != 4 or any(type(v) is not int for v in b):
                raise ValueError("H127 original bbox must use 4 integer coordinates")
            if not (0 <= b[0] <= b[2] < c["image_width"] and
                    0 <= b[1] <= b[3] < c["image_height"]):
                raise ValueError("H127 source bbox outside native image")
        key = secrets.token_hex(16)
        image_name = "images/" + key + "." + ext
        task_records.append({
            "key": key, "image": image_name, "instruction": c["intent"],
            "width": c["image_width"], "height": c["image_height"],
        })
        truth_records.append({
            "key": key, "family": c["family"], "start_bbox": c["start_bbox"],
            "end_bbox": c["end_bbox"], "image_sha256": c["image_sha256"],
            "width": c["image_width"], "height": c["image_height"],
        })
        items.append((image_name, raw))
    # Independently randomized blind task order. Source IDs and target labels
    # never enter the reviewer-facing ZIP.
    secrets.SystemRandom().shuffle(task_records)
    blind = {"schema": SCHEMA, "source_schedule_sha256": EXPECTED_SCHEDULE,
             "tasks": task_records, "label_visibility": "BLIND",
             "reviewer_instruction": "Click START and END or abstain if unclear."}
    pack_file.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(pack_file, "x", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("tasks.json", json.dumps(blind, ensure_ascii=False, sort_keys=True))
        for image_name, raw in items:
            archive.writestr(image_name, raw)
    pack_sha = digest(pack_file.read_bytes())
    truth = {"schema": SCHEMA, "source_schedule_sha256": EXPECTED_SCHEDULE,
             "pack_sha256": pack_sha, "truth": truth_records,
             "reviewer_identities_verified": False,
             "human_review_completed": False}
    with truth_file.open("x", encoding="utf8") as out:
        json.dump(truth, out, indent=2, sort_keys=True)
    return {"status": "PASS_H127_PRIVATE_BLIND_PACK_CREATED_NOT_REVIEWED",
            "tasks": COUNT, "pack_sha256": pack_sha,
            "schedule_sha256": EXPECTED_SCHEDULE, "human_review_completed": False,
            "optimizer_steps": 0, "model_inference_calls": 0}


def _within(point, bbox):
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def validate_submission(submission, truth):
    if submission.get("schema") != SCHEMA or submission.get("pack_sha256") != truth["pack_sha256"]:
        raise ValueError("H127 reviewer submission wrong schema/pack")
    name = submission.get("reviewer_id")
    if not isinstance(name, str) or not 4 <= len(name) <= 80 or not name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("H127 reviewer ID missing or invalid")
    if submission.get("independent_blind_attestation") is not True:
        raise ValueError("H127 reviewer must self-attest blind independent work")
    answers = submission.get("annotations")
    if not isinstance(answers, list) or len(answers) != COUNT:
        raise ValueError("H127 complete 64-scene review required")
    lookup = {c["key"]: c for c in truth["truth"]}
    if len(lookup) != COUNT:
        raise ValueError("H127 duplicate keys in private ground truth")
    seen = set()
    for item in answers:
        if not isinstance(item, dict) or item.get("key") not in lookup or item["key"] in seen:
            raise ValueError("H127 invalid or duplicate review key")
        seen.add(item["key"])
        status = item.get("status")
        if status not in ("located", "ambiguous", "unreadable"):
            raise ValueError("H127 missing explicit review status")
        original = lookup[item["key"]]
        if status == "located":
            for field, bound in (("start", "width"), ("end", "width")):
                p = item.get(field)
                if not isinstance(p, list) or len(p) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in p):
                    raise ValueError("H127 invalid annotation coordinate")
                if not (0 <= p[0] < original["width"] and 0 <= p[1] < original["height"]):
                    raise ValueError("H127 pixel coordinates outside screenshot")
        elif item.get("start") is not None or item.get("end") is not None:
            raise ValueError("H127 abstention cannot contain coordinates")
    return {a["key"]: a for a in answers}


def grade(truth, a, b):
    """Aggregate screening evidence only, not independent human verification."""
    if truth.get("schema") != SCHEMA or truth.get("source_schedule_sha256") != EXPECTED_SCHEDULE:
        raise ValueError("H127 hidden truth does not match frozen original sources")
    amap, bmap = validate_submission(a, truth), validate_submission(b, truth)
    if a["reviewer_id"] == b["reviewer_id"]:
        raise ValueError("H127 two distinct reviewers required")
    totals = {f: {"count": 0, "a_correct": 0, "b_correct": 0,
                  "both_correct": 0, "a_abstain": 0, "b_abstain": 0} for f in FAMILIES}
    for case in truth["truth"]:
        family = case["family"]
        if family not in totals:
            raise ValueError("H127 unknown TRAIN source family")
        t = totals[family]
        t["count"] += 1
        for item, prefix in ((amap[case["key"]], "a"), (bmap[case["key"]], "b")):
            if item["status"] != "located":
                t[prefix + "_abstain"] += 1
                continue
            if _within(item["start"], case["start_bbox"]) and _within(item["end"], case["end_bbox"]):
                t[prefix + "_correct"] += 1
        if (amap[case["key"]]["status"] == "located" and
            bmap[case["key"]]["status"] == "located" and
            all(_within(amap[case["key"]][k], case[k + "_bbox"]) and
                _within(bmap[case["key"]][k], case[k + "_bbox"])
                for k in ("start", "end"))):
            t["both_correct"] += 1
    if any(v["count"] != 16 for v in totals.values()):
        raise ValueError("H127 16 original TRAIN scenes per family required")
    aggregate = {k: sum(v[k] for v in totals.values()) for k in
                 ("count", "a_correct", "b_correct", "both_correct", "a_abstain", "b_abstain")}
    # Conservative preregistered screening threshold, not publication/model gate:
    # 58/64 jointly correct, >=13/16 per domain, and <=4 abstentions per reviewer.
    threshold = (aggregate["both_correct"] >= 58 and
                 all(v["both_correct"] >= 13 for v in totals.values()) and
                 aggregate["a_abstain"] <= 4 and aggregate["b_abstain"] <= 4)
    return {
        "schema": "hala.h127.blind-original-TRAIN64-identifiability-score.v1",
        "status": "PROVISIONAL_SELF_ATTESTED_SCREEN_PASS" if threshold else "FAIL_OR_AMBIGUOUS_SOURCE_IDENTIFIABILITY",
        "source_schedule_sha256": EXPECTED_SCHEDULE,
        "pack_sha256": truth["pack_sha256"],
        "source_is_original_TRAIN_only": True,
        "per_family": totals, "aggregate": aggregate,
        "two_distinct_self_attested_reviewers": True,
        "reviewer_identities_independently_verified": False,
        "model_skill_qualified": False, "new_optimizer_steps": 0,
        "new_model_inference_calls": 0,
    }
