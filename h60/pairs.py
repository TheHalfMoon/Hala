"""H60 original DragOn TRAIN image/task pairing rules.

Every rule here is derived from the original upstream source format that was
independently observed in the pinned TRAIN archives. Nothing in this module
invents a label, a box, a dimension or a task: it only accepts what the
original source actually contains and records an explicit rejection reason for
everything else.
"""
from __future__ import annotations

import hashlib
import io
import json
import re

from PIL import Image

UUID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")
CREDENTIAL_MARKERS = (
    "password",
    "api key",
    "social security",
    "@gmail.",
    "otp",
    "private key",
)
MIN_DIMENSION = 300
MAX_DIMENSION = 5000
MAX_PIXELS = 12_000_000
MAX_TASKS_PER_IMAGE = 100


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def is_uuid(value: str) -> bool:
    return bool(UUID.fullmatch(value))


def image_digest(payload: bytes) -> tuple[int, int, str]:
    """Return (width, height, format) from the original JPEG header."""
    if payload[:3] != b"\xff\xd8\xff":
        raise ValueError("not an original JPEG byte stream")
    with Image.open(io.BytesIO(payload)) as picture:
        size = picture.size
        image_format = picture.format
    if image_format != "JPEG":
        raise ValueError("original member is not a JPEG image")
    return int(size[0]), int(size[1]), image_format


def image_tail_ok(payload: bytes) -> bool:
    return payload[-2:] == b"\xff\xd9"


def verify_jpeg_structure(payload: bytes) -> None:
    """Full structural verification of one original JPEG (no pixel decode)."""
    with Image.open(io.BytesIO(payload)) as picture:
        picture.verify()


def decode_jpeg_pixels(payload: bytes) -> None:
    """Full pixel decode of one original JPEG."""
    with Image.open(io.BytesIO(payload)) as picture:
        picture.load()


def parse_source_tasks(payload: bytes) -> list[dict]:
    """Parse the original per-image task JSON exactly as the source stores it."""
    try:
        source = json.loads(payload.decode("utf8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("original task JSON is not valid UTF-8 JSON") from error
    if not isinstance(source, list):
        raise ValueError("original task JSON is not a list")
    if not 1 <= len(source) <= MAX_TASKS_PER_IMAGE:
        raise ValueError("original task JSON has an implausible task count")
    return source


def candidate_from_source_task(domain: str, key: str, index: int, task: object) -> dict:
    """Normalise one original source task into a candidate training record."""
    if not isinstance(task, dict):
        raise ValueError("original task entry is not an object")
    if task.get("image_id") != key:
        raise ValueError("original task image_id does not match its source file name")
    if task.get("domain") != domain:
        raise ValueError("original task domain does not match its source archive")
    if type(task.get("ordered")) is not bool:
        raise ValueError("original task has no boolean ordered flag")
    intent = task.get("intent")
    if not isinstance(intent, str) or not 4 <= len(intent) <= 1000:
        raise ValueError("original task intent is missing or implausible")
    lowered = intent.lower()
    if any(marker in lowered for marker in CREDENTIAL_MARKERS):
        raise ValueError("original task instruction is credential bearing")
    boxes = []
    for box_key in ("start_bbox", "end_bbox"):
        box = task.get(box_key)
        if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError("original task bounding box is missing or non integral")
        boxes.append([int(v) for v in box])
    subtype = task.get("subtype")
    if subtype is not None and (not isinstance(subtype, str) or len(subtype) > 140):
        raise ValueError("original task subtype is implausible")
    return {
        "source_task_index": index,
        "intent": intent,
        "ordered": bool(task["ordered"]),
        "start_bbox": boxes[0],
        "end_bbox": boxes[1],
        "subtype": subtype,
    }


def geometry_ok(candidate: dict, width: int, height: int) -> tuple[list, list]:
    """Validate the original drag geometry against the original image bounds."""
    for box in (candidate["start_bbox"], candidate["end_bbox"]):
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("original drag box leaves the original image bounds")
    start_centre = [
        (candidate["start_bbox"][0] + candidate["start_bbox"][2]) // 2,
        (candidate["start_bbox"][1] + candidate["start_bbox"][3]) // 2,
    ]
    end_centre = [
        (candidate["end_bbox"][0] + candidate["end_bbox"][2]) // 2,
        (candidate["end_bbox"][1] + candidate["end_bbox"][3]) // 2,
    ]
    if start_centre == end_centre:
        raise ValueError("original drag has a degenerate start and end point")
    return start_centre, end_centre


def build_row(
    domain: str,
    key: str,
    candidate: dict,
    width: int,
    height: int,
    image_sha: str,
    task_sha: str,
    image_offset: int,
    task_offset: int,
) -> dict:
    """Build one authentic training record from an original image/task pair."""
    start_centre, end_centre = geometry_ok(candidate, width, height)
    return {
        "id": f"h60-{domain}-{key}-t{candidate['source_task_index']:03d}",
        "split": "train",
        "domain": domain,
        "subtype": candidate["subtype"],
        "language": "en",
        "image_id": key,
        "image_path": f"images/{domain}/{key}.jpg",
        "image_sha256": image_sha,
        "image_width": width,
        "image_height": height,
        "source_task_json_path": f"tasks/{domain}/{key}.json",
        "source_task_sha256": task_sha,
        "source_task_index": candidate["source_task_index"],
        "source_image_tar_offset": image_offset,
        "source_task_tar_offset": task_offset,
        "intent": candidate["intent"],
        "action_type": "drag",
        "ordered": candidate["ordered"],
        "start_bbox": candidate["start_bbox"],
        "end_bbox": candidate["end_bbox"],
        "completion": {
            "type": "drag",
            "start_point": start_centre,
            "end_point": end_centre,
            "coordinate_space": "source_image_pixels",
            "ordered": candidate["ordered"],
        },
    }
