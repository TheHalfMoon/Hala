"""H61 evaluation against the upstream DragOn official public evaluation split.

The first-party PUBLIC DEV sets measure the Hala agent against scenarios it
authored itself. That is a source-disjoint held-out set, but it is not an
independent benchmark, and the founder's requirement is independently
reproducible benchmark performance.

The pinned upstream repository publishes an official `eval/` split with the
same image and task format as `train/`: original screenshots, original task
JSONs, original boxes and the original `ordered` flag. It is third-party,
disjoint from TRAIN, and it was never read during the H60 corpus work.

This module therefore:

* fetches a bounded number of official evaluation scenes from the pinned
  revision and records every fetched file and digest;
* builds the prompt with the one shared H61 prompt contract, so the official
  evaluation, the public DEV evaluation and the training curriculum all use
  the same contract;
* scores the raw model outputs with an explicit geometry rule, and counts an
  unordered drag as successful in either direction only because the original
  task says the direction does not matter.

Nothing here trains, and no answer is ever placed in a prompt.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.request
from pathlib import Path

from h61.prompt_contract import drag_prompt, source_body

SCHEMA = "hala.h61.official-dragon-eval.v1"
SCENES_SCHEMA = "hala.h61.official-dragon-eval-scenes.v1"
REPO = "Hcompany/DragOn"
REV = "3451c1fc3543c18fc3361a32bfe1f12cc69b64b5"
API = "https://huggingface.co/api/datasets/{repo}/tree/{rev}/{path}"
RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"
DOMAINS = ("slider", "sheet", "slide_resize", "text_highlight")
MAX_SCENES = 400
PAGE = 250
USER_AGENT = "Hala-H61-official-dragon-eval"
STRICT_FIELDS = {"type", "start_point", "end_point", "coordinate_space", "ordered"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(obj) -> bytes:
    return (
        json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf8")


def _get(url: str, timeout: int = 120, attempts: int = 6) -> bytes:
    """Fetch one upstream file or listing, retrying a transient refusal.

    The source provider refuses a burst of parallel or repeated requests with
    a transient status, which is a transport condition and not a source
    condition, so it is retried with a bounded backoff.
    """
    import urllib.error

    last: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            last = error
            if error.code not in (400, 408, 429, 500, 502, 503, 504):
                raise
        except OSError as error:
            last = error
        time.sleep(min(30.0, 2.0 * (attempt + 1)))
    raise IOError(f"upstream request failed after {attempts} attempts: {last}")


def list_eval_files(domain: str, kind: str) -> list[dict]:
    """List the official evaluation files of one domain and kind.

    The provider answers one bounded page of 250 entries per listing and does
    not offer a usable continuation cursor, so the bounded page is the sample.
    The listing is ordered by path and the path is a content identifier, so the
    sample is not selected by task difficulty or by answer.
    """
    base = API.format(repo=REPO, rev=REV, path=f"eval/{domain}/{kind}")
    payload = json.loads(_get(base))
    return [row for row in payload if row.get("type") == "file"]


def fetch_domain(domain: str, limit: int, out_dir) -> dict:
    """Fetch a bounded number of official evaluation scenes for one domain.

    The image and the task file of one scene share the same content
    identifier, so the task is fetched by that identifier and not looked up in
    a listing. A bounded listing would only intersect the image list by
    chance, and the intersection would silently shrink the sample.
    """
    if type(limit) is not int or not 1 <= limit <= MAX_SCENES:
        raise ValueError("Bounded official evaluation scene count required")
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "tasks").mkdir(parents=True, exist_ok=True)
    images = list_eval_files(domain, "images")
    scenes = []
    missing_tasks = 0
    for row in images[:limit]:
        name = row["path"].split("/")[-1]
        key = name[:-4]
        task_path = f"eval/{domain}/tasks/{key}.json"
        image_bytes = _get(RESOLVE.format(repo=REPO, rev=REV, path=row["path"]))
        try:
            task_bytes = _get(RESOLVE.format(repo=REPO, rev=REV, path=task_path))
        except Exception:  # noqa: BLE001
            missing_tasks += 1
            continue
        (out_dir / "images" / name).write_bytes(image_bytes)
        (out_dir / "tasks" / f"{key}.json").write_bytes(task_bytes)
        source_tasks = json.loads(task_bytes.decode("utf8"))
        if not isinstance(source_tasks, list) or not source_tasks:
            missing_tasks += 1
            continue
        scenes.append(
            {
                "id": key,
                "domain": domain,
                "image_path": f"images/{name}",
                "image_sha256": sha(image_bytes),
                "image_bytes": len(image_bytes),
                "task_path": f"tasks/{key}.json",
                "task_sha256": sha(task_bytes),
                "source_tasks": source_tasks,
            }
        )
    receipt = {
        "schema": SCENES_SCHEMA,
        "source_repo": REPO,
        "source_commit_sha": REV,
        "source_split": f"eval/{domain}",
        "domain": domain,
        "scenes": scenes,
        "scene_count": len(scenes),
        "scenes_without_a_readable_task": missing_tasks,
        "official_eval_or_test_used": False,
        "benchmark_answer_leakage": False,
    }
    (out_dir / "scenes.json").write_bytes(canonical(receipt))
    return receipt


def build_prompt(scene: dict, task: dict) -> str:
    """The one shared contract, applied to one official evaluation task."""
    width, height = scene["image_width"], scene["image_height"]
    body = source_body(task["intent"], bool(task["ordered"]))
    return drag_prompt(body, width, height, bool(task["ordered"]))


def strict_json(text):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("Duplicate JSON key")
            value[key] = item
        return value

    return json.loads(text, object_pairs_hook=pairs)


def parse_action(raw, width: int, height: int) -> dict | None:
    """Strict parse of one raw model output, exactly like the public DEV scorer."""
    if not isinstance(raw, str) or len(raw) > 8192:
        return None
    try:
        obj = strict_json(raw)
    except (ValueError, TypeError):
        return None
    if (
        not isinstance(obj, dict)
        or set(obj) != STRICT_FIELDS
        or obj["type"] != "drag"
        or obj["coordinate_space"] != "source_image_pixels"
        or type(obj["ordered"]) is not bool
    ):
        return None
    for field in ("start_point", "end_point"):
        point = obj[field]
        if (
            not isinstance(point, list)
            or len(point) != 2
            or not all(type(value) is int for value in point)
            or not (0 <= point[0] < width and 0 <= point[1] < height)
        ):
            return None
    return obj


def _inside(point, box) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def score(scenes: list[dict], outputs: dict[str, str]) -> dict:
    """Geometry score of one official evaluation run.

    A scene succeeds when the predicted start point is inside the original
    start box and the predicted end point is inside the original end box. For
    an original unordered task the reversed direction is accepted, because the
    original task states that the direction does not matter.
    """
    per_scene = []
    for scene in scenes:
        raw = outputs.get(scene["id"])
        action = parse_action(raw, scene["image_width"], scene["image_height"]) if raw else None
        task = scene["task"]
        start_box = task["start_bbox"]
        end_box = task["end_bbox"]
        record = {
            "id": scene["id"],
            "domain": scene["domain"],
            "ordered": bool(task["ordered"]),
            "action_valid": action is not None,
            "degenerate": bool(
                action is not None and action["start_point"] == action["end_point"]
            ),
            "start_hit": False,
            "end_hit": False,
            "task_success": False,
        }
        if action is not None:
            start_hit = _inside(action["start_point"], start_box)
            end_hit = _inside(action["end_point"], end_box)
            reversed_hit = _inside(action["start_point"], end_box) and _inside(
                action["end_point"], start_box
            )
            record["start_hit"] = bool(start_hit)
            record["end_hit"] = bool(end_hit)
            record["task_success"] = bool(
                (start_hit and end_hit)
                or (not task["ordered"] and reversed_hit)
            )
        per_scene.append(record)
    total = len(per_scene)
    success = sum(1 for row in per_scene if row["task_success"])
    valid = sum(1 for row in per_scene if row["action_valid"])
    result = {
        "schema": SCHEMA,
        "source_repo": REPO,
        "source_commit_sha": REV,
        "source_split": "eval",
        "scenes": total,
        "action_valid": valid,
        "start_hit": sum(1 for row in per_scene if row["start_hit"]),
        "end_hit": sum(1 for row in per_scene if row["end_hit"]),
        "task_success": success,
        "task_success_rate": (success / total) if total else 0.0,
        "degenerate_start_equals_end": sum(1 for row in per_scene if row["degenerate"]),
        "ordered_scenes": sum(1 for row in per_scene if row["ordered"]),
        "unordered_scenes": sum(1 for row in per_scene if not row["ordered"]),
        "per_scene": per_scene,
        "official_public_eval_used": True,
        "hidden_test_used": False,
        "training_executed_by_scorer": False,
    }
    return result


def load_scenes(path) -> list[dict]:
    """Load the fetched official evaluation scenes with their measured dimensions."""
    payload = json.loads(Path(path).read_text("utf8"))
    if payload.get("schema") != SCENES_SCHEMA:
        raise ValueError("An official evaluation scene receipt is required")
    from PIL import Image

    root = Path(path).parent
    scenes = []
    for scene in payload["scenes"]:
        task = scene["source_tasks"][0]
        with Image.open(root / scene["image_path"]) as picture:
            width, height = picture.size
        scenes.append(
            {
                "id": scene["id"],
                "domain": scene["domain"],
                "image_path": str(root / scene["image_path"]),
                "image_sha256": scene["image_sha256"],
                "image_width": width,
                "image_height": height,
                "task": task,
                "prompt": build_prompt(scene | {"image_width": width, "image_height": height}, task),
            }
        )
    return scenes


def apply_frame(scenes: list[dict], out_dir, frame=(1024, 640)) -> list[dict]:
    """Move the official scenes into the H61 evaluation frame.

    The official evaluation screenshots are 1280x800 and the first-party public
    DEV screenshots are 1024x640. Both keep the same aspect ratio, so one
    uniform scale maps the official scenes into the evaluation frame, and the
    original start and end boxes are scaled with the original screenshot.

    The original bytes and the original boxes stay in the fetched receipt, and
    the framed scene records the original digest, the scale and the framed
    digest, so the transformation is auditable and exactly reversible.
    """
    from PIL import Image

    from h61.source_export import frame_target

    out_dir = Path(out_dir)
    (out_dir / "framed").mkdir(parents=True, exist_ok=True)
    frame_width, frame_height = frame
    framed_scenes = []
    for scene in scenes:
        with Image.open(scene["image_path"]) as picture:
            original_width, original_height = picture.size
            target_width, target_height, _ = frame_target(
                original_width, original_height, (frame_width, frame_height)
            )
            if (target_width, target_height) == (original_width, original_height):
                framed = picture.convert("RGB")
            else:
                framed = picture.convert("RGB").resize(
                    (target_width, target_height), Image.LANCZOS
                )
            target = out_dir / "framed" / f"{scene['id']}.jpg"
            framed.save(target, format="JPEG", quality=95, subsampling=0)
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        task = dict(scene["task"])
        for box_key in ("start_bbox", "end_bbox"):
            box = task[box_key]
            task[box_key] = [
                int(round(box[0] * scale_x)),
                int(round(box[1] * scale_y)),
                int(round(box[2] * scale_x)),
                int(round(box[3] * scale_y)),
            ]
        framed_scene = dict(scene)
        framed_scene["image_path"] = str(target)
        framed_scene["image_width"] = target_width
        framed_scene["image_height"] = target_height
        framed_scene["image_sha256"] = sha(target.read_bytes())
        framed_scene["original_image_sha256"] = scene["image_sha256"]
        framed_scene["original_image_width"] = original_width
        framed_scene["original_image_height"] = original_height
        framed_scene["frame_scale_x"] = round(scale_x, 6)
        framed_scene["frame_scale_y"] = round(scale_y, 6)
        framed_scene["task"] = task
        framed_scene["prompt"] = build_prompt(framed_scene, task)
        framed_scenes.append(framed_scene)
    return framed_scenes
