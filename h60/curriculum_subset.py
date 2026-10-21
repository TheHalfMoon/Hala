"""H60 bounded authentic curriculum subset from a verified original TRAIN archive.

This module streams a pinned original DragOn TRAIN archive, pairs the original
JPEG with the original task JSON exactly as the full ingest does, and stops
after a bounded number of original screenshots. The result is written in the
same shard layout, so the ordinary H60 training bridge derives the SFT export
from it without any special case.

The subset is a *prefix of a fully verified original archive*: the whole
archive's SHA-256 was already matched against the official upstream digest, and
every subset member keeps its original byte offset inside that archive, so its
provenance is the same as the full corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from h59.master_data_plan import check_plan
from h60 import pairs
from h60.source_fetch import REV, REPO, WindowedSource
from h60.tar_stream import TarStreamReader

SHARD_SCHEMA = "hala.h60.dragon-train-shard.v1"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build_subset(
    plan_path: str,
    domain: str,
    output: str,
    *,
    max_images: int,
    labels_per_image: int,
    workers: int = 6,
) -> dict:
    plan = json.loads(Path(plan_path).read_text("utf8"))
    check_plan(plan)
    total = int(plan["source_domains"][domain]["bytes"])
    official = str(plan["source_domains"][domain]["upstream_lfs_sha256"])
    windows = [w for w in plan["windows"] if w["source_domain"] == domain]
    output = Path(output)
    if output.exists():
        raise FileExistsError("An H60 curriculum subset is create-only")
    receipts = output.parent / f"{output.name}-window-receipts"
    source = WindowedSource(domain, windows, total, receipts)
    (output / "images" / domain).mkdir(parents=True)
    (output / "tasks" / domain).mkdir(parents=True)

    image_prefix = f"{domain}/images/"
    task_prefix = f"{domain}/tasks/"
    pending = output / ".pending"
    pending.mkdir(parents=True, exist_ok=True)
    tasks: dict[str, list[dict]] = {}
    task_offset: dict[str, int] = {}
    image_records: list[dict] = []
    task_records: list[dict] = []
    rows: list[dict] = []
    rejections: list[dict] = []
    selected: set[str] = set()
    pending_images: set[str] = set()
    state = {"images": 0, "tasks": 0, "pairs": 0}

    def keep(member) -> bool:
        name = member.name
        if "/._" in name or "/PaxHeader/" in name:
            return False
        for prefix, suffix in ((image_prefix, ".jpg"), (task_prefix, ".json")):
            if name.startswith(prefix) and name.endswith(suffix):
                return pairs.is_uuid(name[len(prefix) : -len(suffix)])
        return False

    def emit_pair(key: str) -> None:
        """Write the original image and its rows once both members are present."""
        if key in selected or key not in tasks:
            return
        source_image = pending / f"{key}.jpg"
        source_task = pending / f"{key}.json"
        if not source_image.exists() or not source_task.exists():
            return
        payload = source_image.read_bytes()
        task_payload = source_task.read_bytes()
        try:
            width, height, _ = pairs.image_digest(payload)
        except ValueError as error:
            rejections.append({"image_id": key, "reason": str(error)[:160]})
            return
        if not pairs.image_tail_ok(payload):
            rejections.append({"image_id": key, "reason": "incomplete original JPEG"})
            return
        (output / "images" / domain / f"{key}.jpg").write_bytes(payload)
        image_records.append(
            {
                "image_id": key,
                "image_path": f"images/{domain}/{key}.jpg",
                "image_sha256": _sha(payload),
                "image_bytes": len(payload),
                "image_width": width,
                "image_height": height,
                "source_image_tar_offset": image_offset[key],
                "source_image_tar_size": image_size[key],
            }
        )
        selected.add(key)
        for candidate in tasks[key][:labels_per_image]:
            try:
                row = pairs.build_row(
                    domain, key, candidate, width, height,
                    image_records[-1]["image_sha256"], _sha(task_payload),
                    image_offset[key], task_offset[key],
                )
            except ValueError as error:
                rejections.append(
                    {"image_id": key, "source_task_index": candidate["source_task_index"],
                     "reason": str(error)[:160]}
                )
                continue
            rows.append(row)
            state["pairs"] += 1

    image_offset: dict[str, int] = {}
    image_size: dict[str, int] = {}

    def sink(member, payload) -> None:
        name = member.name
        if name.startswith(task_prefix):
            key = name[len(task_prefix) : -len(".json")]
            state["tasks"] += 1
            if key in selected or key not in pending_images:
                return
            try:
                source_tasks = pairs.parse_source_tasks(payload)
                candidates = [
                    pairs.candidate_from_source_task(domain, key, index, task)
                    for index, task in enumerate(source_tasks)
                ]
            except ValueError as error:
                rejections.append({"image_id": key, "reason": str(error)[:160]})
                return
            tasks[key] = candidates
            task_offset[key] = member.offset
            task_records.append(
                {
                    "image_id": key,
                    "source_task_json_path": f"tasks/{domain}/{key}.json",
                    "source_task_sha256": _sha(payload),
                    "source_task_bytes": len(payload),
                    "source_task_tar_offset": member.offset,
                    "source_task_tar_size": member.size,
                }
            )
            (pending / f"{key}.json").write_bytes(payload)
            emit_pair(key)
            return
        key = name[len(image_prefix) : -len(".jpg")]
        state["images"] += 1
        if len(pending_images) >= max_images:
            return
        pending_images.add(key)
        image_offset[key] = member.offset
        image_size[key] = member.size
        (pending / f"{key}.jpg").write_bytes(payload)
        emit_pair(key)

    reader = TarStreamReader(keep, sink)
    for _window, payload in source.stream(workers=workers):
        reader.feed(payload)
    reader.close()
    shutil.rmtree(pending, ignore_errors=True)

    labels = b"".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf8")
        + b"\n"
        for row in rows
    )
    (output / "labels.jsonl").write_bytes(labels)
    manifest = {
        "schema": SHARD_SCHEMA,
        "shard_id": f"h60-dragon-train-{domain}-curriculum",
        "source_repo": REPO,
        "source_commit_sha": REV,
        "source_domain": domain,
        "split": "train",
        "source_license": "Apache-2.0",
        "upstream_full_tar_lfs_sha256": official,
        "whole_tar_sha256_verified": source.archive_sha256 == official,
        "subset_of_verified_archive": True,
        "subset_original_images": max_images,
        "official_eval_or_test_used": False,
        "benchmark_answer_leakage": False,
        "original_images": len(image_records),
        "original_task_members": len(task_records),
        "native_drag_actions": len(rows),
        "ordered_true": sum(1 for row in rows if row["ordered"]),
        "ordered_false": sum(1 for row in rows if not row["ordered"]),
        "image_bytes": sum(record["image_bytes"] for record in image_records),
        "jpeg_structure_verified": 0,
        "jpeg_pixels_decoded": 0,
        "labels_sha256": _sha(labels),
        "rejected": rejections,
        "source_window_ids": [],
        "original_image_members": image_records,
        "original_task_member_records": task_records,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )
    shutil.rmtree(receipts, ignore_errors=True)
    return {
        "source_domain": domain,
        "whole_tar_sha256_verified": manifest["whole_tar_sha256_verified"],
        "original_images": len(image_records),
        "native_drag_actions": len(rows),
        "rejected": len(rejections),
        "output": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="h59/full_source_windows.json")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-images", type=int, default=4000)
    parser.add_argument("--labels-per-image", type=int, default=2)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    result = build_subset(
        args.plan,
        args.domain,
        args.output,
        max_images=args.max_images,
        labels_per_image=args.labels_per_image,
        workers=args.workers,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
