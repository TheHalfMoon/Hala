"""H61 bounded, verified export of original DragOn TRAIN material.

The upstream TRAIN archives are 61.2 GB in total and the images are laid out
before the task JSONs, so a full archive stream is the only way to reach a
task JSON. A curriculum does not need the whole corpus: it needs a bounded,
balanced set of original screenshots with their original task JSONs, and it
needs proof that every byte came from the pinned immutable revision.

This module therefore reads the archive in two bounded, contiguous, verified
ranges:

* the image region, read from offset 0 until the requested number of original
  screenshots has been collected, and
* the task region, read from the exact offset of the first original `tasks/`
  member until every collected screenshot has its original task JSON.

Every range is fetched with explicit HTTP 206 byte-range requests against the
pinned revision, every fetched range gets its own SHA-256 receipt, and the
original image bytes, the original task JSON bytes, the original `ordered`
flag, the original start and end boxes and the original screenshot dimensions
are preserved. The upstream public evaluation and hidden test splits are never
read.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from h60 import pairs
from h60.source_fetch import REV, REPO, source_url
from h60.tar_stream import BLOCK, TarStreamReader, _checksum_ok, _member_name

SCHEMA = "hala.h61.source-export.v1"
RECEIPT_SCHEMA = "hala.h61.source-range-receipt.v1"
DEFAULT_CHUNK = 16 * 1024 * 1024
TASK_CHUNK = 32 * 1024 * 1024
PROBE_BYTES = 1024 * 1024
MAX_PROBE_BYTES = 64 * 1024 * 1024
LOOKAHEAD = 4
FETCH_ATTEMPTS = 8
MAX_IMAGES = 200_000
MAX_RECEIPTS = 4096
# The public DEV evaluation frames are 1024x640 and the pinned processor keeps
# both 1024x640 and 1280x800 unchanged, because the processor rounds to a
# multiple of 32 and both sizes already are. Training in the evaluation frame
# therefore removes any train/evaluate pixel-frame shift, and it removes it
# without inventing a coordinate: the scale is applied to the original boxes
# and to the original screenshot together.
DEFAULT_FRAME = (1024, 640)


def frame_target(original_width: int, original_height: int, frame=(1024, 640)):
    """The training frame of one original screenshot, aspect ratio preserved.

    A uniform scale is used, so the original boxes and the original screenshot
    are always scaled by the same factor and the screenshot is never
    distorted. A screenshot whose aspect ratio differs from the frame keeps
    its own aspect ratio and is simply smaller, and the prompt states the size
    that the model actually sees.
    """
    frame_width, frame_height = frame
    if original_width <= 0 or original_height <= 0:
        raise ValueError("Original screenshot dimensions required")
    scale = min(frame_width / original_width, frame_height / original_height)
    new_width = max(1, int(round(original_width * scale)))
    new_height = max(1, int(round(original_height * scale)))
    return new_width, new_height, scale


def scale_row_to_frame(row: dict, width: int, height: int) -> dict:
    """Scale one original row into the training frame, boxes and points alike.

    The original screenshot bytes are never altered in the private shard; this
    transform only produces the training frame, and it is recorded per row so
    the transform is auditable rather than implicit.
    """
    original_width = row["image_width"]
    original_height = row["image_height"]
    if (original_width, original_height) == (width, height):
        return row
    new_width, new_height, scale = frame_target(original_width, original_height, (width, height))
    scale_x = new_width / original_width
    scale_y = new_height / original_height
    scaled = dict(row)
    for box_key in ("start_bbox", "end_bbox"):
        box = row[box_key]
        scaled[box_key] = [
            int(round(box[0] * scale_x)),
            int(round(box[1] * scale_y)),
            int(round(box[2] * scale_x)),
            int(round(box[3] * scale_y)),
        ]
    completion = dict(row["completion"])
    for point_key in ("start_point", "end_point"):
        point = completion[point_key]
        completion[point_key] = [
            int(round(point[0] * scale_x)),
            int(round(point[1] * scale_y)),
        ]
    scaled["completion"] = completion
    scaled["image_width"] = new_width
    scaled["image_height"] = new_height
    scaled["original_image_width"] = original_width
    scaled["original_image_height"] = original_height
    scaled["frame_scale_x"] = round(scale_x, 6)
    scaled["frame_scale_y"] = round(scale_y, 6)
    scaled["frame_uniform_scale"] = round(scale, 6)
    return scaled


def _canonical(obj) -> bytes:
    return (
        json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf8")


def fetch_range_resilient(url: str, start: int, end: int, total: int) -> bytes:
    """Fetch one exact byte range, proving the provider honoured it.

    The source provider throttles a burst of parallel range requests, and a
    throttled answer is a transport condition rather than a source condition.
    A throttled or transient answer is therefore retried with a bounded
    backoff, while a range that is answered with the wrong status, the wrong
    content range, the wrong length or a truncated body is refused.
    """
    expected_range = f"bytes {start}-{end}/{total}"
    expected_length = end - start + 1
    last_error: Exception | None = None
    for attempt in range(FETCH_ATTEMPTS):
        request = urllib.request.Request(
            url,
            headers={"Range": f"bytes={start}-{end}", "User-Agent": "Hala-H61-bounded-TRAIN-source"},
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                if response.status != 206:
                    raise ValueError("source provider did not answer with HTTP 206")
                if response.headers.get("Content-Range") != expected_range:
                    raise ValueError("source provider returned an unexpected Content-Range")
                declared = response.headers.get("Content-Length")
                if declared is not None and int(declared) != expected_length:
                    raise ValueError("source provider returned an unexpected Content-Length")
                buffer = bytearray()
                while len(buffer) < expected_length:
                    chunk = response.read(min(1 << 20, expected_length - len(buffer)))
                    if not chunk:
                        break
                    buffer += chunk
                if len(buffer) != expected_length:
                    raise IOError("source provider truncated the requested range")
                return bytes(buffer)
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in (429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, IOError, ValueError) as error:  # noqa: PERF203
            last_error = error
        time.sleep(min(60.0, 3.0 * (2 ** attempt)))
    raise IOError(f"bounded source range {start}-{end} failed: {last_error}")


class RangeReader:
    """Bounded, resumable reader over one contiguous range of the pinned source."""

    def __init__(self, domain: str, start: int, end: int, total: int, receipt_dir: Path,
                 chunk_bytes: int = DEFAULT_CHUNK):
        self.domain = domain
        self.start = start
        self.end = end
        self.total = total
        self.chunk_bytes = chunk_bytes
        self.receipt_dir = Path(receipt_dir)
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        self.url = source_url(domain)
        self.digest = hashlib.sha256()
        self.bytes_read = 0
        self.receipts: list[dict] = []

    def chunks(self):
        """Yield the range in order, with a bounded parallel look-ahead.

        The source provider serves one byte range at a time quickly but a
        sequential stream of 32 MiB ranges measured around 4 MB/s, which is
        slower than the corpus needs. A bounded look-ahead keeps the memory
        small while the ranges are fetched in parallel, and the digest is
        still computed over the exact bytes in archive order.
        """
        from concurrent.futures import ThreadPoolExecutor

        cursor = self.start
        index = 0
        spans = []
        while cursor <= self.end:
            stop = min(cursor + self.chunk_bytes - 1, self.end)
            spans.append((cursor, stop))
            cursor = stop + 1
        with ThreadPoolExecutor(max_workers=LOOKAHEAD) as pool:
            pending: list = []
            next_index = 0
            while next_index < len(spans) and len(pending) < LOOKAHEAD:
                span_start, span_stop = spans[next_index]
                pending.append(
                    (next_index, span_start, span_stop,
                     pool.submit(fetch_range_resilient, self.url, span_start, span_stop, self.total))
                )
                next_index += 1
            while pending:
                order, span_start, span_stop, future = pending.pop(0)
                payload = future.result()
                if next_index < len(spans):
                    nxt_start, nxt_stop = spans[next_index]
                    pending.append(
                        (next_index, nxt_start, nxt_stop,
                         pool.submit(fetch_range_resilient, self.url, nxt_start, nxt_stop, self.total))
                    )
                    next_index += 1
                self.digest.update(payload)
                self.bytes_read += len(payload)
                receipt = {
                    "schema": RECEIPT_SCHEMA,
                    "source_repo": REPO,
                    "source_commit_sha": REV,
                    "source_domain": self.domain,
                    "split": "train",
                    "range_start": span_start,
                    "range_end_inclusive": span_stop,
                    "bytes_verified": len(payload),
                    "range_sha256": hashlib.sha256(payload).hexdigest(),
                    "http_status": 206,
                }
                self.receipt_dir.joinpath(
                    f"{self.domain}-{'image' if span_start < self.total // 2 else 'task'}-"
                    f"{order:05d}.receipt.json"
                ).write_text(_canonical(receipt).decode("utf8"), encoding="utf8")
                self.receipts.append(receipt)
                index += 1
                yield payload

    @property
    def sha256(self) -> str:
        return self.digest.hexdigest()


def official_lfs_digest(domain: str, plan_path=None) -> str:
    """The official upstream LFS SHA-256 of one pinned TRAIN archive."""
    if plan_path is None:
        plan_path = Path(__file__).resolve().parents[1] / "h59" / "full_source_windows.json"
    plan = json.loads(Path(plan_path).read_text("utf8"))
    entry = plan["source_domains"][domain]
    return str(entry["upstream_lfs_sha256"])


def find_task_region_start(domain: str, total_bytes: int) -> tuple[int, str]:
    """The exact byte offset of the first original `tasks/` member.

    The archive holds the images before the task JSONs, so the task region can
    only be reached after the image region. The offset is located with a
    bounded binary search.

    A probe has to GROW until it finds a header: a probe that lands inside a
    large original member finds nothing in a small window, and treating that
    as "no member here" would drive the search the wrong way. Every probe is
    nevertheless bounded, so the cost of the search stays bounded too.
    """
    url = source_url(domain)
    prefix = f"{domain}/tasks/"

    def scan(payload: bytes, base: int) -> tuple[int, str] | None:
        for index in range(0, len(payload) - BLOCK + 1, BLOCK):
            block = payload[index : index + BLOCK]
            if not block.strip(b"\0") and not _checksum_ok(block):
                continue
            if _checksum_ok(block) and block[257:262] == b"ustar":
                return base + index, _member_name(block)
        return None

    def probe(offset: int) -> tuple[int, str] | None:
        """The first original member header at or after `offset`.

        The window GROWS until a header is found. A window that lands inside
        a large original member finds nothing, and treating that as "no
        member here" would drive the search the wrong way, so every window is
        bounded and the growth stops as soon as a header is found.
        """
        size = PROBE_BYTES
        while size <= MAX_PROBE_BYTES:
            window = min(size, total_bytes - offset)
            if window <= 0:
                return None
            payload = fetch_range_resilient(url, offset, offset + window - 1, total_bytes)
            found = scan(payload, offset)
            if found is not None:
                return found
            size *= 8
        return None

    low, high = 0, total_bytes
    while low < high:
        middle = ((low + high) // 2 // BLOCK) * BLOCK
        found = probe(middle)
        if found is None:
            high = middle
            continue
        _, name = found
        if name.startswith(prefix):
            high = middle
        else:
            low = middle + BLOCK
    found = probe(low)
    if found is None:
        raise ValueError(f"no original task member found in {domain}")
    offset, name = found
    if not name.startswith(prefix):
        raise ValueError(f"task region boundary probe resolved to {name}")
    return offset, name


def export_domain(domain: str, total_bytes: int, out_dir, *, images_wanted: int = 10_000,
                  chunk_bytes: int = DEFAULT_CHUNK, receipt_dir=None,
                  frame: tuple = DEFAULT_FRAME,
                  max_task_bytes: int = 4 * 1024 * 1024 * 1024) -> dict:
    """Materialise a bounded original corpus slice for one domain.

    The result is a directory that holds the training-frame screenshots under
    `images/<domain>/`, the derived `labels.jsonl` rows and a `receipt.json`
    that binds every fetched range and every derived count.
    """
    if type(images_wanted) is not int or not 1 <= images_wanted <= MAX_IMAGES:
        raise ValueError("Bounded original image count required")
    if type(total_bytes) is not int or total_bytes <= 0:
        raise ValueError("Original archive size required")
    if (not isinstance(frame, tuple) or len(frame) != 2 or
            any(type(value) is not int or value <= 0 for value in frame)):
        raise ValueError("A bounded training frame is required")
    frame_width, frame_height = frame
    out_dir = Path(out_dir)
    (out_dir / "images" / domain).mkdir(parents=True, exist_ok=True)
    receipt_dir = Path(receipt_dir) if receipt_dir else out_dir / "range-receipts"
    started = time.time()
    task_start, first_task_name = find_task_region_start(domain, total_bytes)
    image_prefix = f"{domain}/images/"
    task_prefix = f"{domain}/tasks/"
    wanted: dict[str, dict] = {}
    rejections: list[dict] = []

    def keep_image(member) -> bool:
        name = member.name
        if "/._" in name or "/PaxHeader/" in name:
            return False
        return (
            name.startswith(image_prefix)
            and name.endswith(".jpg")
            and pairs.is_uuid(name[len(image_prefix) : -len(".jpg")])
        )

    def sink_image(member, payload) -> None:
        key = member.name[len(image_prefix) : -len(".jpg")]
        if not pairs.image_tail_ok(payload):
            rejections.append({"image_id": key, "reason": "no end-of-image marker"})
            return
        try:
            width, height, _ = pairs.image_digest(payload)
        except ValueError as error:
            rejections.append({"image_id": key, "reason": str(error)[:120]})
            return
        stored = payload
        stored_width, stored_height = width, height
        if (width, height) != (frame_width, frame_height):
            from PIL import Image
            import io

            target_width, target_height, _ = frame_target(width, height, frame)
            with Image.open(io.BytesIO(payload)) as picture:
                resized = picture.convert("RGB").resize(
                    (target_width, target_height), Image.LANCZOS
                )
                buffer = io.BytesIO()
                resized.save(buffer, format="JPEG", quality=95, subsampling=0)
                stored = buffer.getvalue()
            stored_width, stored_height = target_width, target_height
        target = out_dir / "images" / domain / f"{key}.jpg"
        target.write_bytes(stored)
        wanted[key] = {
            "image_id": key,
            "image_path": f"images/{domain}/{key}.jpg",
            "image_sha256": pairs.sha256_bytes(stored),
            "source_image_sha256": pairs.sha256_bytes(payload),
            "image_bytes": len(stored),
            "source_image_bytes": len(payload),
            "image_width": stored_width,
            "image_height": stored_height,
            "original_image_width": width,
            "original_image_height": height,
            "source_image_tar_offset": member.offset,
            "source_image_tar_size": member.size,
            "tasks": [],
        }

    image_reader = RangeReader(domain, 0, total_bytes - 1, total_bytes, receipt_dir, chunk_bytes)
    reader = TarStreamReader(keep_image, sink_image)
    for payload in image_reader.chunks():
        reader.feed(payload)
        if len(wanted) >= images_wanted:
            break

    missing = set(wanted) - {
        key for key, entry in wanted.items() if entry["tasks"]
    }
    task_reader = RangeReader(domain, task_start, total_bytes - 1, total_bytes, receipt_dir,
                              max(chunk_bytes, TASK_CHUNK))

    def keep_task(member) -> bool:
        name = member.name
        if "/._" in name or "/PaxHeader/" in name:
            return False
        if not (name.startswith(task_prefix) and name.endswith(".json")):
            return False
        return name[len(task_prefix) : -len(".json")] in wanted

    def sink_task(member, payload) -> None:
        key = member.name[len(task_prefix) : -len(".json")]
        entry = wanted.get(key)
        if entry is None:
            return
        try:
            source_tasks = pairs.parse_source_tasks(payload)
        except ValueError as error:
            rejections.append({"image_id": key, "reason": str(error)[:120]})
            return
        for index, task in enumerate(source_tasks):
            try:
                candidate = pairs.candidate_from_source_task(domain, key, index, task)
            except ValueError as error:
                rejections.append({"image_id": key, "reason": str(error)[:120]})
                continue
            entry["tasks"].append(
                {
                    "source_task_index": index,
                    "intent": candidate["intent"],
                    "ordered": candidate["ordered"],
                    "start_bbox": candidate["start_bbox"],
                    "end_bbox": candidate["end_bbox"],
                    "subtype": candidate["subtype"],
                    "source_task_sha256": pairs.sha256_bytes(payload),
                    "source_task_tar_offset": member.offset,
                    "source_task_tar_size": member.size,
                }
            )

    task_stream = TarStreamReader(keep_task, sink_task)
    resolved = 0
    for payload in task_reader.chunks():
        task_stream.feed(payload)
        resolved = sum(1 for entry in wanted.values() if entry["tasks"])
        if resolved >= len(wanted):
            break
        if task_reader.bytes_read >= max_task_bytes:
            break

    rows: list[dict] = []
    for key in sorted(wanted):
        entry = wanted[key]
        for candidate in entry["tasks"]:
            try:
                row = pairs.build_row(
                    domain,
                    key,
                    candidate,
                    entry["original_image_width"],
                    entry["original_image_height"],
                    entry["source_image_sha256"],
                    candidate["source_task_sha256"],
                    entry["source_image_tar_offset"],
                    candidate["source_task_tar_offset"],
                )
                row = scale_row_to_frame(row, frame_width, frame_height)
                row["image_sha256"] = entry["image_sha256"]
                row["source_image_sha256"] = entry["source_image_sha256"]
                rows.append(row)
            except ValueError as error:
                rejections.append({"image_id": key, "reason": str(error)[:120]})
    labels = b"".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf8")
        + b"\n"
        for row in rows
    )
    (out_dir / "labels.jsonl").write_bytes(labels)
    receipt = {
        "schema": SCHEMA,
        "source_repo": REPO,
        "source_commit_sha": REV,
        "source_domain": domain,
        "split": "train",
        "source_license": "Apache-2.0",
        "source_tar_bytes": total_bytes,
        "upstream_lfs_sha256": official_lfs_digest(domain),
        "bounded_export": True,
        "source_tar_sha256_matches_official": False,
        "whole_archive_streamed": False,
        "task_region_start": task_start,
        "first_task_member": first_task_name,
        "images_wanted": images_wanted,
        "training_frame": [frame_width, frame_height],
        "original_images": len(wanted),
        "images_with_original_tasks": sum(1 for entry in wanted.values() if entry["tasks"]),
        "native_drag_actions": len(rows),
        "rejected": rejections,
        "image_region_bytes_read": image_reader.bytes_read,
        "image_region_sha256": image_reader.sha256,
        "task_region_bytes_read": task_reader.bytes_read,
        "task_region_sha256": task_reader.sha256,
        "range_receipts": len(image_reader.receipts) + len(task_reader.receipts),
        "labels_sha256": hashlib.sha256(labels).hexdigest(),
        "official_eval_or_test_used": False,
        "benchmark_answer_leakage": False,
        "model_gpu_steps": 0,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (out_dir / "receipt.json").write_bytes(_canonical(receipt))
    return receipt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--total-bytes", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--images", type=int, default=10_000)
    parser.add_argument("--frame-width", type=int, default=DEFAULT_FRAME[0])
    parser.add_argument("--frame-height", type=int, default=DEFAULT_FRAME[1])
    parser.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK)
    args = parser.parse_args()
    result = export_domain(
        args.domain, args.total_bytes, args.out, images_wanted=args.images,
        frame=(args.frame_width, args.frame_height), chunk_bytes=args.chunk_bytes,
    )
    print(json.dumps(result, sort_keys=True))
