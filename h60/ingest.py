"""H60 full-corpus ingest of the pinned original DragOn TRAIN archives.

One domain at a time:
  1. locate the exact first original `tasks/` member of the pinned TRAIN TAR;
  2. read that original task region once and keep its parsed source tasks;
  3. stream the whole archive in the pinned 512 MiB windows, proving the
     official upstream LFS SHA-256 of the complete original TAR while pairing
     every original JPEG with its original task JSON;
  4. write bounded shards (images + labels + manifest) and publish each shard
     as a NEW private provider dataset with a full byte-level readback proof.

The archive bytes are never trusted because a header looked plausible: every
member header is checksum validated and every task payload seen in the full
stream is re-hashed against the independently read task region.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import json
import shutil
import time
import threading
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from h59.master_data_plan import DOMAINS, check_plan
from h60 import pairs, publish
from h60.source_fetch import REV, REPO, WindowedSource, fetch_range, source_url
from h60.tar_stream import BLOCK, TarStreamReader, _checksum_ok, _member_name

SCHEMA = "hala.h60.dragon-train-ingest.v1"
SHARD_SCHEMA = "hala.h60.dragon-train-shard.v1"
TASK_REGION_PROBE = 1 << 20
TASK_REGION_PROBE_LARGE = 8 << 20
TASK_REGION_PROBE_MAX = 64 << 20
TASK_REGION_CHUNK = 64 << 20
# Every shard member carries the same fixed timestamp, so an identical shard
# body always produces identical archive bytes. That makes a shard SHA a
# stable resume key: a rebuilt shard can be matched against the receipt of an
# already published shard instead of being uploaded twice.
FIXED_MEMBER_TIME = (2026, 10, 21, 0, 0, 0)


def member_info(name: str, compress: int = zipfile.ZIP_STORED) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=FIXED_MEMBER_TIME)
    info.compress_type = compress
    info.external_attr = 0o644 << 16
    info.create_system = 0
    return info


# ---------------------------------------------------------------- probing
def _scan_first_header(payload: bytes, base: int) -> tuple[int, str] | None:
    for index in range(0, len(payload) - BLOCK + 1, BLOCK):
        block = payload[index : index + BLOCK]
        if not block.strip(b"\0"):
            continue
        if not _checksum_ok(block) or block[257:262] != b"ustar":
            continue
        return base + index, _member_name(block)
    return None


def probe_first_header(url: str, offset: int, total: int) -> tuple[int, str] | None:
    """First original TAR header at or after `offset` (probe size grows)."""
    for probe in (TASK_REGION_PROBE, TASK_REGION_PROBE_LARGE, TASK_REGION_PROBE_MAX):
        if offset + probe > total:
            probe = total - offset
            if probe <= 0:
                return None
        payload = fetch_range(url, offset, offset + probe - 1, total)
        found = _scan_first_header(payload, offset)
        if found is not None:
            return found
    return None


def find_tasks_region_start(domain: str, total: int) -> tuple[int, str]:
    """Exact byte offset of the first original `tasks/` member of the archive."""
    url = source_url(domain)
    prefix = f"{domain}/tasks/"
    low, high = 0, total
    while low < high:
        middle = ((low + high) // 2) // BLOCK * BLOCK
        found = probe_first_header(url, middle, total)
        if found is None:
            high = middle
            continue
        _, name = found
        if name.startswith(prefix):
            high = middle
        else:
            low = middle + BLOCK
    found = probe_first_header(url, low, total)
    if found is None:
        raise ValueError(f"no original task member found in {domain}")
    offset, name = found
    if not name.startswith(prefix):
        raise ValueError(f"task region boundary probe resolved to {name}")
    return offset, name


# ------------------------------------------------------------ task region
def read_task_region(domain: str, start: int, total: int) -> dict:
    """Read every original task member of one domain exactly once."""
    url = source_url(domain)
    prefix = f"{domain}/tasks/"
    collected: dict[str, dict] = {}

    def keep(member) -> bool:
        name = member.name
        if not name.startswith(prefix) or not name.endswith(".json"):
            return False
        if "/._" in name or "/PaxHeader/" in name:
            return False
        return pairs.is_uuid(name[len(prefix) : -len(".json")])

    def sink(member, payload) -> None:
        key = member.name[len(prefix) : -len(".json")]
        if key in collected:
            raise ValueError(f"duplicate original task member {key}")
        source_tasks = pairs.parse_source_tasks(payload)
        candidates = [
            pairs.candidate_from_source_task(domain, key, index, task)
            for index, task in enumerate(source_tasks)
        ]
        collected[key] = {
            "payload_sha256": pairs.sha256_bytes(payload),
            "payload": payload,
            "member_size": member.size,
            "member_offset": member.offset,
            "candidates": candidates,
        }

    reader = TarStreamReader(keep, sink)
    cursor = start
    while cursor < total:
        end = min(cursor + TASK_REGION_CHUNK, total) - 1
        reader.feed(fetch_range(url, cursor, end, total))
        cursor = end + 1
    reader.close()
    return {
        "tasks": collected,
        "task_member_count": len(collected),
        "task_region_start": start,
        "task_region_bytes": total - start,
    }


# ------------------------------------------------------------------ shard
class ShardBuilder:
    """One bounded, create-only private shard of original images and labels."""

    def __init__(self, shard_id: str, domain: str, staging: Path, target_bytes: int):
        self.shard_id = shard_id
        self.domain = domain
        self.staging = staging
        self.target_bytes = target_bytes
        self.zip_path = staging / f"{shard_id}.zip"
        self.staging.mkdir(parents=True, exist_ok=True)
        self.archive = zipfile.ZipFile(self.zip_path, "w", zipfile.ZIP_STORED, allowZip64=True)
        self.image_records: list[dict] = []
        self.task_records: list[dict] = []
        self.rows: list[dict] = []
        self.rejections: list[dict] = []
        self.image_bytes = 0
        self.ordered_true = 0
        self.ordered_false = 0
        self.structure_verified = 0
        self.pixels_decoded = 0
        self.image_shas: dict[str, str] = {}
        self.window_ids: list[str] = []

    def add_image(self, key: str, payload: bytes, member) -> dict:
        image_sha = pairs.sha256_bytes(payload)
        width, height, _ = pairs.image_digest(payload)
        if not pairs.image_tail_ok(payload):
            raise ValueError(f"original JPEG {key} has no end-of-image marker")
        name = f"images/{self.domain}/{key}.jpg"
        with self.archive.open(member_info(name), "w") as member_out:
            member_out.write(payload)
        self.image_bytes += len(payload)
        self.image_shas[name] = image_sha
        record = {
            "image_id": key,
            "image_path": name,
            "image_sha256": image_sha,
            "image_bytes": len(payload),
            "image_width": width,
            "image_height": height,
            "source_image_tar_offset": member.offset,
            "source_image_tar_size": member.size,
        }
        self.image_records.append(record)
        return record

    def add_task(self, key: str, payload: bytes, offset: int, size: int) -> None:
        name = f"tasks/{self.domain}/{key}.json"
        self.task_records.append(
            {
                "image_id": key,
                "source_task_json_path": name,
                "source_task_sha256": pairs.sha256_bytes(payload),
                "source_task_bytes": len(payload),
                "source_task_tar_offset": offset,
                "source_task_tar_size": size,
            }
        )
        self.image_shas.setdefault(name, pairs.sha256_bytes(payload))
        with self.archive.open(member_info(name), "w") as member_out:
            member_out.write(payload)

    def add_row(self, row: dict) -> None:
        self.rows.append(row)
        if row["ordered"]:
            self.ordered_true += 1
        else:
            self.ordered_false += 1

    def add_rejection(self, record: dict) -> None:
        self.rejections.append(record)

    @property
    def full(self) -> bool:
        return self.image_bytes >= self.target_bytes

    def close(self, window_ids: list[str], archive_sha: str, official_sha: str) -> dict:
        labels = b"".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf8")
            + b"\n"
            for row in self.rows
        )
        self.archive.writestr(
            member_info("labels.jsonl", zipfile.ZIP_DEFLATED),
            labels,
        )
        manifest = {
            "schema": SHARD_SCHEMA,
            "shard_id": self.shard_id,
            "source_repo": REPO,
            "source_commit_sha": REV,
            "source_domain": self.domain,
            "split": "train",
            "source_license": "Apache-2.0",
            "upstream_full_tar_lfs_sha256": official_sha,
            "whole_tar_sha256_verified": archive_sha == official_sha,
            "official_eval_or_test_used": False,
            "benchmark_answer_leakage": False,
            "original_images": len(self.image_records),
            "original_task_members": len(self.task_records),
            "native_drag_actions": len(self.rows),
            "ordered_true": self.ordered_true,
            "ordered_false": self.ordered_false,
            "image_bytes": self.image_bytes,
            "jpeg_structure_verified": self.structure_verified,
            "jpeg_pixels_decoded": self.pixels_decoded,
            "labels_sha256": hashlib.sha256(labels).hexdigest(),
            "rejected": self.rejections,
            "source_window_ids": list(dict.fromkeys(window_ids)),
            "original_image_members": self.image_records,
            "original_task_member_records": self.task_records,
        }
        blob = (
            json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
        ).encode("utf8")
        self.archive.writestr(
            member_info("manifest.json", zipfile.ZIP_DEFLATED),
            blob,
        )
        self.archive.close()
        self.image_shas["labels.jsonl"] = hashlib.sha256(labels).hexdigest()
        self.image_shas["manifest.json"] = hashlib.sha256(blob).hexdigest()
        return manifest


# --------------------------------------------------------------- publish
def shard_inventory(zip_path: Path) -> tuple[dict[str, int], dict[str, str]]:
    """Exact published inventory: member name -> size, and member name -> SHA."""
    sizes: dict[str, int] = {}
    shas: dict[str, str] = {}
    with zipfile.ZipFile(zip_path) as bundle:
        for info in bundle.infolist():
            sizes[info.filename] = info.file_size
            digest = hashlib.sha256()
            with bundle.open(info) as member:
                for chunk in iter(lambda: member.read(1 << 20), b""):
                    digest.update(chunk)
            shas[info.filename] = digest.hexdigest()
    return sizes, shas


def publish_shard(
    shard_id: str,
    domain: str,
    zip_path: Path,
    staging: Path,
    token: str,
    manifest: dict,
    do_publish: bool = True,
) -> dict:
    """Publish one shard as a NEW private dataset and prove the readback.

    The local shard is released only after the provider has been proven to
    hold exactly the published bytes, so an interrupted publication can be
    retried from the same local body instead of re-downloading the source.
    """
    sizes, shas = shard_inventory(zip_path)
    ref = f"{publish.OWNER}/hala-h60-dragon-train-{domain}-{shard_id.rsplit('-', 1)[-1]}-private"
    receipt = {
        "shard_id": shard_id,
        "source_domain": domain,
        "provider_ref": ref,
        "provider_published": False,
        "provider_is_private": False,
        "anonymous_provider_http_status": None,
        "provider_readback_verified": False,
        "local_zip_sha256": publish.sha256_file(zip_path),
        "published_members": len(sizes),
        "published_bytes": sum(sizes.values()),
        "original_images": manifest["original_images"],
        "native_drag_actions": manifest["native_drag_actions"],
    }
    receipt_dir = staging / "shard-receipts" / domain
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"{shard_id}.json"
    if receipt_path.exists():
        try:
            prior = json.loads(receipt_path.read_text("utf8"))
        except (OSError, ValueError):
            prior = {}
        if (
            prior.get("provider_readback_verified")
            and prior.get("local_zip_sha256") == receipt["local_zip_sha256"]
        ):
            receipt.update(prior)
            receipt["resumed_from_verified_receipt"] = True
            zip_path.unlink(missing_ok=True)
            return receipt
    if not do_publish:
        return receipt
    upload_dir = staging / f"upload-{shard_id}"
    if upload_dir.exists():
        shutil.rmtree(upload_dir, ignore_errors=True)
    upload_dir.mkdir(parents=True)
    # The shard body is uploaded as ONE opaque member. The provider expands a
    # `.zip` member into its own tree and then materialises a downloadable
    # artifact for every expanded member, which was measured not to scale: a
    # 3,000 member archive became downloadable after 309 s and a 28,806 member
    # archive was still refused with "No gcs url found" after an hour. One
    # opaque member measured 17 s and is verified by one SHA-256.
    member_name = f"{shard_id}{publish.SHARD_MEMBER_SUFFIX}"
    linked = upload_dir / member_name
    try:
        os.link(zip_path, linked)
    except OSError:
        shutil.copy2(zip_path, linked)
    readme = {
        "schema": "hala.h60.dragon-train-shard-package.v1",
        "shard_id": shard_id,
        "source_domain": domain,
        "source_repo": REPO,
        "source_commit_sha": REV,
        "split": "train",
        "source_license": "Apache-2.0",
        "packaging": {
            "member": member_name,
            "inner_format": "zip",
            "inner_members": len(sizes),
            "inner_labels_sha256": manifest.get("labels_sha256"),
            "inner_manifest_sha256": publish.sha256_bytes_of_member(
                zip_path, "manifest.json"
            ),
        },
        "instructions": (
            "The shard body is one ZIP member. Unpack it, then read the inner "
            "manifest.json for the per-member SHA-256 of every original image, "
            "original task JSON, the derived labels and the manifest."
        ),
    }
    (upload_dir / publish.README_MEMBER).write_text(
        json.dumps(readme, indent=2, sort_keys=True) + "\n", encoding="utf8"
    )
    provider_members = {member_name: linked.stat().st_size}
    provider_shas = {member_name: receipt["local_zip_sha256"]}
    readme_bytes = (upload_dir / publish.README_MEMBER).read_bytes()
    provider_members[publish.README_MEMBER] = len(readme_bytes)
    provider_shas[publish.README_MEMBER] = publish.sha256_bytes(readme_bytes)
    receipt["provider_member"] = member_name
    receipt["provider_members"] = provider_members
    try:
        # A shard body is byte-reproducible, so a provider dataset that already
        # holds exactly these bytes is reused instead of uploaded again. The
        # upload is the expensive part and must never be repeated for a
        # verification that can simply be re-run.
        reused = publish.reusable_private_dataset(
            ref, token, expected_bytes=sum(provider_members.values()),
            expected_member=member_name,
        )
        receipt["provider_reused_existing_dataset"] = bool(reused)
        if not reused:
            if publish.dataset_exists(ref, token):
                # A previous attempt left this shard slug behind. It is not a
                # verified copy, so it is removed before the shard is rebuilt.
                publish.delete_dataset(ref)
            publish.create_private_dataset(
                upload_dir,
                ref,
                f"Hala H60 DragOn TRAIN {domain} {shard_id.rsplit('-', 1)[-1]} PRIVATE",
                "Original Hcompany DragOn TRAIN images and drag labels, private verified shard",
            )
        publish.wait_until_visible(ref, token)
        privacy = publish.verify_private(ref, token)
        receipt["provider_is_private"] = privacy["provider_is_private"]
        receipt["anonymous_provider_http_status"] = privacy["anonymous_view_http_status"]
        receipt["provider_version"] = privacy["provider_version"]
        receipt["provider_view_total_bytes"] = privacy["provider_view_total_bytes"]
        if not privacy["provider_is_private"]:
            raise ValueError("provider did not report the shard as private")
        if privacy["anonymous_view_http_status"] != 403:
            raise ValueError("unauthenticated provider access was not rejected with HTTP 403")
        readback = publish.verify_readback(
            ref, token, provider_members, provider_shas, staging / "readback" / shard_id
        )
        receipt.update(readback)
        receipt["provider_published"] = True
        receipt["provider_readback_verified"] = True
    except Exception as error:  # noqa: BLE001
        # A provider dataset that already holds exactly the published bytes is
        # kept: the upload is correct and only its verification was not
        # finished. It is never presented as a stored shard until the byte
        # proof succeeds. Anything else is removed so a retry cannot mistake
        # it for a verified private copy.
        if publish.reusable_private_dataset(
            ref, token, expected_bytes=sum(provider_members.values()),
            expected_member=member_name,
        ):
            receipt["provider_retained_for_retry"] = True
        else:
            publish.delete_dataset(ref)
            receipt["provider_deleted_after_failure"] = True
        receipt["failure"] = f"{type(error).__name__}: {error}"[:400]
        receipt_path.write_text(
            json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf8"
        )
        raise
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf8"
    )
    # The provider copy is now proven byte-exact, so the local shard is
    # released and the bounded disk budget covers the next shards.
    zip_path.unlink(missing_ok=True)
    return receipt


def publish_enabled(flag) -> bool:
    return bool(flag)


def publish_enabled(flag) -> bool:
    return bool(flag)


# ---------------------------------------------------------------- ingest
def ingest_domain(
    domain: str,
    plan: dict,
    staging: Path,
    target_bytes: int,
    workers: int,
    sample_structure_every: int,
    sample_pixels_every: int,
    do_publish: bool,
    token: str,
    max_shards: int | None = None,
    max_windows: int | None = None,
    upload_workers: int = 2,
) -> dict:
    total = int(plan["source_domains"][domain]["bytes"])
    official_sha = str(plan["source_domains"][domain]["upstream_lfs_sha256"])
    windows = [w for w in plan["windows"] if w["source_domain"] == domain]
    receipt_dir = staging / "window-receipts" / domain
    source = WindowedSource(domain, windows, total, receipt_dir)
    image_prefix = f"{domain}/images/"
    task_prefix = f"{domain}/tasks/"

    started = time.time()
    task_start, first_task_name = find_tasks_region_start(domain, total)
    region = read_task_region(domain, task_start, total)
    task_map = region["tasks"]
    print(
        f"H60_TASK_REGION {domain} start={task_start} bytes={region['task_region_bytes']} "
        f"members={region['task_member_count']} first={first_task_name}",
        flush=True,
    )

    state = {
        "images": 0,
        "tasks": 0,
        "rows": 0,
        "rejected": 0,
        "orphan_tasks": 0,
        "images_without_tasks": 0,
    }
    shards: list[dict] = []
    builder: ShardBuilder | None = None
    shard_index = 0
    window_receipts: list[dict] = []
    paired_task_keys: set[str] = set()
    receipt_lock = threading.Lock()
    slots = threading.BoundedSemaphore(max(1, upload_workers))
    pool = ThreadPoolExecutor(max_workers=max(1, upload_workers))
    pending: list[Future] = []

    def start_shard() -> ShardBuilder:
        slots.acquire()
        return ShardBuilder(
            f"h60-dragon-train-{domain}-{shard_index:04d}", domain, staging, target_bytes
        )

    def close_shard() -> None:
        nonlocal builder, shard_index
        if builder is None:
            return
        manifest = builder.close(builder.window_ids, source.archive_sha256, official_sha)
        shard_id = builder.shard_id
        zip_path = builder.zip_path
        builder = None
        shard_index += 1

        def task() -> None:
            try:
                receipt = publish_shard(
                    shard_id, domain, zip_path, staging, token, manifest, do_publish
                )
                receipt["manifest"] = {
                    key: manifest[key]
                    for key in (
                        "original_images",
                        "original_task_members",
                        "native_drag_actions",
                        "ordered_true",
                        "ordered_false",
                        "image_bytes",
                        "labels_sha256",
                        "whole_tar_sha256_verified",
                    )
                }
                with receipt_lock:
                    shards.append(receipt)
                print(
                    f"H60_SHARD {shard_id} images={manifest['original_images']} "
                    f"actions={manifest['native_drag_actions']} "
                    f"published={receipt['provider_published']} "
                    f"readback={receipt['provider_readback_verified']}",
                    flush=True,
                )
            except Exception as error:  # noqa: BLE001
                with receipt_lock:
                    shards.append(
                        {
                            "shard_id": shard_id,
                            "source_domain": domain,
                            "provider_published": False,
                            "provider_readback_verified": False,
                            "error": f"{type(error).__name__}: {error}"[:400],
                        }
                    )
                print(f"H60_SHARD_FAILED {shard_id} {type(error).__name__}: {error}"[:400], flush=True)
            finally:
                slots.release()

        pending.append(pool.submit(task))

    def keep(member) -> bool:
        name = member.name
        if "/._" in name or "/PaxHeader/" in name:
            return False
        for prefix, suffix in ((image_prefix, ".jpg"), (task_prefix, ".json")):
            if name.startswith(prefix) and name.endswith(suffix):
                return pairs.is_uuid(name[len(prefix) : -len(suffix)])
        return False

    def sink(member, payload) -> None:
        nonlocal builder
        name = member.name
        if name.startswith(image_prefix):
            key = name[len(image_prefix) : -len(".jpg")]
            state["images"] += 1
            if builder is None:
                builder = start_shard()
            record = builder.add_image(key, payload, member)
            if sample_structure_every and state["images"] % sample_structure_every == 0:
                pairs.verify_jpeg_structure(payload)
                builder.structure_verified += 1
            if sample_pixels_every and state["images"] % sample_pixels_every == 0:
                pairs.decode_jpeg_pixels(payload)
                builder.pixels_decoded += 1
            entry = task_map.get(key)
            if entry is None:
                state["images_without_tasks"] += 1
                builder.add_rejection(
                    {"image_id": key, "reason": "no original task member for this source image"}
                )
                return
            paired_task_keys.add(key)
            builder.add_task(key, entry["payload"], entry["member_offset"], entry["member_size"])
            width, height, _ = pairs.image_digest(payload)
            for candidate in entry["candidates"]:
                try:
                    row = pairs.build_row(
                        domain,
                        key,
                        candidate,
                        width,
                        height,
                        record["image_sha256"],
                        entry["payload_sha256"],
                        member.offset,
                        entry["member_offset"],
                    )
                except ValueError as error:
                    state["rejected"] += 1
                    builder.add_rejection(
                        {
                            "image_id": key,
                            "source_task_index": candidate["source_task_index"],
                            "reason": str(error)[:200],
                        }
                    )
                    continue
                builder.add_row(row)
                state["rows"] += 1
            if builder.full and (max_shards is None or len(shards) < max_shards):
                close_shard()
            return
        key = name[len(task_prefix) : -len(".json")]
        state["tasks"] += 1
        entry = task_map.get(key)
        if entry is None:
            state["orphan_tasks"] += 1
            raise ValueError(f"original task member {key} was absent from the task region read")
        digest = pairs.sha256_bytes(payload)
        if digest != entry["payload_sha256"]:
            raise ValueError(f"original task member {key} differs from the independently read copy")

    reader = TarStreamReader(keep, sink)
    window_index = 0
    current_window: str | None = None
    for window, payload in source.stream(workers=workers):
        if builder is not None:
            builder.window_ids.append(window["id"])
        if current_window is None:
            current_window = window["id"]
        reader.feed(payload)
        if max_windows is not None and len(source.window_receipts) >= max_windows:
            break
    partial = max_windows is not None and len(source.window_receipts) >= max_windows
    if not partial:
        reader.close()
    close_shard()
    for future in pending:
        future.result()
    pool.shutdown(wait=True)
    shards.sort(key=lambda item: item.get("shard_id", ""))

    archive_sha = source.archive_sha256
    official = official_sha
    result = {
        "schema": SCHEMA,
        "source_repo": REPO,
        "source_commit_sha": REV,
        "source_domain": domain,
        "split": "train",
        "source_tar_bytes": total,
        "source_tar_sha256": archive_sha,
        "source_tar_lfs_sha256_official": official,
        "source_tar_sha256_matches_official": archive_sha == official and not partial,
        "partial_run": partial,
        "source_bytes_streamed": source.archive_bytes,
        "task_region_start": task_start,
        "task_region_bytes": region["task_region_bytes"],
        "task_region_members": region["task_member_count"],
        "task_members_seen_in_full_stream": state["tasks"],
        "source_window_receipts": source.window_receipts,
        "original_images": state["images"],
        "images_paired_with_original_tasks": len(paired_task_keys),
        "images_without_original_tasks": state["images_without_tasks"],
        "orphan_task_members": state["orphan_tasks"],
        "native_drag_actions": state["rows"],
        "rejected_rows": state["rejected"],
        "shards": shards,
        "shard_count": len(shards),
        "official_eval_or_test_used": False,
        "model_gpu_steps": 0,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="h59/full_source_windows.json")
    parser.add_argument("--staging", required=True)
    parser.add_argument("--domain", required=True, choices=sorted(DOMAINS))
    parser.add_argument("--target-shard-bytes", type=int, default=2 * 1024 * 1024 * 1024)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--sample-structure-every", type=int, default=20)
    parser.add_argument("--sample-pixels-every", type=int, default=500)
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--upload-workers", type=int, default=2)
    parser.add_argument("--result", default=None)
    args = parser.parse_args()

    plan = json.loads(Path(args.plan).read_text("utf8"))
    check_plan(plan)
    staging = Path(args.staging)
    staging.mkdir(parents=True, exist_ok=True)
    token = ""
    if not args.no_publish:
        import os

        token = os.environ["KAGGLE_API_TOKEN"]
    result = ingest_domain(
        args.domain,
        plan,
        staging,
        args.target_shard_bytes,
        args.workers,
        args.sample_structure_every,
        args.sample_pixels_every,
        not args.no_publish,
        token,
        args.max_shards,
        args.max_windows,
        args.upload_workers,
    )
    blob = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.result:
        Path(args.result).write_text(blob, encoding="utf8")
    print("H60_DOMAIN_RESULT " + json.dumps(
        {key: result[key] for key in (
            "source_domain",
            "source_tar_sha256_matches_official",
            "original_images",
            "native_drag_actions",
            "shard_count",
            "rejected_rows",
            "elapsed_seconds",
        )},
        sort_keys=True,
    ), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
