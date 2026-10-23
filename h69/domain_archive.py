"""H69 CPU-only original DragOn TRAIN domain archival and strict reload.

Independent from H67 mixed curriculum: one full verified original TRAIN domain
can live in private durable storage while other domains are produced. No GPU,
no DEV/test answers, no optimistic success if the owner cannot read bytes back.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path

from h60.source_fetch import REV
from h67.stage import _strict_json as strict_json

SCHEMA = "hala.h69.original-cpu-domain.v1"
INDEX = "h69-domain-inventory.json"
EXPECTED_IMAGES = {"slider": 3500, "slide_resize": 2500,
                   "sheet": 2500, "text_highlight": 2500}
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_BYTES = 5 * 1024**3
HEX = re.compile(r"^[0-9a-f]{64}$")
GIT = re.compile(r"^[0-9a-f]{40}$")
SAFE = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def member_safe(name):
    return (isinstance(name, str) and 0 < len(name) <= 240 and
            SAFE.fullmatch(name) is not None and
            all(part not in (".", "..") for part in name.split("/")))


def validate(root: Path, domain: str) -> dict:
    """Real H61 original source receipt AND every JPEG's derived media hash."""
    if domain not in EXPECTED_IMAGES:
        raise ValueError("only four original pinned TRAIN domains allowed")
    root = Path(root)
    receipt = strict_json(root / "receipt.json")
    if (receipt.get("schema") != "hala.h61.source-export.v1"
            or receipt.get("source_commit_sha") != REV
            or receipt.get("source_domain") != domain
            or receipt.get("split") != "train"
            or receipt.get("official_eval_or_test_used") is not False
            or receipt.get("benchmark_answer_leakage") is not False
            or receipt.get("images_wanted") != EXPECTED_IMAGES[domain]
            or receipt.get("original_images") != EXPECTED_IMAGES[domain]
            or receipt.get("images_with_original_tasks") != EXPECTED_IMAGES[domain]
            or receipt.get("native_drag_actions", 0) < EXPECTED_IMAGES[domain]):
        raise ValueError("original TRAIN receipt not full, pinned or properly qualified")
    labels = root / "labels.jsonl"
    if labels.is_symlink() or not labels.is_file() or sha(labels) != receipt.get("labels_sha256"):
        raise ValueError("source TRAIN labels corrupted")
    expected = {}
    rows = 0
    with labels.open("r", encoding="utf8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            name = row.get("image_path")
            digest = row.get("image_sha256")
            image_id = row.get("image_id")
            if (not isinstance(name, str) or not member_safe(name)
                    or not name.startswith("images/" + domain + "/")
                    or not name.endswith(".jpg")
                    or not isinstance(image_id, str)
                    or name != f"images/{domain}/{image_id}.jpg"
                    or not isinstance(digest, str) or HEX.fullmatch(digest) is None
                    or row.get("domain") != domain
                    or row.get("action_type") != "drag"
                    or row.get("split") != "train"):
                raise ValueError("corrupt original TRAIN row/media reference")
            if name in expected and expected[name] != digest:
                raise ValueError("inconsistent image SHA across original task labels")
            expected[name] = digest
            rows += 1
            if rows > 2_000_000:
                raise ValueError("unbounded source label count")
    actual = root / "images" / domain
    files = {p.name for p in actual.glob("*.jpg")}
    if len(expected) != EXPECTED_IMAGES[domain] or files != {
        Path(n).name for n in expected
    } or rows != receipt["native_drag_actions"]:
        raise ValueError("original TRAIN image/row inventory mismatch")
    for name, digest in expected.items():
        image = root / name
        if image.is_symlink() or not image.is_file() or sha(image) != digest:
            raise ValueError("original TRAIN JPEG byte mismatch")
    receipts = list((root / "range-receipts").glob("*.receipt.json"))
    if len(receipts) != receipt.get("range_receipts"):
        raise ValueError("source HTTP range proof receipt count differs")
    return {"images": len(expected), "actions": rows,
            "source_revision": REV, "domain": domain,
            "labels_sha256": sha(labels),
            "range_receipts": len(receipts)}


def inventory(root: Path):
    members = {}
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("symbolic links forbidden in original source")
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        if not member_safe(rel) or size > MAX_FILE_BYTES:
            raise ValueError("untrusted or oversized original source file")
        total += size
        if total > MAX_BYTES or len(members) >= 100000:
            raise ValueError("original CPU shard exceeds safe space budget")
        members[rel] = {"bytes": size, "sha256": sha(path)}
    return members


def pack(root: Path, domain: str, bundle: Path, git_sha: str) -> dict:
    """Deterministic immutable ZIP-as-binary; create only."""
    if not isinstance(git_sha, str) or GIT.fullmatch(git_sha) is None:
        raise ValueError("exact Hala source Git SHA required")
    qualified = validate(root, domain)
    members = inventory(Path(root))
    bundle = Path(bundle).resolve()
    if bundle.exists():
        raise FileExistsError("refuse overwrite of immutable CPU source archive")
    bundle.parent.mkdir(parents=True, exist_ok=True)
    marker = {
        "schema": SCHEMA, "domain": domain, "source_git_sha": git_sha,
        "source_revision": REV, "qualification": qualified, "members": members,
    }
    tmp = bundle.with_name("." + bundle.name + ".tmp")
    if tmp.exists():
        raise FileExistsError("archive is already being staged")
    try:
        with zipfile.ZipFile(tmp, "w", allowZip64=True) as output:
            for relative in members:
                info = zipfile.ZipInfo(relative, (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o100644 << 16
                with output.open(info, "w", force_zip64=True) as dst, (Path(root) / relative).open("rb") as src:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
            info = zipfile.ZipInfo(INDEX, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            output.writestr(info, json.dumps(marker, sort_keys=True, separators=(",", ":")).encode())
        os.replace(tmp, bundle)
    finally:
        tmp.unlink(missing_ok=True)
    return {"schema": SCHEMA, "status": "CPU_SOURCE_PACKED",
            "sha256": sha(bundle), "bytes": bundle.stat().st_size,
            "source_git_sha": git_sha, **qualified,
            "provider_readback_verified": False}


def unpack(bundle: Path, digest: str, dest: Path, git_sha: str, domain: str) -> dict:
    if (not isinstance(digest, str) or HEX.fullmatch(digest) is None or
            not isinstance(git_sha, str) or GIT.fullmatch(git_sha) is None):
        raise ValueError("operator SHA256 and exact Git SHA required")
    if domain not in EXPECTED_IMAGES:
        raise ValueError("original TRAIN domain unknown")
    bundle, dest = Path(bundle).resolve(), Path(dest).resolve()
    if dest.exists():
        raise FileExistsError("destination must not already exist")
    if not bundle.is_file() or sha(bundle) != digest:
        raise ValueError("untrusted source archive SHA256")
    with zipfile.ZipFile(bundle) as archive:
        infos = archive.infolist()
        if not 0 < len(infos) <= 100001 or len({x.filename for x in infos}) != len(infos):
            raise ValueError("duplicate or unbounded original archive")
        meta = json.loads(archive.read(INDEX))
        if (not isinstance(meta, dict) or meta.get("schema") != SCHEMA
                or meta.get("domain") != domain or meta.get("source_git_sha") != git_sha
                or meta.get("source_revision") != REV or not isinstance(meta.get("members"), dict)
                or {i.filename for i in infos} != set(meta["members"]) | {INDEX}):
            raise ValueError("CPU archive manifest or immutable source identity differs")
        size = 0
        for info in infos:
            if info.filename == INDEX:
                continue
            item = meta["members"].get(info.filename, {})
            if (not member_safe(info.filename) or
                    not isinstance(item, dict) or set(item) != {"bytes", "sha256"} or
                    type(item["bytes"]) is not int or item["bytes"] != info.file_size or
                    info.file_size > MAX_FILE_BYTES or info.compress_type != zipfile.ZIP_STORED or
                    stat.S_IFMT(info.external_attr >> 16) not in (0, stat.S_IFREG)):
                raise ValueError("hostile original CPU archive member")
            size += info.file_size
            if size > MAX_BYTES:
                raise ValueError("source import exceeds safe unpack budget")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".hala-h69-", dir=dest.parent) as temp:
            root = Path(temp) / "domain"
            root.mkdir()
            for info in infos:
                if info.filename == INDEX:
                    continue
                target = root / info.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as src, target.open("xb") as out:
                    shutil.copyfileobj(src, out, 1024 * 1024)
                if sha(target) != meta["members"][info.filename]["sha256"]:
                    raise ValueError("source member SHA256 changed during transport")
            verified = validate(root, domain)
            if verified != meta["qualification"]:
                raise ValueError("source provenance changed during import")
            os.replace(root, dest)
    return {"schema": SCHEMA, "status": "VERIFIED_CPU_SOURCE_RESTORED",
            "sha256": digest, "domain": domain, **verified}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_subparsers(dest="mode", required=True)
    build = mode.add_parser("pack")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--domain", choices=sorted(EXPECTED_IMAGES), required=True)
    build.add_argument("--bundle", type=Path, required=True)
    build.add_argument("--git-sha", required=True)
    restore = mode.add_parser("unpack")
    restore.add_argument("--bundle", type=Path, required=True)
    restore.add_argument("--sha256", required=True)
    restore.add_argument("--dest", type=Path, required=True)
    restore.add_argument("--git-sha", required=True)
    restore.add_argument("--domain", choices=sorted(EXPECTED_IMAGES), required=True)
    args = parser.parse_args(argv)
    result = (pack(args.source, args.domain, args.bundle, args.git_sha)
              if args.mode == "pack" else
              unpack(args.bundle, args.sha256, args.dest, args.git_sha, args.domain))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
