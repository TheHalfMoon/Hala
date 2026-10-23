"""H67: reproducible CPU-prepared H61 TRAIN bundle; no GPU or public release.

Build on CPU after H61 export, firstparty, curriculum and baseline. Import on
an already approved GPU only after binding the ZIP SHA256 and pinned Git SHA.
The original H61 curriculum verifier is called on *both* ends. A successful
import does not mean trained model, valid model checkpoint, or private upload.
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

SCHEMA = "hala.h67.cpu-stage.v1"
BASELINE = "public-dev-baseline-receipt.json"
CURRICULUM = "export/h61_drag_curriculum.jsonl"
MANIFEST = "export/h61_drag_curriculum_manifest.json"
INDEX = "h67-stage.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
GITSHA = re.compile(r"^[0-9a-f]{40}$")
SAFE = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")
MAX_FILES = 200_000
MAX_FILE_BYTES = 2 * 1024**3
MAX_TOTAL_BYTES = 100 * 1024**3
BLOCK = 1024 * 1024


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(BLOCK), b""):
            value.update(part)
    return value.hexdigest()


def name_ok(name: str) -> bool:
    return (isinstance(name, str) and len(name) <= 255
            and SAFE.fullmatch(name) is not None
            and all(part not in (".", "..") for part in name.split("/"))
            and (name == BASELINE or name.startswith("export/"))
            and not any(part.startswith(".") or part in ("__pycache__", ".git")
                        for part in name.split("/")))


def _strict_json(path: Path, limit=16 * 1024**2):
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError("missing or oversized stage metadata")
    return json.loads(path.read_text(encoding="utf8"))


def validate_train(source: Path, *, rederive: bool) -> dict:
    """Validate the exact H61 TRAIN corpus; never accept evaluation media."""
    source = Path(source)
    manifest = _strict_json(source / MANIFEST)
    if (not isinstance(manifest, dict)
            or manifest.get("schema") != "hala.training.manifest.v1"
            or manifest.get("split") != "train"
            or manifest.get("no_benchmark_eval_source") is not True
            or manifest.get("training_executed") is not False
            or not isinstance(manifest.get("dataset_sha256"), str)
            or HEX64.fullmatch(manifest["dataset_sha256"]) is None):
        raise ValueError("H61 TRAIN-only manifest required")
    rows = source / CURRICULUM
    if not rows.is_file() or digest(rows) != manifest["dataset_sha256"]:
        raise ValueError("curriculum labels differ from manifest")
    media = manifest.get("media")
    if not isinstance(media, list) or not media:
        raise ValueError("nonempty H61 media manifest required")
    used = set()
    for entry in media:
        if not isinstance(entry, dict):
            raise ValueError("invalid H61 media entry")
        name = entry.get("path")
        sha = entry.get("sha256")
        if (not isinstance(name, str) or not name_ok("export/" + name)
                or name in used or not isinstance(sha, str)
                or HEX64.fullmatch(sha) is None):
            raise ValueError("invalid or duplicated H61 media reference")
        used.add(name)
        candidate = source / "export" / name
        if candidate.is_symlink() or not candidate.is_file() or digest(candidate) != sha:
            raise ValueError("H61 media mismatch or missing screenshot")
    if not isinstance(_strict_json(source / BASELINE), dict):
        raise ValueError("validated H61 baseline receipt required")
    if rederive:
        from h61.curriculum import load_export
        load_export(rows, manifest)
    return {"dataset_sha256": manifest["dataset_sha256"], "media": len(media),
            "rows": manifest.get("train_rows", 0)}


def _inventory(source: Path) -> dict:
    files = {}
    total = 0
    for root in (source / "export", source / BASELINE):
        if not root.exists():
            raise ValueError("required CPU-prepared artifact absent")
        paths = [root] if root.is_file() or root.is_symlink() else root.rglob("*")
        for p in paths:
            if p.is_symlink():
                raise ValueError("symlink forbidden inside stage")
            if p.is_dir():
                continue
            if not p.is_file():
                raise ValueError("irregular CPU-stage input")
            relative = p.relative_to(source).as_posix()
            size = p.stat().st_size
            if not name_ok(relative) or size > MAX_FILE_BYTES:
                raise ValueError("invalid staged file or excessive member")
            total += size
            if total > MAX_TOTAL_BYTES or len(files) >= MAX_FILES:
                raise ValueError("CPU-stage byte/file budget exceeded")
            files[relative] = {"sha256": digest(p), "size": size}
    if not {CURRICULUM, MANIFEST, BASELINE}.issubset(files):
        raise ValueError("incomplete CPU-stage inventory")
    return dict(sorted(files.items()))


def make_stage(source: Path, bundle: Path, git_sha: str, *,
               rederive: bool = True) -> dict:
    """Create-only; deterministic content order, exact byte SHA receipts."""
    if not isinstance(git_sha, str) or GITSHA.fullmatch(git_sha) is None:
        raise ValueError("pinned source Git SHA required")
    source = Path(source).resolve()
    bundle = Path(bundle).resolve()
    if bundle.exists():
        raise FileExistsError("never overwrite a previously prepared stage")
    qualification = validate_train(source, rederive=rederive)
    inventory = _inventory(source)
    metadata = {
        "schema": SCHEMA, "git_sha": git_sha, "source_split": "train",
        "dataset_sha256": qualification["dataset_sha256"],
        "media_count": qualification["media"], "row_count": qualification["rows"],
        "files": inventory,
    }
    bundle.parent.mkdir(parents=True, exist_ok=True)
    temporary = bundle.with_name("." + bundle.name + ".tmp")
    if temporary.exists():
        raise FileExistsError("staging archive already exists")
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as z:
            for relative in inventory:
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = (0o100644 << 16)
                with z.open(info, "w", force_zip64=True) as target, (source / relative).open("rb") as stream:
                    shutil.copyfileobj(stream, target, BLOCK)
            info = zipfile.ZipInfo(INDEX, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = (0o100644 << 16)
            z.writestr(info, json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf8"))
        os.replace(temporary, bundle)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "schema": SCHEMA, "status": "CPU_STAGE_CREATED",
        "sha256": digest(bundle), "bytes": bundle.stat().st_size,
        "git_sha": git_sha, "dataset_sha256": qualification["dataset_sha256"],
        "files": len(inventory), "media": qualification["media"],
        "private_provider_readback_verified": False,
    }


def _check_zip(z, *, pinned_sha: str) -> tuple[dict, list]:
    infos = z.infolist()
    if not infos or len(infos) > MAX_FILES + 1 or len(set(i.filename for i in infos)) != len(infos):
        raise ValueError("duplicate or oversized ZIP inventory")
    index = [i for i in infos if i.filename == INDEX]
    if len(index) != 1 or index[0].file_size > 16 * 1024**2:
        raise ValueError("missing or oversized stage index")
    metadata = json.loads(z.read(index[0]))
    if (not isinstance(metadata, dict) or set(metadata) !=
            {"schema", "git_sha", "source_split", "dataset_sha256",
             "media_count", "row_count", "files"}
            or metadata["schema"] != SCHEMA or metadata["source_split"] != "train"
            or metadata["git_sha"] != pinned_sha
            or not isinstance(metadata["dataset_sha256"], str)
            or HEX64.fullmatch(metadata["dataset_sha256"]) is None
            or not isinstance(metadata["files"], dict)):
        raise ValueError("stage does not match pinned source or TRAIN contract")
    expected = metadata["files"]
    if set(i.filename for i in infos) != set(expected) | {INDEX}:
        raise ValueError("ZIP member-set mismatch")
    if not {CURRICULUM, MANIFEST, BASELINE}.issubset(expected):
        raise ValueError("ZIP missing TRAIN artifacts")
    total = 0
    for info in infos:
        if info.filename == INDEX:
            continue
        name = info.filename
        entry = expected.get(name)
        mode = info.external_attr >> 16
        if (not name_ok(name) or not isinstance(entry, dict)
                or set(entry) != {"sha256", "size"}
                or not isinstance(entry["sha256"], str)
                or HEX64.fullmatch(entry["sha256"]) is None
                or type(entry["size"]) is not int or entry["size"] < 0
                or entry["size"] != info.file_size
                or info.file_size > MAX_FILE_BYTES
                or info.compress_type != zipfile.ZIP_STORED
                or stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
            raise ValueError("untrusted or altered ZIP member")
        total += info.file_size
        if total > MAX_TOTAL_BYTES:
            raise ValueError("ZIP exceeds uncompressed byte budget")
    return metadata, infos


def restore_stage(bundle: Path, expected_sha256: str, destination: Path,
                  pinned_sha: str, *, rederive: bool = True) -> dict:
    """Import into a brand-new run directory, atomic rename after full proof."""
    bundle, destination = Path(bundle).resolve(), Path(destination).resolve()
    if (not isinstance(expected_sha256, str) or HEX64.fullmatch(expected_sha256) is None
            or not GITSHA.fullmatch(pinned_sha)):
        raise ValueError("explicit expected bundle SHA and source Git SHA required")
    if destination.exists():
        raise FileExistsError("refusing to overwrite a populated run directory")
    if not bundle.is_file() or digest(bundle) != expected_sha256:
        raise ValueError("stage archive differs from operator SHA-256")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".h67-stage-", dir=destination.parent) as temp:
        temporary = Path(temp) / "run"
        temporary.mkdir()
        with zipfile.ZipFile(bundle) as archive:
            metadata, infos = _check_zip(archive, pinned_sha=pinned_sha)
            for info in infos:
                if info.filename == INDEX:
                    continue
                target = temporary / info.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                h = hashlib.sha256()
                with archive.open(info) as stream, target.open("xb") as result:
                    while True:
                        part = stream.read(BLOCK)
                        if not part:
                            break
                        h.update(part)
                        result.write(part)
                if h.hexdigest() != metadata["files"][info.filename]["sha256"]:
                    raise ValueError("stage file SHA-256 mismatch")
        checked = validate_train(temporary, rederive=rederive)
        if checked["dataset_sha256"] != metadata["dataset_sha256"]:
            raise ValueError("derived TRAIN SHA differs after extraction")
        # Only move into place after every member and source derivation passes.
        os.replace(temporary, destination)
    return {"schema": SCHEMA, "status": "RESTORED_TRAIN_STAGE",
            "sha256": expected_sha256, "git_sha": pinned_sha,
            "dataset_sha256": checked["dataset_sha256"],
            "files": len(metadata["files"]), "private_provider_readback_verified": False}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    build = sub.add_parser("build")
    build.add_argument("--work", type=Path, required=True)
    build.add_argument("--bundle", type=Path, required=True)
    build.add_argument("--git-sha", required=True)
    restore = sub.add_parser("restore")
    restore.add_argument("--bundle", type=Path, required=True)
    restore.add_argument("--expected-sha256", required=True)
    restore.add_argument("--dest", type=Path, required=True)
    restore.add_argument("--git-sha", required=True)
    args = p.parse_args(argv)
    result = (make_stage(args.work, args.bundle, args.git_sha)
              if args.mode == "build" else
              restore_stage(args.bundle, args.expected_sha256, args.dest, args.git_sha))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
