"""H73: Kaggle CPU-only, internet-off admission of the immutable H71 TRAIN ZIP.

This script is intentionally self-contained (Python stdlib only) so Kaggle
does not need GitHub access, pip installations, internet or model weights.
It never unpacks screenshots, runs an optimizer, reads DEV images, or
uploads anything. Its ONLY output is a non-sensitive aggregate proof.

The immutable expected H71 source is independently archived privately.
The full H67 member inventory and H61 TRAIN manifest/rows are rechecked
using streaming SHA256 with zero raw screenshot output. This is a separate
CPU archive verifier, NOT a replacement for H67's stronger full restore
and H61 source re-derivation (already performed on WSL and Kaggle readback).
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import zipfile

SCHEMA = "hala.h73.kaggle-cpu-admission.v1"
INPUT = Path("/kaggle/input/hala-h71-drag-train-10000-private/hala-h71-train-10000.hala.bin")
OUTPUT = Path("/kaggle/working/h73-cpu-admission.json")
DATA_ROOT = Path("/kaggle/input")
DATASET_OWNER = "abdulazizshehri"
DATASET_SLUG = "hala-h71-drag-train-10000-private"
DATASET_FILE = "hala-h71-train-10000.hala.bin"

def locate_attached_h71(root: Path = DATA_ROOT) -> Path:
    """Only the exact owner/slug/file Kaggle input; never scan other datasets."""
    choices = (
        root / "datasets" / DATASET_OWNER / DATASET_SLUG / DATASET_FILE,
        root / DATASET_SLUG / DATASET_FILE,
        root / DATASET_OWNER / DATASET_SLUG / DATASET_FILE,
    )
    for path in choices:
        if path.is_file() and not path.is_symlink():
            return path
    raise ValueError("exact private H71 TRAIN Kaggle input was not mounted")
INDEX = "h67-stage.json"
MANIFEST = "export/h61_drag_curriculum_manifest.json"
ROWS = "export/h61_drag_curriculum.jsonl"
BASELINE = "public-dev-baseline-receipt.json"
EXPECTED = {
    "sha256": "4a8e602c785c2b90a60e49f78381a7af983401cb99b16c53f3df69e930cc073a",
    "git_sha": "acd694fbae10752ff4ee3fe6cc3e52142ab7f0fe",
    "dataset_sha256": "9888ac9d3b5ec7b8efe677287d54dc350c222a1804619e6217a84ec8f95eddcd",
    "bytes": 1_759_484_545, "files": 15_224, "rows": 10_000,
    "language_counts": {"ar": 1000, "en": 9000},
    "firstparty_rows": 2000,
}
HEX = re.compile(r"^[0-9a-f]{64}$")
NAME = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")
BLOCK = 1024 * 1024
MAX_TOTAL = 5 * 1024**3
MAX_MEMBER = 2 * 1024**3
MAX_FILES = 30000


def digest(stream) -> str:
    h = hashlib.sha256()
    for part in iter(lambda: stream.read(BLOCK), b""):
        h.update(part)
    return h.hexdigest()


def json_object(raw: bytes, limit: int = 16 * 1024 * 1024):
    if len(raw) > limit:
        raise ValueError("oversized H73 archive metadata")
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate archive metadata key")
            value[key] = item
        return value
    def bad_constant(_):
        raise ValueError("non-finite archive metadata")
    obj = json.loads(raw.decode("utf8"), object_pairs_hook=pairs,
                     parse_constant=bad_constant)
    if not isinstance(obj, dict):
        raise ValueError("archive metadata must be an object")
    return obj


def valid_member(name: str) -> bool:
    return (isinstance(name, str) and 0 < len(name) <= 255
            and NAME.fullmatch(name) is not None
            and all(segment not in (".", "..") and not segment.startswith(".")
                    for segment in name.split("/"))
            and (name == BASELINE or name.startswith("export/")))


def qualify(path: Path, expected: dict = EXPECTED) -> dict:
    """Fail-closed streaming, CPU-only proof of an exact pinned H71 ZIP."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size != expected["bytes"]:
        raise ValueError("H71 file absent or wrong byte count")
    with path.open("rb") as source:
        original_sha = digest(source)
    if original_sha != expected["sha256"]:
        raise ValueError("H71 complete archive SHA256 mismatch")
    with zipfile.ZipFile(path, "r", allowZip64=True) as z:
        infos = z.infolist()
        if (len(infos) != expected["files"] + 1 or len(infos) > MAX_FILES
                or len({entry.filename for entry in infos}) != len(infos)):
            raise ValueError("unexpected number or duplication of H71 members")
        if {info.filename for info in infos} != set(z.namelist()):
            raise ValueError("ambiguous ZIP member namespace")
        if INDEX not in z.namelist():
            raise ValueError("H67 source-index missing")
        index = json_object(z.read(INDEX))
        if (index.get("schema") != "hala.h67.cpu-stage.v1"
                or index.get("git_sha") != expected["git_sha"]
                or index.get("source_split") != "train"
                or index.get("dataset_sha256") != expected["dataset_sha256"]
                or index.get("row_count") != expected["rows"]
                or index.get("media_count") != expected["rows"]
                or not isinstance(index.get("files"), dict)
                or len(index["files"]) != expected["files"]):
            raise ValueError("H67 TRAIN source identity differs")
        if set(index["files"]) | {INDEX} != {x.filename for x in infos}:
            raise ValueError("H67 index/member inventory differs")
        total = 0
        hashes = {}
        for info in infos:
            if info.filename == INDEX:
                continue
            meta = index["files"][info.filename]
            if (not valid_member(info.filename)
                    or not isinstance(meta, dict)
                    or set(meta) != {"sha256", "size"}
                    or type(meta["size"]) is not int
                    or meta["size"] != info.file_size
                    or not isinstance(meta["sha256"], str)
                    or HEX.fullmatch(meta["sha256"]) is None
                    or not 0 <= info.file_size <= MAX_MEMBER
                    or info.compress_type != zipfile.ZIP_STORED
                    or stat.S_IFMT(info.external_attr >> 16) not in (0, stat.S_IFREG)):
                raise ValueError("hostile or corrupt H67 ZIP member")
            total += info.file_size
            if total > MAX_TOTAL:
                raise ValueError("H73 unbounded source bytes")
            with z.open(info, "r") as part:
                actual = digest(part)
            if actual != meta["sha256"]:
                raise ValueError("H67 member byte proof failed")
            hashes[info.filename] = actual
        if not {MANIFEST, ROWS, BASELINE}.issubset(hashes):
            raise ValueError("H61 required TRAIN labels/receipts missing")
        manifest = json_object(z.read(MANIFEST))
        if (manifest.get("schema") != "hala.training.manifest.v1"
                or manifest.get("source_schema") != "hala.h61.native-drag-curriculum.v1"
                or manifest.get("split") != "train"
                or manifest.get("train_rows") != expected["rows"]
                or manifest.get("unique_source_images") != expected["rows"]
                or manifest.get("dataset_sha256") != expected["dataset_sha256"]
                or manifest.get("language_counts") != expected["language_counts"]
                or manifest.get("training_executed") is not False
                or manifest.get("no_benchmark_eval_source") is not True
                or manifest.get("model_improvement_qualified") is not False
                or manifest.get("family_targets", {}).get(
                    "h61_firstparty_bilingual_drag_train") != expected["firstparty_rows"]):
            raise ValueError("H61 TRAIN curriculum admission guard failed")
        if hashes[ROWS] != expected["dataset_sha256"]:
            raise ValueError("H61 curriculum row-data SHA mismatch")
        media = manifest.get("media")
        if not isinstance(media, list) or len(media) != expected["rows"]:
            raise ValueError("H61 media manifest is incomplete")
        media_hashes = {}
        for media_item in media:
            if not isinstance(media_item, dict):
                raise ValueError("untrusted H61 media item")
            name = media_item.get("path")
            digest_value = media_item.get("sha256")
            if (not isinstance(name, str) or not valid_member("export/" + name)
                    or not isinstance(digest_value, str)
                    or HEX.fullmatch(digest_value) is None
                    or name in media_hashes
                    or hashes.get("export/" + name) != digest_value):
                raise ValueError("H61 selected screenshot digest changed")
            media_hashes[name] = digest_value
        seen_rows = set()
        families, languages, capabilities = Counter(), Counter(), Counter()
        count = 0
        # Avoid emitting training text, screenshot paths or dataset prompts.
        with z.open(ROWS, "r") as source:
            for line in source:
                row = json_object(line, 512 * 1024)
                path_name = row.get("image_path")
                row_id = row.get("id")
                if (row.get("split") != "train"
                        or row.get("language") not in ("ar", "en")
                        or not isinstance(row_id, str) or not row_id
                        or row_id in seen_rows
                        or path_name not in media_hashes
                        or row.get("image_sha256") != media_hashes[path_name]
                        or not isinstance(row.get("family"), str)
                        or not isinstance(row.get("capability"), str)):
                    raise ValueError("H61 non-TRAIN or malformed source row")
                seen_rows.add(row_id)
                families[row["family"]] += 1
                languages[row["language"]] += 1
                capabilities[row["capability"]] += 1
                count += 1
                if count > expected["rows"]:
                    raise ValueError("H61 excess rows")
        if count != expected["rows"] or len(seen_rows) != expected["rows"]:
            raise ValueError("H61 selected row count mismatch")
        if (dict(languages) != manifest.get("language_counts")
                or dict(families) != manifest.get("selected_by_family")
                or dict(capabilities) != manifest.get("capability_counts")
                or sum(families.values()) != len(media_hashes)
                or families.get("h61_firstparty_bilingual_drag_train")
                != expected["firstparty_rows"]):
            raise ValueError("H61 selected rows diverge from selected TRAIN ledger")
    return {"schema": SCHEMA, "status": "KAGGLE_PRIVATE_CPU_SOURCE_ADMITTED",
            "source_archive_sha256": original_sha,
            "dataset_sha256": expected["dataset_sha256"],
            "source_git_sha": expected["git_sha"],
            "members_sha_verified": expected["files"],
            "media_sha_verified": len(media_hashes),
            "rows_verified": count, "language_counts": dict(languages),
            "original_dataset_unchanged": True, "gpu_allocated": False,
            "training_executed": False, "model_checkpoint_produced": False,
            "provider_visibility_verified_by_owner": False}


def main():
    # This script may only be executed as a CPU-only Kaggle source proof.
    if os.getenv("KAGGLE_KERNEL_RUN_TYPE") is None:
        raise RuntimeError("H73 requires a Kaggle kernel, not a local model run")
    if os.getenv("CUDA_VISIBLE_DEVICES") not in (None, "", "-1"):
        raise RuntimeError("H73 cannot execute in a GPU-exposed notebook")
    result = qualify(locate_attached_h71())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        raise FileExistsError("refusing overwrite of existing H73 proof")
    OUTPUT.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                      encoding="utf8")
    print("H73_CPU_PROOF=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
