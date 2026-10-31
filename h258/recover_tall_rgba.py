"""H258 owner-only recovery of bounded tall historical PNG visual exclusions.

DENIAL ONLY: never admits data, opens scored H173/gold, or publishes media.
The H256 receipt remains immutable. This tool writes a NEW private receipt.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import zlib

from PIL import Image, UnidentifiedImageError

from h256.pre173_visual_history import _safe_directory, safe_file

OUTPUT_NAME = "H258_OWNER_PRIVATE_TALL_RGBA_EXCLUSIONS_NOT_ADMITTED.json"
STATUS = "H258_TALL_HISTORICAL_RGBA_EXCLUSIONS_ONLY_NO_ADMISSION"
SOURCE_STATUS = "H256_PRE173_SURVIVING_MEDIA_EXCLUSIONS_ONLY_NO_ADMISSION"
SHA = re.compile(r"[0-9a-f]{64}\Z")
PHASE = re.compile(r"(?:private-h|hala-h)([0-9]{2,3})(?:-private)?\Z", re.I)
SENSITIVE = ("h173", "gold", "label", "scored", "ood", "reference", "prediction")
SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_SIDE = 8192
MAX_PIXELS = 16_777_216
MAX_PNG_BYTES = 8 * 1024 * 1024
MAX_ROWS = 1024


def require(ok: bool) -> None:
    if not ok:
        raise ValueError("H258_NOT_QUALIFIED_NO_ADMISSION")


def is_sha(value: object) -> bool:
    return type(value) is str and SHA.fullmatch(value) is not None


def safe_private(path: Path, *, output: bool = False) -> None:
    require(isinstance(path, Path) and path.is_absolute())
    require(not any(t in str(path).casefold() for t in SENSITIVE))
    require(not any((p / ".git").exists() for p in (path.parent, *path.parent.parents)))
    require(_safe_directory(path.parent))
    if output:
        require(path.name == OUTPUT_NAME and not path.exists() and not path.is_symlink())
    else:
        require(safe_file(path) and 0 < path.stat().st_size <= 2 * 1024 * 1024)


def resolve_image(parents: list[Path], phase: str, relative: str) -> Path:
    match = PHASE.fullmatch(phase) if type(phase) is str else None
    require(match is not None and int(match.group(1)) < 173)
    require(type(relative) is str and "\\" not in relative)
    rel = PurePosixPath(relative)
    require(not rel.is_absolute() and bool(rel.parts)
            and rel.as_posix() == relative
            and all(p not in ("", ".", "..") and ":" not in p
                    and not any(ord(ch) < 32 for ch in p) for p in rel.parts))
    require(not any(t in p.casefold() for p in rel.parts for t in SENSITIVE))
    matches = []
    for parent in parents:
        require(parent.is_absolute() and _safe_directory(parent))
        phase_dir = parent / phase
        if not phase_dir.exists():
            continue
        require(_safe_directory(phase_dir))
        current = phase_dir
        for part in rel.parts[:-1]:
            current = current / part
            require(_safe_directory(current))
        image = current / rel.parts[-1]
        if image.exists():
            require(safe_file(image) and 0 < image.stat().st_size <= MAX_PNG_BYTES)
            require(image.resolve().is_relative_to(phase_dir.resolve()))
            matches.append(image)
    require(len(matches) == 1)
    return matches[0]


def inspect_png_structure(blob: bytes) -> tuple[str, int, int]:
    """Validate signature, all chunk CRCs, IEND and bounded IHDR; no pixel decode."""
    if not blob.startswith(SIGNATURE):
        return "not_png_signature", 0, 0
    p = len(SIGNATURE)
    first = True
    saw_data = False
    width = height = 0
    while p < len(blob):
        if len(blob) - p < 12:
            return "invalid_structure", 0, 0
        length = struct.unpack(">I", blob[p:p+4])[0]
        if length > MAX_PNG_BYTES or p + length + 12 > len(blob):
            return "invalid_structure", 0, 0
        kind = blob[p+4:p+8]
        if not all(65 <= x <= 90 or 97 <= x <= 122 for x in kind):
            return "invalid_structure", 0, 0
        payload = blob[p+8:p+8+length]
        checksum = struct.unpack(">I", blob[p+8+length:p+12+length])[0]
        if zlib.crc32(kind + payload) & 0xffffffff != checksum:
            return "invalid_structure", 0, 0
        if first:
            if kind != b"IHDR" or length != 13:
                return "invalid_structure", 0, 0
            width, height = struct.unpack(">II", payload[:8])
            if not width or not height:
                return "invalid_structure", 0, 0
            first = False
        elif kind == b"IHDR":
            return "invalid_structure", 0, 0
        if kind == b"IDAT":
            saw_data = True
        p += length + 12
        if kind == b"IEND":
            if length or not saw_data or p != len(blob):
                return "invalid_structure", 0, 0
            return "valid_structure", width, height
    return "invalid_structure", 0, 0


def rgba_digest(blob: bytes, width: int, height: int) -> str:
    require(0 < width <= MAX_SIDE and 0 < height <= MAX_SIDE
            and width * height <= MAX_PIXELS)
    with Image.open(BytesIO(blob)) as image:
        require(image.format == "PNG" and image.size == (width, height))
        image.verify()
    with Image.open(BytesIO(blob)) as image:
        require(image.format == "PNG" and image.size == (width, height))
        pixels = image.convert("RGBA").tobytes()
    require(len(pixels) == width * height * 4)
    return sha256(width.to_bytes(4, "big")
                  + height.to_bytes(4, "big") + pixels).hexdigest()


def audit(h256_receipt: Path, parents: list[Path], output: Path) -> dict:
    safe_private(h256_receipt)
    safe_private(output, output=True)
    require(type(parents) is list and 0 < len(parents) <= 4)
    require(len({str(p.resolve()).casefold() for p in parents}) == len(parents))
    source = json.loads(h256_receipt.read_text(encoding="utf-8"))
    summary, rows = source.get("summary"), source.get("owner_only_images")
    require(type(summary) is dict and type(rows) is list
            and summary.get("status") == SOURCE_STATUS
            and summary.get("complete_old_visual_history") is False
            and summary.get("complete_old_source_history") is False
            and summary.get("train_dev_ood_admitted") is False
            and summary.get("h173_ood_accessed") is False
            and summary.get("gold_opened") is False
            and 0 < len(rows) <= MAX_ROWS
            and len(rows) == summary.get("surviving_png_files"))
    missing = [r for r in rows if type(r) is dict
               and r.get("decoded_rgba_sha256") is None]
    require(len(missing) == summary.get("undecodable_png_named_files"))
    old_png, old_rgba = set(), set()
    for row in rows:
        require(type(row) is dict and is_sha(row.get("png_sha256")))
        phase = row.get("phase")
        match = PHASE.fullmatch(phase) if type(phase) is str else None
        require(match is not None and int(match.group(1)) < 173)
        relative = row.get("relative_png_path")
        require(type(relative) is str and "\\" not in relative)
        rel = PurePosixPath(relative)
        require(not rel.is_absolute() and bool(rel.parts)
                and rel.as_posix() == relative
                and all(p not in ("", ".", "..") and ":" not in p
                        and not any(ord(ch) < 32 for ch in p) for p in rel.parts)
                and not any(t in p.casefold() for p in rel.parts for t in SENSITIVE))
        rgba = row.get("decoded_rgba_sha256")
        require(rgba is None or is_sha(rgba))
        old_png.add(row["png_sha256"])
        if rgba is not None:
            old_rgba.add(rgba)
    require(len(old_png) == summary.get("distinct_surviving_png_sha")
            and len(old_rgba) == summary.get("distinct_surviving_decoded_rgba_sha"))
    categories = {"not_png_signature": 0, "invalid_structure": 0,
                  "oversize_policy_recovered": 0, "unsafe_dimensions": 0,
                  "unexpected_missing_rgba": 0}
    recovered = []
    seen = set()
    for row in missing:
        image = resolve_image(parents, row.get("phase"), row.get("relative_png_path"))
        blob = image.read_bytes()
        require(sha256(blob).hexdigest() == row["png_sha256"])
        kind, width, height = inspect_png_structure(blob)
        if kind != "valid_structure":
            categories[kind] += 1
            continue
        if max(width, height) <= 4096 and width * height <= 4096 * 4096:
            categories["unexpected_missing_rgba"] += 1
            continue
        if max(width, height) > MAX_SIDE or width * height > MAX_PIXELS:
            categories["unsafe_dimensions"] += 1
            continue
        try:
            rgba = rgba_digest(blob, width, height)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            categories["invalid_structure"] += 1
            continue
        categories["oversize_policy_recovered"] += 1
        if row["png_sha256"] not in seen:
            recovered.append({"phase": row["phase"],
                              "relative_png_path": row["relative_png_path"],
                              "png_sha256": row["png_sha256"],
                              "decoded_rgba_sha256": rgba,
                              "width": width, "height": height})
            seen.add(row["png_sha256"])
    require(sum(categories.values()) == len(missing))
    additional_rgba = {r["decoded_rgba_sha256"] for r in recovered}
    result = {
        "status": STATUS,
        "source_h256_png_files": len(rows),
        "source_h256_missing_rgba_files": len(missing),
        "classification": categories,
        "recovered_distinct_png": len(recovered),
        "recovered_distinct_rgba": len(additional_rgba),
        "new_distinct_rgba_exclusions": len(additional_rgba - old_rgba),
        "combined_distinct_rgba_exclusions": len(old_rgba | additional_rgba),
        "historical_completeness_proven": False,
        "source_disjointness_proven": False,
        "media_rights_proven": False,
        "train_admitted": False, "dev_admitted": False, "ood_admitted": False,
        "h173_opened": False, "gold_opened": False,
        "model_calls": 0, "optimizer_steps": 0,
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump({"summary": result, "owner_only_recovered": recovered},
                  handle, sort_keys=True)
        handle.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h256-receipt", required=True, type=Path)
    parser.add_argument("--historical-parent", required=True, type=Path, action="append")
    parser.add_argument("--private-output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.h256_receipt, args.historical_parent, args.private_output)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError,
            Image.DecompressionBombError):
        print('{"status":"H258_NOT_QUALIFIED_NO_ADMISSION"}')
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
