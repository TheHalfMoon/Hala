"""H256 owner-only surviving PRE-H173 PNG/RGBA exclusion inventory.

Reads pre-H173 PNG pixels ONLY to derive hashes. Never reads gold, scored
H173, private labels, or models; never publishes any private receipt.
This is a denial-only historical census, NEVER an admission gate.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import stat

from PIL import Image, UnidentifiedImageError
from h178.owner_git_inventory import _safe_directory
from h185.manifest_sha_only import scan_legacy_manifest_sha_fields
from h241.expanded_pre173_metadata import find_pre_h173_phases

MAX_FILES = 1024
MAX_DEPTH = 8
MAX_PNG_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_PIXELS = 4096 * 4096
RECEIPT_NAME = "H256_OWNER_PRIVATE_PRE173_VISUAL_HASHES_NOT_ADMITTED.json"
SENSITIVE_TOKENS = ("gold", "label", "ood", "scored", "prediction", "reference", "answer")
DENY_DIRS = frozenset({
    ".git", ".venv", "venv", "node_modules", "gold", "labels",
    "predictions", "scored", "ood", "h173", "checkpoints", "weights",
    "models", "site-packages", "__pycache__",
})


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("H256 fail-closed: " + message)


def safe_file(path: Path) -> bool:
    st = path.lstat()
    return (not path.is_symlink()
            and not (getattr(st, "st_file_attributes", 0) & 0x400)
            and stat.S_ISREG(st.st_mode))


def digest_png(blob: bytes) -> tuple[str, str, int, int]:
    """H231-compatible decoded RGBA: width|height|pixel bytes, not PNG SHA."""
    require(type(blob) is bytes and 0 < len(blob) <= MAX_PNG_BYTES,
            "PNG bytes out of bounds")
    with Image.open(BytesIO(blob)) as image:
        require(image.format == "PNG", "PNG decoder format mismatch")
        width, height = image.size
        require(0 < width <= 4096 and 0 < height <= 4096
                and width * height <= MAX_PIXELS, "PNG dimensions unsafe")
        rgba = image.convert("RGBA")
        pixels = rgba.tobytes()
    return (
        sha256(blob).hexdigest(),
        sha256(width.to_bytes(4, "big") +
               height.to_bytes(4, "big") + pixels).hexdigest(),
        width, height,
    )


def collect_pngs(phases: list[Path]) -> tuple[list[tuple[Path, Path]], int]:
    require(type(phases) is list and 0 < len(phases) <= 100,
            "bounded phase list required")
    result: list[tuple[Path, Path]] = []
    excluded_dir_count = 0
    seen: set[str] = set()
    for phase in phases:
        require(_safe_directory(phase), "unsafe phase root")
        match = re.fullmatch(r"(?:private-h|hala-h)([0-9]{2,3})(?:-private)?",
                             phase.name, re.I)
        require(match is not None and int(match.group(1)) < 173,
                "only strictly pre-H173 phase names allowed")
        for current, dirs, files in os.walk(phase, topdown=True, followlinks=False):
            directory = Path(current)
            depth = len(directory.relative_to(phase).parts)
            require(depth <= MAX_DEPTH, "historical tree too deep")
            allowed = []
            for name in sorted(dirs):
                child = directory / name
                if (name.casefold() in DENY_DIRS
                        or any(token in name.casefold() for token in SENSITIVE_TOKENS)):
                    excluded_dir_count += 1
                    continue
                require(_safe_directory(child), "unsafe directory/reparse")
                require(depth < MAX_DEPTH, "historical tree too deep")
                allowed.append(name)
            dirs[:] = allowed
            for name in sorted(files):
                if Path(name).suffix.casefold() != ".png":
                    continue
                require(not any(token in name.casefold() for token in SENSITIVE_TOKENS),
                        "sensitive PNG filename refused")
                path = directory / name
                require(safe_file(path), "unsafe PNG/reparse")
                require(0 < path.stat().st_size <= MAX_PNG_BYTES,
                        "PNG file size unsafe")
                key = os.path.normcase(str(path.resolve())).casefold()
                require(key not in seen, "duplicate PNG path")
                seen.add(key)
                result.append((phase, path))
                require(len(result) <= MAX_FILES, "too many PNGs")
    require(bool(result), "no surviving historical PNGs")
    return result, excluded_dir_count


def safe_output_path(output: Path, phases: list[Path]) -> Path:
    require(output.is_absolute() and output.name == RECEIPT_NAME,
            "explicit owner-only receipt filename required")
    require(_safe_directory(output.parent), "private output parent unsafe")
    require(not output.exists() and not output.is_symlink(),
            "no overwrite of private receipt")
    target = output.resolve(strict=False)
    for phase in phases:
        require(not target.is_relative_to(phase.resolve()),
                "receipt cannot be inside historical source")
    for ancestor in (output.parent, *output.parent.parents):
        require(not (ancestor / ".git").exists(),
                "receipt cannot be inside a Git checkout")
    return output


def audit(parents: list[Path], output: Path) -> dict:
    phases = find_pre_h173_phases(parents)  # NEVER traverses H173
    target = safe_output_path(output, phases)
    files, excluded_dirs = collect_pngs(phases)
    legacy = scan_legacy_manifest_sha_fields(phases)
    known = legacy["_owner_only_png_sha_candidates"]
    require(type(known) is frozenset and len(known) > 0,
            "historical manifest SHA floor missing")
    rows: list[dict] = []
    png_seen: set[str] = set()
    rgba_seen: set[str] = set()
    total = 0
    undecodable = 0
    for phase, path in files:
        size = path.stat().st_size
        total += size
        require(total <= MAX_TOTAL_BYTES, "total historical PNG bytes exceeded")
        blob = path.read_bytes()
        require(len(blob) == size, "historical PNG changed during read")
        try:
            png, rgba, width, height = digest_png(blob)
        except (UnidentifiedImageError, OSError, ValueError):
            # Surviving filename is not proof of a decodable PNG.
            # Preserve raw-byte exclusion, never invent a visual digest.
            png, rgba, width, height = sha256(blob).hexdigest(), None, None, None
            undecodable += 1
        png_seen.add(png)
        if rgba is not None:
            rgba_seen.add(rgba)
        rows.append({
            "phase": phase.name,
            "relative_png_path": path.relative_to(phase).as_posix(),
            "png_sha256": png,
            "decoded_rgba_sha256": rgba,
            "width": width,
            "height": height,
        })
    summary = {
        "status": "H256_PRE173_SURVIVING_MEDIA_EXCLUSIONS_ONLY_NO_ADMISSION",
        "historical_phases": len(phases),
        "manifest_png_sha_floor": len(known),
        "surviving_png_files": len(rows),
        "decodable_png_files": len(rows) - undecodable,
        "undecodable_png_named_files": undecodable,
        "distinct_surviving_png_sha": len(png_seen),
        "distinct_surviving_decoded_rgba_sha": len(rgba_seen),
        "known_manifest_png_sha_found_on_disk": len(known & png_seen),
        "known_manifest_png_sha_missing_on_disk": len(known - png_seen),
        "surviving_png_sha_not_in_known_manifests": len(png_seen - known),
        "excluded_sensitive_directories": excluded_dirs,
        "complete_old_source_history": False,
        "complete_old_visual_history": False,
        "train_dev_ood_admitted": False,
        "h173_ood_accessed": False,
        "gold_opened": False,
        "model_calls": 0,
        "optimizer_steps": 0,
    }
    private_receipt = {"summary": summary, "owner_only_images": rows}
    with target.open("x", encoding="utf-8") as handle:
        json.dump(private_receipt, handle, ensure_ascii=True, sort_keys=True)
        handle.write("\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-parent", type=Path, action="append", required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = audit(args.historical_parent, args.private_output)
    except (OSError, ValueError, TypeError, KeyError, Image.DecompressionBombError):
        print('{"status":"H256_NOT_QUALIFIED_NO_ADMISSION"}')
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
