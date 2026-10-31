"""H257 owner-only, fail-closed PNG/RGBA collision screen. DENIAL ONLY.

Never consumes H173, gold, labels, predictions or model outputs.
Never admits TRAIN/DEV/OOD; never prints private paths or digests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import stat

from h250.native_original_todo_app import STATUS as H250_STATUS, STAGES as H250_STAGES
from h251.original_lifeos_rtl import STATUS as H251_STATUS, STAGES as H251_STAGES
from h256.pre173_visual_history import digest_png

OUTPUT_NAME = "H257_OWNER_PRIVATE_VISUAL_COLLISION_SCREEN_NOT_ADMITTED.json"
SOURCES = {"h250": (H250_STATUS, H250_STAGES), "h251": (H251_STATUS, H251_STAGES)}
SHA = re.compile(r"[0-9a-f]{64}\Z")
MAX_JSON = 2 * 1024 * 1024
DENY = ("h173", "gold", "scored", "ood", "prediction", "label", "reference")


def require(ok: bool) -> None:
    if not ok:
        raise ValueError("H257_NOT_QUALIFIED_NO_ADMISSION")


def private_path(path: Path, *, output: bool = False) -> Path:
    require(isinstance(path, Path) and path.is_absolute())
    require(not any(t in part.casefold() for part in path.parts for t in DENY))
    for parent in (path.parent, *path.parent.parents):
        require(not (parent / ".git").exists())
        if parent.exists():
            st = parent.lstat()
            require(parent.is_dir() and not parent.is_symlink()
                    and not (getattr(st, "st_file_attributes", 0) & 0x400))
    if output:
        require(path.name == OUTPUT_NAME and not path.exists() and not path.is_symlink())
    else:
        require(path.is_file() and not path.is_symlink())
        st = path.lstat()
        require(stat.S_ISREG(st.st_mode)
                and not (getattr(st, "st_file_attributes", 0) & 0x400)
                and 0 < st.st_size <= MAX_JSON)
    return path


def read_private_json(path: Path) -> dict:
    private_path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    require(type(value) is dict)
    return value


def sha(value: object) -> bool:
    return type(value) is str and SHA.fullmatch(value) is not None


def audit(historical: Path, sources: dict[str, Path], output: Path) -> dict:
    private_path(output, output=True)
    require(type(sources) is dict and set(sources) == set(SOURCES))
    for path in (historical, *sources.values()):
        private_path(path)
    require(len({str(p.resolve(strict=False)).casefold() for p in
                 [historical, *sources.values(), output.parent]}) == 4)
    history = read_private_json(historical)
    summary = history.get("summary")
    rows = history.get("owner_only_images")
    require(type(summary) is dict and type(rows) is list
            and summary.get("status") ==
                "H256_PRE173_SURVIVING_MEDIA_EXCLUSIONS_ONLY_NO_ADMISSION"
            and summary.get("h173_ood_accessed") is False
            and summary.get("gold_opened") is False
            and summary.get("train_dev_ood_admitted") is False
            and summary.get("complete_old_visual_history") is False
            and summary.get("complete_old_source_history") is False
            and type(summary.get("surviving_png_files")) is int
            and len(rows) == summary["surviving_png_files"]
            and 0 < len(rows) <= 1024)
    old_png, old_rgba = set(), set()
    for row in rows:
        require(type(row) is dict and sha(row.get("png_sha256")))
        phase = row.get("phase")
        match = re.fullmatch(r"(?:private-h|hala-h)([0-9]{2,3})(?:-private)?",
                             phase, re.I) if type(phase) is str else None
        require(match is not None and int(match.group(1)) < 173)
        rgba = row.get("decoded_rgba_sha256")
        require(rgba is None or sha(rgba))
        old_png.add(row["png_sha256"])
        if rgba is not None:
            old_rgba.add(rgba)
    require(len(old_png) == summary.get("distinct_surviving_png_sha")
            and len(old_rgba) == summary.get("distinct_surviving_decoded_rgba_sha")
            and len(old_png) >= summary.get("manifest_png_sha_floor", 1))
    seen_png, seen_rgba = set(), set()
    png_hits = rgba_hits = 0
    repeated_png = repeated_rgba = 0
    frames_checked = 0
    for name in sorted(SOURCES):
        status, stages = SOURCES[name]
        receipt_path = sources[name]
        require(receipt_path.name == "PRIVATE_RESEARCH_NOT_ADMITTED.json")
        data = read_private_json(receipt_path)
        require(data.get("status") == status
                and data.get("train") is False and data.get("dev") is False
                and data.get("ood") is False
                and data.get("model_calls") == 0
                and data.get("optimizer_steps") == 0)
        cases = data.get("cases")
        require(type(cases) is list and len(cases) == 2
                and {c.get("language") for c in cases if type(c) is dict}
                == {"ar", "en"})
        for case in cases:
            require(type(case) is dict and
                    all(case.get(k) is False for k in ("train", "dev", "ood")))
            frames = case.get("frames")
            require(type(frames) is dict and set(frames) == set(stages))
            case_id = case.get("case_id")
            require(type(case_id) is str and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", case_id) is not None)
            for stage in stages:
                frame = frames[stage]
                require(type(frame) is dict
                        and sha(frame.get("png_sha256"))
                        and sha(frame.get("rgba_sha256")))
                media = receipt_path.parent / (case_id + "-" + stage + ".png")
                private_path(media)
                require(media.parent.resolve() == receipt_path.parent.resolve()
                        and media.stat().st_size <= 8 * 1024 * 1024)
                png, rgba, _, _ = digest_png(media.read_bytes())
                require(png == frame["png_sha256"] and rgba == frame["rgba_sha256"])
                repeated_png += int(png in seen_png)
                repeated_rgba += int(rgba in seen_rgba)
                seen_png.add(png)
                seen_rgba.add(rgba)
                png_hits += int(png in old_png)
                rgba_hits += int(rgba in old_rgba)
                frames_checked += 1
    require(frames_checked == 24)
    result = {
        "status": "H257_PRIVATE_VISUAL_SCREEN_DENIAL_ONLY_NOT_ADMITTED",
        "historical_surviving_png_exclusions": len(old_png),
        "historical_surviving_rgba_exclusions": len(old_rgba),
        "candidate_source_families": len(SOURCES),
        "candidate_frames_verified_against_actual_png": frames_checked,
        "candidate_duplicate_png_frames": repeated_png,
        "candidate_duplicate_rgba_frames": repeated_rgba,
        "candidate_unique_png": len(seen_png),
        "candidate_unique_rgba": len(seen_rgba),
        "known_old_png_collisions": png_hits,
        "known_old_decoded_rgba_collisions": rgba_hits,
        "known_surviving_visual_collision_free": png_hits == rgba_hits == 0,
        "historical_completeness_proven": False,
        "source_disjointness_proven": False,
        "transitive_media_rights_proven": False,
        "train_admitted": False, "dev_admitted": False, "ood_admitted": False,
        "h173_opened": False, "gold_opened": False,
        "model_calls": 0, "optimizer_steps": 0,
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True)
        handle.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-receipt", required=True, type=Path)
    parser.add_argument("--h250-receipt", required=True, type=Path)
    parser.add_argument("--h251-receipt", required=True, type=Path)
    parser.add_argument("--private-output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.historical_receipt,
                       {"h250": args.h250_receipt, "h251": args.h251_receipt},
                       args.private_output)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        print('{"status":"H257_NOT_QUALIFIED_NO_ADMISSION"}')
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
