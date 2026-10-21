"""H61 capability-balanced curriculum over the verified original DragOn TRAIN shards.

The founder's curriculum requirement is capability balance, not row count: a
million rows of the same screenshot teach nothing. This module therefore
balances on the capability of a row, caps how many rows one screenshot may
contribute, and records the resulting balance so the mixture can be audited
instead of asserted.

Every row is derived from the original upstream record: the original
instruction, the original `ordered` flag, the original start and end boxes and
the original screenshot bytes. Nothing is invented, and the upstream public
evaluation and hidden test sources are never read.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from h61.prompt_contract import PROMPT_REVISION, drag_prompt, source_body

SCHEMA = "hala.h61.native-drag-curriculum.v1"
ROWS_FILE = "h61_drag_curriculum.jsonl"
MANIFEST_FILE = "h61_drag_curriculum_manifest.json"
SOURCE_SCHEMA = "hala.h60.dragon-train-shard.v1"
EXPORT_SCHEMA = "hala.h61.source-export.v1"
PINNED_REVISION = "3451c1fc3543c18fc3361a32bfe1f12cc69b64b5"
DOMAINS = ("sheet", "slide_resize", "slider", "text_highlight")
MAX_ROWS = 500_000
MAX_ROWS_PER_IMAGE = 8
DEFAULT_TARGET_ROWS = 10_000
FAMILY_PREFIX = "hcompany_dragon_h61_"
RESERVED_DEV_FAMILY = "original_disjoint_drag_holdout_reserved"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(obj) -> bytes:
    return (
        json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf8")


def strict_json(text: str):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("Duplicate JSON key")
            value[key] = item
        return value

    def constant(name):
        raise ValueError("Non finite JSON constant")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def capability_of(row: dict) -> str:
    """The capability a row teaches.

    A capability is the original domain plus the original direction
    sensitivity, and the original source subtype when the source recorded one.
    Nothing here is invented: every part comes from the original record.
    """
    domain = row.get("domain")
    if domain not in DOMAINS:
        raise ValueError("Only the original DragOn TRAIN domains are allowed")
    ordered = row.get("ordered")
    if type(ordered) is not bool:
        raise ValueError("The original ordered flag must be boolean")
    subtype = row.get("subtype")
    if subtype is not None and (not isinstance(subtype, str) or not subtype.strip()):
        raise ValueError("The original source subtype is implausible")
    key = f"{domain}/{'ordered' if ordered else 'unordered'}"
    if subtype:
        key = f"{key}/{subtype.strip()}"
    return key


def shard_roots(root: Path) -> list[Path]:
    """Every verified shard or export under `root`.

    A shard is an extracted H60 shard directory or a shard ZIP; an export is
    an H61 source export directory, which carries the same labels and a
    verified range receipt instead of a shard manifest.
    """
    root = Path(root)
    candidates: list[Path] = []
    if (root / "labels.jsonl").exists():
        candidates.append(root)
    if root.is_dir():
        for candidate in sorted(root.iterdir()):
            if candidate.is_dir() and (candidate / "labels.jsonl").exists():
                candidates.append(candidate)
            elif candidate.is_file() and candidate.suffix.lower() in (".zip", ".bin"):
                candidates.append(candidate)
    if not candidates:
        raise ValueError("No verified shard or export with labels was found")
    return candidates


def read_member(shard: Path, name: str) -> bytes:
    """Read one shard member from an extracted shard directory or a shard ZIP."""
    shard = Path(shard)
    if shard.is_dir():
        return (shard / name).read_bytes()
    import zipfile

    with zipfile.ZipFile(shard) as bundle:
        return bundle.read(name)


def has_member(shard: Path, name: str) -> bool:
    if Path(shard).is_dir():
        return (Path(shard) / name).exists()
    import zipfile

    with zipfile.ZipFile(shard) as bundle:
        return name in bundle.namelist()


def read_shard(shard: Path) -> tuple[list[dict], dict]:
    """Read one shard or export and refuse anything that is not original TRAIN."""
    shard = Path(shard)
    if has_member(shard, "manifest.json"):
        manifest = strict_json(read_member(shard, "manifest.json").decode("utf8"))
        if (
            manifest.get("schema") != SOURCE_SCHEMA
            or manifest.get("split") != "train"
            or manifest.get("official_eval_or_test_used") is not False
            or manifest.get("benchmark_answer_leakage") is not False
            or manifest.get("source_commit_sha") != PINNED_REVISION
            or manifest.get("source_domain") not in DOMAINS
        ):
            raise ValueError("Shard is not a verified original TRAIN-only DragOn shard")
        normalised = dict(manifest)
        normalised["manifest_schema"] = SOURCE_SCHEMA
    else:
        receipt = strict_json(read_member(shard, "receipt.json").decode("utf8"))
        if (
            receipt.get("schema") != EXPORT_SCHEMA
            or receipt.get("split") != "train"
            or receipt.get("official_eval_or_test_used") is not False
            or receipt.get("benchmark_answer_leakage") is not False
            or receipt.get("source_commit_sha") != PINNED_REVISION
            or receipt.get("source_domain") not in DOMAINS
        ):
            raise ValueError("Export is not a verified original TRAIN-only DragOn export")
        normalised = {
            "shard_id": f"h61-export-{receipt['source_domain']}",
            "source_domain": receipt["source_domain"],
            "split": "train",
            "source_commit_sha": receipt["source_commit_sha"],
            "labels_sha256": receipt.get("labels_sha256"),
            "original_images": receipt.get("original_images"),
            "native_drag_actions": receipt.get("native_drag_actions"),
            "upstream_full_tar_lfs_sha256": receipt.get("upstream_lfs_sha256"),
            "whole_tar_sha256_verified": receipt.get("source_tar_sha256_matches_official") is True,
            "manifest_schema": EXPORT_SCHEMA,
        }
    rows = []
    for line in read_member(shard, "labels.jsonl").decode("utf8").splitlines():
        if line.strip():
            rows.append(strict_json(line))
    return rows, normalised


def select(rows: list[dict], target_rows: int, max_rows_per_image: int) -> tuple[list[dict], dict]:
    """Capability-balanced, screenshot-deduplicated selection.

    Rows are grouped by capability, ordered deterministically inside a
    capability, and then drawn round robin so that every capability
    contributes until the target is reached. One screenshot may contribute at
    most `max_rows_per_image` rows, so a dense screenshot cannot saturate the
    mixture.
    """
    if type(target_rows) is not int or not 1 <= target_rows <= MAX_ROWS:
        raise ValueError("Bounded curriculum target required")
    if (type(max_rows_per_image) is not int or
            not 1 <= max_rows_per_image <= MAX_ROWS_PER_IMAGE):
        raise ValueError("Bounded per-screenshot row cap required")
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(capability_of(row), []).append(row)
    for capability in grouped:
        grouped[capability].sort(key=lambda item: (item["image_id"], item["source_task_index"]))
    per_image: dict[str, int] = {}
    taken: dict[str, list[dict]] = {}
    for capability in grouped:
        kept: list[dict] = []
        for row in grouped[capability]:
            key = row["image_id"]
            if per_image.get(key, 0) >= max_rows_per_image:
                continue
            per_image[key] = per_image.get(key, 0) + 1
            kept.append(row)
        taken[capability] = kept
    ordered_capabilities = sorted(taken)
    selected: list[dict] = []
    position = 0
    while len(selected) < target_rows:
        progressed = False
        for capability in ordered_capabilities:
            bucket = taken[capability]
            if position < len(bucket):
                selected.append(bucket[position])
                progressed = True
                if len(selected) == target_rows:
                    break
        if not progressed:
            break
        position += 1
    selected.sort(key=lambda item: (item["domain"], item["image_id"], item["source_task_index"]))
    return selected, {
        "available_capabilities": {key: len(value) for key, value in sorted(taken.items())},
        "available_rows": sum(len(value) for value in taken.values()),
        "selected_rows": len(selected),
        "unique_screenshots": len({row["image_id"] for row in selected}),
    }


def build(root, *, target_rows: int = DEFAULT_TARGET_ROWS,
          max_rows_per_image: int = 2, domain_receipts: dict | None = None) -> tuple[bytes, dict, list[dict]]:
    """Build the curriculum rows, the manifest and the decoded rows."""
    root = Path(root)
    shards = shard_roots(root)
    all_rows: list[dict] = []
    shard_receipts = []
    for shard in shards:
        rows, manifest = read_shard(shard)
        all_rows.extend(rows)
        shard_receipts.append(
            {
                "shard_id": manifest.get("shard_id"),
                "source_domain": manifest.get("source_domain"),
                "labels_sha256": manifest.get("labels_sha256"),
                "original_images": manifest.get("original_images"),
                "native_drag_actions": manifest.get("native_drag_actions"),
                "upstream_full_tar_lfs_sha256": manifest.get("upstream_full_tar_lfs_sha256"),
                "whole_tar_sha256_verified_at_shard_close": manifest.get("whole_tar_sha256_verified"),
            }
        )
    if not all_rows:
        raise ValueError("No original TRAIN rows were found")
    selected, balance = select(all_rows, target_rows, max_rows_per_image)
    if not selected:
        raise ValueError("The curriculum selection is empty")
    domain_receipts = domain_receipts or {}
    missing_receipt = sorted(
        {
            row["domain"]
            for row in selected
            if row["domain"] not in domain_receipts
        }
    )
    out_rows: list[dict] = []
    families: set[str] = set()
    capability_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    ordered_true = 0
    ordered_false = 0
    language_counts: dict[str, int] = {}
    screenshots: set[str] = set()
    for row in selected:
        if row.get("split") != "train" or row.get("action_type") != "drag":
            raise ValueError("Only original TRAIN drag actions are allowed")
        width = row["image_width"]
        height = row["image_height"]
        body = source_body(row["intent"], row["ordered"])
        prompt = drag_prompt(body, width, height, row["ordered"])
        completion = {
            "type": "drag",
            "start_point": row["completion"]["start_point"],
            "end_point": row["completion"]["end_point"],
            "coordinate_space": "source_image_pixels",
            "ordered": row["ordered"],
        }
        capability = capability_of(row)
        capability_counts[capability] = capability_counts.get(capability, 0) + 1
        domain_counts[row["domain"]] = domain_counts.get(row["domain"], 0) + 1
        language = row.get("language", "en")
        language_counts[language] = language_counts.get(language, 0) + 1
        if row["ordered"]:
            ordered_true += 1
        else:
            ordered_false += 1
        screenshots.add(row["image_id"])
        family = FAMILY_PREFIX + row["domain"] + "_train"
        families.add(family)
        out_rows.append(
            {
                "id": row["id"],
                "split": "train",
                "family": family,
                "language": language,
                "capability": capability,
                "prompt": prompt,
                "completion": json.dumps(
                    completion, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
                "image_path": row["image_path"],
                "image_sha256": row["image_sha256"],
            }
        )
    blob = b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                   allow_nan=False).encode("utf8") + b"\n"
        for row in out_rows
    )
    manifest = {
        "schema": "hala.training.manifest.v1",
        "source_schema": SCHEMA,
        "split": "train",
        "rights_cleared": True,
        "source_kind": "rights_cleared_public",
        "source_dataset": "Hcompany/DragOn",
        "source_revision": PINNED_REVISION,
        "source_split": "train/{sheet,slide_resize,slider,text_highlight}.tar",
        "source_license": "Apache-2.0",
        "dataset_sha256": sha(blob),
        "token_budget": 4096,
        "families": sorted(families),
        "held_out_families": [RESERVED_DEV_FAMILY],
        "independent_mixed_drag_dev_available": False,
        "media": sorted(
            [
                {
                    "path": row["image_path"],
                    "family": row["family"],
                    "sha256": row["image_sha256"],
                }
                for row in out_rows
            ],
            key=lambda item: (item["path"], item["sha256"]),
        ),
        "unique_source_images": len(screenshots),
        "train_rows": len(out_rows),
        "ordered_true": ordered_true,
        "ordered_false": ordered_false,
        "language_counts": dict(sorted(language_counts.items())),
        "capability_counts": dict(sorted(capability_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "max_rows_per_image": max_rows_per_image,
        "target_rows": target_rows,
        "prompt_revision": PROMPT_REVISION,
        "verified_source_shards": shard_receipts,
        "domain_verification_receipts": {
            domain: receipt for domain, receipt in sorted(domain_receipts.items())
        },
        "domains_without_whole_archive_verification": missing_receipt,
        "no_benchmark_eval_source": True,
        "training_executed": False,
        "model_improvement_qualified": False,
    }
    return blob, manifest, out_rows


def load_export(examples_path, manifest):
    """Re-derive and byte-compare the curriculum export before any weight is touched."""
    examples_path = Path(examples_path)
    if examples_path.name != ROWS_FILE or examples_path.is_symlink() or not isinstance(manifest, dict):
        raise ValueError("H61 source-bound native drag curriculum only")
    if manifest.get("schema") != "hala.training.manifest.v1":
        raise ValueError("H61 curriculum manifest schema required")
    blob, expected, rows = build(
        examples_path.parent,
        target_rows=manifest["target_rows"],
        max_rows_per_image=manifest["max_rows_per_image"],
        domain_receipts=manifest.get("domain_verification_receipts") or {},
    )
    if manifest != expected or examples_path.read_bytes() != blob:
        raise ValueError("H61 curriculum labels/media/derived SFT were modified")
    return rows


def export(root, *, target_rows: int = DEFAULT_TARGET_ROWS, max_rows_per_image: int = 2,
           domain_receipts: dict | None = None) -> dict:
    """Create-only export of the H61 curriculum rows and manifest."""
    root = Path(root)
    rows_path = root / ROWS_FILE
    manifest_path = root / MANIFEST_FILE
    if rows_path.exists() or manifest_path.exists():
        raise FileExistsError("The H61 curriculum export must be create-only")
    blob, manifest, rows = build(
        root, target_rows=target_rows, max_rows_per_image=max_rows_per_image,
        domain_receipts=domain_receipts,
    )
    with rows_path.open("xb") as handle:
        handle.write(blob)
    try:
        with manifest_path.open("xb") as handle:
            handle.write(canonical(manifest))
    except BaseException:
        rows_path.unlink()
        raise
    return {
        "schema": SCHEMA,
        "curriculum_rows": len(rows),
        "unique_source_images": manifest["unique_source_images"],
        "capability_counts": manifest["capability_counts"],
        "domain_counts": manifest["domain_counts"],
        "ordered_true": manifest["ordered_true"],
        "ordered_false": manifest["ordered_false"],
        "language_counts": manifest["language_counts"],
        "derived_train_sha256": manifest["dataset_sha256"],
        "model_training_executed": False,
        "independent_model_evaluation_completed": False,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["export", "audit"])
    parser.add_argument("--source", required=True)
    parser.add_argument("--target-rows", type=int, default=DEFAULT_TARGET_ROWS)
    parser.add_argument("--max-rows-per-image", type=int, default=2)
    args = parser.parse_args()
    if args.command == "export":
        print(json.dumps(export(args.source, target_rows=args.target_rows,
                                max_rows_per_image=args.max_rows_per_image), sort_keys=True))
    else:
        blob, manifest, rows = build(args.source, target_rows=args.target_rows,
                                     max_rows_per_image=args.max_rows_per_image)
        print(json.dumps({"rows": len(rows), "images": manifest["unique_source_images"],
                          "capabilities": manifest["capability_counts"],
                          "dataset_sha256": manifest["dataset_sha256"]}, sort_keys=True))
