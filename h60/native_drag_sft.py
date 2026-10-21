"""H60 native drag SFT bridge for the verified full-corpus DragOn TRAIN shards.

This bridge turns one or more verified H60 private shards into the exact
Qwen SFT export that `training/scripts/train_adapter.py` consumes. It only
re-labels what the original upstream source already contains: the original
instruction, the original source-image pixel boxes and the original `ordered`
flag. It never invents a label and never reads a public evaluation or hidden
test source.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

SCHEMA = "hala.h60.native-millions-drag-sft.v1"
ROWS_FILE = "h60_drag_sft.jsonl"
MANIFEST_FILE = "h60_drag_training_manifest.json"
LABELS_FILE = "labels.jsonl"
SHARD_MANIFEST = "manifest.json"
PROOF_FILE = "domain-proof.json"
FAMILY_PREFIX = "hcompany_dragon_h60_"
RESERVED_DEV_FAMILY = "original_disjoint_drag_holdout_reserved"
DOMAINS = ("sheet", "slide_resize", "slider", "text_highlight")
MAX_ROWS = 200_000
MAX_RECORDS = 200_000
MAX_DATASET_BYTES = 1024 * 1024 * 1024
TOKEN_BUDGET = 4096
MAX_LABELS_PER_IMAGE = 2
MAX_IMAGES_PER_DOMAIN = 60_000


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(obj) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf8")


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


def _shard_roots(root: Path) -> list[Path]:
    root = Path(root)
    if (root / LABELS_FILE).exists() and (root / SHARD_MANIFEST).exists():
        return [root]
    found = []
    for candidate in sorted(root.glob("shard-*")):
        if (candidate / LABELS_FILE).exists() and (candidate / SHARD_MANIFEST).exists():
            found.append(candidate)
    if not found:
        raise ValueError("No verified H60 shard with labels and manifest was found")
    return found


def load_domain_proof(root: Path) -> tuple[dict | None, str | None]:
    """Load the domain proof that binds the shards to one verified archive."""
    for candidate in (Path(root) / PROOF_FILE, Path(root).parent / PROOF_FILE):
        if candidate.is_file():
            proof = strict_json(candidate.read_text("utf8"))
            if proof.get("schema") != "hala.h60.dragon-train-domain-proof.v1":
                raise ValueError("Unknown H60 domain proof schema")
            return proof, hashlib.sha256(candidate.read_bytes()).hexdigest()
    return None, None


def shard_bound_to_verified_archive(manifest: dict, proof: dict | None) -> bool:
    """True when this shard is bound to the one SHA-verified original archive.

    A shard that carries the whole-archive digest itself is self-proving. Any
    other shard is accepted only when the domain proof binds it to the very
    same archive whose recomputed digest equals the official upstream LFS
    digest, and when the proof also carries the provider readback result for
    that shard.
    """
    if manifest.get("whole_tar_sha256_verified") is True:
        return True
    if not proof:
        return False
    if proof.get("archive_sha256_matches_official") is not True:
        return False
    if proof.get("recomputed_archive_sha256") != proof.get("official_upstream_lfs_sha256"):
        return False
    if proof.get("source_commit_sha") != manifest.get("source_commit_sha"):
        return False
    if proof.get("official_upstream_lfs_sha256") != manifest.get("upstream_full_tar_lfs_sha256"):
        return False
    window_ids = {
        row.get("original_window_id") for row in (proof.get("source_window_receipts") or [])
    }
    if not window_ids or not set(manifest.get("source_window_ids") or []) <= window_ids:
        return False
    entry = next(
        (
            row
            for row in (proof.get("shards") or [])
            if row.get("shard_id") == manifest.get("shard_id")
        ),
        None,
    )
    if entry is None:
        return False
    if entry.get("provider_readback_verified") is not True:
        return False
    return (
        entry.get("labels_sha256") == manifest.get("labels_sha256")
        and entry.get("original_images") == manifest.get("original_images")
        and entry.get("native_drag_actions") == manifest.get("native_drag_actions")
    )


def _read_shard(shard: Path, proof: dict | None = None) -> tuple[list[dict], dict]:
    manifest = strict_json((shard / SHARD_MANIFEST).read_text("utf8"))
    if (
        manifest.get("schema") != "hala.h60.dragon-train-shard.v1"
        or manifest.get("split") != "train"
        or manifest.get("official_eval_or_test_used") is not False
        or manifest.get("benchmark_answer_leakage") is not False
        or manifest.get("source_commit_sha") != "3451c1fc3543c18fc3361a32bfe1f12cc69b64b5"
    ):
        raise ValueError("H60 shard is not a verified original TRAIN-only shard")
    if not shard_bound_to_verified_archive(manifest, proof):
        raise ValueError("H60 shard was not produced from a fully SHA-verified original TAR")
    rows = []
    for line in (shard / LABELS_FILE).read_text("utf8").splitlines():
        if line.strip():
            rows.append(strict_json(line))
    return rows, manifest


def _image_index(shard: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for record in strict_json((shard / SHARD_MANIFEST).read_text("utf8")).get(
        "original_image_members", []
    ):
        index[record["image_id"]] = record
    return index


def derive(root, *, max_images_per_domain: int = MAX_IMAGES_PER_DOMAIN,
           max_labels_per_image: int = MAX_LABELS_PER_IMAGE) -> tuple[bytes, dict, list[dict]]:
    """Build the SFT blob, its manifest and the decoded rows from H60 shards."""
    root = Path(root)
    shards = _shard_roots(root)
    proof, proof_sha256 = load_domain_proof(root)
    rows: list[dict] = []
    media: list[dict] = []
    families: set[str] = set()
    domain_counts: dict[str, int] = {}
    ordered_true = 0
    ordered_false = 0
    seen_images: set[str] = set()
    per_domain_images: dict[str, int] = {}
    shard_receipts = []
    for shard in shards:
        shard_rows, shard_manifest = _read_shard(shard, proof)
        images = _image_index(shard)
        shard_receipts.append(
            {
                "shard_id": shard_manifest.get("shard_id"),
                "labels_sha256": shard_manifest.get("labels_sha256"),
                "original_images": shard_manifest.get("original_images"),
                "native_drag_actions": shard_manifest.get("native_drag_actions"),
                "whole_tar_sha256_verified": shard_manifest.get("whole_tar_sha256_verified"),
                "upstream_full_tar_lfs_sha256": shard_manifest.get("upstream_full_tar_lfs_sha256"),
            }
        )
        # Deterministic curriculum: bound the labels taken from one screenshot so
        # a single dense screenshot cannot saturate a microbatch, and bound the
        # number of screenshots per domain so no domain dominates the mixture.
        per_image: dict[str, int] = {}
        for source in shard_rows:
            domain = source.get("domain")
            if domain not in DOMAINS:
                raise ValueError("Only original H60 DragOn TRAIN domains are allowed")
            if source.get("split") != "train" or source.get("action_type") != "drag":
                raise ValueError("Only original H60 TRAIN drag actions are allowed")
            if type(source.get("ordered")) is not bool:
                raise ValueError("Original H60 ordered flag must be boolean")
            if source.get("language") != "en":
                raise ValueError("Only the original English source instruction is allowed")
            key = source.get("image_id")
            if key in per_image and per_image[key] >= max_labels_per_image:
                continue
            record = images.get(key)
            if record is None:
                raise ValueError("H60 label references an image that the shard does not hold")
            if (key, source.get("source_task_index")) in seen_images:
                continue
            per_image[key] = per_image.get(key, 0) + 1
            if key not in seen_images:
                if per_domain_images.get(domain, 0) >= max_images_per_domain:
                    continue
                per_domain_images[domain] = per_domain_images.get(domain, 0) + 1
                seen_images.add(key)
                media.append(
                    {
                        "path": record["image_path"],
                        "family": FAMILY_PREFIX + domain + "_train",
                        "sha256": record["image_sha256"],
                    }
                )
            width = source["image_width"]
            height = source["image_height"]
            completion = {
                "type": "drag",
                "start_point": source["completion"]["start_point"],
                "end_point": source["completion"]["end_point"],
                "coordinate_space": "source_image_pixels",
                "ordered": source["ordered"],
            }
            order = "direction-sensitive" if source["ordered"] else "direction-insensitive"
            prompt = (
                f"Inspect the ORIGINAL {width}x{height} GUI screenshot. "
                f"Complete the {domain} drag action: {source['intent']}\n"
                "Return ONLY one JSON object: type=drag, start_point=[x,y], "
                "end_point=[x,y], coordinate_space=source_image_pixels, "
                f"ordered={str(source['ordered']).lower()}. The task is {order}. "
                "Use integer coordinates from the ORIGINAL screenshot, not the resized "
                "vision tokens. Never click, browse or access outside services."
            )
            rows.append(
                {
                    "id": source["id"],
                    "split": "train",
                    "family": FAMILY_PREFIX + domain + "_train",
                    "language": "en",
                    "prompt": prompt,
                    "completion": json.dumps(completion, ensure_ascii=False, sort_keys=True,
                                             separators=(",", ":")),
                    "image_path": source["image_path"],
                    "image_sha256": source["image_sha256"],
                }
            )
            families.add(FAMILY_PREFIX + domain + "_train")
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            if source["ordered"]:
                ordered_true += 1
            else:
                ordered_false += 1
    if not rows:
        raise ValueError("No original H60 training rows were derived")
    if len(rows) > MAX_ROWS:
        raise ValueError("H60 curriculum exceeds the supported record budget")
    blob = b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                   allow_nan=False).encode("utf8") + b"\n"
        for row in rows
    )
    if len(blob) > MAX_DATASET_BYTES:
        raise ValueError("H60 curriculum exceeds the supported dataset size budget")
    manifest = {
        "schema": "hala.training.manifest.v1",
        "source_schema": SCHEMA,
        "split": "train",
        "rights_cleared": True,
        "source_kind": "rights_cleared_public",
        "source_dataset": "Hcompany/DragOn",
        "source_revision": "3451c1fc3543c18fc3361a32bfe1f12cc69b64b5",
        "source_split": "train/{sheet,slide_resize,slider,text_highlight}.tar",
        "source_license": "Apache-2.0",
        "dataset_sha256": sha(blob),
        "token_budget": TOKEN_BUDGET,
        "families": sorted(families),
        "held_out_families": [RESERVED_DEV_FAMILY],
        "independent_mixed_drag_dev_available": False,
        "media": sorted(media, key=lambda item: item["path"]),
        "unique_source_images": len(media),
        "train_rows": len(rows),
        "ordered_true": ordered_true,
        "ordered_false": ordered_false,
        "native_action_type": "drag",
        "coordinate_space": "source_image_pixels",
        "original_source_domain_counts": domain_counts,
        "max_labels_per_image": max_labels_per_image,
        "max_images_per_domain": max_images_per_domain,
        "verified_source_shards": shard_receipts,
        "no_benchmark_eval_source": True,
        "training_executed": False,
        "model_improvement_qualified": False,
    }
    return blob, manifest, rows


def load_export(examples_path, manifest):
    """Re-derive and byte-compare the SFT export before any weight is touched."""
    examples_path = Path(examples_path)
    if examples_path.name != ROWS_FILE or examples_path.is_symlink() or not isinstance(manifest, dict):
        raise ValueError("H60 source-bound private native drag SFT only")
    # Re-derive with the very caps the export was created under, so a bounded
    # curriculum is verified against exactly the same derivation and not
    # against a different one.
    blob, expected, rows = derive(
        examples_path.parent,
        max_images_per_domain=manifest.get("max_images_per_domain", MAX_IMAGES_PER_DOMAIN),
        max_labels_per_image=manifest.get("max_labels_per_image", MAX_LABELS_PER_IMAGE),
    )
    if manifest != expected or examples_path.read_bytes() != blob:
        raise ValueError("H60 source train labels/media/derived SFT were modified")
    return rows


def export(root, *, max_images_per_domain: int = MAX_IMAGES_PER_DOMAIN,
           max_labels_per_image: int = MAX_LABELS_PER_IMAGE) -> dict:
    """Create-only export of the H60 native drag SFT rows and manifest."""
    root = Path(root)
    rows_path = root / ROWS_FILE
    manifest_path = root / MANIFEST_FILE
    if rows_path.exists() or manifest_path.exists():
        raise FileExistsError("H60 model candidate must be create-only")
    blob, manifest, rows = derive(root, max_images_per_domain=max_images_per_domain,
                                   max_labels_per_image=max_labels_per_image)
    with rows_path.open("xb") as handle:
        handle.write(blob)
    try:
        with manifest_path.open("xb") as handle:
            handle.write(canonical(manifest))
    except BaseException:
        rows_path.unlink()
        raise
    if load_export(rows_path, manifest) != rows:
        raise ValueError("H60 create-only original source-to-Qwen conversion failed")
    return {
        "schema": SCHEMA,
        "verified_original_screens": manifest["unique_source_images"],
        "genuine_native_drag_rows": len(rows),
        "ordered_true": manifest["ordered_true"],
        "ordered_false": manifest["ordered_false"],
        "derived_train_sha256": manifest["dataset_sha256"],
        "model_training_executed": False,
        "independent_model_evaluation_completed": False,
    }


def validate_drag_baseline(baseline) -> None:
    """Honest receipt gate for the H60 curriculum.

    The accepted held-out source is the first-party PUBLIC DEV drag set. It is
    genuinely source-disjoint from DragOn TRAIN, but it is NOT a blind test,
    so the receipt says so explicitly and no blind-test claim is implied.
    """
    if (
        not isinstance(baseline, dict)
        or baseline.get("evidence") != "real_inference"
        or baseline.get("task_kind") != "mixed_ordered_unordered_drag"
        or baseline.get("held_out_source")
        != "original_firstparty_public_dev_h55_h58_source_disjoint_not_training"
        or type(baseline.get("evaluated_tasks")) is not int
        or baseline["evaluated_tasks"] < 24
        or not isinstance(baseline.get("held_out_dataset_sha256"), str)
        or len(baseline["held_out_dataset_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in baseline["held_out_dataset_sha256"])
        or baseline.get("independent_blind_test_completed") is not False
    ):
        raise ValueError("H60 curriculum requires a real public-dev source-disjoint baseline receipt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["export", "audit"])
    parser.add_argument("--source", required=True)
    parser.add_argument("--max-images-per-domain", type=int, default=MAX_IMAGES_PER_DOMAIN)
    parser.add_argument("--max-labels-per-image", type=int, default=MAX_LABELS_PER_IMAGE)
    args = parser.parse_args()
    if args.command == "export":
        print(json.dumps(export(args.source, max_images_per_domain=args.max_images_per_domain,
                                max_labels_per_image=args.max_labels_per_image), sort_keys=True))
    else:
        blob, manifest, rows = derive(args.source, max_images_per_domain=args.max_images_per_domain,
                                      max_labels_per_image=args.max_labels_per_image)
        print(json.dumps({"rows": len(rows), "dataset_sha256": manifest["dataset_sha256"],
                          "images": manifest["unique_source_images"],
                          "ordered_true": manifest["ordered_true"],
                          "ordered_false": manifest["ordered_false"]}, sort_keys=True))
