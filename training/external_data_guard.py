"""Metadata-only competitor dataset admission gate for *future* lawful TRAIN batches.

No network, no downloads, no trainer integration, no model loading, and no
automatic rights clearance. Benchmark evaluation traces fail closed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

REGISTRY = Path(__file__).resolve().parent.parent / "research" / "competitor_data" / "2026-10-21" / "registry.json"
SHA40 = re.compile(r"[0-9a-f]{40}\Z")
SHA64 = re.compile(r"[0-9a-f]{64}\Z")
ATTESTATIONS = frozenset({
    "dataset_revision_verified",
    "source_license_verified",
    "upstream_media_rights_verified",
    "personal_data_screened",
    "credentials_and_secrets_removed",
    "app_episode_train_dev_test_disjoint",
    "benchmark_overlap_screened",
    "source_images_sha256_verified",
})
# Reserved public benchmark IDs are not silently admitted as train rows.
DENIED_ID_MARKERS = (
    "osworld", "androidworld", "automationbench", "pinchbench",
    "agents-last-exam", "screenspot", "agentnetbench",
    "dragon-eval", "guiodyssey-test",
)


def registry():
    data = json.loads(REGISTRY.read_text("utf-8"))
    if (data.get("schema") != "hala.competitor-data-source-registry.v1" or
            data.get("downloaded_raw_data") is not False or
            data.get("ran_training") is not False or
            data.get("benchmarks_test_reserved") is not True):
        raise ValueError("Research-only registry integrity requirements")
    rows = data["sources"]
    if type(rows) is not list or len(rows) < 6 or len({x["id"] for x in rows}) != len(rows):
        raise ValueError("Duplicate/incomplete source registry")
    return {row["id"]: row for row in rows}


def decision(source_id):
    source = registry().get(source_id)
    if not source:
        raise ValueError("Unknown external dataset identity")
    role = source["allowed_use"]
    if source["release_use_status"] != "CANDIDATE_NOT_IMPORTED" or role != "candidate_train_after_audit":
        return "DENY"
    if source["declared_license"] not in ("mit", "apache-2.0", "cc-by-4.0"):
        return "DENY"
    if not source["training_partition"] or source["third_party_rights_review"] is not True:
        return "DENY"
    return "CANDIDATE_ONLY"


def admit_metadata(source_id, revision, split, blob_sha256, attestations, items):
    """Metadata-only check. Returns evidence, never raw content or a training object.

    NOTE: attestations remain external operator CLAIMS; this function cannot
    establish actual copyright, consent, privacy, provenance or legal validity.
    """
    if decision(source_id) != "CANDIDATE_ONLY":
        raise ValueError("Research-only, benchmark or restricted external data rejected")
    source = registry()[source_id]
    if (not isinstance(revision, str) or not SHA40.fullmatch(revision) or
            not isinstance(blob_sha256, str) or not SHA64.fullmatch(blob_sha256)):
        raise ValueError("Immutable upstream dataset commit and original blob SHA256 required")
    if split not in source["training_partition"]:
        raise ValueError("Only explicitly verified official TRAIN partition admitted")
    if type(attestations) is not dict or set(attestations) != ATTESTATIONS:
        raise ValueError("Complete source/privacy/benchmark review receipt required")
    if any(v is not True for v in attestations.values()):
        raise ValueError("Unqualified external dataset batch review")
    if (type(items) is not list or not 1 <= len(items) <= 256):
        raise ValueError("Bounded candidate batch required")
    ids, pixels = set(), set()
    for item in items:
        if (type(item) is not dict or
                set(item) != {"id", "app", "episode", "image_sha256",
                              "benchmark_family", "partition", "source_revision"}):
            raise ValueError("Exact minimal metadata only; raw pictures not admitted")
        if (not all(isinstance(item[k], str) and 1 <= len(item[k]) <= 256
                    for k in ("id", "app", "episode")) or
                not SHA64.fullmatch(str(item["image_sha256"])) or
                item["benchmark_family"] is not None or
                item["partition"] != split or item["source_revision"] != revision):
            raise ValueError("Benchmark or unverifiable metadata/partition rejected")
        identity = (item["app"], item["episode"], item["id"])
        if any(marker in " ".join(identity).casefold() for marker in DENIED_ID_MARKERS):
            raise ValueError("Public benchmark identifier cannot be training data")
        if identity in ids or item["image_sha256"] in pixels:
            raise ValueError("Duplicate episode step or screenshot in batch")
        ids.add(identity)
        pixels.add(item["image_sha256"])
    # Declared downstream attribution must persist even if user renames Hala.
    return {
        "schema": "hala.external-metadata-candidate.v1",
        "training_data_written": False,
        "trained_model": False,
        "rights_verified_by_tool": False,
        "attestations_operator_claimed_only": True,
        "candidate_not_yet_imported": True,
        "source_id": source_id,
        "source_repo": source["repo"],
        "source_revision": revision,
        "split": split,
        "blob_sha256": blob_sha256,
        "declared_license": source["declared_license"],
        "retain_attribution": source["declared_license"] == "cc-by-4.0" or
                              source["declared_license"] == "mit" or
                              source["declared_license"] == "apache-2.0",
        "reviewed_metadata_rows": len(ids),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="Summarize decisions; never download")
    arguments = parser.parse_args()
    if not arguments.list:
        parser.error("This phase supports metadata-only --list; no import is implemented")
    for key in sorted(registry()):
        entry = registry()[key]
        print(key, decision(key), entry["declared_license"], entry["repository_role"])


if __name__ == "__main__":
    main()
