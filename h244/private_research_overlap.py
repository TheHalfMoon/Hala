"""H244: owner-only cross-cohort research PNG/RGBA leakage audit.

Compares H242 real browser screenshots against (a) pre-H173 manifest-only
known old SHA exclusions, (b) the 20 quarantined H188-H197 original source
metadata rows, and (c) H236 original Dragula research QA if supplied.
It may read ONLY NEW H242 owner-private images and allowlisted source-only
metadata; NEVER historical/consumed/OOD images, old gold, or weights.
No dataset admission, no model calls, no optimizer, no uploaded receipts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path

from h174.split_firewall import canonical_repository_family
from h178.owner_git_inventory import _safe_directory, inventory_local_original_sources
from h185.manifest_sha_only import scan_legacy_manifest_sha_fields
from h194.owner_private_snap_capture import _safe_lexical_path
from h198.owner_source_metadata_readiness import (
    COHORTS, _regular_small, parse_source_manifest, census,
)
from h231.native_replay import _rgba_sha
from h236.original_dragula_native import REPO, COMMIT, PINNED
from h241.expanded_pre173_metadata import find_pre_h173_phases
from h242.private_dragula_prospects import CASES, CASE_COUNT

RECEIPT = "PRIVATE_RESEARCH_PROSPECTS_NOT_ADMITTED.json"
MAX_RECEIPT_BYTES = 256 * 1024
MAX_PNG_BYTES = 5 * 1024 * 1024
HEX = frozenset("0123456789abcdef")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError("H244 owner research overlap gate: " + message)


def sha_ok(value: object) -> bool:
    return type(value) is str and len(value) == 64 and set(value) <= HEX


def private_dir(path: Path) -> None:
    require(isinstance(path, Path) and path.is_absolute()
            and _safe_directory(path), "unsafe owner-only directory")
    _safe_lexical_path(path)
    require(not any(os.path.lexists(p / ".git") for p in (path, *path.parents)),
            "owner research directory cannot be inside a Git checkout")


def private_json(path: Path) -> dict:
    require(os.path.lexists(path), "required private source-only receipt missing")
    _regular_small(path, "bounded private research receipt")
    require(path.stat().st_size <= MAX_RECEIPT_BYTES,
            "receipt size limit")
    return parse_source_manifest(path.read_text(encoding="utf-8"))


def current_20_metadata(owner_root: Path) -> tuple[set[str], set[str]]:
    # H198 strict schema checks all 20 prior case metadata and explicitly
    # refuses anyone promoting them; NEVER opens private gold/image content.
    checked = census(owner_root)
    require(checked["total_quarantined_original_preaction_png_metadata"] == 20
            and checked["training_approved"] is False
            and checked["dev_approved"] is False
            and checked["ood_approved"] is False
            and checked["data_gold_or_image_files_opened"] == 0,
            "current original research cohort has unauthorized admission")
    png: set[str] = set()
    rgba: set[str] = set()
    for folder, count, _family, _status in COHORTS:
        source = private_json(owner_root / folder / "source-prospects-not-admitted.json")
        rows = source["cases"]
        require(type(rows) is list and len(rows) == count,
                "prior quarantine manifest changed")
        for row in rows:
            require(type(row) is dict
                    and sha_ok(row.get("image_sha256"))
                    and sha_ok(row.get("visual_rgba_sha256")),
                    "prior research candidate has malformed hashes")
            png.add(row["image_sha256"])
            rgba.add(row["visual_rgba_sha256"])
    require(len(png) == len(rgba) == 20, "prior source cohort duplicates")
    return png, rgba


def historical_h236_research(folder: Path | None) -> tuple[set[str], set[str]]:
    if folder is None:
        return set(), set()
    private_dir(folder)
    source = private_json(folder / "receipt.json")
    require(source.get("status") == "H236_ORIGINAL_DRAGULA_NATIVE_QA_NOT_ADMISSION"
            and source.get("source_repository") == REPO
            and source.get("source_git_sha") == COMMIT
            and source.get("source_sha256") == PINNED
            and source.get("official_train_dev_admitted") is False
            and source.get("new_ood_approved") is False
            and source.get("model_calls") == 0
            and source.get("optimizer_updates") == 0,
            "H236 prior research receipt not immutable/research only")
    contexts = source.get("contexts")
    require(type(contexts) is list and len(contexts) == 2,
            "expected two old source-native research contexts")
    png: set[str] = set()
    rgba: set[str] = set()
    for item in contexts:
        require(item.get("original_native_dom_drop_verified") is True
                and item.get("network_attempts") == 0,
                "prior H236 context was not genuine offline QA")
        for stage in ("before", "after"):
            p, v = item.get(stage+"_png_sha256"), item.get(stage+"_rgba_sha256")
            require(sha_ok(p) and sha_ok(v), "invalid old research source hashes")
            # H236 used one unchanged authored UI fixture for two language
            # labels; duplicate images are real historical reuse, not two
            # independent visual contexts. Deduplicate as EXCLUSIONS only.
            png.add(p)
            rgba.add(v)
    return png, rgba


def verify_h242_receipt(
    data: dict, known_png: set[str], known_rgba: set[str],
) -> dict:
    """Exact fail-closed metadata overlap gate; never returns an allowlist."""
    require(type(data) is dict and
            data.get("status") == "H242_OWNER_PRIVATE_ORIGINAL_DRAGULA_PROSPECTS_NOT_ADMITTED"
            and data.get("source_repository") == REPO
            and data.get("source_git_sha") == COMMIT
            and data.get("source_asset_sha256") == PINNED
            and data.get("original_js_runtime_unmodified") is True
            and data.get("rendered_html_self_authored") is True,
            "unexpected source identity or unaudited original JS")
    for key in ("official_train_dev_dataset_admitted","train_eligible",
                "dev_eligible","ood_eligible","train_optimizer_approved",
                "full_historical_source_and_visual_history_proven",
                "transitive_license_and_media_rights_audited"):
        require(data.get(key) is False, "research status unexpectedly promoted")
    require(data.get("training_model_calls") == 0
            and data.get("optimizer_steps") == 0,
            "model calls/gradients not permitted")
    rows = data.get("cases")
    require(type(rows) is list and len(rows) == CASE_COUNT,
            "expected eight unadmitted private source-native tasks")
    p_seen: set[str] = set()
    v_seen: set[str] = set()
    langs: Counter[str] = Counter()
    for i, case in enumerate(rows):
        spec = CASES[i]
        require(type(case) is dict
                and case.get("case_id") == spec["case_id"]
                and case.get("language") == spec["lang"]
                and case.get("instruction") == spec["task"]
                and case.get("native_drag_verified") is True
                and case.get("network_requests") == 0
                and case.get("training_eligible") is False
                and case.get("dev_eligible") is False
                and case.get("ood_eligible") is False,
                "bilingual original source research case mutated or promoted")
        for position in ("source_point", "destination_point"):
            pt = case.get(position)
            require(type(pt) is list and len(pt) == 2
                    and all(type(x) is int for x in pt)
                    and 0 <= pt[0] < 960 and 0 <= pt[1] < 720,
                    "invalid native screenshot coordinate")
        langs[case["language"]] += 1
        for stage in ("preaction", "postaction"):
            p, v = case.get(stage+"_png_sha256"), case.get(stage+"_rgba_sha256")
            require(sha_ok(p) and sha_ok(v), "missing private scene digest")
            require(p not in known_png and v not in known_rgba,
                    "private source case duplicates known old/current research image")
            require(p not in p_seen and v not in v_seen,
                    "private source case duplicates another new frame")
            p_seen.add(p)
            v_seen.add(v)
    require(langs == {"ar":4,"en":4} and len(p_seen) == len(v_seen) == 16,
            "bilingual eight-scene integrity incomplete")
    return {
        "source_native_private_cases_checked": CASE_COUNT,
        "new_private_png_frames_compared": len(p_seen),
        "new_private_decoded_rgba_frames_compared": len(v_seen),
        "unique_en_source_cases": langs["en"],
        "unique_ar_source_cases": langs["ar"],
        "known_prior_png_overlaps": 0,
        "known_prior_rgba_overlaps": 0,
        "training_approved": False,
        "dev_approved": False,
        "ood_approved": False,
        "historical_completeness_proven": False,
        "model_calls": 0,
        "optimizer_steps": 0,
    }


def validate_new_private_media(folder: Path, data: dict) -> None:
    """Read H242 NEW images only; never read H236/old cohort pictures."""
    for case in data["cases"]:
        for stage, key in (("before", "preaction"),("after", "postaction")):
            path = folder / (case["case_id"] + "-" + stage + ".png")
            require(os.path.lexists(path), "new private frame missing")
            _regular_small(path, "new private PNG")
            require(path.stat().st_size < MAX_PNG_BYTES, "new private frame too large")
            frame = path.read_bytes()
            require(sha256(frame).hexdigest() == case[key+"_png_sha256"]
                    and _rgba_sha(frame) == case[key+"_rgba_sha256"],
                    "new H242 image bytes changed since native capture")


def audit_owner_research_overlap(
    owner_root: Path, h242_dir: Path, historical_parents: list[Path],
    h236_dir: Path | None = None,
) -> dict:
    private_dir(h242_dir)
    require(isinstance(owner_root, Path) and owner_root.is_absolute()
            and _safe_directory(owner_root), "owner root invalid")
    require(not os.path.lexists(h242_dir / "INCOMPLETE_NEVER_ADMIT"),
            "failed private research capture cannot be screened")
    phases = find_pre_h173_phases(historical_parents)
    old = scan_legacy_manifest_sha_fields(phases)
    require(old["complete_historical_manifest_coverage_proven"] is False
            and old["old_image_media_opened"] is False
            and old["training_dataset_admitted"] is False,
            "partial older hash history cannot authorize admission")
    current_png,current_rgba = current_20_metadata(owner_root)
    h236_png,h236_rgba = historical_h236_research(h236_dir)
    known_png = set(old["_owner_only_png_sha_candidates"]) | current_png | h236_png
    known_rgba = set(old["_owner_only_rgba_sha_candidates"]) | current_rgba | h236_rgba
    receipt = private_json(h242_dir / RECEIPT)
    result = verify_h242_receipt(receipt, known_png, known_rgba)
    validate_new_private_media(h242_dir, receipt)
    # Fail closed if any evidence unexpectedly vanished from the receipt.
    require(result["new_private_png_frames_compared"] == 16
            and result["new_private_decoded_rgba_frames_compared"] == 16,
            "not all private frames accounted for")
    return {
        "status": "H244_OWNER_PRIVATE_CROSS_COHORT_INTEGRITY_NOT_ADMITTED",
        **result,
        "pre_h173_known_png_exclusions": len(old["_owner_only_png_sha_candidates"]),
        "pre_h173_known_rgba_exclusions": len(old["_owner_only_rgba_sha_candidates"]),
        "other_quarantined_case_sources": len(current_png),
        "earlier_h236_native_research_frames": len(h236_png),
        "new_private_media_bytes_verified": 16,
        "old_private_media_or_gold_opened": False,
        "h173_consumed_ood_opened": False,
    }


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--owner-root", type=Path, required=True)
    a.add_argument("--h242-private", type=Path, required=True)
    a.add_argument("--h236-private", type=Path)
    a.add_argument("--historical-parent", type=Path, action="append", required=True)
    ns=a.parse_args()
    try:
        result=audit_owner_research_overlap(
            ns.owner_root, ns.h242_private, ns.historical_parent, ns.h236_private)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError):
        a.exit(2, "H244_OWNER_RESEARCH_OVERLAP_NOT_QUALIFIED\n")
    # Aggregate counts only. NEVER print raw hashes, source path, instructions.
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
