"""H198 owner-only 20-case quarantine and training-readiness census.

No screenshot pixel, old benchmark, reference action, label or model bytes
may be opened. The only owner files read are bounded new cohort *source*
manifest JSONs and optional old SHA-only historical manifests via H185.

Copy-first H194 safe owner paths, H185 historical SHA-only exclusion and
H182 frozen source-role preregistration. This is NOT data admission.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

from h174.source_candidates import CANDIDATES
from h185.manifest_sha_only import (
    scan_legacy_manifest_sha_fields, screen_unadmitted_new_case_hashes,
)
from h194.owner_private_snap_capture import _safe_lexical_path
from h195.original_vanilla_datepicker_native import PIN

COHORTS = (
    ("Hala-H188-Private", 2, "pickr",
     "H188_PRIVATE_ORIGINAL_SOURCE_QUARANTINE_NOT_ADMISSION"),
    ("Hala-H190-Private", 2, "interact-dropzones",
     "H190_PRIVATE_INTERACT_SOURCE_QUARANTINE_NOT_ADMISSION"),
    ("Hala-H191-Private", 12, "pickr",
     "H191_TWELVE_PRIVATE_PICKR_PREACTION_QUARANTINE_ONLY"),
    ("Hala-H194-Private", 2, "interact-dropzones",
     "H194_OWNER_PRIVATE_ORIGINAL_SLIDER_DEV_PROSPECTS_NOT_ADMITTED"),
    ("Hala-H197-Private", 2, "vanilla-datepicker",
     "H197_RESEARCH_CALENDAR_NOT_ADMITTED"),
)
SOURCE_ROLES = {
    "pickr": "PROSPECTIVE_TRAIN_NOT_ADMITTED",
    "interact-dropzones": "PROSPECTIVE_DEV_NOT_ADMITTED",
    "vanilla-datepicker": "RESEARCH_UNASSIGNED_NOT_ADMITTED",
}
EXPECTED_FAMILY_COUNTS = {
    "pickr": 14,
    "interact-dropzones": 4,
    "vanilla-datepicker": 2,
}
EXPECTED_LANG_COUNTS = {"ar": 10, "en": 10}
EXPECTED_TOTAL = 20
MAX_METADATA_BYTES = 256 * 1024


def require(ok: bool, message: str) -> None:
    if not ok:
        # Never echo owner image/file paths, labels or SHA values.
        raise ValueError("H198 private source readiness: " + message)


def _regular_small(path: Path, label: str) -> None:
    # The inherited reparse guard requires lstat on the leaf. Check
    # missing files first so a missing owner manifest fails CLOSED with
    # a redacted error instead of leaking the owner path in OSError.
    require(os.path.lexists(path), "required bounded " + label + " missing or unsafe")
    _safe_lexical_path(path)
    require(path.is_file() and 0 < path.stat().st_size < MAX_METADATA_BYTES,
            "required bounded " + label + " missing or unsafe")


def _unique_source_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject ambiguous JSON keys at every depth, including escaped aliases."""
    result: dict[str, object] = {}
    for key, value in pairs:
        require(key not in result, "duplicate source metadata JSON key")
        result[key] = value
    return result


def _reject_source_constant(value: str) -> None:
    """NaN/Infinity are Python extensions, not valid JSON provenance."""
    raise ValueError("H198 private source readiness: nonstandard JSON constant")


def parse_source_manifest(raw: str) -> dict:
    """Fail closed on ambiguous source provenance before inspecting any case."""
    require(type(raw) is str and len(raw.encode("utf-8")) < MAX_METADATA_BYTES,
            "source manifest is not bounded")
    result = json.loads(
        raw, object_pairs_hook=_unique_source_object,
        parse_constant=_reject_source_constant,
    )
    require(type(result) is dict, "source manifest must be an object")
    return result


def _sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        c in "0123456789abcdef" for c in value
    )


def _source_pin(family: str) -> dict:
    if family == "vanilla-datepicker":
        return PIN
    return CANDIDATES[family]


def census(
    owner_root: Path,
    historical_root: Path | None = None,
) -> dict:
    """Read *only* new provenance manifests and old allowlisted SHA manifests.

    All 20 new source PNGs and ALL private reference/gold files stay closed.
    Any corruption, unexpectedly promoted split or duplicate must fail closed.
    """
    require(isinstance(owner_root, Path) and owner_root.is_absolute(),
            "absolute owner root required")
    _safe_lexical_path(owner_root)
    require(owner_root.is_dir(), "owner original source root missing")
    seen_cases: set[str] = set()
    png: set[str] = set()
    rgba: set[str] = set()
    langs: Counter[str] = Counter()
    families: Counter[str] = Counter()
    cohorts: Counter[str] = Counter()
    sha_records: list[dict[str, str]] = []
    for name, expected_count, family, expected_status in COHORTS:
        folder = owner_root / name
        _safe_lexical_path(folder)
        require(folder.is_dir() and not (folder / ".git").exists(),
                "original source quarantine not a regular private directory")
        source = folder / "source-prospects-not-admitted.json"
        reference = folder / "OWNER_PRIVATE_REFERENCE_NOT_MODEL_INPUT.json"
        _regular_small(source, "source SHA provenance")
        # Reference sidecar is ONLY checked for a safe name/regular existence,
        # and NEVER read; an intentionally invalid JSON gold file is fine.
        _regular_small(reference, "sealed private reference sidecar")
        manifest = parse_source_manifest(source.read_text(encoding="utf-8"))
        require(type(manifest) is dict
                and manifest.get("status") == expected_status
                and manifest.get("official_train_dev_dataset_admitted") is False
                and manifest.get("historical_source_complete") is not True
                and manifest.get("historical_decoded_rgba_complete") is not True,
                "original source provenance or admission flag changed")
        rows = manifest.get("cases")
        require(type(rows) is list and len(rows) == expected_count,
                "original cohort case count changed")
        pin = _source_pin(family)
        for row in rows:
            require(type(row) is dict
                    and type(row.get("case_id")) is str
                    and 0 < len(row["case_id"]) <= 140
                    and row["case_id"] not in seen_cases,
                    "missing or repeated original case identity")
            seen_cases.add(row["case_id"])
            require(row.get("source_repo") == pin["repo"]
                    and row.get("source_git_sha") == pin["git_sha"],
                    "upstream original app identity or SHA changed")
            language = row.get("language")
            require(language in ("ar", "en") and type(language) is str,
                    "missing bilingual source language")
            require(type(row.get("instruction")) is str
                    and 0 < len(row["instruction"]) <= 600,
                    "source task instruction missing or unexpectedly large")
            require(row.get("train_eligible") is False
                    and row.get("dev_eligible") is False
                    and row.get("ood_eligible") is False,
                    "unauthorized TRAIN/DEV/OOD promotion")
            png_hash = row.get("image_sha256")
            rgba_hash = row.get("visual_rgba_sha256")
            require(_sha(png_hash) and _sha(rgba_hash),
                    "original source PNG/RGBA fingerprints invalid")
            require(png_hash not in png and rgba_hash not in rgba,
                    "original source overlap across new cohorts")
            png.add(png_hash)
            rgba.add(rgba_hash)
            langs[language] += 1
            families[family] += 1
            cohorts[name] += 1
            sha_records.append({
                "image_sha256": png_hash,
                "visual_rgba_sha256": rgba_hash,
            })
    require(len(seen_cases) == len(png) == len(rgba) == EXPECTED_TOTAL,
            "total original source case or pixel SHA count mismatch")
    require(dict(families) == EXPECTED_FAMILY_COUNTS,
            "prospective source family composition changed")
    require(dict(langs) == EXPECTED_LANG_COUNTS,
            "current source bilingual balance changed")
    checked_old = False
    old_total = 0
    if historical_root is not None:
        require(isinstance(historical_root, Path)
                and historical_root.is_absolute(),
                "actual owner historical source path required")
        old = scan_legacy_manifest_sha_fields([historical_root])
        result = screen_unadmitted_new_case_hashes(sha_records, old)
        require(result["training_approved"] is False
                and result["historical_completeness_proven"] is False,
                "historical SHA check cannot approve model training")
        checked_old = True
        old_total = len(old["_owner_only_png_sha_candidates"])
    # No private source path, SHA, case ID, task, gold or model response printed.
    return {
        "status": "H198_20_PRIVATE_GUI_PROSPECTS_RESEARCH_ONLY",
        "total_quarantined_original_preaction_png_metadata": EXPECTED_TOTAL,
        "source_implementation_families": len(families),
        "cohorts_checked": len(cohorts),
        "source_family_counts": dict(families),
        "languages": dict(langs),
        "unique_png_shas": len(png),
        "unique_rgba_shas": len(rgba),
        "known_old_sha_collision_check_ran": checked_old,
        "known_old_png_shas_checked": old_total,
        "prospective_source_roles": dict(SOURCE_ROLES),
        "training_approved": False,
        "dev_approved": False,
        "ood_approved": False,
        "historic_source_and_rgba_history_complete": False,
        "data_gold_or_image_files_opened": 0,
        "private_files_uploaded": 0,
        "model_calls": 0,
        "gpu_optimizer_steps": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Owner-only strict original-source SHA metadata readiness"
    )
    parser.add_argument("--owner-root", required=True, type=Path)
    parser.add_argument("--historical-root", type=Path)
    args = parser.parse_args()
    result = census(args.owner_root, args.historical_root)
    print(
        f"{result['status']} "
        f"total={result['total_quarantined_original_preaction_png_metadata']} "
        f"families={result['source_implementation_families']} "
        f"ar={result['languages']['ar']} en={result['languages']['en']} "
        f"distinct_png={result['unique_png_shas']} "
        f"distinct_rgba={result['unique_rgba_shas']} "
        f"known_old_png={result['known_old_png_shas_checked']} "
        "NO_DATA_ADMISSION_NO_PRIVATE_MEDIA_NO_OLD_OOD_NO_MODEL"
    )


if __name__ == "__main__":
    main()
