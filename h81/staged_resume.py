"""H81: SHA-verified reuse of a previously restored original TRAIN stage.

Reuses H67's real archive/member/curriculum validators. No GPU or third-party
libraries needed; existing training outputs remain untouched for true resume.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

from h67 import stage

SCHEMA = "hala.h81.reused-train-stage.v1"
# These are the H61 run's outputs, never part of original TRAIN admission.
ALLOWED_OUTPUTS = frozenset({
    "training", "training-config.json", "run-receipt.json",
    "eval", "official-eval",
})


def admit_stage(bundle: Path, expected_sha256: str, destination: Path,
                pinned_sha: str, *, rederive: bool = True) -> dict:
    """Restore once, or recheck ALL source bytes on subsequent session retry.

    A reused folder is never trusted based on its path or a prior receipt.
    Any missing, extra, linked or altered original source artifact is refused.
    """
    destination, bundle = Path(destination), Path(bundle)
    if destination.is_symlink():
        raise ValueError("linked H61 run workspace forbidden")
    if not destination.is_dir():
        # H67's create-only gate still protects a regular existing file,
        # incomplete import, symlinks, and all first-admission integrity.
        return stage.restore_stage(bundle, expected_sha256, destination, pinned_sha,
                                   rederive=rederive)
    if (bundle.is_symlink() or not bundle.is_file()
            or not isinstance(expected_sha256, str)
            or stage.HEX64.fullmatch(expected_sha256) is None
            or not isinstance(pinned_sha, str)
            or stage.GITSHA.fullmatch(pinned_sha) is None
            or stage.digest(bundle) != expected_sha256):
        raise ValueError("original TRAIN bundle digest or source Git SHA invalid")
    for top in destination.iterdir():
        if top.is_symlink() or top.name not in (
                {"export", stage.BASELINE} | ALLOWED_OUTPUTS):
            raise ValueError("untrusted H61 run workspace entry")
    if (destination / "export").is_symlink() or not (destination / "export").is_dir():
        raise ValueError("original TRAIN export directory missing or linked")
    if (destination / stage.BASELINE).is_symlink():
        raise ValueError("original TRAIN baseline may not be linked")
    with zipfile.ZipFile(bundle) as archive:
        metadata, _ = stage._check_zip(archive, pinned_sha=pinned_sha)
    expected = metadata["files"]
    current = stage._inventory(destination)
    if current != expected:
        raise ValueError("existing TRAIN source inventory differs from pinned ZIP")
    qualified = stage.validate_train(destination, rederive=rederive)
    if (qualified["dataset_sha256"] != metadata["dataset_sha256"]
            or qualified["rows"] != metadata["row_count"]
            or qualified["media"] != metadata["media_count"]):
        raise ValueError("existing TRAIN data does not match independently derived archive")
    return {"schema": stage.SCHEMA, "status": "RESTORED_TRAIN_STAGE",
            "sha256": expected_sha256, "git_sha": pinned_sha,
            "dataset_sha256": qualified["dataset_sha256"],
            "files": len(expected), "private_provider_readback_verified": False,
            "reused_verified_existing": True}
