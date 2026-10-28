"""Never recycle quarantined H173 candidate screenshots by SHA identity."""
import json
import pytest

from h173.capture_ood import (
    CAPTURE_ATTEMPT_LOCK, MANIFEST, quarantined_h173_image_shas,
)


def make_attempt(root, name, offset=0, *, marker=False):
    folder = root / name
    folder.mkdir()
    if marker:
        (folder / CAPTURE_ATTEMPT_LOCK).write_text("reserved", encoding="utf8")
    rows = [{"image_sha256": f"{i + offset:064x}"} for i in range(1, 33)]
    (folder / "manifest.json").write_text(
        json.dumps({"schema": MANIFEST, "private_cases": rows}), encoding="utf8"
    )
    return folder


def test_legacy_and_locked_attempts_are_excluded_without_reading_png(tmp_path):
    make_attempt(tmp_path, "ood-32", 0)
    make_attempt(tmp_path, "ood-32-v2", 2, marker=True)
    make_attempt(tmp_path, "ood-32-v3", 100, marker=True)
    hashes = quarantined_h173_image_shas(tmp_path / "ood-32-v4")
    assert len(hashes) == 66
    assert f"{1:064x}" in hashes
    assert f"{132:064x}" in hashes


def test_current_capture_not_self_excluded_during_replay(tmp_path):
    folder = make_attempt(tmp_path, "ood-32-v4", marker=True)
    assert quarantined_h173_image_shas(folder) == set()


def test_incomplete_prior_attempt_fails_closed(tmp_path):
    folder = tmp_path / "ood-32-v2"
    folder.mkdir()
    (folder / CAPTURE_ATTEMPT_LOCK).write_text("reserved", encoding="utf8")
    with pytest.raises(ValueError, match="incomplete quarantined"):
        quarantined_h173_image_shas(tmp_path / "ood-32-v4")


def test_malformed_prior_hash_fails_closed(tmp_path):
    folder = make_attempt(tmp_path, "ood-32", marker=False)
    manifest = folder / "manifest.json"
    doc = json.loads(manifest.read_text("utf8"))
    doc["private_cases"][0]["image_sha256"] = "not-a-sha"
    manifest.write_text(json.dumps(doc), encoding="utf8")
    with pytest.raises(ValueError, match="SHA malformed"):
        quarantined_h173_image_shas(tmp_path / "ood-32-v4")


def test_unrelated_manifest_does_not_count_as_h173(tmp_path):
    folder = tmp_path / "pytest-source"
    folder.mkdir()
    (folder / "manifest.json").write_text(
        json.dumps({"schema": "unrelated"}), encoding="utf8"
    )
    assert quarantined_h173_image_shas(tmp_path / "ood-32-v4") == set()
