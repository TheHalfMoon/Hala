"""H174 donor source-byte preflight; synthetic fixtures only, never old OOD."""
from copy import deepcopy
import hashlib

import pytest

import h174.source_candidates as candidates
from h173.source_preflight import SOURCE_PINS


def test_three_distinct_pinned_mit_candidates_without_split_assignment():
    assert candidates.verify_catalog() == 3
    assert len({p["git_sha"] for p in candidates.CANDIDATES.values()}) == 3
    assert len({p["html_sha256"] for p in candidates.CANDIDATES.values()}) == 3


@pytest.fixture
def pristine(tmp_path, monkeypatch):
    html = b"<html><button>New source only</button></html>\n"
    license_bytes = b"MIT License\nPermission is hereby granted\n"
    pins = deepcopy(candidates.CANDIDATES)
    pin = pins["pickr"]
    pin["html_sha256"] = hashlib.sha256(html).hexdigest()
    pin["license_sha256"] = hashlib.sha256(license_bytes).hexdigest()
    monkeypatch.setattr(candidates, "CANDIDATES", pins)
    root = tmp_path / "pickr"
    root.mkdir()
    (root / ".git").mkdir()
    (root / pin["html_path"]).write_bytes(html)
    (root / pin["license_path"]).write_bytes(license_bytes)
    answers = {
        ("rev-parse", "HEAD"): pin["git_sha"],
        ("status", "--porcelain", "--untracked-files=all"): "",
        ("remote", "get-url", "origin"): "https://github.com/" + pin["repo"] + ".git",
    }

    def fake_git(cmd, **kwargs):
        assert cmd[:2] == ["git", "-C"]
        assert cmd[2] == str(root)
        return answers[tuple(cmd[3:])] + "\n"

    monkeypatch.setattr(candidates.subprocess, "check_output", fake_git)
    return root, pin, answers


def test_source_bytes_only_success_is_not_native_or_dataset_qualification(pristine):
    root, pin, _ = pristine
    result = candidates.verify_local_source_only("pickr", root)
    assert result["status"] == "H174_PINNED_SOURCE_BYTES_ONLY_NO_NATIVE_QA"
    assert result["source_git_sha"] == pin["git_sha"]
    assert result["source_native_qa_qualified"] is False
    assert result["split_assigned"] is False
    assert result["train_dev_images_created"] == 0
    assert result["ood_images_created"] == 0
    assert result["model_calls"] == 0


def test_corrupted_original_html_rejected(pristine):
    root, pin, _ = pristine
    (root / pin["html_path"]).write_bytes(b"<html>modified</html>")
    with pytest.raises(ValueError, match="HTML/license bytes mismatch"):
        candidates.verify_local_source_only("pickr", root)


def test_corrupted_license_bytes_rejected(pristine):
    root, pin, _ = pristine
    (root / pin["license_path"]).write_bytes(b"NOT AN MIT LICENSE")
    with pytest.raises(ValueError, match="HTML/license bytes mismatch"):
        candidates.verify_local_source_only("pickr", root)


@pytest.mark.parametrize("command,value,pattern", [
    (("rev-parse", "HEAD"), "f" * 40, "immutable Git revision"),
    (("status", "--porcelain", "--untracked-files=all"), " M index.html", "source dirty"),
    (("remote", "get-url", "origin"), "https://github.com/attacker/repo.git", "remote mismatch"),
])
def test_wrong_commit_dirty_source_and_wrong_origin_fail_closed(pristine, command, value, pattern):
    root, _, answers = pristine
    answers[command] = value
    with pytest.raises(ValueError, match=pattern):
        candidates.verify_local_source_only("pickr", root)


def test_unknown_source_and_missing_git_checkout_rejected(pristine):
    root, _, _ = pristine
    with pytest.raises(ValueError, match="unknown candidate"):
        candidates.verify_local_source_only("not-registered", root)
    with pytest.raises(ValueError, match="Git checkout missing"):
        candidates.verify_local_source_only("pickr", root / "nonexistent")


def test_historical_git_alias_rejected_without_historical_pixels(monkeypatch):
    pins = deepcopy(candidates.CANDIDATES)
    pins["pickr"]["git_sha"] = SOURCE_PINS["tingle"]["git_sha"]
    monkeypatch.setattr(candidates, "CANDIDATES", pins)
    with pytest.raises(ValueError, match="aliases historical"):
        candidates.verify_catalog()


def test_duplicate_original_html_alias_rejected(monkeypatch):
    pins = deepcopy(candidates.CANDIDATES)
    pins["pickr"]["html_sha256"] = pins["medium-zoom"]["html_sha256"]
    monkeypatch.setattr(candidates, "CANDIDATES", pins)
    with pytest.raises(ValueError, match="aliases historical"):
        candidates.verify_catalog()


def test_invalid_digest_rejected_before_any_source_reads(monkeypatch):
    pins = deepcopy(candidates.CANDIDATES)
    pins["pickr"]["license_sha256"] = "not-a-sha"
    monkeypatch.setattr(candidates, "CANDIDATES", pins)
    with pytest.raises(ValueError, match="invalid provenance"):
        candidates.verify_catalog()


def test_no_escaping_candidate_html_path(pristine):
    root, pin, _ = pristine
    pin["html_path"] = "../private-gold.json"
    with pytest.raises(ValueError, match="escapes checkout"):
        candidates.verify_local_source_only("pickr", root)
