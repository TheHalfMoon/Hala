"""H173 exact source commit is pinned *outside* its own immutable tree."""
import json

import pytest

from h173.source_pin import (
    SOURCE_ENV, pinned_source_commit, validate_source_commit
)
from h173 import audit_gpu


def test_owner_pins_actual_immutable_source_commit_via_environment(monkeypatch):
    expected = "c" * 40
    monkeypatch.setenv(SOURCE_ENV, expected)
    assert pinned_source_commit() == expected
    assert validate_source_commit(expected) == expected


@pytest.mark.parametrize("value", [
    "", "UNQUALIFIED", "main", "ps/h173-branch", "f"*39, "x"*40,
    "0"*40, "F"*40, "a"*41, "a"*39+" ", 1234, None,
])
def test_bad_or_mutable_source_commit_is_refused(value):
    with pytest.raises(ValueError):
        validate_source_commit(value)


def test_gpu_auditor_cannot_use_untrusted_job_self_pin(monkeypatch, tmp_path):
    monkeypatch.delenv(SOURCE_ENV, raising=False)
    monkeypatch.setattr(audit_gpu, "SOURCE_SHA", "UNQUALIFIED")
    with pytest.raises(ValueError, match="owner must explicitly pin"):
        audit_gpu.audit(tmp_path)


def test_private_gpu_auditor_allows_explicit_expected_git_identity(monkeypatch):
    monkeypatch.delenv(SOURCE_ENV, raising=False)
    assert validate_source_commit("b"*40) == "b"*40
