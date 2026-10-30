"""Synthetic-only H254 unit tests; never touch owner media or historical OOD."""
from pathlib import Path

import pytest

from h254.original_media_runtime_evidence import (
    history_status, summarize, within_root,
)


def fixture_static():
    return {
        "status": "INCOMPLETE_NO_ADMISSION",
        "pinned_source_verified": True,
        "referenced_local_media": ["assets/logo.png", "assets/icon.ico"],
        "remote_runtime_dependencies": ["https://fonts.example.test/css"],
        "inline_svg_elements": 1,
    }


def call(**overrides):
    args = dict(static=fixture_static(), shallow=True,
                history={"assets/logo.png": ["a" * 40]},
                images=[{"loaded": True, "local": True},
                        {"loaded": False, "local": True}],
                blocked=2, remote_responses=0,
                file_escapes=0, websocket_attempts=0)
    args.update(overrides)
    return summarize(**args)


def test_realistic_observation_does_not_admit():
    result = call()
    assert result["browser_loaded_local_images"] == 1
    assert result["browser_failed_or_nonlocal_images"] == 1
    assert result["blocked_external_http_requests_observed"] == 2
    assert result["referenced_media_history_status"]["assets/logo.png"] == "INCOMPLETE_GIT_HISTORY"
    assert result["asset_authorship_and_transitive_rights_proven"] is False
    assert result["train_dev_ood_admitted"] is False
    assert result["model_calls"] == result["optimizer_steps"] == 0


@pytest.mark.parametrize("field", ["remote_responses", "file_escapes", "websocket_attempts"])
def test_forbidden_browser_activity_fail_closed(field):
    with pytest.raises(ValueError, match="Forbidden"):
        call(**{field: 1})


def test_malformed_observation_fail_closed():
    with pytest.raises(ValueError, match="Malformed"):
        call(images=[{"loaded": "yes", "local": True}])


def test_source_preflight_fail_closed():
    bad = fixture_static()
    bad["pinned_source_verified"] = False
    with pytest.raises(ValueError, match="preflight"):
        call(static=bad)


def test_shallow_git_is_not_proven():
    assert history_status(True, ["a" * 40]) == "INCOMPLETE_GIT_HISTORY"
    assert history_status(False, []) == "INCOMPLETE_GIT_HISTORY"
    assert history_status(False, ["a" * 40]) == "HISTORY_VISIBLE_RIGHTS_UNPROVEN"


def test_file_scope_rejects_nonfile_and_traversal(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    inside = root / "index.html"
    outside = tmp_path / "private.txt"
    assert within_root(inside.as_uri(), root)
    assert not within_root(outside.as_uri(), root)
    assert not within_root("https://example.com/asset.png", root)
    assert not within_root("data:image/png;base64,AAAA", root)
    assert not within_root("file:///../outside", root)
