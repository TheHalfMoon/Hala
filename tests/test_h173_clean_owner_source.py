"""Owner one-shot browser scoring requires exact, unchanged Git source.

Synthetic temporary Git repository only; no model files or owner D-drive.
"""
import subprocess

import pytest
from h173.source_pin import verify_clean_local_source_checkout


def local_repo(tmp_path):
    root = tmp_path / "owner-checkout"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    source = root / "module.py"
    source.write_text("value = 'known'\n", encoding="utf8")
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()
    git("add", "module.py")
    subprocess.run([
        "git", "-C", str(root),
        "-c", "user.name=Hala QA", "-c", "user.email=qa@example.invalid",
        "commit", "-qm", "freeze source"
    ], check=True)
    return root, source, git("rev-parse", "HEAD")


def test_owner_one_shot_accepts_exact_clean_pinned_commit(tmp_path):
    root, _, revision = local_repo(tmp_path)
    assert verify_clean_local_source_checkout(root, revision) == revision
    # Untracked local PRIVATE scratch is deliberately not fetched or inspected.
    (root / "private-local-not-in-git.txt").write_text(
        "untracked owner content remains private", encoding="utf8"
    )
    assert verify_clean_local_source_checkout(root, revision) == revision


def test_owner_one_shot_rejects_tracked_source_edit(tmp_path):
    root, path, revision = local_repo(tmp_path)
    path.write_text("value = 'tampered'\n", encoding="utf8")
    with pytest.raises(ValueError, match="exact clean immutable Git"):
        verify_clean_local_source_checkout(root, revision)


def test_owner_one_shot_rejects_staged_tracked_source_edit(tmp_path):
    root, path, revision = local_repo(tmp_path)
    path.write_text("value = 'tampered'\n", encoding="utf8")
    subprocess.run(["git", "-C", str(root), "add", "module.py"], check=True)
    with pytest.raises(ValueError, match="exact clean immutable Git"):
        verify_clean_local_source_checkout(root, revision)


def test_owner_one_shot_rejects_wrong_or_missing_commit(tmp_path):
    root, _, revision = local_repo(tmp_path)
    assert len(revision) == 40
    with pytest.raises(ValueError, match="exact clean immutable Git"):
        verify_clean_local_source_checkout(root, "b" * 40)
    with pytest.raises(ValueError, match="exact 40-hex Git"):
        verify_clean_local_source_checkout(root, "main")
