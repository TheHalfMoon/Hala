"""H173 native OOD must act on exactly the independently-audited GPU output.

These are fake local path fixtures. They never touch private OOD, models,
Kaggle, browser gold or previously consumed sources.
"""
import pytest
from h173.native_reward import audited_gpu_proposals_root


def test_exact_private_gpu_job_file_is_admitted(tmp_path):
    folder = tmp_path / "hala-h173-private-paired-ood"
    folder.mkdir()
    canonical = folder / "h173-paired-ood-predictions.json"
    canonical.write_text("{}", encoding="utf8")
    assert audited_gpu_proposals_root(canonical) == tmp_path


@pytest.mark.parametrize("filename,dirname", [
    ("substituted-predictions.json", "hala-h173-private-paired-ood"),
    ("h173-paired-ood-predictions.json", "unqualified-sibling-job"),
    ("h173-paired-ood-predictions.json.bak", "hala-h173-private-paired-ood"),
])
def test_other_file_cannot_be_scored_after_sibling_audit(
    tmp_path, filename, dirname
):
    folder = tmp_path / dirname
    folder.mkdir()
    forged = folder / filename
    forged.write_text("{}", encoding="utf8")
    with pytest.raises(ValueError, match="exact non-symlink audited GPU"):
        audited_gpu_proposals_root(forged)


def test_symlink_gpu_proposals_cannot_substitute_another_job(tmp_path):
    outside = tmp_path / "other-output.json"
    outside.write_text('{"wrong":"gpu-job"}', encoding="utf8")
    folder = tmp_path / "hala-h173-private-paired-ood"
    folder.mkdir()
    alias = folder / "h173-paired-ood-predictions.json"
    alias.symlink_to(outside)
    with pytest.raises(ValueError, match="exact non-symlink audited GPU"):
        audited_gpu_proposals_root(alias)


def test_symlink_gpu_job_directory_is_refused(tmp_path):
    actual = tmp_path / "other-gpu"
    actual.mkdir()
    (actual / "h173-paired-ood-predictions.json").write_text(
        "{}", encoding="utf8"
    )
    alias = tmp_path / "hala-h173-private-paired-ood"
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match="exact non-symlink audited GPU"):
        audited_gpu_proposals_root(alias / "h173-paired-ood-predictions.json")
