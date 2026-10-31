"""H258 synthetic-only tests: no owner receipts, media, gold or H173 scored OOD."""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import pytest

from h256.pre173_visual_history import RECEIPT_NAME as H256_NAME, audit as h256_audit
from h258.recover_tall_rgba import (
    OUTPUT_NAME, STATUS, audit, inspect_png_structure, rgba_digest,
)


def png(width: int, height: int, color: str = "red") -> bytes:
    out = BytesIO()
    Image.new("RGB", (width, height), color).save(out, format="PNG")
    return out.getvalue()


def fixture(tmp_path: Path, *, extra_tall: bool = False):
    phase = tmp_path / "private-h168"
    phase.mkdir()
    small = png(8, 8)
    tall = png(2, 4100)
    (phase / "small.png").write_bytes(small)
    (phase / "tall.png").write_bytes(tall)
    (phase / "bad.png").write_bytes(b"not actually PNG")
    if extra_tall:
        (phase / "tall-copy.png").write_bytes(tall)
    (phase / "manifest.json").write_text(
        json.dumps({"png_sha256": sha256(small).hexdigest()}), encoding="utf-8")
    h256_dir = tmp_path / "h256-private"
    h256_dir.mkdir()
    h256_receipt = h256_dir / H256_NAME
    h256_audit([tmp_path], h256_receipt)
    output_dir = tmp_path / "h258-private"
    output_dir.mkdir()
    return h256_receipt, [tmp_path], output_dir / OUTPUT_NAME, small, tall


def test_real_tall_synthetic_png_is_recovered_without_admission(tmp_path):
    source, parents, output, small, tall = fixture(tmp_path)
    result = audit(source, parents, output)
    assert result["status"] == STATUS
    assert result["source_h256_missing_rgba_files"] == 2
    assert result["classification"]["not_png_signature"] == 1
    assert result["classification"]["oversize_policy_recovered"] == 1
    assert result["new_distinct_rgba_exclusions"] == 1
    assert result["combined_distinct_rgba_exclusions"] == 2
    assert result["train_admitted"] is False
    assert result["dev_admitted"] is False
    assert result["ood_admitted"] is False
    assert result["h173_opened"] is False
    assert result["model_calls"] == 0
    assert sha256(tall).hexdigest() not in json.dumps(result)
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert len(receipt["owner_only_recovered"]) == 1
    assert receipt["owner_only_recovered"][0]["decoded_rgba_sha256"] == (
        rgba_digest(tall, 2, 4100))


def test_identical_tall_bytes_are_not_independent(tmp_path):
    source, parents, output, _, _ = fixture(tmp_path, extra_tall=True)
    result = audit(source, parents, output)
    assert result["classification"]["oversize_policy_recovered"] == 2
    assert result["recovered_distinct_png"] == 1
    assert result["new_distinct_rgba_exclusions"] == 1


def test_valid_structure_and_crc_corruption():
    data = png(2, 4100)
    assert inspect_png_structure(data) == ("valid_structure", 2, 4100)
    altered = bytearray(data)
    altered[29] ^= 1  # corrupt IHDR CRC (never touch source image)
    assert inspect_png_structure(bytes(altered))[0] == "invalid_structure"


def test_trailing_bytes_rejected():
    assert inspect_png_structure(png(2, 4100) + b"garbage")[0] == "invalid_structure"


def test_no_png_signature():
    assert inspect_png_structure(b"not a png")[0] == "not_png_signature"


def test_oversize_dimensions_refused(tmp_path):
    source, parents, output, _, _ = fixture(tmp_path)
    phase = tmp_path / "private-h168"
    (phase / "tall.png").write_bytes(png(2, 8200))
    # Rebuild source receipt from altered bytes without overwriting it.
    source.unlink()
    h256_audit(parents, source)
    result = audit(source, parents, output)
    assert result["classification"]["unsafe_dimensions"] == 1
    assert result["new_distinct_rgba_exclusions"] == 0


def test_private_output_never_overwritten(tmp_path):
    source, parents, output, _, _ = fixture(tmp_path)
    audit(source, parents, output)
    with pytest.raises(ValueError, match="H258_NOT_QUALIFIED"):
        audit(source, parents, output)


def test_modified_media_hash_fails_closed(tmp_path):
    source, parents, output, _, _ = fixture(tmp_path)
    (tmp_path / "private-h168" / "tall.png").write_bytes(png(2, 4100, "blue"))
    with pytest.raises(ValueError, match="H258_NOT_QUALIFIED"):
        audit(source, parents, output)
    assert not output.exists()


def test_h173_phase_in_receipt_refused(tmp_path):
    source, parents, output, _, _ = fixture(tmp_path)
    data = json.loads(source.read_text(encoding="utf-8"))
    data["owner_only_images"][0]["phase"] = "private-h173"
    source.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="H258_NOT_QUALIFIED"):
        audit(source, parents, output)
    assert not output.exists()


def test_output_in_git_checkout_refused(tmp_path):
    source, parents, output, _, _ = fixture(tmp_path)
    (output.parent / ".git").mkdir()
    with pytest.raises(ValueError, match="H258_NOT_QUALIFIED"):
        audit(source, parents, output)


def test_bad_source_receipt_rgba_rejected(tmp_path):
    source, parents, output, _, _ = fixture(tmp_path)
    data = json.loads(source.read_text(encoding="utf-8"))
    data["owner_only_images"][0]["decoded_rgba_sha256"] = "not-a-sha"
    source.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="H258_NOT_QUALIFIED"):
        audit(source, parents, output)


def test_bad_output_name_refused(tmp_path):
    source, parents, output, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="H258_NOT_QUALIFIED"):
        audit(source, parents, output.with_name("public.json"))


@pytest.mark.parametrize("bad_relative", ["../tall.png", "C:/tall.png",
                                         "sub/../tall.png", "sub\\\\tall.png"])
def test_path_traversal_and_drive_rejected(tmp_path, bad_relative):
    source, parents, output, _, _ = fixture(tmp_path)
    data = json.loads(source.read_text(encoding="utf-8"))
    data["owner_only_images"][0]["relative_png_path"] = bad_relative
    source.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="H258_NOT_QUALIFIED"):
        audit(source, parents, output)
    assert not output.exists()
