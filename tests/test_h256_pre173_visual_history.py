"""Synthetic-only H256 tests. Never access any owner-private phase/media."""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import pytest

from h256.pre173_visual_history import (
    RECEIPT_NAME, audit, collect_pngs, digest_png, safe_output_path,
)


def synthetic_png(color: str, *, compress_level: int = 6) -> bytes:
    buffer = BytesIO()
    Image.new("RGBA", (12, 9), color).save(
        buffer, format="PNG", compress_level=compress_level
    )
    return buffer.getvalue()


def fixture_history(tmp_path: Path) -> tuple[Path, Path, Path, bytes]:
    phase = tmp_path / "private-h168"
    phase.mkdir()
    first = synthetic_png("blue", compress_level=0)
    (phase / "a.png").write_bytes(first)
    (phase / "manifest.json").write_text(
        json.dumps({"png_sha256": sha256(first).hexdigest()}),
        encoding="utf-8",
    )
    private = tmp_path / "owner-only"
    private.mkdir()
    return phase, private / RECEIPT_NAME, tmp_path, first


def test_rgba_digest_includes_dimensions_and_pixels():
    blob = synthetic_png("red")
    raw, rgba, w, h = digest_png(blob)
    assert raw == sha256(blob).hexdigest()
    assert (w, h) == (12, 9)
    with Image.open(BytesIO(blob)) as image:
        expected = sha256(
            w.to_bytes(4, "big") + h.to_bytes(4, "big")
            + image.convert("RGBA").tobytes()
        ).hexdigest()
    assert rgba == expected


def test_png_compression_differs_but_rgba_same():
    a = synthetic_png("green", compress_level=0)
    b = synthetic_png("green", compress_level=9)
    assert digest_png(a)[0] != digest_png(b)[0]
    assert digest_png(a)[1] == digest_png(b)[1]


def test_surviving_history_private_receipt_only(tmp_path):
    phase, output, parent, first = fixture_history(tmp_path)
    second = synthetic_png("blue", compress_level=9)
    (phase / "b.png").write_bytes(second)
    result = audit([parent], output)
    assert result["surviving_png_files"] == 2
    assert result["decodable_png_files"] == 2
    assert result["manifest_png_sha_floor"] == 1
    assert result["known_manifest_png_sha_found_on_disk"] == 1
    assert result["surviving_png_sha_not_in_known_manifests"] == 1
    assert result["distinct_surviving_decoded_rgba_sha"] == 1
    assert result["complete_old_visual_history"] is False
    assert result["train_dev_ood_admitted"] is False
    assert result["h173_ood_accessed"] is False
    assert sha256(first).hexdigest() not in json.dumps(result)
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert len(receipt["owner_only_images"]) == 2
    assert all("decoded_rgba_sha256" in row for row in receipt["owner_only_images"])


def test_sealed_h173_never_traversed(tmp_path):
    _, output, parent, _ = fixture_history(tmp_path)
    sealed = parent / "private-h173"
    sealed.mkdir()
    (sealed / "manifest.json").write_text("NOT JSON", encoding="utf-8")
    (sealed / "forbidden.png").write_bytes(b"not a PNG")
    result = audit([parent], output)
    assert result["historical_phases"] == 1
    assert result["surviving_png_files"] == 1
    assert result["h173_ood_accessed"] is False


def test_gold_dir_not_opened(tmp_path):
    phase, output, parent, _ = fixture_history(tmp_path)
    gold = phase / "gold"
    gold.mkdir()
    (gold / "private.png").write_bytes(b"not a PNG")
    result = audit([parent], output)
    assert result["surviving_png_files"] == 1
    assert result["excluded_sensitive_directories"] >= 1
    assert result["gold_opened"] is False


def test_undecodable_png_not_falsely_given_rgba(tmp_path):
    phase, output, parent, _ = fixture_history(tmp_path)
    (phase / "bad.png").write_bytes(b"not a valid PNG")
    result = audit([parent], output)
    assert result["undecodable_png_named_files"] == 1
    assert result["decodable_png_files"] == 1
    rows = json.loads(output.read_text(encoding="utf-8"))["owner_only_images"]
    assert sum(row["decoded_rgba_sha256"] is None for row in rows) == 1


def test_private_receipt_cannot_overwrite(tmp_path):
    _, output, parent, _ = fixture_history(tmp_path)
    audit([parent], output)
    with pytest.raises(ValueError, match="no overwrite"):
        audit([parent], output)


def test_output_in_git_checkout_rejected(tmp_path):
    _, output, parent, _ = fixture_history(tmp_path)
    (parent / ".git").mkdir()
    with pytest.raises(ValueError, match="Git checkout"):
        audit([parent], output)


def test_h173_direct_phase_rejected(tmp_path):
    sealed = tmp_path / "private-h173"
    sealed.mkdir()
    (sealed / "a.png").write_bytes(synthetic_png("red"))
    with pytest.raises(ValueError, match="pre-H173"):
        collect_pngs([sealed])


def test_sensitive_gold_filename_refused_before_open(tmp_path):
    phase, output, parent, _ = fixture_history(tmp_path)
    (phase / "reference_gold.png").write_bytes(b"private bytes not opened")
    with pytest.raises(ValueError, match="sensitive PNG filename"):
        audit([parent], output)
    assert not output.exists()


def test_output_filename_fail_closed(tmp_path):
    phase, output, _, _ = fixture_history(tmp_path)
    with pytest.raises(ValueError, match="filename"):
        safe_output_path(output.with_name("public.json"), [phase])


def test_non_png_bytes_refused_by_decoder():
    with pytest.raises((ValueError, OSError)):
        digest_png(b"definitely not a PNG")
