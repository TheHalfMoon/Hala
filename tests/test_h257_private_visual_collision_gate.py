"""H257 synthetic-only tests: invented PNG pixels, no owner media or OOD."""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import pytest

from h250.native_original_todo_app import STATUS as H250_STATUS, STAGES as H250_STAGES
from h251.original_lifeos_rtl import STATUS as H251_STATUS, STAGES as H251_STAGES
from h256.pre173_visual_history import digest_png
from h257.private_visual_collision_gate import OUTPUT_NAME, audit


def png(index: int, *, compress: int = 6) -> bytes:
    buf = BytesIO()
    Image.new("RGBA", (12, 9),
              (index * 7 % 256, index * 13 % 256, index * 17 % 256, 255)
              ).save(buf, format="PNG", compress_level=compress)
    return buf.getvalue()


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, sort_keys=True), encoding="utf-8")


def fixture(tmp: Path):
    old = tmp / "history"
    old.mkdir()
    history = old / "H256_OWNER_PRIVATE_PRE173_VISUAL_HASHES_NOT_ADMITTED.json"
    raw, rgba, _, _ = digest_png(png(200))
    historical = {
        "summary": {
            "status": "H256_PRE173_SURVIVING_MEDIA_EXCLUSIONS_ONLY_NO_ADMISSION",
            "h173_ood_accessed": False, "gold_opened": False,
            "train_dev_ood_admitted": False,
            "complete_old_visual_history": False,
            "complete_old_source_history": False,
            "surviving_png_files": 1, "distinct_surviving_png_sha": 1,
            "distinct_surviving_decoded_rgba_sha": 1,
            "manifest_png_sha_floor": 1,
        },
        "owner_only_images": [{"phase": "private-h168", "png_sha256": raw, "decoded_rgba_sha256": rgba}],
    }
    write_json(history, historical)
    sources = {}
    for name, (status, stages) in {
        "h250": (H250_STATUS, H250_STAGES),
        "h251": (H251_STATUS, H251_STAGES),
    }.items():
        folder = tmp / name
        folder.mkdir()
        receipt = folder / "PRIVATE_RESEARCH_NOT_ADMITTED.json"
        data = {"status": status, "train": False, "dev": False, "ood": False,
                "model_calls": 0, "optimizer_steps": 0, "cases": []}
        for case_i, language in enumerate(("en", "ar")):
            case_id = name + "-" + language
            frames = {}
            for stage_i, stage in enumerate(stages):
                index = 1 + (0 if name == "h250" else 12) + case_i * 6 + stage_i
                blob = png(index)
                (folder / (case_id + "-" + stage + ".png")).write_bytes(blob)
                p, r, _, _ = digest_png(blob)
                frames[stage] = {"png_sha256": p, "rgba_sha256": r}
            data["cases"].append({
                "case_id": case_id, "language": language,
                "train": False, "dev": False, "ood": False, "frames": frames,
            })
        write_json(receipt, data)
        sources[name] = receipt
    output_folder = tmp / "output"
    output_folder.mkdir()
    return history, historical, sources, output_folder / OUTPUT_NAME


def test_full_synthetic_private_screen(tmp_path):
    history, _, sources, output = fixture(tmp_path)
    result = audit(history, sources, output)
    assert result["candidate_frames_verified_against_actual_png"] == 24
    assert result["candidate_unique_rgba"] == 24
    assert result["known_old_png_collisions"] == 0
    assert result["known_old_decoded_rgba_collisions"] == 0
    assert result["train_admitted"] is False
    assert result["historical_completeness_proven"] is False
    saved = output.read_text(encoding="utf-8")
    assert "png_sha256" not in saved and "rgba_sha256" not in saved
    assert "case_id" not in saved and "source_git_sha" not in saved


def test_same_pixels_reencoded_detected(tmp_path):
    history, historical, sources, output = fixture(tmp_path)
    original = (sources["h250"].parent / "h250-en-before_focus.png").read_bytes()
    assert digest_png(original)[0] != digest_png(png(1, compress=0))[0]
    raw, rgba, _, _ = digest_png(png(1, compress=0))
    historical["owner_only_images"][0] = {
        "phase": "private-h168", "png_sha256": raw, "decoded_rgba_sha256": rgba,
    }
    write_json(history, historical)
    result = audit(history, sources, output)
    assert result["known_old_png_collisions"] == 0
    assert result["known_old_decoded_rgba_collisions"] == 1
    assert result["known_surviving_visual_collision_free"] is False


def test_altered_png_rejected(tmp_path):
    history, _, sources, output = fixture(tmp_path)
    media = sources["h250"].parent / "h250-en-before_focus.png"
    media.write_bytes(png(240))
    with pytest.raises(ValueError):
        audit(history, sources, output)
    assert not output.exists()


def test_candidate_promoted_to_train_rejected(tmp_path):
    history, _, sources, output = fixture(tmp_path)
    data = json.loads(sources["h251"].read_text(encoding="utf-8"))
    data["train"] = True
    write_json(sources["h251"], data)
    with pytest.raises(ValueError):
        audit(history, sources, output)


def test_history_claiming_completeness_rejected(tmp_path):
    history, historical, sources, output = fixture(tmp_path)
    historical["summary"]["complete_old_visual_history"] = True
    write_json(history, historical)
    with pytest.raises(ValueError):
        audit(history, sources, output)


def test_history_count_tamper_rejected(tmp_path):
    history, historical, sources, output = fixture(tmp_path)
    historical["summary"]["distinct_surviving_png_sha"] = 999
    write_json(history, historical)
    with pytest.raises(ValueError):
        audit(history, sources, output)


def test_repeated_candidate_frame_is_counted_not_admitted(tmp_path):
    history, _, sources, output = fixture(tmp_path)
    receipt = sources["h250"]
    data = json.loads(receipt.read_text(encoding="utf-8"))
    first, second = H250_STAGES[:2]
    case_id = data["cases"][0]["case_id"]
    (receipt.parent / (case_id + "-" + second + ".png")).write_bytes(
        (receipt.parent / (case_id + "-" + first + ".png")).read_bytes()
    )
    data["cases"][0]["frames"][second] = data["cases"][0]["frames"][first]
    write_json(receipt, data)
    result = audit(history, sources, output)
    assert result["candidate_duplicate_png_frames"] == 1
    assert result["candidate_duplicate_rgba_frames"] == 1
    assert result["candidate_unique_rgba"] == 23
    assert result["train_admitted"] is False


def test_existing_output_refused(tmp_path):
    history, _, sources, output = fixture(tmp_path)
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(ValueError):
        audit(history, sources, output)
    assert output.read_text(encoding="utf-8") == "existing"


def test_sealed_path_refused(tmp_path):
    history, _, sources, output = fixture(tmp_path)
    forbidden = tmp_path / "h173-sealed"
    forbidden.mkdir()
    with pytest.raises(ValueError):
        audit(forbidden / history.name, sources, output)
    assert not output.exists()


def test_git_checkout_output_refused(tmp_path):
    history, _, sources, output = fixture(tmp_path)
    (output.parent / ".git").mkdir()
    with pytest.raises(ValueError):
        audit(history, sources, output)


def test_missing_media_refused(tmp_path):
    history, _, sources, output = fixture(tmp_path)
    (sources["h251"].parent / "h251-ar-after_scroll.png").unlink()
    with pytest.raises(ValueError):
        audit(history, sources, output)


def test_unexpected_source_family_refused(tmp_path):
    history, _, sources, output = fixture(tmp_path)
    sources["unknown"] = sources["h250"]
    with pytest.raises(ValueError):
        audit(history, sources, output)

def test_sealed_phase_row_refused(tmp_path):
    history, historical, sources, output = fixture(tmp_path)
    historical["owner_only_images"][0]["phase"] = "private-h173"
    write_json(history, historical)
    with pytest.raises(ValueError):
        audit(history, sources, output)
    assert not output.exists()
