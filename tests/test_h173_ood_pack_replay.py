"""H173 never ships OOD image input without 32/32 pixel replay proof."""
import hashlib
import io
import json
import zipfile

import pytest
from PIL import Image

from h173.capture_ood import (
    MANIFEST, OWNER, SLIDERS,
    CAPTURE_ATTEMPT_LOCK, REPLAY_ATTEMPT_LOCK,
    CAPTURE_LOCK_CONTENT, REPLAY_LOCK_CONTENT,
)
from h173.pack_ood import (
    pack, REPLAY_SCHEMA, SCHEMA, PACK_ATTEMPT_LOCK,
)
from h173.source_preflight import SOURCE_PINS, preregister_provisional_cases


def sha(content):
    return hashlib.sha256(content).hexdigest()


def source(tmp_path):
    base = tmp_path / "source"
    (base / "images").mkdir(parents=True)
    rows = []
    for i, spec in enumerate(preregister_provisional_cases()):
        image = Image.new(
            "RGB", (960, 720),
            (i * 7 % 256, i * 13 % 256, i * 19 % 256)
        )
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data = buf.getvalue()
        name = f"images/{spec['index']}.png"
        (base / name).write_bytes(data)
        rows.append({
            **spec, "split": "OOD", "image": name,
            "image_sha256": sha(data),
            "instruction": "Close original dialog" if i < 16
                else "Advance original carousel",
            "original_reference_success": True,
        })
    doc = {
        "schema": MANIFEST, "split": "OOD", "count": 32,
        "no_old_train_dev_ood_images": True,
        "no_model_calls_before_frozen_capture": True,
        "owner_gold_uploaded": False, "private_cases": rows,
        "source_proof": {
            app: {
                "original_git_sha": pin["git_sha"],
                "original_html_sha256": pin["original_html_sha256"],
                "source_license": "MIT",
                "action_type": pin["action"],
                "native_state_family": pin["family"],
                "source_role": "OOD_CANDIDATE_NOT_EVALUATED",
                "model_gpu_calls": 0,
                "reference_labels_uploaded": False,
            }
            for app, pin in SOURCE_PINS.items()
        },
    }
    manifest = base / "manifest.json"
    manifest.write_text(json.dumps(doc), encoding="utf8")
    reference = {
        "schema": OWNER,
        "owner_reference_actions": [],
    }
    for spec in preregister_provisional_cases():
        if spec["app"] == "tingle":
            state = {"kind": "modal_visible", "value": True}
        else:
            previous = spec["variant"] % 2 == 0
            start = 4
            state = {
                "kind": "carousel_index",
                "name": SLIDERS[spec["variant"] // 2],
                "direction": "prev" if previous else "next",
                "reference_control_hit": "prev" if previous else "next",
                "value": start,
                "reference_after_index": start - 1 if previous else start + 1,
            }
        reference["owner_reference_actions"].append({
            "index": spec["index"], "kind": "click",
            "point": [12, 20], "prestate": state,
            "original_native_reference_success": True,
        })
    (base / "owner-reference-private.json").write_text(
        json.dumps(reference), encoding="utf8"
    )
    receipt = {
        "schema": REPLAY_SCHEMA,
        "status": "H173_PRIVATE_NEW_OOD_32_EXACT_PIXEL_REPLAY_PASS",
        "source_manifest_sha256": sha(manifest.read_bytes()),
        "cases": 32,
        "independent_rgba_pixel_replays": 32,
        "prior_train_dev_ood_excluded": True,
        "owner_reference_actions_not_in_model_input": True,
        "model_calls": 0,
        "optimizer_steps": 0,
    }
    (base / "owner-source-replay-receipt.json").write_text(
        json.dumps(receipt), encoding="utf8"
    )
    (base / CAPTURE_ATTEMPT_LOCK).write_text(
        CAPTURE_LOCK_CONTENT, encoding="utf8"
    )
    (base / REPLAY_ATTEMPT_LOCK).write_text(
        REPLAY_LOCK_CONTENT, encoding="utf8"
    )
    return base, doc, receipt


def test_new_ood_zip_is_only_32_images_and_gold_free(tmp_path):
    base, _, _ = source(tmp_path)
    target = tmp_path / "private-ood.zip"
    result = pack(base, target)
    assert result["model_calls"] == 0
    assert result["ood_cases"] == 32
    with zipfile.ZipFile(target) as archive:
        assert len(archive.namelist()) == 33
        assert all("owner" not in name and "reference" not in name
                   for name in archive.namelist())
        record = json.loads(archive.read("manifest.json"))
        assert record["schema"] == SCHEMA
        assert record["owner_gold_uploaded"] is False
        assert all("reference" not in str(row).lower()
                   for row in record["private_cases"])


def test_missing_independent_pixel_receipt_never_packages(tmp_path):
    base, _, _ = source(tmp_path)
    (base / "owner-source-replay-receipt.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        pack(base, tmp_path / "invalid.zip")


@pytest.mark.parametrize("key,value", [
    ("independent_rgba_pixel_replays", 31),
    ("cases", 31),
    ("model_calls", 1),
    ("optimizer_steps", 1),
    ("prior_train_dev_ood_excluded", False),
    ("owner_reference_actions_not_in_model_input", False),
    ("source_manifest_sha256", "f" * 64),
])
def test_mutated_pixel_receipt_fails_closed(tmp_path, key, value):
    base, _, receipt = source(tmp_path)
    receipt[key] = value
    (base / "owner-source-replay-receipt.json").write_text(
        json.dumps(receipt), encoding="utf8"
    )
    with pytest.raises(ValueError):
        pack(base, tmp_path / "invalid.zip")


@pytest.mark.parametrize("change", [
    lambda d: d.update(split="TRAIN"),
    lambda d: d.update(owner_gold_uploaded=True),
    lambda d: d["private_cases"][0].update(training_supervision={"click": 3}),
    lambda d: d["private_cases"][0].update(image_sha256="0" * 64),
    lambda d: d["private_cases"][0].update(original_reference_success=False),
    lambda d: d["private_cases"][0].update(app="pikaday"),
    lambda d: d["private_cases"][1].update(
        image_sha256=d["private_cases"][0]["image_sha256"]
    ),
])
def test_modified_or_duplicate_unseen_ood_fails_closed(tmp_path, change):
    base, doc, receipt = source(tmp_path)
    change(doc)
    manifest = base / "manifest.json"
    manifest.write_text(json.dumps(doc), encoding="utf8")
    receipt["source_manifest_sha256"] = sha(manifest.read_bytes())
    (base / "owner-source-replay-receipt.json").write_text(
        json.dumps(receipt), encoding="utf8"
    )
    with pytest.raises(ValueError):
        pack(base, tmp_path / "invalid.zip")


def test_visually_identical_images_with_different_png_encoding_refused(tmp_path):
    """Both raw file SHAs differ, yet exactly identical original pixels."""
    base, doc, receipt = source(tmp_path)
    from PIL import Image

    p0 = base / doc["private_cases"][0]["image"]
    p1 = base / doc["private_cases"][1]["image"]
    with Image.open(p0) as image:
        image.save(p1, format="PNG", compress_level=0)
    p0_raw = p0.read_bytes()
    p1_raw = p1.read_bytes()
    assert p0_raw != p1_raw
    assert sha(p0_raw) != sha(p1_raw)
    doc["private_cases"][1]["image_sha256"] = sha(p1_raw)
    manifest = base / "manifest.json"
    manifest.write_text(json.dumps(doc), encoding="utf8")
    receipt["source_manifest_sha256"] = sha(manifest.read_bytes())
    (base / "owner-source-replay-receipt.json").write_text(
        json.dumps(receipt), encoding="utf8"
    )
    with pytest.raises(ValueError, match="visual pixels"):
        pack(base, tmp_path / "visually-duplicated-ood.zip")


@pytest.mark.parametrize("mutation", [
    lambda d: d["owner_reference_actions"].pop(),
    lambda d: d["owner_reference_actions"][0].update(point=[960, 1]),
    lambda d: d["owner_reference_actions"][0].update(point=[True, 2]),
    lambda d: d["owner_reference_actions"][0].update(original_native_reference_success=False),
    lambda d: d["owner_reference_actions"][0].update(index=173101),
    lambda d: d["owner_reference_actions"][16]["prestate"].update(direction="wrong"),
    lambda d: d["owner_reference_actions"][16]["prestate"].update(reference_after_index=4),
    # Reject wrong native button, even if the index happened to change.
    lambda d: d["owner_reference_actions"][16]["prestate"].update(reference_control_hit="next"),
    lambda d: d["owner_reference_actions"][17]["prestate"].update(reference_control_hit="prev"),
    lambda d: d["owner_reference_actions"][16]["prestate"].pop("reference_control_hit"),
])
def test_pack_rejects_missing_or_forged_owner_reference(tmp_path, mutation):
    base, _, _ = source(tmp_path)
    private_path = base / "owner-reference-private.json"
    data = json.loads(private_path.read_text("utf8"))
    mutation(data)
    private_path.write_text(json.dumps(data), encoding="utf8")
    with pytest.raises(ValueError):
        pack(base, tmp_path / "invalid-source-owner-gold.zip")


@pytest.mark.parametrize("mutation", [
    lambda d: d["source_proof"]["tingle"].update(original_git_sha="a"*40),
    lambda d: d["source_proof"]["tiny-slider"].update(reference_labels_uploaded=True),
    lambda d: d.pop("source_proof"),
])
def test_pack_rejects_fake_pristine_original_source_evidence(tmp_path, mutation):
    base, doc, receipt = source(tmp_path)
    mutation(doc)
    source_manifest = base / "manifest.json"
    source_manifest.write_text(json.dumps(doc), encoding="utf8")
    receipt["source_manifest_sha256"] = sha(source_manifest.read_bytes())
    (base / "owner-source-replay-receipt.json").write_text(
        json.dumps(receipt), encoding="utf8"
    )
    with pytest.raises(ValueError):
        pack(base, tmp_path / "invalid-source-pins.zip")


@pytest.mark.parametrize("marker", [
    CAPTURE_ATTEMPT_LOCK,
    REPLAY_ATTEMPT_LOCK,
])
def test_pack_refuses_receipt_without_original_one_shot_attempt_marker(tmp_path, marker):
    base, _, _ = source(tmp_path)
    (base / marker).unlink()
    target = tmp_path / "no-attempt-provenance.zip"
    with pytest.raises(ValueError, match="attempt marker"):
        pack(base, target)
    assert not target.exists()
    assert not (base / PACK_ATTEMPT_LOCK).exists()


@pytest.mark.parametrize("marker", [
    CAPTURE_ATTEMPT_LOCK,
    REPLAY_ATTEMPT_LOCK,
])
def test_pack_refuses_altered_attempt_marker(tmp_path, marker):
    base, _, _ = source(tmp_path)
    (base / marker).write_text("FORGED_OR_TRUNCATED\n", encoding="utf8")
    with pytest.raises(ValueError, match="attempt marker"):
        pack(base, tmp_path / "altered-marker.zip")
    assert not (base / PACK_ATTEMPT_LOCK).exists()


def test_pack_same_private_source_cannot_be_frozen_twice_under_new_filename(tmp_path):
    base, _, _ = source(tmp_path)
    first = tmp_path / "first-ood.zip"
    second = tmp_path / "second-ood.zip"
    assert pack(base, first)["ood_cases"] == 32
    assert (base / PACK_ATTEMPT_LOCK).is_file()
    with pytest.raises(FileExistsError, match="already attempted"):
        pack(base, second)
    assert first.is_file()
    assert not second.exists()


def test_pack_crash_after_source_reservation_permanently_blocks_retry(
    tmp_path, monkeypatch
):
    base, _, _ = source(tmp_path)
    import h173.pack_ood as module

    def injected_archive_failure(*args, **kwargs):
        raise OSError("injected archive open failure")

    with monkeypatch.context() as patch:
        patch.setattr(module.zipfile, "ZipFile", injected_archive_failure)
        with pytest.raises(OSError, match="injected archive"):
            pack(base, tmp_path / "first-attempt.zip")
    assert (base / PACK_ATTEMPT_LOCK).is_file()
    with pytest.raises(FileExistsError, match="already attempted"):
        pack(base, tmp_path / "retry-after-crash.zip")
    assert not (tmp_path / "retry-after-crash.zip").exists()
