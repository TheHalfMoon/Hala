"""H173 never ships OOD image input without 32/32 pixel replay proof."""
import hashlib
import io
import json
import zipfile

import pytest
from PIL import Image

from h173.capture_ood import MANIFEST
from h173.pack_ood import pack, REPLAY_SCHEMA, SCHEMA
from h173.source_preflight import preregister_provisional_cases


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
    }
    manifest = base / "manifest.json"
    manifest.write_text(json.dumps(doc), encoding="utf8")
    (base / "owner-reference-private.json").write_text(
        '{"owner_reference_actions":[{"point":[12,20]}]}', encoding="utf8"
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
