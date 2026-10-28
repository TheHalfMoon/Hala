"""H173 FREE T4 OOD source accepts ONLY new all-32 no-gold ZIP."""
import hashlib
import io
import json
import zipfile

import pytest
from PIL import Image

from h173 import infer_gpu
from h173.pack_ood import SCHEMA
from h173.source_preflight import preregister_provisional_cases


def sample(tmp_path, monkeypatch, mutate=None):
    raw_scenes = {}
    rows = []
    for n, source in enumerate(preregister_provisional_cases()):
        img = Image.new(
            "RGB", (960, 720), ((n * 7) % 255, (n * 13) % 255, 180)
        )
        data = io.BytesIO()
        img.save(data, "PNG")
        blob = data.getvalue()
        name = f"images/{source['index']}.png"
        raw_scenes[name] = blob
        rows.append({
            **source,
            "split": "OOD",
            "image": name,
            "image_sha256": hashlib.sha256(blob).hexdigest(),
            "instruction": "Close dialog" if n < 16 else "Advance carousel",
        })
    manifest = {
        "schema": SCHEMA,
        "split": "OOD",
        "count": 32,
        "never_previously_used_original_app_families": True,
        "previous_train_dev_ood_excluded": True,
        "owner_gold_uploaded": False,
        "owner_source_manifest_sha256": "a" * 64,
        "independent_rgba_replay_receipt_sha256": "b" * 64,
        "private_cases": rows,
    }
    if mutate is not None:
        mutate(manifest)
    path = tmp_path / "ood-gold-free.pkg"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for name, blob in raw_scenes.items():
            archive.writestr(name, blob)
    monkeypatch.setattr(infer_gpu, "OOD_ZIP_SHA",
                        hashlib.sha256(path.read_bytes()).hexdigest())
    return path


def test_all_32_new_bilingual_ood_scenes_admitted(tmp_path, monkeypatch):
    path = sample(tmp_path, monkeypatch)
    rows = infer_gpu.admit_unseen_ood(path)
    assert len(rows) == 32
    assert len({row["image_sha256"] for row in rows}) == 32
    assert sum(row["language"] == "ar" for row in rows) == 16


def test_real_kernel_cannot_run_with_unqualified_sha(tmp_path, monkeypatch):
    path = sample(tmp_path, monkeypatch)
    monkeypatch.setattr(infer_gpu, "OOD_ZIP_SHA", "UNQUALIFIED")
    with pytest.raises(ValueError):
        infer_gpu.admit_unseen_ood(path)


@pytest.mark.parametrize("mutate", [
    lambda m: m.update(split="TRAIN"),
    lambda m: m.update(owner_gold_uploaded=True),
    lambda m: m.update(previous_train_dev_ood_excluded=False),
    lambda m: m.update(never_previously_used_original_app_families=False),
    lambda m: m.update(independent_rgba_replay_receipt_sha256="x"),
    lambda m: m["private_cases"][0].update(
        training_supervision={"actions": []}
    ),
    lambda m: m["private_cases"][0].update(reference_point=[10, 20]),
    lambda m: m["private_cases"][0].update(image_sha256="0" * 64),
    lambda m: m["private_cases"][0].update(app="Pikaday"),
    lambda m: m["private_cases"][1].update(
        image_sha256=m["private_cases"][0]["image_sha256"]
    ),
])
def test_mutated_ood_source_or_gold_refused(tmp_path, monkeypatch, mutate):
    path = sample(tmp_path, monkeypatch, mutate)
    with pytest.raises(ValueError):
        infer_gpu.admit_unseen_ood(path)


def test_duplicate_zip_member_never_enters_frozen_gpu(tmp_path, monkeypatch):
    """ZipFile names as a SET can hide two different manifest entries."""
    import zipfile
    import hashlib

    path = sample(tmp_path, monkeypatch)
    with zipfile.ZipFile(path) as archive:
        original = archive.read("manifest.json")
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("manifest.json", original)
    monkeypatch.setattr(
        infer_gpu,
        "OOD_ZIP_SHA",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="unique original OOD PNGs"):
        infer_gpu.admit_unseen_ood(path)
