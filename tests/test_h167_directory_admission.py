"""Kaggle auto-unpacked PRIVATE H164 adapter must be admitted by exact file SHA."""
import hashlib
import json
from pathlib import Path
import pytest
import h167.combined_probe as m

def sha(blob):
    return hashlib.sha256(blob).hexdigest()

def owner_like_files(tmp_path,monkeypatch):
    source=tmp_path/"owner-research-adapter"
    source.mkdir()
    objects={
        "README.md":b"owner private research-only license metadata",
        "adapter_model.safetensors":b"dummy bytes: NOT a real torch model",
        "adapter_config.json":json.dumps({
            "r":8,"peft_type":"LORA","inference_mode":True,
            "base_model_name_or_path":"patrickballou/gui-assist-grounder",
            "target_modules":["q_proj"]}).encode()}
    for name,blob in objects.items():
        (source/name).write_bytes(blob)
    monkeypatch.setattr(m,"LANGUAGE_README_SHA",sha(objects["README.md"]))
    monkeypatch.setattr(m,"LANGUAGE_WEIGHTS_SHA",sha(objects["adapter_model.safetensors"]))
    monkeypatch.setattr(m,"LANGUAGE_CONFIG_SHA",sha(objects["adapter_config.json"]))
    return source

def test_admit_only_three_hash_identical_extracted_files(tmp_path,monkeypatch):
    source=owner_like_files(tmp_path,monkeypatch)
    extracted=tmp_path/"copy"
    assert m.admit_frozen_language_directory(source,extracted)==extracted
    assert {p.name for p in extracted.iterdir()}==m.EXPECTED_ZIP_FILES
    for p in extracted.iterdir():
        assert p.read_bytes()==(source/p.name).read_bytes()

@pytest.mark.parametrize("target",[
    "README.md","adapter_config.json","adapter_model.safetensors"])
def test_reject_any_modified_original_adapter_file(tmp_path,monkeypatch,target):
    source=owner_like_files(tmp_path,monkeypatch)
    (source/target).write_bytes((source/target).read_bytes()+b"X")
    dest=tmp_path/"copy"
    with pytest.raises(ValueError):
        m.admit_frozen_language_directory(source,dest)
    assert not dest.exists()

def test_reject_extra_media_even_when_adapter_files_match(tmp_path,monkeypatch):
    source=owner_like_files(tmp_path,monkeypatch)
    (source/"dev-image.png").write_bytes(b"forbidden")
    with pytest.raises(ValueError):
        m.admit_frozen_language_directory(source,tmp_path/"copy")

def test_reject_preexisting_destination_and_missing_source(tmp_path,monkeypatch):
    source=owner_like_files(tmp_path,monkeypatch)
    dest=tmp_path/"copy"
    dest.mkdir()
    with pytest.raises(ValueError):
        m.admit_frozen_language_directory(source,dest)
    (source/"adapter_config.json").unlink()
    with pytest.raises(ValueError):
        m.admit_frozen_language_directory(source,tmp_path/"new")
