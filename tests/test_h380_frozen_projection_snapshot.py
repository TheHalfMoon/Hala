"""H380: frozen training projection must consume only manifest-bound bytes.

Invented in-memory Mind/Type rows and temporary files only. Does not
admit TRAIN, load models, open media or exercise external OOD.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from training.development_data import (
    FROZEN_PROJECTION_BYTE_LIMIT, _verified_projection_bytes, load_rows,
)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def fixture(root: Path):
    settings = b'{"model":"invented-cpu-only","flags":[]}'
    rows = (
        {"id":"invented-001", "family":"A", "language":"en",
         "component":"mind", "prompt":"Select Save", "completion":"ok",
         "image":None, "image_sha256":None},
        {"id":"invented-002", "family":"B", "language":"ar",
         "component":"type", "prompt":"اختر حفظ", "completion":"تم",
         "image":"images/invented.png", "image_sha256":"a"*64},
    )
    raw_rows = b"".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")).encode("utf-8")+b"\n" for r in rows)
    (root/"settings.json").write_bytes(settings)
    (root/"training-rows.jsonl").write_bytes(raw_rows)
    manifest = {"files":{"settings.json":sha(settings), "training-rows.jsonl":sha(raw_rows)}}
    return manifest, rows


class H380FrozenProjectionTests(unittest.TestCase):
    def test_valid_exact_hash_bound_bilingual_rows_not_changed(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            manifest, rows=fixture(root)
            with patch("training.development_data.verify_bundle", return_value=manifest) as verify, (
                 patch("training.development_data.check_settings") as settings):
                actual, metadata=load_rows(root, "b"*64)
            self.assertEqual(actual[0]["prompt"], rows[0]["prompt"])
            self.assertEqual(actual[1]["prompt"], rows[1]["prompt"])
            self.assertEqual({row["language"] for row in actual},{"en","ar"})
            self.assertEqual(metadata, manifest)
            verify.assert_called_once_with(root, require_media=True, expected_manifest_sha256="b"*64)
            settings.assert_called_once_with({"model":"invented-cpu-only","flags":[]})

    def test_modified_training_rows_after_freeze_verification_denied(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            manifest, _=fixture(root)
            def after_verified(*_args, **_kwargs):
                data=(root/"training-rows.jsonl").read_bytes()
                # In-place same length, different model-visible goal/target.
                (root/"training-rows.jsonl").write_bytes(data.replace(b"Select Save", b"Select Halt"))
                return manifest
            with patch("training.development_data.verify_bundle", side_effect=after_verified), (
                 patch("training.development_data.check_settings") as settings):
                with self.assertRaisesRegex(ValueError,"Frozen projection bytes differ"):
                    load_rows(root,"b"*64)
                settings.assert_not_called()

    def test_modified_settings_after_verification_denied(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            manifest,_=fixture(root)
            def after_verified(*_args, **_kwargs):
                (root/"settings.json").write_bytes(b'{"model":"UNTRUSTED","flags":[]}')
                return manifest
            with patch("training.development_data.verify_bundle", side_effect=after_verified), (
                 patch("training.development_data.check_settings") as settings):
                with self.assertRaisesRegex(ValueError,"Frozen projection bytes differ"):
                    load_rows(root,"b"*64)
                settings.assert_not_called()

    def test_duplicate_json_keys_in_projection_refused(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            manifest,_=fixture(root)
            duplicate=b'{"model":"invented","model":"override"}'
            (root/"settings.json").write_bytes(duplicate)
            manifest["files"]["settings.json"]=sha(duplicate)
            with patch("training.development_data.verify_bundle", return_value=manifest), (
                 patch("training.development_data.check_settings") as settings):
                with self.assertRaisesRegex(ValueError,"Duplicate JSON key"):
                    load_rows(root,"b"*64)
                settings.assert_not_called()

    def test_oversize_projection_refused_without_processing(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            path=root/"training-rows.jsonl"
            with path.open("wb") as handle:
                handle.seek(FROZEN_PROJECTION_BYTE_LIMIT)
                handle.write(b"x")
            manifest={"files":{"training-rows.jsonl":"a"*64}}
            with self.assertRaisesRegex(ValueError,"Frozen projection bytes differ"):
                _verified_projection_bytes(root,manifest,"training-rows.jsonl")

    def test_unknown_path_invalid_manifest_and_symlink_refused(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            manifest,_=fixture(root)
            with self.assertRaisesRegex(ValueError,"Unregistered"):
                _verified_projection_bytes(root,manifest,"other.json")
            manifest["files"]["settings.json"]="not-a-hash"
            with self.assertRaisesRegex(ValueError,"Invalid freeze"):
                _verified_projection_bytes(root,manifest,"settings.json")
            source=root/"training-rows.jsonl"
            shortcut=root/"alias.json"
            try:
                shortcut.symlink_to(source)
            except (OSError, NotImplementedError):
                return  # Symlinks need admin privileges on some Windows hosts.
            # Force a registered filename to be a link, never follow it.
            source.unlink()
            source.symlink_to(shortcut)
            manifest["files"]["training-rows.jsonl"]="a"*64
            with self.assertRaisesRegex(ValueError,"Symlinked freeze"):
                _verified_projection_bytes(root,manifest,"training-rows.jsonl")


if __name__ == "__main__":
    unittest.main()
