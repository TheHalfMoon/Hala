"""H218 all eight pinned processor-file SHA256 adversarial regression tests.

Real files verified separately on owner Windows, this CI is fake bytes
in disposable Python temp directories; no weights, network or owner data.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

from h218.pinned_qwen_processor_bytes import (
    PINNED_PUBLIC_FILES, SNAPSHOT_REVISION, verify_files,
    verify_pinned_snapshot,
)
from h216.real_qwen_processor_alignment import verify_cached_processor_root


class PinnedPublicProcessorByteTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/SNAPSHOT_REVISION
        self.root.mkdir()
        self.synthetic={}
        for i,name in enumerate(sorted(PINNED_PUBLIC_FILES)):
            data=(b"H218 synthetic safe test "+str(i).encode())*(i+1)
            (self.root/name).write_bytes(data)
            self.synthetic[name]=(len(data),sha256(data).hexdigest())

    def good(self):
        return verify_files(self.root,self.synthetic)

    def deny(self,root=None,expected=None):
        with self.assertRaises(ValueError):
            verify_files(self.root if root is None else root,
                         self.synthetic if expected is None else expected)

    def test_eight_unique_bytes_checked_exactly(self):
        result=self.good()
        self.assertEqual(result["verified_files"],8)
        self.assertEqual(result["verified_total_bytes"],
                         sum(len((self.root/name).read_bytes())
                             for name in self.synthetic))
        self.assertFalse(result["model_weights_loaded"])
        self.assertEqual(result["optimizer_steps"],0)

    def test_public_production_manifest_pins_all_known_hf_assets(self):
        self.assertEqual(set(PINNED_PUBLIC_FILES),{
            "chat_template.json","config.json","merges.txt",
            "preprocessor_config.json","tokenizer.json",
            "tokenizer_config.json","video_preprocessor_config.json",
            "vocab.json",
        })
        self.assertEqual(len(PINNED_PUBLIC_FILES),8)
        self.assertEqual(sum(item[0] for item in PINNED_PUBLIC_FILES.values()),
                         5502+1505+1671839+390+7032403+10868+385+2776833)
        with self.assertRaises(TypeError):
            PINNED_PUBLIC_FILES["fake"]=(1,"0"*64)

    def test_modified_file_with_same_length_refused(self):
        p=self.root/"tokenizer.json"
        data=p.read_bytes()
        p.write_bytes(bytes([data[0]^1])+data[1:])
        self.deny()

    def test_truncation_and_growth_refused(self):
        p=self.root/"config.json"
        original=p.read_bytes()
        p.write_bytes(original[:-1])
        self.deny()
        p.write_bytes(original+b"x")
        self.deny()

    def test_removed_file_refused(self):
        (self.root/"merges.txt").unlink()
        self.deny()

    def test_extra_safetensors_weight_refused_without_open(self):
        (self.root/"model-00001-of-00002.safetensors").write_bytes(b"FAKE")
        self.deny()

    def test_extra_unrecognized_directory_refused_without_walk(self):
        (self.root/"owner-private-folder").mkdir()
        self.deny()

    def test_symlink_file_rejected_even_if_digest_matches(self):
        p=self.root/"vocab.json"
        content=p.read_bytes()
        p.unlink()
        target=Path(self.tmp.name)/"public-target.txt"
        target.write_bytes(content)
        try:
            p.symlink_to(target)
        except (OSError, NotImplementedError):
            self.skipTest("test Windows account disallows symlinks")
        self.deny()

    def test_symlink_snapshot_rejected_even_if_all_bytes_match(self):
        other=Path(self.tmp.name)/"alt-revision"
        try:
            other.symlink_to(self.root, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("test Windows account disallows directory links")
        self.deny(root=other)

    def test_wrong_digest_in_trusted_manifest_fails(self):
        e=dict(self.synthetic)
        size,_=e["chat_template.json"]
        e["chat_template.json"]=(size,"0"*64)
        self.deny(expected=e)

    def test_malformed_trusted_spec_and_traversal_refused(self):
        for name in ("../owner", "/absolute","x\\y", ".secret"):
            e=dict(self.synthetic)
            e.pop("chat_template.json")
            e[name]=(1,"a"*64)
            with self.subTest(name=name):
                self.deny(expected=e)
        e=dict(self.synthetic)
        e["merges.txt"]=(True,"a"*64)
        self.deny(expected=e)

    def test_wrong_filename_count_or_path_refused(self):
        self.deny(expected={"file":(1,"a"*64)})
        with self.assertRaises(ValueError):
            verify_files(Path("relative"),self.synthetic)

    def test_pinned_production_rejects_synthetic_bytes(self):
        # Deliberately no fake PASS with the real production digest list.
        with self.assertRaises(ValueError):
            verify_pinned_snapshot(self.root)

    def test_preflight_rejects_untrusted_bytes_before_transformers_import(self):
        with patch.dict(os.environ,{
            "HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"
        }):
            with self.assertRaisesRegex(ValueError,
                                        "public Qwen processor bytes blocked"):
                verify_cached_processor_root(self.root)

    def test_offline_guard_still_refuses_even_correct_shape(self):
        with patch.dict(os.environ,{
            "HF_HUB_OFFLINE":"0","TRANSFORMERS_OFFLINE":"1"
        }):
            with self.assertRaises(ValueError):
                verify_cached_processor_root(self.root)


if __name__=="__main__":
    unittest.main()
