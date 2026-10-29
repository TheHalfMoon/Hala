"""H215 strict research-source PNG-byte vs actual-RGBA distinction tests."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import sys
import json
import unittest

from h212.procedural_native_gui_candidates import candidate_manifest
from h215.portable_png_rgba_gate import (
    FROZEN_WINDOWS_JSON_SHA256, verify_linux_frozen_raster,
)


class LinuxVisualParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p = Path(__file__).resolve().parents[1] / "h212" / (
            "RESEARCH_CANDIDATE_METADATA_NOT_ADMITTED.json"
        )
        cls.frozen = p.read_bytes()
        cls.frozen_doc = json.loads(cls.frozen.decode("utf-8"))
        cls.observed = candidate_manifest()

    def test_h212_original_frozen_windows_metadata_bytes_are_pinned(self):
        self.assertEqual(sha256(self.frozen.replace(b"\r\n", b"\n")).hexdigest(),
                         FROZEN_WINDOWS_JSON_SHA256)
        self.assertEqual(len(self.observed["cases"]),96)

    def test_owner_windows_bit_identical_source_is_accepted(self):
        result = verify_linux_frozen_raster(self.frozen, self.observed)
        self.assertEqual(result["candidate_cases_checked"], 96)
        # Observed OS differences: Windows PNG matches 96/96,
        # Linux PNG 0/96, while raw RGBA matches 96/96 on both.
        self.assertIn(sys.platform, ("win32", "linux"))
        expected_png = 96 if sys.platform == "win32" else 0
        self.assertEqual(result["png_sha_matches"],expected_png)
        self.assertEqual(result["png_encoding_differences"],96-expected_png)
        self.assertEqual(result["rgba_sha_matches"],96)
        self.assertTrue(result["frozen_manifest_sha256_pinned"])
        self.assertFalse(result["official_train_admitted"])
        self.assertFalse(result["official_ood_evaluated"])

    def test_simulated_linux_codec_bytes_differ_but_rgba_exact(self):
        # A MOCK CODEC delta, not claiming actual Ubuntu inference here.
        rendered = deepcopy(self.observed)
        for idx, row in enumerate(rendered["cases"]):
            row["image_png_sha256"] = sha256(
                f"independent valid PNG stream {idx}".encode()
            ).hexdigest()
        result = verify_linux_frozen_raster(self.frozen, rendered)
        self.assertEqual(result["png_sha_matches"],0)
        self.assertEqual(result["png_encoding_differences"],96)
        self.assertEqual(result["rgba_sha_matches"],96)

    def deny(self, frozen=None, generated=None):
        with self.assertRaises(ValueError):
            verify_linux_frozen_raster(
                self.frozen if frozen is None else frozen,
                self.observed if generated is None else generated,
            )

    def test_frozen_png_hash_change_denied_even_if_linux_encoding_differs(self):
        # Mutate the original frozen Windows PNG hash, not Linux codec output.
        sha=self.frozen_doc["cases"][0]["image_png_sha256"].encode()
        self.deny(frozen=self.frozen.replace(sha, b"0"*64,1))

    def test_frozen_rgba_hash_change_denied(self):
        sha=self.observed["cases"][0]["image_decoded_rgba_sha256"].encode()
        self.deny(frozen=self.frozen.replace(sha, b"e"*64,1))

    def test_actual_rgba_change_is_never_masked_by_linux_png_exception(self):
        source=deepcopy(self.observed)
        source["cases"][12]["image_decoded_rgba_sha256"] = "0"*64
        self.deny(generated=source)

    def test_actual_bilingual_language_or_role_change_denied(self):
        for field,value in (("language","xx"),("role","TRAIN"),
                            ("family","owner private"),("index",200)):
            with self.subTest(field=field):
                source=deepcopy(self.observed)
                source["cases"][41][field]=value
                self.deny(generated=source)

    def test_actual_generator_source_sha_change_denied(self):
        source=deepcopy(self.observed)
        source["generator_source_sha256"]="a"*64
        self.deny(generated=source)

    def test_training_approval_forgery_cannot_pass_source_parity(self):
        for field in ("official_admitted","dev_gold_sealed",
                      "official_ood_evaluated","historical_source_history_complete",
                      "old_rgba_history_complete"):
            with self.subTest(field=field):
                source=deepcopy(self.observed)
                source[field]=True
                self.deny(generated=source)

    def test_repeated_generated_png_digest_denied(self):
        source=deepcopy(self.observed)
        source["cases"][1]["image_png_sha256"] = (
            source["cases"][0]["image_png_sha256"])
        self.deny(generated=source)

    def test_extra_key_removed_key_and_reordered_cases_denied(self):
        for mode in ("extra","remove","swap"):
            with self.subTest(mode=mode):
                source=deepcopy(self.observed)
                if mode=="extra":source["cases"][0]["gold"]=[2,3]
                elif mode=="remove":source["cases"][0].pop("family")
                else:source["cases"][0],source["cases"][1]=(
                    source["cases"][1],source["cases"][0])
                self.deny(generated=source)

    def test_truncated_oversized_and_incorrect_frozen_input_denied(self):
        self.deny(frozen=self.frozen[:-30])
        self.deny(frozen=self.frozen+b" "*70000)
        self.deny(frozen="not bytes")

    def test_true_conclusion_stays_research_not_train(self):
        r=verify_linux_frozen_raster(self.frozen,self.observed)
        self.assertEqual(r["optimizer_steps"],0)
        self.assertEqual(r["model_calls"],0)
        self.assertEqual(r["owner_private_media_read"],0)
        self.assertFalse(r["used_consumed_h173_ood"])


if __name__ == "__main__":
    unittest.main()
