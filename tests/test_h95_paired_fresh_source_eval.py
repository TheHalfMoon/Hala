"""H95: never rewrite strict score, and never evaluate unbound adapter."""
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
from h95 import eval as h95


class FreshPairedSourceGates(unittest.TestCase):
    def test_new_scenes_are_disjoint_from_original_h93_four(self):
        self.assertEqual(h95.START_INDEX, 4)
        self.assertEqual(h95.PER_DOMAIN, 4)
        self.assertEqual(h95.FETCH_COUNT, 8)
        self.assertEqual(h95.DOMAiNS if hasattr(h95, "DOMAiNS") else len(h95.DOMAINS), 4)

    def test_h94_exact_sha_provenance(self):
        self.assertEqual(h95.H94_CHECKPOINT_SHA,
            "c13fb4ce619318075006af3ac24f5fd0ce43757cb5746201c9d595328d3594ac")
        self.assertEqual(h95.H94_ADAPTER_SHA,
            "d73cc9a302782bf7eca08f434a2050a9c3f212368c25afe65014f71319eb673c")
        self.assertEqual(h95.STEPS, 64)

    def test_only_exact_json_fence_stripped_in_separate_diagnostic(self):
        fenced = '```json\n{"type":"drag"}\n```'
        self.assertEqual(h95.normalize_fenced_json(fenced), '{"type":"drag"}')
        for s in ('plain not JSON', '```json\n{"bad":true}\n```more',
                  '{"type":"drag"}', '```javascript\n{}\n```'):
            with self.subTest(s=s):
                self.assertEqual(h95.normalize_fenced_json(s), s)

    def test_strict_parser_still_rejects_fenced_output(self):
        from h61.official_eval import parse_action
        direct = '{"type":"drag","start_point":[10,20],"end_point":[30,40],"coordinate_space":"source_image_pixels","ordered":true}'
        self.assertIsNotNone(parse_action(direct, 1024, 640))
        fenced = '```json\n' + direct + '\n```'
        self.assertIsNone(parse_action(fenced, 1024, 640))
        self.assertIsNotNone(parse_action(
            h95.normalize_fenced_json(fenced), 1024, 640))

    def test_unapproved_provider_cannot_evaluate(self):
        for env, system, work, output in (
            ({}, "Linux", "/kaggle/tmp/h95", "/kaggle/working/h95"),
            ({"KAGGLE_KERNEL_RUN_TYPE":"Batch","HALA_NONCOMMERCIAL_RESEARCH":"0"},
             "Linux", "/kaggle/tmp/h95", "/kaggle/working/h95"),
            ({"KAGGLE_KERNEL_RUN_TYPE":"Batch","HALA_NONCOMMERCIAL_RESEARCH":"1"},
             "Windows", "/kaggle/tmp/h95", "/kaggle/working/h95"),
            ({"KAGGLE_KERNEL_RUN_TYPE":"Batch","HALA_NONCOMMERCIAL_RESEARCH":"1"},
             "Linux", "/tmp/h95", "/kaggle/working/h95")
        ):
            with self.subTest(env=env,system=system), self.assertRaises(ValueError):
                h95.require_private_eval(env=env, system=system,
                    work=work, output=output)

    def test_unattached_h94_checkpoint_denied(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                h95.admit_h94_adapter(Path(td), Path(td))


if __name__ == "__main__":
    unittest.main()
