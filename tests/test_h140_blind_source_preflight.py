"""Verify source-only one-shot OOD preflight contract without browser."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from h140.blind_source_preflight import (
    SOURCES,check_original_source,original_seed_for,is_moved,SCHEMA,
)


class H140IndependentOriginalOODSourceTests(unittest.TestCase):
    def test_independent_mit_revisions_and_no_training_capture_claims(self):
        self.assertEqual(len(SOURCES),2)
        self.assertNotEqual(SOURCES["ood-vanilla-complete"]["sha"],
                            SOURCES["ood-second-drag"]["sha"])
        self.assertTrue(SCHEMA.endswith(".v1"))
        for name,policy in SOURCES.items():
            with tempfile.TemporaryDirectory() as scratch:
                root=Path(scratch)
                (root/"LICENSE").write_text("MIT License\nOriginal attribution",encoding="utf8")
                for f in policy["files"]:
                    (root/f).write_text("original unmodified source",encoding="utf8")
                with patch("h140.blind_source_preflight.subprocess.check_output",
                           side_effect=[policy["sha"]+"\n",""]):
                    self.assertEqual(check_original_source(root,policy),policy["sha"])
                with patch("h140.blind_source_preflight.subprocess.check_output",
                           side_effect=[policy["sha"]+"\n","modified file.js"]):
                    with self.assertRaises(ValueError):
                        check_original_source(root,policy)
                self.assertIn("localStorage",original_seed_for(name))
                self.assertNotIn("blind_bbox",original_seed_for(name))

    def test_real_owner_state_transition_must_be_observed(self):
        true_a=[{"id":"h140_source_qa","status":"in-progress"}]
        wrong_a=[{"id":"h140_source_qa","status":"todo"}]
        self.assertTrue(is_moved("ood-vanilla-complete",json.dumps(true_a)))
        self.assertFalse(is_moved("ood-vanilla-complete",json.dumps(wrong_a)))
        true_b=[{"tasks":[{"id":"h140_source_qa","status":"inprogress"}]}]
        wrong_b=[{"tasks":[{"id":"h140_source_qa","status":"todo"}]}]
        self.assertTrue(is_moved("ood-second-drag",json.dumps(true_b)))
        self.assertFalse(is_moved("ood-second-drag",json.dumps(wrong_b)))
        with self.assertRaises(ValueError):
            is_moved("another source",json.dumps(true_a))


if __name__=="__main__":
    unittest.main()
