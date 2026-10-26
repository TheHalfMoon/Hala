"""One-shot H140 blind source seal gates, CPU only; never neural training."""
import json
import tempfile
import unittest
from pathlib import Path
from h140.freeze_blind import SOURCES,CANDIDATES,seal,admit_source


class H140SealedOriginalOODTests(unittest.TestCase):
    def test_two_completely_independent_blind_sources_preregistered(self):
        self.assertEqual(len(CANDIDATES),2)
        self.assertEqual(set(CANDIDATES),set(SOURCES))
        self.assertNotEqual(SOURCES["ood-vanilla-complete"]["sha"],
                            SOURCES["ood-second-drag"]["sha"])
        self.assertEqual(set(CANDIDATES.values()),
                         {"blind-candidate-aisha16-v1",
                          "blind-candidate-michael16-v3"})

    def test_refuse_missing_original_train_dev_seal(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"private-h140"
            p.mkdir()
            with self.assertRaisesRegex(ValueError,"create-only"):
                seal(p)
            (p/"frozen-source-v1").mkdir()
            (p/"frozen-source-v1"/"blind_test").mkdir()
            with self.assertRaisesRegex(ValueError,"create-only"):
                seal(p)

    def test_unregistered_or_model_evaluated_blind_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"private-h140"
            p.mkdir()
            d=p/CANDIDATES["ood-vanilla-complete"]
            d.mkdir()
            (d/"manifest.json").write_text(json.dumps({
                "status":"PRIVATE_ORIGINAL_BLIND_SOURCE_GEOMETRY_READY",
                "private":True,"model_neural_calls":1,
                "blind_model_evaluated":True,"cases":[]
            }),encoding="utf8")
            with self.assertRaisesRegex(ValueError,"truly independent"):
                admit_source(p,"ood-vanilla-complete",set(),set())


if __name__=="__main__":
    unittest.main()
