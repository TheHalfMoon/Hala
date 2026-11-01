"""H308 invented-only RGB identity tests; no user media or model calls."""
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image, PngImagePlugin

from h285.train_only_browser_candidates import ROLE, SOURCE_BLOB
from h307 import one_shot_train_candidate_handoff as unit


def frame(index, *, alpha=255, size=(960, 720), metadata=False):
    with BytesIO() as output:
        info = None
        if metadata:
            info = PngImagePlugin.PngInfo()
            info.add_text("invented-only", "re-encoded-with-metadata")
        Image.new("RGBA", size, (index, 19, 33, alpha)).save(
            output, format="PNG", pnginfo=info)
        return output.getvalue()


def proof():
    return {
        "status": "H286_RELATIVE_ALIGNED_FIRSTPARTY_TRAIN_CANDIDATES_ONLY",
        "original_source_blob": SOURCE_BLOB,
        "role": ROLE,
        "actual_browser_supervisions": 24,
        "relative_targets_roundtrip_verified": 24,
        "english_candidates": 12,
        "arabic_candidates": 12,
        "real_model_generations": 0,
        "optimizer_steps": 0,
        "official_train_admitted": False,
        "source_disjoint_blind_ood": False,
        "adapted_gui_gain_proven": False,
        "media_or_labels_saved_to_public_output": False,
    }


def producer(*, replacement=None):
    def run(_browser, _source, sink, *, role):
        assert role == ROLE
        for i in range(24):
            png = replacement if i == 1 and replacement is not None else frame(i)
            sink(png, "invented-only prompt", "invented-only completion")
        return proof()
    return run


class ModelVisibleTrainIdentityTests(unittest.TestCase):
    def handoff(self, replacement=None):
        calls = []
        def consumer(batch):
            calls.append(len(batch))
            return True
        with patch.object(unit, "produce_relative_train_candidates",
                          side_effect=producer(replacement=replacement)):
            try:
                result = unit.handoff_train_candidate_batch_once(
                    object(), Path("invented/original.html"), consumer,
                    role=ROLE)
                return result, calls
            except unit.CandidateHandoffRefused as error:
                return str(error), calls

    def test_24_distinct_model_visible_rgb_frames_single_consumer(self):
        result, calls = self.handoff()
        self.assertEqual(calls, [24])
        self.assertEqual(result["candidate_count"], 24)
        self.assertFalse(result["official_train_admitted"])

    def test_different_alpha_same_rgb_refused_before_consumer(self):
        result, calls = self.handoff(frame(0, alpha=60))
        self.assertEqual(result, "H307 SOURCE_BATCH_FAILED")
        self.assertEqual(calls, [])

    def test_metadata_only_png_reencode_refused_before_consumer(self):
        result, calls = self.handoff(frame(0, metadata=True))
        self.assertEqual(result, "H307 SOURCE_BATCH_FAILED")
        self.assertEqual(calls, [])

    def test_wrong_viewport_refused_before_consumer(self):
        result, calls = self.handoff(frame(1, size=(959, 720)))
        self.assertIn("SOURCE_BATCH_FAILED", result)
        self.assertEqual(calls, [])

    def test_truncated_png_refused_before_consumer(self):
        result, calls = self.handoff(frame(1)[:200])
        self.assertIn("SOURCE_BATCH_FAILED", result)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
