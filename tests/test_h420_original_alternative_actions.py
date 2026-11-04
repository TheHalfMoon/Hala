"""H420 authentic second native target vs duplicate original action tests."""
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from h420 import curate_alternative_drags as h
from h418 import curate as src

ROOT=Path(__file__).resolve().parents[1]
ZIP="h60-dragon-train-slider-0005.hala.bin"
IMAGE="a"*64
TASK="b"*64


def baseline():
    return {
        "schema":"hala.h418.original-native-drag-train.v1",
        "id":"h60-slider-original-source-first-task",
        "source_domain":"slider",
        "source_zip":ZIP,
        "source_zip_sha256":src.ALL_SHAS[ZIP],
        "split":"train","language":"en",
        "image_path":"images/slider/example-original.jpg",
        "image_sha256":IMAGE,"image_width":1024,"image_height":640,
        "intent":"Drag the original slider to the initial target",
        "original_source_task_sha256":TASK,
        "ordered":True,
        "native_drag_completion":{
            "type":"drag","start_point":[100,300],
            "end_point":[500,300],"ordered":True,
            "coordinate_space":"source_image_pixels"}
    }


def raw_second(endpoint=800,intent=None):
    one=baseline()
    return {
        "id":"h60-slider-original-source-second-task",
        "split":"train","domain":"slider","language":"en",
        "image_path":one["image_path"],"image_sha256":one["image_sha256"],
        "image_width":1024,"image_height":640,
        "intent":intent or "Drag the original slider to another position",
        "ordered":True,"source_task_sha256":TASK,
        "completion":{"type":"drag","start_point":[100,300],
                      "end_point":[endpoint,300],"ordered":True,
                      "coordinate_space":"source_image_pixels"}
    }


class H420AlternativeTargets(unittest.TestCase):
    def test_true_native_other_endpoint_not_duplicate(self):
        rank,case=h.alternative(baseline(),raw_second(),ZIP)
        self.assertGreater(rank[0],0.15)
        self.assertEqual(case["native_drag_completion"]["end_point"],[800,300])
        self.assertEqual(case["h419_first_action_id"],baseline()["id"])
        self.assertTrue(case["different_instruction_text"])
        self.assertTrue(case["no_original_h417_or_h419_dev_image_or_task"])
        self.assertEqual(case["image_sha256"],IMAGE)

    def test_pixel_identical_original_jpeg_alias_uses_actual_action_path(self):
        # H60 original source contains same SHA under two distinct member
        # names; never assume the first task and second task share filename.
        # The actual archive member for the new action must be preserved.
        raw=raw_second()
        raw["image_path"]="images/slider/another-original-path.jpg"
        rank,case=h.alternative(baseline(),raw,ZIP)
        self.assertEqual(case["image_path"],raw["image_path"])
        self.assertEqual(case["image_sha256"],baseline()["image_sha256"])
        self.assertGreater(rank[0],0)
        traversal=dict(raw,image_path="images/slider/../bad.jpg")
        with self.assertRaises(ValueError):
            h.alternative(baseline(),traversal,ZIP)

    def test_same_native_endpoint_with_different_words_is_not_new_training(self):
        self.assertIsNone(h.alternative(baseline(),raw_second(
            endpoint=500,intent="Same target with a different sentence"),ZIP))

    def test_mismatched_original_source_image_or_zip_rejected(self):
        corrupted=dict(raw_second(),image_sha256="c"*64)
        with self.assertRaisesRegex(ValueError,"different source image"):
            h.alternative(baseline(),corrupted,ZIP)
        with self.assertRaisesRegex(ValueError,"different source image"):
            h.alternative(baseline(),raw_second(),
                "h60-dragon-train-sheet-0000.hala.bin")

    def test_fake_click_and_outside_screen_native_endpoint_rejected(self):
        click=raw_second()
        click["completion"]={**click["completion"],"type":"click"}
        with self.assertRaises(ValueError):
            h.alternative(baseline(),click,ZIP)
        outside=raw_second(endpoint=9000)
        with self.assertRaises(ValueError):
            h.alternative(baseline(),outside,ZIP)

    def test_source_receipt_only_has_never_trained_or_released_model(self):
        rank,case=h.alternative(baseline(),raw_second(),ZIP)
        with tempfile.TemporaryDirectory() as tmp:
            state=h.export([case],{"slider":1},{"slider":1},tmp)
            self.assertEqual(state["new_original_distinct_complementary_native_actions"],1)
            self.assertFalse(state["model_training_executed"])
            self.assertFalse(state["external_independent_ood"])
            self.assertFalse(state["weights_public"])
            self.assertFalse(state["model_release_qualified"])
            blob=(Path(tmp)/h.OUTPUT).read_bytes()
            self.assertEqual(state["original_source_second_actions_jsonl_sha256"],
                             sha256(blob).hexdigest())
            rec=(Path(tmp)/h.RECEIPT).read_bytes()
            self.assertEqual((Path(tmp)/"receipt.sha256").read_text().strip(),
                             sha256(rec).hexdigest())
            with self.assertRaises(FileExistsError):
                h.export([case],{},{"slider":1},tmp)

    def test_original_task_isolation_and_prior_30k_source_receipts_pinned(self):
        proof=ROOT/"h419/verified_receipt/h419-owner-source-task-isolation/receipt.json"
        self.assertEqual(sha256(proof.read_bytes()).hexdigest(),h.H419_SOURCE_TASK_AUDIT_SHA)
        original=ROOT/"h418/verified_receipt/h418-owner-original-30k-source/receipt.json"
        self.assertEqual(sha256(original.read_bytes()).hexdigest(),h.H418_RECEIPT_SHA)
        self.assertEqual(h.FROZEN_CANDIDATE_N,29827)
        self.assertEqual(h.FROZEN_COUNT_BY_DOMAIN,
                          {"slider":9914,"slide_resize":9913,"text_highlight":10000})
        self.assertEqual(h.ORIGINAL_NATIVE_ACTIONS,500554)
        self.assertEqual(h.MIN_DISTINCT_ALTERNATIVES,29000)

if __name__=="__main__":
    unittest.main()
