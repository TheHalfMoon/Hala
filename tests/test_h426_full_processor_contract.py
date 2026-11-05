"""Synthetic falsifications for the image-expanded original H425 token boundary."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "h426_full_processor", ROOT/"h426/full_processor_contract.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class FakeTokenizer:
    def convert_tokens_to_ids(self, token):
        return 999999 if token == "<|image_pad|>" else -1

    def decode(self, ids, skip_special_tokens=False):
        if skip_special_tokens:
            raise AssertionError("No special token stripping allowed")
        return "".join(chr(x) for x in ids)


class FakeImageProcessor:
    tokenizer = FakeTokenizer()

    def __init__(self, *, shift=False, omit_patch=False, leak_image=False,
                 hide_grid=False, masked_answer=False, malformed_grid=False, drift_grid=False):
        self.shift = shift
        self.omit_patch = omit_patch
        self.leak_image = leak_image
        self.hide_grid = hide_grid
        self.masked_answer = masked_answer
        self.malformed_grid = malformed_grid
        self.drift_grid = drift_grid

    def apply_chat_template(self, msgs, *, tokenize, add_generation_prompt):
        if tokenize:
            raise AssertionError("Input expected as formatted text")
        if add_generation_prompt:
            return "PREFIX"
        return "PREFIX" + msgs[-1]["content"][0]["text"]

    def __call__(self, *, text, images, return_tensors):
        assert len(images) == 1 and return_tensors == "pt"
        raw = text[0]
        prefix = [80, 82, 69, 70, 73, 88]
        if not self.omit_patch:
            prefix.insert(3, 999999)
        is_full = raw != "PREFIX"
        if self.shift and is_full:
            prefix[0] = 79
        tokens = prefix
        if is_full:
            tokens += [ord(x) for x in raw[len("PREFIX"):]]
        if is_full and self.leak_image:
            tokens += [999999]
        attention = [1] * len(tokens)
        if is_full and self.masked_answer:
            attention[-1] = 0
        out = {"input_ids": [tokens], "attention_mask": [attention],
               "pixel_values": [[0.5]]}
        if not self.hide_grid:
            out["image_grid_thw"] = [[1, 0, 2]] if self.malformed_grid else ([[1, 3, 2]] if self.drift_grid and is_full else [[1, 2, 2]])
        return out


def answer():
    return json.dumps({"type": "drag", "start_point": [100, 250],
                       "end_point": [500, 750], "ordered": True,
                       "coordinate_space": "normalized_1000"},
                      separators=(",", ":"))


class FullImageBoundaryTests(unittest.TestCase):
    def test_synthetic_expanded_image_boundary_and_weights(self):
        r = c.check(FakeImageProcessor(), object(), "synthetic", answer(), "slide_resize")
        self.assertEqual(r["image_patch_tokens"], 1)
        self.assertEqual(r["weighted_coordinate_tokens"], 12)
        self.assertFalse(r["new_model_trained"])
        self.assertFalse(r["model_release_qualified"])

    def test_shifted_image_expanded_prefix_rejected(self):
        with self.assertRaisesRegex(ValueError, "prefix drift"):
            c.check(FakeImageProcessor(shift=True), object(), "synthetic", answer(), "slider")

    def test_missing_image_patch_rejected(self):
        with self.assertRaisesRegex(ValueError, "image patch"):
            c.check(FakeImageProcessor(omit_patch=True), object(), "synthetic", answer(), "slider")

    def test_image_token_leaking_into_supervision_rejected(self):
        with self.assertRaisesRegex(ValueError, "supervised"):
            c.check(FakeImageProcessor(leak_image=True), object(), "synthetic", answer(), "slider")

    def test_missing_image_grid_rejected(self):
        with self.assertRaisesRegex(ValueError, "patch/grid"):
            c.check(FakeImageProcessor(hide_grid=True), object(), "synthetic", answer(), "slider")

    def test_malformed_grid_rejected(self):
        with self.assertRaisesRegex(ValueError, "grid malformed"):
            c.check(FakeImageProcessor(malformed_grid=True), object(), "synthetic", answer(), "slider")

    def test_grid_drift_rejected(self):
        with self.assertRaisesRegex(ValueError, "grid drift"):
            c.check(FakeImageProcessor(drift_grid=True), object(), "synthetic", answer(), "slider")

    def test_masked_final_answer_rejected(self):
        with self.assertRaisesRegex(ValueError, "prefix drift"):
            c.check(FakeImageProcessor(masked_answer=True), object(), "synthetic", answer(), "slider")

    def test_invalid_target_rejected_by_source_loss(self):
        with self.assertRaises(ValueError):
            c.check(FakeImageProcessor(), object(), "synthetic", "not json", "slider")

    def test_no_runtime_cuda_or_dataset_egress(self):
        code = (ROOT/"h426/full_processor_contract.py").read_text("utf8")
        self.assertNotIn("kaggle kernels push", code)
        self.assertNotIn("model.generate(", code)
        self.assertNotIn("cuda:0", code)
        self.assertIn("local_files_only=True", code)
        self.assertIn("new_model_trained", code)


if __name__ == "__main__":
    unittest.main()
