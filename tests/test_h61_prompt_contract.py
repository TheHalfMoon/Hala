"""H61 tests for the one shared native drag prompt contract.

The training curriculum and the public DEV evaluation must use the same
contract, otherwise a measured training gain could be a prompt mismatch. These
tests assert the contract against the H60 public DEV runner itself.
"""
import unittest
from pathlib import Path

from h61.prompt_contract import CONTRACT_SUFFIX, PROMPT_REVISION, drag_prompt, source_body

REPO = Path(__file__).resolve().parents[1]


class PromptContractTests(unittest.TestCase):
    def test_drag_prompt_matches_the_h60_public_dev_prompt(self):
        body = ("Drag the fictional interface slider to 27%. "
                "Return only ONE ordered drag action in original-image pixel coordinates.")
        expected = (
            body
            + "\nThe image is 1024x640 pixels. Return ONLY a JSON object with type=drag, "
            "start_point=[x,y], end_point=[x,y], coordinate_space=source_image_pixels, "
            "ordered=true. Coordinates must be integer pixels of the ORIGINAL screenshot; "
            "do not click, navigate, or execute anything."
        )
        self.assertEqual(drag_prompt(body, 1024, 640, True), expected)

    def test_h60_runner_still_carries_the_same_contract_literals(self):
        source = (REPO / "h60" / "real_drag_dev_baseline.py").read_text("utf8")
        for literal in (
            "The image is 1024x640 pixels. Return ONLY a JSON object with type=drag, ",
            "start_point=[x,y], end_point=[x,y], coordinate_space=source_image_pixels, ",
            "Coordinates must be integer pixels of the ORIGINAL screenshot; ",
            "do not click, navigate, or execute anything.",
        ):
            self.assertIn(literal, source)

    def test_unordered_flag_is_never_rewritten_as_ordered(self):
        body = source_body("Highlight the paragraph", False)
        self.assertIn("ONE unordered drag action", body)
        self.assertNotIn("ONE ordered drag action", body)
        prompt = drag_prompt(body, 1024, 640, False)
        self.assertIn("ordered=false", prompt)
        self.assertNotIn("ordered=true", prompt)

    def test_ordered_flag_is_preserved_for_ordered_source_tasks(self):
        body = source_body("Move the slider to 160", True)
        self.assertIn("ordered drag action", body)
        self.assertIn("ordered=true", drag_prompt(body, 1024, 640, True))

    def test_source_instruction_is_used_verbatim(self):
        intent = "Knowing the brightness max is 20, drag to 12"
        self.assertTrue(source_body(intent, True).startswith(intent))

    def test_bounded_inputs_are_enforced(self):
        with self.assertRaises(ValueError):
            drag_prompt("x", 1024, 640, True)
        with self.assertRaises(ValueError):
            drag_prompt("body text", 0, 640, True)
        with self.assertRaises(ValueError):
            drag_prompt("body text", 1024, 640, "true")
        with self.assertRaises(ValueError):
            source_body("x", True)

    def test_contract_revision_is_declared(self):
        self.assertEqual(PROMPT_REVISION, "hala-h61-native-drag-prompt-contract-v1")
        self.assertIn("ordered={ordered}", CONTRACT_SUFFIX)


if __name__ == "__main__":
    unittest.main()
