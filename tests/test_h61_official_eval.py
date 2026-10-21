"""H61 tests for the official upstream evaluation scorer and the frame.

The official DragOn public evaluation split is the independent benchmark, so
its scorer must be explicit: a scene succeeds when the predicted start point
is inside the original start box and the predicted end point is inside the
original end box, and an unordered original task may be satisfied in either
direction because the original task says the direction does not matter.
"""
import json
import tempfile
import unittest
from pathlib import Path

from h61.official_eval import (
    SCENES_SCHEMA,
    apply_frame,
    build_prompt,
    parse_action,
    score,
)


def action(start, end, ordered=True):
    return json.dumps(
        {
            "type": "drag",
            "start_point": list(start),
            "end_point": list(end),
            "coordinate_space": "source_image_pixels",
            "ordered": ordered,
        }
    )


def scene(scene_id, start_box, end_box, ordered=True, width=1024, height=640):
    return {
        "id": scene_id,
        "domain": "slider",
        "image_path": "images/x.jpg",
        "image_sha256": "0" * 64,
        "image_width": width,
        "image_height": height,
        "task": {
            "intent": "Set it to 4",
            "ordered": ordered,
            "start_bbox": list(start_box),
            "end_bbox": list(end_box),
        },
    }


class OfficialScorerTests(unittest.TestCase):
    def test_a_correct_drag_is_a_task_success(self):
        scenes = [scene("a", [100, 100, 110, 110], [400, 100, 410, 110])]
        result = score(scenes, {"a": action([105, 105], [405, 105])})
        self.assertEqual(result["task_success"], 1)
        self.assertEqual(result["task_success_rate"], 1.0)
        self.assertEqual(result["start_hit"], 1)
        self.assertEqual(result["end_hit"], 1)

    def test_a_wrong_endpoint_is_not_a_task_success(self):
        scenes = [scene("a", [100, 100, 110, 110], [400, 100, 410, 110])]
        result = score(scenes, {"a": action([105, 105], [405, 500])})
        self.assertEqual(result["task_success"], 0)
        self.assertEqual(result["start_hit"], 1)
        self.assertEqual(result["end_hit"], 0)

    def test_an_unordered_original_task_accepts_the_reversed_direction(self):
        scenes = [scene("a", [100, 100, 110, 110], [400, 100, 410, 110], ordered=False)]
        result = score(scenes, {"a": action([405, 105], [105, 105], ordered=False)})
        self.assertEqual(result["task_success"], 1)
        self.assertEqual(result["unordered_scenes"], 1)

    def test_an_ordered_original_task_does_not_accept_the_reversed_direction(self):
        scenes = [scene("a", [100, 100, 110, 110], [400, 100, 410, 110], ordered=True)]
        result = score(scenes, {"a": action([405, 105], [105, 105])})
        self.assertEqual(result["task_success"], 0)

    def test_a_degenerate_drag_is_never_a_task_success(self):
        scenes = [scene("a", [100, 100, 110, 110], [400, 100, 410, 110])]
        result = score(scenes, {"a": action([105, 105], [105, 105])})
        self.assertEqual(result["task_success"], 0)
        self.assertEqual(result["degenerate_start_equals_end"], 1)

    def test_an_invalid_output_is_counted_and_never_succeeds(self):
        scenes = [scene("a", [100, 100, 110, 110], [400, 100, 410, 110])]
        result = score(scenes, {"a": "```json\n{\"type\": \"drag\"}\n```"})
        self.assertEqual(result["action_valid"], 0)
        self.assertEqual(result["task_success"], 0)

    def test_an_out_of_range_point_is_not_a_valid_action(self):
        scenes = [scene("a", [100, 100, 110, 110], [400, 100, 410, 110])]
        result = score(scenes, {"a": action([105, 105], [4000, 105])})
        self.assertEqual(result["action_valid"], 0)

    def test_the_scorer_never_reports_a_benchmark_success_without_scenes(self):
        result = score([], {})
        self.assertEqual(result["scenes"], 0)
        self.assertEqual(result["task_success_rate"], 0.0)
        self.assertTrue(result["official_public_eval_used"])
        self.assertFalse(result["hidden_test_used"])


class OfficialFrameTests(unittest.TestCase):
    def test_the_prompt_states_the_frame_of_the_screenshot(self):
        prompt = build_prompt(
            {"image_width": 1280, "image_height": 800},
            {"intent": "Set it to 4", "ordered": True},
        )
        self.assertIn("The image is 1280x800 pixels.", prompt)
        self.assertIn("ordered=true", prompt)

    def test_the_frame_scales_the_boxes_with_the_screenshot(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "images" / "x.jpg"
            image.parent.mkdir(parents=True)
            Image.new("RGB", (1280, 800), (200, 200, 200)).save(image)
            scenes = [scene("a", [666, 232, 688, 254], [527, 238, 537, 248], width=1280, height=800)]
            scenes[0]["image_path"] = str(image)
            framed = apply_frame(scenes, root, (1024, 640))
            self.assertEqual(framed[0]["image_width"], 1024)
            self.assertEqual(framed[0]["frame_scale_x"], 0.8)
            self.assertEqual(framed[0]["task"]["start_bbox"], [533, 186, 550, 203])
            self.assertEqual(framed[0]["task"]["end_bbox"], [422, 190, 430, 198])
            self.assertEqual(framed[0]["original_image_width"], 1280)
            self.assertIn("The image is 1024x640 pixels.", framed[0]["prompt"])

    def test_a_different_aspect_ratio_is_never_distorted(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "images" / "x.jpg"
            image.parent.mkdir(parents=True)
            Image.new("RGB", (1280, 720), (200, 200, 200)).save(image)
            scenes = [scene("a", [666, 232, 688, 254], [527, 238, 537, 248], width=1280, height=720)]
            scenes[0]["image_path"] = str(image)
            framed = apply_frame(scenes, root, (1024, 640))
            self.assertEqual(framed[0]["image_width"], 1024)
            self.assertEqual(framed[0]["image_height"], 576)
            self.assertIn("The image is 1024x576 pixels.", framed[0]["prompt"])


if __name__ == "__main__":
    unittest.main()
