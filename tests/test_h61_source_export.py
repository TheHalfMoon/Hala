"""H61 tests for the training-frame transform of the source exporter.

The training frame must be the evaluation frame, and the original boxes and
points must be scaled together with the original screenshot. These tests prove
that transform instead of assuming it.
"""
import unittest

from h61.source_export import DEFAULT_FRAME, scale_row_to_frame


def row(width=1280, height=800):
    return {
        "id": "h60-slider-000be6ed-a326-530c-89e7-299c79c0f49a-t000",
        "image_width": width,
        "image_height": height,
        "start_bbox": [591, 246, 601, 256],
        "end_bbox": [711, 246, 721, 256],
        "completion": {
            "type": "drag",
            "start_point": [596, 251],
            "end_point": [716, 251],
            "coordinate_space": "source_image_pixels",
            "ordered": True,
        },
    }


class FrameTests(unittest.TestCase):
    def test_the_default_frame_is_the_public_dev_frame(self):
        self.assertEqual(DEFAULT_FRAME, (1024, 640))

    def test_boxes_and_points_scale_with_the_screenshot(self):
        scaled = scale_row_to_frame(row(), 1024, 640)
        self.assertEqual(scaled["image_width"], 1024)
        self.assertEqual(scaled["image_height"], 640)
        self.assertEqual(scaled["frame_scale_x"], 0.8)
        self.assertEqual(scaled["frame_scale_y"], 0.8)
        self.assertEqual(scaled["start_bbox"], [473, 197, 481, 205])
        self.assertEqual(scaled["completion"]["start_point"], [477, 201])
        self.assertEqual(scaled["completion"]["end_point"], [573, 201])

    def test_the_point_stays_inside_its_own_scaled_box(self):
        scaled = scale_row_to_frame(row(), 1024, 640)
        for box_key, point_key in (("start_bbox", "start_point"), ("end_bbox", "end_point")):
            box = scaled[box_key]
            point = scaled["completion"][point_key]
            self.assertTrue(box[0] <= point[0] <= box[2])
            self.assertTrue(box[1] <= point[1] <= box[3])

    def test_a_screenshot_in_the_target_frame_is_left_alone(self):
        original = row(1024, 640)
        scaled = scale_row_to_frame(original, 1024, 640)
        self.assertEqual(scaled["start_bbox"], original["start_bbox"])
        self.assertEqual(scaled["completion"], original["completion"])
        self.assertNotIn("frame_scale_x", scaled)

    def test_the_original_dimensions_are_recorded(self):
        scaled = scale_row_to_frame(row(), 1024, 640)
        self.assertEqual(scaled["original_image_width"], 1280)
        self.assertEqual(scaled["original_image_height"], 800)

    def test_a_distorted_aspect_ratio_is_refused(self):
        with self.assertRaises(ValueError):
            scale_row_to_frame(row(1280, 800), 1024, 600)

    def test_the_row_is_not_mutated_in_place(self):
        original = row()
        scale_row_to_frame(original, 1024, 640)
        self.assertEqual(original["image_width"], 1280)
        self.assertEqual(original["start_bbox"], [591, 246, 601, 256])


if __name__ == "__main__":
    unittest.main()
