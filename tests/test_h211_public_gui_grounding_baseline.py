"""H211 self-authored 960x720 GUI raster + strict coordinate-reward tests.

No model weights, no owner media, no OOD and no browser. This tests
grading; public pretrained performance is a separate owner CPU run.
"""
import json
import unittest

from h211.public_gui_grounding_baseline import (
    MAX_NEW_TOKENS, MODEL_TASK, SCENES, VIEWPORT,
    grade_native_gui_action, original_synthetic_gui,
)


def click(x, y):
    return json.dumps({"actions": [{"kind": "click", "point": [x, y]}]})


class OriginalSyntheticGuiTests(unittest.TestCase):
    def test_two_new_image_panels_have_distinct_visual_pixels(self):
        a = original_synthetic_gui("LEFT")
        b = original_synthetic_gui("RIGHT")
        self.assertEqual(a.size, (960, 720))
        self.assertEqual(b.size, (960, 720))
        self.assertEqual(a.mode, "RGB")
        self.assertEqual(b.mode, "RGB")
        self.assertNotEqual(a.getpixel((100, 300)), b.getpixel((100, 300)))
        self.assertNotEqual(a.getpixel((780, 300)), b.getpixel((780, 300)))

    def test_images_only_from_self_authored_generator(self):
        self.assertEqual(set(SCENES), {"LEFT", "RIGHT"})
        self.assertEqual(VIEWPORT, (960, 720))
        self.assertNotIn("left", MODEL_TASK.lower())
        self.assertNotIn("right", MODEL_TASK.lower())
        self.assertNotIn("620", MODEL_TASK)
        self.assertNotIn("250", MODEL_TASK)
        self.assertTrue("SAVE" in MODEL_TASK)
        self.assertLessEqual(MAX_NEW_TOKENS, 192)

    def test_source_buttons_are_actual_contrasting_rasters(self):
        for side in SCENES:
            with self.subTest(side=side):
                img = original_synthetic_gui(side)
                bbox = SCENES[side]
                center = ((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2)
                # Avoid central white SAVE letters: background offset.
                sample = (bbox[0]+25, bbox[1]+28)
                self.assertEqual(img.getpixel(sample), (28, 97, 213))
                self.assertNotEqual(img.getpixel(center), (245, 246, 248))

    def test_correct_click_center_reward_only_within_target(self):
        for side, rect in SCENES.items():
            with self.subTest(side=side):
                x = (rect[0]+rect[2])//2
                y = (rect[1]+rect[3])//2
                self.assertEqual(grade_native_gui_action(click(x,y),side),"HIT")

    def test_wrong_button_is_miss_not_hit(self):
        for side in SCENES:
            other = "RIGHT" if side=="LEFT" else "LEFT"
            bbox = SCENES[other]
            x = (bbox[0]+bbox[2])//2
            y = (bbox[1]+bbox[3])//2
            self.assertEqual(grade_native_gui_action(click(x,y),side),"MISS")

    def test_off_screen_coordinate_does_not_pass(self):
        for wrong in (click(-1,300),click(961,300),click(200,721),
                      click(200,-1)):
            self.assertEqual(grade_native_gui_action(wrong,"LEFT"),"MALFORMED")

    def test_free_form_or_duplicate_or_nonstandard_json_fails_closed(self):
        bad = (
            "click at 100,300",
            '{"actions":[{"kind":"click","point":[40,30],"point":[80,80]}]}',
            '{"actions":[{"kind":"click","point":[NaN,3]}]}',
            '{"actions":[{"kind":"click","point":[100,310]}],"explanation":"yes"}',
        )
        for response in bad:
            with self.subTest(response=response):
                self.assertEqual(grade_native_gui_action(response,"LEFT"),"MALFORMED")

    def test_abstention_is_never_misgraded_as_click(self):
        self.assertEqual(grade_native_gui_action(
            '{"actions":[{"kind":"abstain"}]}',"LEFT"),"ABSTAIN")

    def test_keyboard_and_drag_not_equated_with_click(self):
        self.assertEqual(grade_native_gui_action(
            '{"actions":[{"kind":"key_press","point":[100,300],"key":"ArrowRight"}]}',
            "LEFT"),"NONCLICK")
        self.assertEqual(grade_native_gui_action(
            '{"actions":[{"kind":"drag","start":[100,300],"end":[150,350]}]}',
            "LEFT"),"NONCLICK")

    def test_unknown_source_scene_rejected_before_grading(self):
        for bad in ("right", "H173", None, 2):
            with self.subTest(side=str(bad)):
                with self.assertRaises(ValueError):
                    grade_native_gui_action(click(100,300),bad)
                with self.assertRaises(ValueError):
                    original_synthetic_gui(bad)


if __name__=="__main__":
    unittest.main()
