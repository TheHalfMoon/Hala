"""H61 tests for the first-party bilingual drag corpus and the family mixture.

The measured H60 result is that the trained adapter recovered the English
public DEV scenes and almost not the Arabic ones, because the original corpus
is English only. These tests prove that the first-party bilingual corpus is a
verified TRAIN source, that its captured screenshots really show the authored
handle and target, and that the curriculum can bound how much one family
contributes so a large English family cannot crowd out the Arabic rows.
"""
import unittest

from h61.curriculum import select
from h61.firstparty_drag import (
    KINDS,
    LANGUAGES,
    audit_scene,
    build_scenes,
    scene_html,
)

ARABIC_START = 0x0600
ARABIC_END = 0x0700


def is_arabic(text: str) -> bool:
    return any(ARABIC_START <= ord(character) < ARABIC_END for character in text)


def row(index: int, family: str, language: str) -> dict:
    return {
        "id": f"{family}{index}", "split": "train", "domain": "slider", "subtype": None,
        "language": language, "family": family, "image_id": f"{family}{index}",
        "image_path": f"images/slider/{family}{index}.jpg", "image_sha256": "0" * 64,
        "image_width": 1024, "image_height": 640, "source_task_index": 0, "intent": "x",
        "action_type": "drag", "ordered": True, "start_bbox": [0, 0, 2, 2],
        "end_bbox": [4, 4, 6, 6],
        "completion": {"type": "drag", "start_point": [1, 1], "end_point": [5, 5],
                       "coordinate_space": "source_image_pixels", "ordered": True},
    }


class FirstPartySceneTests(unittest.TestCase):
    def test_the_scenes_are_balanced_over_language_and_kind(self):
        scenes = build_scenes(240)
        self.assertEqual(len(scenes), 240)
        self.assertEqual(len({scene["language"] for scene in scenes}), len(LANGUAGES))
        self.assertEqual({scene["kind"] for scene in scenes}, set(KINDS))
        self.assertTrue(all(scene["split"] == "train" for scene in scenes))

    def test_the_scene_identities_are_disjoint_from_the_public_dev_families(self):
        scenes = build_scenes(240)
        self.assertTrue(
            all(not scene["id"].startswith(("h55-", "h58-")) for scene in scenes)
        )

    def test_an_unordered_scene_is_never_rewritten_as_ordered(self):
        scenes = build_scenes(240)
        unordered = [scene for scene in scenes if not scene["ordered"]]
        self.assertTrue(unordered)
        self.assertTrue(all(scene["kind"] == "text_highlight" for scene in unordered))

    def test_the_recorded_boxes_are_page_boxes(self):
        scenes = build_scenes(240)
        for scene in scenes:
            self.assertEqual(scene["start_bbox"][0], scene["handle"][0])
            self.assertGreater(scene["start_bbox"][1], scene["handle"][1])
            self.assertEqual(scene["start_bbox"][2] - scene["start_bbox"][0], 26)

    def test_the_arabic_instruction_is_arabic_and_the_english_one_is_not(self):
        scenes = build_scenes(240)
        arabic = [scene for scene in scenes if scene["language"] == "ar"]
        english = [scene for scene in scenes if scene["language"] == "en"]
        self.assertTrue(arabic)
        self.assertTrue(english)
        self.assertTrue(all(is_arabic(scene["instruction"]) for scene in arabic))
        self.assertTrue(all(not is_arabic(scene["instruction"]) for scene in english))

    def test_the_page_contains_the_handle_and_the_target(self):
        scene = build_scenes(1)[0]
        html = scene_html(scene)
        self.assertIn('id="handle"', html)
        self.assertIn('id="target"', html)
        self.assertIn(scene["title"], html)


class FirstPartyAuditTests(unittest.TestCase):
    def test_a_scene_that_did_not_render_is_rejected(self):
        from PIL import Image

        with self.TemporaryDirectory() as directory:
            path = directory / "blank.png"
            Image.new("RGB", (1024, 640), (239, 245, 248)).save(path)
            scene = build_scenes(1)[0]
            problems = audit_scene(path, scene)
            self.assertTrue(problems)
            self.assertIn("handle", problems[0])

    def test_a_scene_that_rendered_is_accepted(self):
        from PIL import Image

        with self.TemporaryDirectory() as directory:
            path = directory / "scene.png"
            picture = Image.new("RGB", (1024, 640), (239, 245, 248))
            scene = build_scenes(1)[0]
            for box, colour in ((scene["start_bbox"], (34, 108, 132)),
                                (scene["end_bbox"], (173, 106, 80))):
                for x in range(box[0] + 4, box[2] - 4):
                    for y in range(box[1] + 4, box[3] - 4):
                        picture.putpixel((x, y), colour)
            picture.save(path)
            self.assertEqual(audit_scene(path, scene), [])

    def TemporaryDirectory(self):
        import tempfile

        class _Holder(tempfile.TemporaryDirectory):
            def __enter__(self):
                value = super().__enter__()
                return Path(value)

        return _Holder()


from pathlib import Path  # noqa: E402


class FamilyMixtureTests(unittest.TestCase):
    def test_a_family_target_bounds_the_contribution_of_one_family(self):
        rows = [row(index, "english_family", "en") for index in range(40)]
        rows += [row(index, "arabic_family", "ar") for index in range(10)]
        selected, balance = select(rows, 50, 1, {"english_family": 40, "arabic_family": 10})
        self.assertEqual(len(selected), 50)
        self.assertEqual(balance["selected_by_family"]["arabic_family"], 10)
        self.assertEqual(balance["selected_by_family"]["english_family"], 40)

    def test_without_a_target_the_large_family_may_dominate(self):
        rows = [row(index, "english_family", "en") for index in range(30)]
        selected, balance = select(rows, 30, 1)
        self.assertEqual(balance["selected_by_family"]["english_family"], 30)


if __name__ == "__main__":
    unittest.main()
