"""H190 only synthetic private DEV-source quarantine tests; no old media."""
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from h189.interact_preaction_native import SPECS
from h190.owner_private_interact_dev_capture import (
    _check_ephemeral_source_capture, _read_previous_train_source,
    private_output_destination,
)


def png(color: str) -> bytes:
    stream = BytesIO()
    Image.new("RGB", (960, 720), color).save(stream, format="PNG")
    return stream.getvalue()


def captured():
    events = []
    for index, spec in enumerate(SPECS):
        for _ in range(2):
            events.append((
                dict(spec), png(("red", "blue")[index]),
                {"start": [120.0 + index, 80.0],
                 "drop": [550.0 + index, 440.0]},
            ))
    return events


def previous_train(path: Path) -> None:
    path.mkdir()
    (path / "source-prospects-not-admitted.json").write_text(json.dumps({
        "status": "H188_PRIVATE_ORIGINAL_SOURCE_QUARANTINE_NOT_ADMISSION",
        "official_train_dev_dataset_admitted": False,
        "cases": [
            {"train_eligible": False, "dev_eligible": False,
             "ood_eligible": False, "image_sha256": "1"*64,
             "visual_rgba_sha256": "a"*64,
             "source_repo": "simonwep/pickr",
             "source_git_sha": "e8ff88d9161ffc6e3c46a4430be8c328c42d6415"},
            {"train_eligible": False, "dev_eligible": False,
             "ood_eligible": False, "image_sha256": "2"*64,
             "visual_rgba_sha256": "b"*64,
             "source_repo": "simonwep/pickr",
             "source_git_sha": "e8ff88d9161ffc6e3c46a4430be8c328c42d6415"},
        ],
    }), encoding="utf-8")


class H190QuarantineMetadataTests(unittest.TestCase):
    def test_strict_original_preaction_two_distinct_source_pngs(self):
        verified = _check_ephemeral_source_capture(captured())
        self.assertEqual(len(verified), 2)
        self.assertEqual({x["spec"]["language"] for x in verified}, {"en", "ar"})
        self.assertEqual(len({x["image_sha256"] for x in verified}), 2)
        self.assertEqual(len({x["visual_rgba_sha256"] for x in verified}), 2)

    def test_preaction_capture_missing_independent_replay_rejected(self):
        with self.assertRaisesRegex(ValueError, "capture missing"):
            _check_ephemeral_source_capture(captured()[:-1])

    def test_same_preaction_image_across_two_tasks_fails(self):
        events = captured()
        for i in (2, 3):
            events[i] = (
                events[i][0], events[0][1], events[i][2],
            )
        with self.assertRaisesRegex(ValueError, "visual collision"):
            _check_ephemeral_source_capture(events)

    def test_png_or_native_target_replay_drift_rejected(self):
        for alter in ("png", "goal"):
            with self.subTest(alter=alter):
                rows = captured()
                if alter == "png":
                    rows[1] = (rows[1][0], png("green"), rows[1][2])
                else:
                    goal = {"start": [120.0, 80.0],
                            "drop": [551.0, 440.0]}
                    rows[1] = (rows[1][0], rows[1][1], goal)
                with self.assertRaisesRegex(ValueError, "replay drift"):
                    _check_ephemeral_source_capture(rows)

    def test_offscreen_or_invalid_original_gold_drag_rejected(self):
        for bad in (-1.0, 960.0, float("nan"), "1"):
            with self.subTest(bad=bad):
                rows = captured()
                goal = {"start": [bad, 80.0],
                        "drop": [550.0, 440.0]}
                rows[0] = (rows[0][0], rows[0][1], goal)
                rows[1] = (rows[1][0], rows[1][1], goal)
                with self.assertRaisesRegex(ValueError, "target point"):
                    _check_ephemeral_source_capture(rows)

    def test_no_cross_source_name_or_existing_destination(self):
        with tempfile.TemporaryDirectory() as root:
            top = Path(root)
            with self.assertRaisesRegex(ValueError, "named owner-private"):
                private_output_destination(top / "public-cases")
            with self.assertRaisesRegex(ValueError, "absolute"):
                private_output_destination(Path("private-h190"))
            (top / "Hala-H190-Private").mkdir()
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                private_output_destination(top / "Hala-H190-Private")

    def test_git_worktree_output_directory_blocked(self):
        with tempfile.TemporaryDirectory() as root:
            top = Path(root)
            (top / ".git").mkdir()
            with self.assertRaisesRegex(ValueError, "Git checkout"):
                private_output_destination(top / "private-h190")

    def test_private_pickr_prospect_source_verified_only_by_sha_meta(self):
        with tempfile.TemporaryDirectory() as root:
            previous_train(Path(root) / "Hala-H188-Private")
            got = _read_previous_train_source(Path(root) / "Hala-H188-Private")
            self.assertEqual(len(got["png"]), 2)
            self.assertEqual(len(got["rgba"]), 2)
            self.assertEqual(
                got["families"],
                {("simonwep/pickr", "e8ff88d9161ffc6e3c46a4430be8c328c42d6415")},
            )

    def test_tampered_train_source_status_and_pin_rejected(self):
        for field in ("status", "source_repo", "image_sha256"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as root:
                base = Path(root) / "Hala-H188-Private"
                previous_train(base)
                path = base / "source-prospects-not-admitted.json"
                doc = json.loads(path.read_text(encoding="utf-8"))
                if field == "status":
                    doc["status"] = "TRAIN_ADMITTED"
                elif field == "source_repo":
                    doc["cases"][0]["source_repo"] = "bad/copy"
                else:
                    doc["cases"][0]["image_sha256"] = "2"*64
                path.write_text(json.dumps(doc), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "quarantine|authenticity"):
                    _read_previous_train_source(base)


if __name__ == "__main__":
    unittest.main()
