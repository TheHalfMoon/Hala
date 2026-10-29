"""H188 synthetic-only private quarantine writer safety tests.

No real owner photos, old media, reference gold or model weights.
Source-runner is mocked ONLY in tests; no fake output can prove real
browser qualification. The hosted H187 source-native workflow is separate.
"""
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from h187.pickr_preaction_native import SPECS
from h188.audit_owner_private_capture import audit_owner_private_capture
from h188.owner_private_pickr_capture import (
    _private_destination, _verify_ephemeral_preview,
    quarantine_owner_source_only,
)


def fake_png(color):
    stream = BytesIO()
    Image.new("RGB", (960, 720), color).save(stream, format="PNG")
    return stream.getvalue()


def source_run_proof(*_args, capture_preaction, **_kwargs):
    for idx, spec in enumerate(SPECS):
        for _ in range(2):
            capture_preaction(
                spec, fake_png(("red", "blue")[idx]),
                {"x": 320.5 + idx*50, "y": 210.5 + idx*100},
            )
    return {
        "original_preaction_scenes": 2,
        "known_owner_history_shas_screened": True,
        "train_dev_examples_admitted": 0,
        "model_training_approved": False,
    }


class H188OwnerPrivateOriginalCaptureTests(unittest.TestCase):
    def test_quarantine_two_genuine_shaped_source_pngs_and_separate_private_gold(self):
        with tempfile.TemporaryDirectory() as root:
            top = Path(root)
            dest = top / "private-h188"
            with patch("h188.owner_private_pickr_capture.run_source_only",
                       side_effect=source_run_proof):
                proof = quarantine_owner_source_only(
                    dist_dir=top, browser_channel="chromium",
                    historical_root=top, owner_private_dir=dest,
                )
            self.assertEqual(proof["owner_private_images_written"], 2)
            self.assertEqual(proof["owner_private_native_reference_goals_written"], 2)
            self.assertFalse(proof["train_dev_admitted"])
            self.assertEqual(proof["model_calls"], 0)
            self.assertEqual(proof["public_images_uploaded"], 0)
            doc = json.loads((dest / "source-prospects-not-admitted.json").read_text(encoding="utf-8"))
            self.assertFalse(doc["official_train_dev_dataset_admitted"])
            self.assertEqual(len(doc["cases"]), 2)
            self.assertEqual({x["language"] for x in doc["cases"]}, {"en", "ar"})
            self.assertEqual(len(list((dest / "images").glob("*.png"))), 2)
            self.assertNotIn("native_source_target_center", str(doc))
            private_gold = json.loads(
                (dest / "OWNER_PRIVATE_REFERENCE_NOT_MODEL_INPUT.json").read_text(encoding="utf-8")
            )
            self.assertTrue(all(row["not_for_model_input"]
                                for row in private_gold["rows"]))
            self.assertEqual(len(private_gold["rows"]), 2)

    def test_never_overwrite_existing_original_photos(self):
        with tempfile.TemporaryDirectory() as root:
            dest = Path(root) / "private-h188"
            dest.mkdir()
            (dest / "important-private-data").write_text("PRESERVE")
            with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                quarantine_owner_source_only(
                    dist_dir=Path(root), browser_channel="chromium",
                    historical_root=Path(root), owner_private_dir=dest,
                )
            self.assertEqual((dest / "important-private-data").read_text(),
                             "PRESERVE")

    def test_no_git_repo_private_output_even_if_folder_name_looks_private(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            (base / ".git").mkdir()
            with self.assertRaisesRegex(ValueError, "within a Git worktree"):
                _private_destination(base / "private-h188")

    def test_refuse_unknown_destination_name(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "owner-private H188 filename"):
                _private_destination(Path(root) / "public_source_images")

    def test_refuse_relative_owner_private_destination(self):
        with self.assertRaisesRegex(ValueError, "absolute"):
            _private_destination(Path("private-h188"))

    def test_incomplete_original_browser_replay_cannot_write(self):
        with tempfile.TemporaryDirectory() as root:
            top = Path(root)
            dest = top / "private-h188"
            def incomplete(*args, capture_preaction, **kwargs):
                capture_preaction(SPECS[0], fake_png("red"), {"x": 10, "y": 10})
                return source_run_proof(*args, capture_preaction=lambda *_: None, **kwargs)
            with patch("h188.owner_private_pickr_capture.run_source_only",
                       side_effect=incomplete):
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    quarantine_owner_source_only(
                        dist_dir=top, browser_channel="chromium",
                        historical_root=top, owner_private_dir=dest,
                    )
            self.assertFalse(dest.exists())

    def test_changed_replay_png_or_target_coord_fails_before_any_write(self):
        with tempfile.TemporaryDirectory() as root:
            top = Path(root)
            dest = top / "private-h188"
            def drift(*args, capture_preaction, **kwargs):
                for idx, spec in enumerate(SPECS):
                    for repeat in range(2):
                        capture_preaction(
                            spec, fake_png("red" if idx == 0 and repeat == 0
                                           else "blue"),
                            {"x": 1.0, "y": 1.0},
                        )
                return source_run_proof(*args, capture_preaction=lambda *_: None, **kwargs)
            with patch("h188.owner_private_pickr_capture.run_source_only",
                       side_effect=drift):
                with self.assertRaisesRegex(ValueError, "drift"):
                    quarantine_owner_source_only(
                        dist_dir=top, browser_channel="chromium",
                        historical_root=top, owner_private_dir=dest,
                    )
            self.assertFalse(dest.exists())

    def test_source_runner_not_allowed_to_skip_historical_old_sha_check(self):
        with tempfile.TemporaryDirectory() as root:
            top = Path(root)
            dest = top / "private-h188"
            def no_history(*args, capture_preaction, **kwargs):
                good = source_run_proof(*args, capture_preaction=capture_preaction, **kwargs)
                good["known_owner_history_shas_screened"] = False
                return good
            with patch("h188.owner_private_pickr_capture.run_source_only",
                       side_effect=no_history):
                with self.assertRaisesRegex(ValueError, "history exclusion"):
                    quarantine_owner_source_only(
                        dist_dir=top, browser_channel="chromium",
                        historical_root=top, owner_private_dir=dest,
                    )
            self.assertFalse(dest.exists())

    def test_offscreen_or_nan_reference_point_rejected(self):
        for target in ({"x": -1, "y": 0},
                       {"x": float("nan"), "y": 100},
                       {"x": 960, "y": 719}):
            with self.subTest(target=target):
                captures = []
                for i, spec in enumerate(SPECS):
                    for _ in range(2):
                        captures.append((
                            spec, fake_png(("red", "blue")[i]), target
                        ))
                with self.assertRaisesRegex(ValueError, "coordinate invalid"):
                    _verify_ephemeral_preview(captures)

    def test_same_preaction_pixels_across_two_source_tasks_rejected(self):
        captures = []
        for spec in SPECS:
            for _ in range(2):
                captures.append((spec, fake_png("red"), {"x": 1, "y": 2}))
        with self.assertRaisesRegex(ValueError, "not actually different"):
            _verify_ephemeral_preview(captures)


    def test_independent_saved_owner_private_rgba_and_png_exact_sha_pass(self):
        with tempfile.TemporaryDirectory() as root:
            top = Path(root)
            (top / "private-h140").mkdir()
            (top / "private-h140" / "manifest.json").write_text(
                json.dumps({"image_sha256": "f"*64}), encoding="utf-8"
            )
            dest = top / "private-h188"
            with patch("h188.owner_private_pickr_capture.run_source_only",
                       side_effect=source_run_proof):
                quarantine_owner_source_only(
                    dist_dir=top, browser_channel="chromium",
                    historical_root=top, owner_private_dir=dest,
                )
            result = audit_owner_private_capture(
                owner_private_dir=dest, historical_root=top,
            )
            self.assertEqual(result["distinct_source_preaction_pngs"], 2)
            self.assertEqual(result["old_owner_png_sha_candidates_compared"], 1)
            self.assertFalse(result["train_dev_admitted"])
            self.assertFalse(result["complete_prior_visual_history_proven"])

    def test_independent_saved_image_corruption_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            top = Path(root)
            (top / "private-h140").mkdir()
            (top / "private-h140" / "manifest.json").write_text(
                json.dumps({"image_sha256": "f"*64}), encoding="utf-8"
            )
            dest = top / "private-h188"
            with patch("h188.owner_private_pickr_capture.run_source_only",
                       side_effect=source_run_proof):
                quarantine_owner_source_only(
                    dist_dir=top, browser_channel="chromium",
                    historical_root=top, owner_private_dir=dest,
                )
            original = dest / "images" / "h188-original-pickr-00.png"
            original.write_bytes(fake_png("green"))
            with self.assertRaisesRegex(ValueError, "visual SHA altered"):
                audit_owner_private_capture(
                    owner_private_dir=dest, historical_root=top
                )

    def test_independent_saved_manifest_eligibility_tamper_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            top = Path(root)
            (top / "private-h140").mkdir()
            (top / "private-h140" / "manifest.json").write_text(
                json.dumps({"image_sha256": "f"*64}), encoding="utf-8"
            )
            dest = top / "private-h188"
            with patch("h188.owner_private_pickr_capture.run_source_only",
                       side_effect=source_run_proof):
                quarantine_owner_source_only(
                    dist_dir=top, browser_channel="chromium",
                    historical_root=top, owner_private_dir=dest,
                )
            manifest_path = dest / "source-prospects-not-admitted.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["cases"][0]["train_eligible"] = True
            manifest_path.write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "admission was altered"):
                audit_owner_private_capture(
                    owner_private_dir=dest, historical_root=top
                )

    def test_independent_saved_owner_gold_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            top = Path(root)
            (top / "private-h140").mkdir()
            (top / "private-h140" / "manifest.json").write_text(
                json.dumps({"image_sha256": "f"*64}), encoding="utf-8"
            )
            dest = top / "private-h188"
            with patch("h188.owner_private_pickr_capture.run_source_only",
                       side_effect=source_run_proof):
                quarantine_owner_source_only(
                    dist_dir=top, browser_channel="chromium",
                    historical_root=top, owner_private_dir=dest,
                )
            gold_path = dest / "OWNER_PRIVATE_REFERENCE_NOT_MODEL_INPUT.json"
            private = json.loads(gold_path.read_text(encoding="utf-8"))
            private["rows"][1]["not_for_model_input"] = False
            gold_path.write_text(json.dumps(private), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "remain separate"):
                audit_owner_private_capture(
                    owner_private_dir=dest, historical_root=top
                )


if __name__ == "__main__":
    unittest.main()
