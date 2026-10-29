"""H197 synthetic source-only data quarantine mutations. NO real media here."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from PIL import Image
from io import BytesIO
from hashlib import sha256

from h194.owner_private_snap_capture import PREVIOUS
from h174.source_candidates import CANDIDATES
from h195.original_vanilla_datepicker_native import SCENES, PIN
from h197.owner_calendar_private import capture, prior_shas, validate_captures
from h197.audit_owner_calendar import audit


def png(color):
    out=BytesIO()
    Image.new("RGB",(960,720),color).save(out,format="PNG")
    return out.getvalue()


def captures():
    return [(dict(s),png(("red","blue")[i]),{
        "action_type":"click",
        "target_center":[422.0+i,320.0+i],
        "native_selected_value":s["expected_value"],
    }) for i,s in enumerate(SCENES) for _ in range(2)]


def prepared(root):
    (root / "private-h140").mkdir()
    (root / "private-h140" / "manifest.json").write_text(
        json.dumps({"image_sha256":"f"*64}),encoding="utf-8")
    for index,(name,status,count,family) in enumerate(PREVIOUS):
        folder=root/name
        folder.mkdir()
        src=CANDIDATES[family]
        cases=[{"source_repo":src["repo"],
                "source_git_sha":src["git_sha"],
                "train_eligible":False,"dev_eligible":False,
                "ood_eligible":False,
                "image_sha256":f"{index*100+n+1:064x}",
                "visual_rgba_sha256":f"{index*100+50+n+1:064x}",
                } for n in range(count)]
        (folder/"source-prospects-not-admitted.json").write_text(
            json.dumps({"status":status,
                        "official_train_dev_dataset_admitted":False,
                        "cases":cases}),encoding="utf-8")
    folder=root/"Hala-H194-Private"
    folder.mkdir()
    (folder/"source-prospects-not-admitted.json").write_text(
        json.dumps({
            "status":"H194_OWNER_PRIVATE_ORIGINAL_SLIDER_DEV_PROSPECTS_NOT_ADMITTED",
            "official_train_dev_dataset_admitted":False,
            "cases":[{
                "source_repo":CANDIDATES["interact-dropzones"]["repo"],
                "source_git_sha":CANDIDATES["interact-dropzones"]["git_sha"],
                "train_eligible":False,"dev_eligible":False,"ood_eligible":False,
                "image_sha256":f"{500+i:064x}",
                "visual_rgba_sha256":f"{600+i:064x}"
            } for i in range(2)]
        }),encoding="utf-8")


def mocked_source(donor, channel, historical, capture_preaction):
    for scene,image,goal in captures():
        capture_preaction(scene,image,goal)
    return {
        "independent_original_date_selection_tasks":2,
        "old_git_checkouts_screened":22,
        "known_old_sha_screened":True,
        "train_dev_ood_admitted":0,
    }


class OwnerOriginalCalendarTests(TestCase):
    def test_two_real_shaped_preaction_scenes(self):
        rows=validate_captures(captures())
        self.assertEqual(len(rows),2)
        self.assertNotEqual(rows[0]["visual_rgba_sha256"],
                            rows[1]["visual_rgba_sha256"])

    def test_replay_image_drift_fails(self):
        rows=captures()
        scene,image,goal=rows[1]
        rows[1]=(scene,png("green"),goal)
        with self.assertRaisesRegex(ValueError,"fresh-context drift"):
            validate_captures(rows)

    def test_native_date_reference_tamper_fails(self):
        rows=captures()
        for index in (0,1):
            scene,image,goal=rows[index]
            rows[index]=(scene,image,{**goal,
                "native_selected_value":"12/25/2030"})
        with self.assertRaisesRegex(ValueError,"not a true original"):
            validate_captures(rows)

    def test_missing_original_browser_replay_fails(self):
        with self.assertRaisesRegex(ValueError,"missing original independent"):
            validate_captures(captures()[:-1])

    def test_existing_output_never_overwritten(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            prepared(root)
            (root/"Hala-H197-Private").mkdir()
            with self.assertRaisesRegex(ValueError,"no overwrite"):
                capture(root,root,root,root/"Hala-H197-Private","chromium")

    def test_prior_source_registry_has_18_distinct_shas(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            prepared(root)
            a,b=prior_shas(root)
            self.assertEqual(len(a),18)
            self.assertEqual(len(b),18)

    def test_real_shaped_writer_and_independent_raster_audit(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            prepared(root)
            with patch("h197.owner_calendar_private.run_original_datepicker",
                       side_effect=mocked_source):
                result=capture(root,root,root,
                               root/"Hala-H197-Private","chromium")
            self.assertEqual(result[0],2)
            self.assertEqual(result[1],18)
            self.assertEqual(len(list((root/"Hala-H197-Private"/"images")
                                      .glob("*.png"))),2)
            a=audit(root/"Hala-H197-Private",root,root)
            self.assertEqual(a["new_png_rgba_rechecked"],2)
            self.assertFalse(a["train_dev_ood_admitted"])

    def test_corrupted_new_png_rejected_by_independent_audit(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            prepared(root)
            with patch("h197.owner_calendar_private.run_original_datepicker",
                       side_effect=mocked_source):
                capture(root,root,root,root/"Hala-H197-Private","chromium")
            (root/"Hala-H197-Private"/"images"/"h197-calendar-00.png").write_bytes(png("green"))
            with self.assertRaisesRegex(ValueError,"PNG/RGBA SHA altered"):
                audit(root/"Hala-H197-Private",root,root)

    def test_old_encoded_png_reuse_fails_before_writing(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            prepared(root)
            (root/"private-h140"/"manifest.json").write_text(
                json.dumps({"image_sha256":sha256(png("red")).hexdigest()}),
                encoding="utf-8")
            with patch("h197.owner_calendar_private.run_original_datepicker",
                       side_effect=mocked_source):
                with self.assertRaisesRegex(ValueError,"reuses known old PNG"):
                    capture(root,root,root,root/"Hala-H197-Private","chromium")
            self.assertFalse((root/"Hala-H197-Private").exists())

    def test_private_goal_cannot_become_model_input(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            prepared(root)
            with patch("h197.owner_calendar_private.run_original_datepicker",
                       side_effect=mocked_source):
                capture(root,root,root,root/"Hala-H197-Private","chromium")
            path=root/"Hala-H197-Private"/"source-prospects-not-admitted.json"
            record=json.loads(path.read_text())
            record["cases"][0]["private_action"]={"action_type":"click"}
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,"source identity"):
                audit(root/"Hala-H197-Private",root,root)

    def test_no_false_dev_promotion(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            prepared(root)
            with patch("h197.owner_calendar_private.run_original_datepicker",
                       side_effect=mocked_source):
                capture(root,root,root,root/"Hala-H197-Private","chromium")
            path=root/"Hala-H197-Private"/"source-prospects-not-admitted.json"
            record=json.loads(path.read_text())
            record["cases"][0]["dev_eligible"]=True
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,"research-only split"):
                audit(root/"Hala-H197-Private",root,root)
