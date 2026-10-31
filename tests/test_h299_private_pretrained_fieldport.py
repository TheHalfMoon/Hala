"""H299 invented CPU receipt/package tests; NO pretrained Qwen or GPU."""
from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from h280.verify_private_native_aggregate import EvidenceRejected
from h299.private_pretrained_fieldport_pilot import (
    SOURCE_SHA,SOURCE_BLOB,MODEL_ID,MODEL_REV,
    SCENE_INDICES,REPORT,DIAGNOSTIC_REPORT,STATUS,
    verified_report,safe_failure_receipt,
)
from h299.verify_private_fieldport_receipt import (
    verify_success,verify_failure,verify_private,
)
from h299.build_private_payload import build,KERNEL_ID,KERNEL_TITLE

ROOT=Path(__file__).resolve().parents[1]


def synthetic_rows():
    return [
        {
            "language":"en","generations":2,"grammar_calls":2,
            "screenshot_steps":2,"focus_clicks":1,"type_calls":1,
            "abstained":False,"task_success":True,
            "all_four_source_fields_checked":True,
        },
        {
            "language":"ar","generations":1,"grammar_calls":1,
            "screenshot_steps":1,"focus_clicks":0,"type_calls":0,
            "abstained":True,"task_success":False,
            "all_four_source_fields_checked":True,
        },
    ]


class H299GenuinePrivateStudyReceiptContractTests(TestCase):
    def setUp(self):
        self.rows=synthetic_rows()
        self.hardware={"free_t4":True,"devices":1}
        self.report=verified_report(self.rows,self.hardware)

    def test_source_model_and_original_scene_pins(self):
        self.assertEqual(SOURCE_SHA,
            "2babf8c91ffb63dd1123070449d1cbc240ab0859")
        self.assertEqual(SOURCE_BLOB,
            "14bc9041fabf59d76adb0fc78b6a16d6d53c6e80")
        self.assertEqual(MODEL_ID,"Qwen/Qwen3-VL-2B-Instruct")
        self.assertEqual(len(MODEL_REV),40)
        self.assertEqual(SCENE_INDICES,(0,6))
        self.assertEqual(KERNEL_TITLE,"Hala H299 Private Pretrained FieldPort T4")
        self.assertEqual(KERNEL_ID,
            "abdulazizshehri/hala-h299-private-pretrained-fieldport-t4")

    def test_independent_original_closed_success_receipt(self):
        safe=verify_success(self.report)
        self.assertEqual(safe["status"],
            "H299_STRICT_SOURCE_MODEL_RESEARCH_RECEIPT_ACCEPTED")
        self.assertEqual(safe["genuine_firstparty_task_successes"],1)
        self.assertEqual(safe["model_time_pretrained_generations"],3)
        self.assertEqual(safe["true_model_time_grammar_calls"],3)
        self.assertEqual([r["language"] for r in safe["per_language"]],
                         ["en","ar"])
        self.assertTrue(safe["per_language"][0]["firstparty_task_success"])
        self.assertFalse(safe["per_language"][1]["firstparty_task_success"])
        self.assertFalse(safe["adapted_model_gain_proven"])
        self.assertFalse(safe["blind_ood_proven"])
        self.assertTrue(safe["owner_private_authentication_external_required"])
        for leak in ("point","FIC-11","FIC-17","image","weights","gold"):
            self.assertNotIn(leak,json.dumps(safe))

    def test_negative_zero_and_positive_actual_firstparty_results_allowed(self):
        for success_en, success_ar in ((False,False),(False,True),(True,True)):
            rows=synthetic_rows()
            rows[0]["task_success"]=success_en
            rows[1]={
                "language":"ar","generations":2,"grammar_calls":2,
                "screenshot_steps":2,"focus_clicks":1,"type_calls":1,
                "abstained":False,"task_success":success_ar,
                "all_four_source_fields_checked":True,
            }
            doc=verified_report(rows,self.hardware)
            safe=verify_success(doc)
            self.assertEqual(safe["genuine_firstparty_task_successes"],
                             int(success_en)+int(success_ar))

    def test_reject_fabricated_native_task_score_and_missing_token_filter(self):
        variants=(
            ("rows",0,"generations",0),
            ("rows",0,"grammar_calls",0),
            ("rows",1,"screenshot_steps",2),
            ("rows",0,"focus_clicks",0),
            ("rows",1,"type_calls",1),
            ("rows",0,"abstained",True),
            ("rows",0,"task_success",1),
            ("rows",1,"all_four_source_fields_checked",False),
        )
        for _,index,key,value in variants:
            doc=deepcopy(self.report)
            doc["rows"][index][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(EvidenceRejected):
                verify_success(doc)

    def test_reject_pinning_exposure_and_unapproved_model_training(self):
        for field,value in (
            ("source_commit","0"*40),
            ("original_html_git_blob","0"*40),
            ("model_id","Different/Model"),
            ("model_revision","0"*40),
            ("source_research_scenes",12),
            ("pretrained_no_optimizer_steps",1),
            ("source_public_previously_exposed",False),
            ("official_train_dev_ood_admitted",True),
            ("independent_blind_ood",True),
            ("trained_adapter_gain_proven",True),
            ("raw_model_output_media_points_gold_or_weights_saved",True),
            ("real_model_visual_generations",0),
            ("original_firstparty_task_successes",2),
        ):
            doc=deepcopy(self.report)
            doc[field]=value
            with self.subTest(field=field), self.assertRaises(EvidenceRejected):
                verify_success(doc)
        doc=deepcopy(self.report)
        doc["rows"][0]["raw_model_action"]={"point":[100,200]}
        with self.assertRaises(EvidenceRejected):
            verify_success(doc)
        doc=deepcopy(self.report)
        doc["hardware"]["free_t4"]=False
        with self.assertRaises(EvidenceRejected):
            verify_success(doc)

    def test_strict_stage_only_source_failure_no_fabricated_score(self):
        original=safe_failure_receipt("REAL_BROWSER","ValueError")
        safe=verify_failure(original)
        self.assertEqual(safe["failure_stage"],"REAL_BROWSER")
        self.assertEqual(safe["exception_class"],"ValueError")
        self.assertFalse(safe["verified_native_model_score"])
        for field,value in (
            ("failure_stage","MODEL_ACCURACY"),
            ("exception_class","private traceback text"),
            ("verified_model_skill_scores",True),
            ("verified_optimizer_steps",8),
            ("blind_ood_proven",True),
        ):
            doc=deepcopy(original)
            doc[field]=value
            with self.subTest(field=field), self.assertRaises(EvidenceRejected):
                verify_failure(doc)
        self.assertEqual(
            safe_failure_receipt("DEPENDENCIES","SensitiveException")["exception_class"],
            "OTHER_ERROR_CLASS")

    def test_actual_private_file_exact_basename_and_raw_data_rejected(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            good=root/REPORT
            good.write_text(json.dumps(self.report),encoding="utf-8")
            self.assertEqual(verify_private(good)["genuine_firstparty_task_successes"],1)
            failed=root/DIAGNOSTIC_REPORT
            failed.write_text(json.dumps(
                safe_failure_receipt("DEPENDENCIES","RuntimeError")),encoding="utf-8")
            self.assertFalse(verify_private(
                failed,diagnostic=True)["verified_native_model_score"])
            wrong=root/"private-model-output.json"
            wrong.write_bytes(good.read_bytes())
            with self.assertRaises(EvidenceRejected):
                verify_private(wrong)
            try:
                link=root/REPORT
                alias=root/"alias-h299.json"
                alias.symlink_to(good)
            except (OSError,NotImplementedError):
                return
            with self.assertRaises(EvidenceRejected):
                verify_private(alias)

    def test_cpu_private_bundle_exactly_two_files_no_model_media(self):
        source=ROOT/"h299"/"private_pretrained_fieldport_pilot.py"
        with TemporaryDirectory() as tmp:
            target=Path(tmp)/"private"
            safe=build(source,target)
            self.assertEqual(safe["media_weights_datasets_gold_uploaded"],0)
            self.assertEqual(safe["python_source_files"],1)
            self.assertEqual(sorted(x.name for x in target.iterdir()),[
                "h299_private_pretrained_fieldport.py","kernel-metadata.json"])
            m=json.loads((target/"kernel-metadata.json").read_text("utf-8"))
            self.assertEqual(m["id"],KERNEL_ID)
            self.assertEqual(m["is_private"],"true")
            self.assertEqual(m["machine_shape"],"NvidiaTeslaT4")
            self.assertEqual(m["model_sources"],[])
            self.assertEqual(m["dataset_sources"],[])
            compiled=(target/"h299_private_pretrained_fieldport.py").read_text("utf-8")
            ast.parse(compiled)
            self.assertIn("def run() -> dict",compiled)
            self.assertIn("H299_PRIVATE_STAGE_ONLY_NOT_A_MODEL_SCORE",compiled)
            with self.assertRaises(ValueError):
                build(source,target)

    def test_duplicate_json_key_nan_and_invalid_private_report(self):
        from h280.verify_private_native_aggregate import decode_bounded
        raw=json.dumps(self.report,separators=(",",":")).encode("utf-8")
        for data in (
            raw.replace(b'"status":',b'"\\u0073tatus":"fake","status":',1),
            raw.replace(b'"real_model_visual_generations":3',
                        b'"real_model_visual_generations":NaN',1),
            b"invalid private report",b" " * 5000,
        ):
            with self.subTest(length=len(data)),self.assertRaises(EvidenceRejected):
                verify_success(decode_bounded(data))
