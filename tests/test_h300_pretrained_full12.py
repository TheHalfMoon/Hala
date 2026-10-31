"""H300 original exposed full12 strict private receipt tests: INVENTED DATA ONLY."""
from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from h280.verify_private_native_aggregate import EvidenceRejected, decode_bounded
from h300.private_pretrained_fieldport_full12 import (
    SOURCE_SHA, SOURCE_BLOB, MODEL_ID, MODEL_REV, REPORT,
    DIAGNOSTIC_REPORT, SCENE_INDICES, verified_report, safe_failure_receipt,
)
from h300.verify_private_full12_receipt import (
    verify_success, verify_failure, verify_private,
)
from h300.build_private_payload import KERNEL_ID, KERNEL_TITLE, build

ROOT = Path(__file__).resolve().parents[1]


def rows_for(value: str):
    result = []
    for i in range(12):
        success = value == "all" or (value == "mixed" and i % 3 != 0)
        if success:
            result.append({
                "language":"en" if i < 6 else "ar",
                "generations":2,"grammar_calls":2,
                "screenshot_steps":2,"focus_clicks":1,"type_calls":1,
                "abstained":False,"task_success":True,
                "all_four_source_fields_checked":True,
            })
        else:
            result.append({
                "language":"en" if i < 6 else "ar",
                "generations":1,"grammar_calls":1,
                "screenshot_steps":1,"focus_clicks":0,"type_calls":0,
                "abstained":True,"task_success":False,
                "all_four_source_fields_checked":True,
            })
    return result


class H300PrivateFull12SourceResearchTests(TestCase):
    def setUp(self):
        self.hw={"free_t4":True,"devices":1}
        self.rows=rows_for("mixed")
        self.doc=verified_report(self.rows,self.hw)

    def test_original_public_source_and_pretrained_model_exact_pins(self):
        self.assertEqual(SCENE_INDICES,tuple(range(12)))
        self.assertEqual(SOURCE_SHA,"2babf8c91ffb63dd1123070449d1cbc240ab0859")
        self.assertEqual(SOURCE_BLOB,"14bc9041fabf59d76adb0fc78b6a16d6d53c6e80")
        self.assertEqual(MODEL_ID,"Qwen/Qwen3-VL-2B-Instruct")
        self.assertEqual(len(MODEL_REV),40)
        self.assertEqual(KERNEL_ID,
            "abdulazizshehri/hala-h300-private-pretrained-fieldport-full12-t4")
        self.assertEqual(KERNEL_TITLE,
            "Hala H300 Private Pretrained FieldPort Full12 T4")

    def test_strict_original_true_model_time_metadata_only_receipt(self):
        safe=verify_success(self.doc)
        self.assertEqual(safe["original_research_cases"],12)
        self.assertEqual(safe["genuine_firstparty_task_successes"],8)
        self.assertEqual(safe["model_time_pretrained_generations"],20)
        self.assertEqual(safe["true_model_time_grammar_calls"],20)
        self.assertEqual(safe["by_language"][0]["language"],"en")
        self.assertEqual(safe["by_language"][1]["language"],"ar")
        self.assertEqual(safe["by_language"][0]["cases"],6)
        self.assertEqual(safe["by_language"][1]["cases"],6)
        self.assertEqual(safe["by_language"][0]["task_successes"],4)
        self.assertEqual(safe["by_language"][1]["task_successes"],4)
        self.assertNotIn("per_language",safe)
        self.assertFalse(safe["blind_ood_proven"])
        self.assertFalse(safe["adapted_model_gain_proven"])
        for leak in ("point","FIC-11","task_text","gold","weights","png"):
            self.assertNotIn(leak,json.dumps(safe))

    def test_zero_some_and_all_source_task_success_prospective(self):
        for mode,total in (("none",0),("mixed",8),("all",12)):
            with self.subTest(mode=mode):
                receipt=verified_report(rows_for(mode),self.hw)
                outcome=verify_success(receipt)
                self.assertEqual(outcome["genuine_firstparty_task_successes"],total)
                self.assertEqual(
                    sum(g["task_successes"] for g in outcome["by_language"]),total)

    def test_no_source_split_role_model_weight_ood_promotion(self):
        for field,value in (
            ("source_commit","0"*40),
            ("original_html_git_blob","0"*40),
            ("model_revision","0"*40),
            ("source_research_scenes",2),
            ("official_train_dev_ood_admitted",True),
            ("independent_blind_ood",True),
            ("trained_adapter_gain_proven",True),
            ("source_public_previously_exposed",False),
            ("pretrained_no_optimizer_steps",8),
            ("real_model_visual_generations",999),
            ("raw_model_output_media_points_gold_or_weights_saved",True),
        ):
            doc=deepcopy(self.doc)
            doc[field]=value
            with self.subTest(field=field),self.assertRaises(EvidenceRejected):
                verify_success(doc)

    def test_no_extra_raw_fields_false_languages_or_fake_browser_actions(self):
        for idx,field,value in (
            (1,"language","ar"),(7,"language","en"),
            (0,"task_success",True),(0,"type_calls",1),
            (0,"generations",2),(1,"grammar_calls",1),
            (1,"all_four_source_fields_checked",False),
        ):
            doc=deepcopy(self.doc)
            doc["rows"][idx][field]=value
            with self.subTest(field=field,idx=idx),self.assertRaises(EvidenceRejected):
                verify_success(doc)
        doc=deepcopy(self.doc)
        doc["rows"][0]["raw_private_model_output"]={"point":[100,200]}
        with self.assertRaises(EvidenceRejected):
            verify_success(doc)
        doc=deepcopy(self.doc)
        doc["rows"].append(deepcopy(doc["rows"][-1]))
        with self.assertRaises(EvidenceRejected):
            verify_success(doc)

    def test_failure_only_receipt_never_model_score(self):
        diag=safe_failure_receipt("MODEL_LOAD","RuntimeError")
        safe=verify_failure(diag)
        self.assertFalse(safe["verified_native_model_score"])
        self.assertEqual(safe["failure_stage"],"MODEL_LOAD")
        for key,value in (
            ("failure_stage","MODEL_SUCCEEDED"),
            ("exception_class","private path or traceback"),
            ("verified_model_skill_scores",True),
            ("blind_ood_proven",True),
        ):
            wrong=deepcopy(diag)
            wrong[key]=value
            with self.subTest(key=key),self.assertRaises(EvidenceRejected):
                verify_failure(wrong)

    def test_only_exact_private_basenames_and_no_symlinks(self):
        with TemporaryDirectory() as root:
            folder=Path(root)
            aggregate=folder/REPORT
            aggregate.write_text(json.dumps(self.doc),encoding="utf-8")
            self.assertEqual(verify_private(aggregate)["original_research_cases"],12)
            failure=folder/DIAGNOSTIC_REPORT
            failure.write_text(json.dumps(
                safe_failure_receipt("SOURCE","ValueError")),encoding="utf-8")
            self.assertFalse(verify_private(failure,diagnostic=True)["verified_native_model_score"])
            copied=folder/"owner-raw-results.json"
            copied.write_bytes(aggregate.read_bytes())
            with self.assertRaises(EvidenceRejected):
                verify_private(copied)
            alias=folder/"alias.h300-private.json"
            try:
                alias.symlink_to(aggregate)
            except (NotImplementedError,OSError):
                return
            with self.assertRaises(EvidenceRejected):
                verify_private(alias)

    def test_private_kaggle_package_two_code_metadata_only(self):
        source=ROOT/"h300"/"private_pretrained_fieldport_full12.py"
        with TemporaryDirectory() as tmp:
            out=Path(tmp)/"private"
            stats=build(source,out)
            self.assertEqual(stats["media_weights_datasets_gold_uploaded"],0)
            self.assertEqual(sorted(x.name for x in out.iterdir()),[
                "h300_private_pretrained_fieldport_full12.py","kernel-metadata.json"])
            metadata=json.loads((out/"kernel-metadata.json").read_text("utf-8"))
            self.assertEqual(metadata["id"],KERNEL_ID)
            self.assertEqual(metadata["is_private"],"true")
            self.assertEqual(metadata["machine_shape"],"NvidiaTeslaT4")
            self.assertEqual(metadata["model_sources"],[])
            self.assertEqual(metadata["dataset_sources"],[])
            bundled=(out/"h300_private_pretrained_fieldport_full12.py").read_text("utf-8")
            ast.parse(bundled)
            self.assertIn("SCENE_INDICES = tuple(range(12))",bundled)
            self.assertIn("H300_PRIVATE_STAGE_ONLY_NOT_A_MODEL_SCORE",bundled)
            with self.assertRaises(ValueError):
                build(source,out)

    def test_duplicate_escaped_json_keys_nan_and_oversized_rejected(self):
        raw=json.dumps(self.doc,sort_keys=True).encode("utf-8")
        for bad in (
            raw.replace(b'"status":',b'"\\u0073tatus":"bad","status":',1),
            raw.replace(b'"real_model_visual_generations": 20',
                        b'"real_model_visual_generations": NaN',1),
            b"NOT_JSON",
            b" "*5000,
        ):
            with self.subTest(n=len(bad)),self.assertRaises(EvidenceRejected):
                verify_success(decode_bounded(bad))
