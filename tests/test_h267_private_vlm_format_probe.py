"""H267 synthetic-only CPU contracts; NEVER run a model or read private media."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import Mock

from h199.native_action_sft import source_task_prompt
from h267.free_kaggle_format_probe import (
    ARMS, ALLOWED_FORMATS, STRICT_SUFFIX,
    ClassifyingProcessor, classify_raw_text, make_private_report,
    prompt_variant,
)
from h267.build_private_payload import make, KERNEL_ID, META, SCRIPT_FILE


def rows():
    return [
        {"arm":ARMS[0],"generate_calls":1,"format":"NOT_JSON",
         "native_input_dispatched":False,"task_succeeded":False},
        {"arm":ARMS[1],"generate_calls":1,"format":"STRICT_NATIVE_JSON",
         "native_input_dispatched":True,"task_succeeded":True},
    ]


class FormatProbeTests(unittest.TestCase):
    def test_read_only_format_classification_never_returns_raw_response(self):
        cases=(
            ('{"actions":[{"kind":"click","point":[11,12]}]}',
             "STRICT_NATIVE_JSON"),
            ('{"actions":[{"kind":"abstain"}]}', "STRICT_NATIVE_JSON"),
            ('{"other":"invalid"}', "JSON_WRONG_SCHEMA"),
            ("\n```json\n{}\n```", "MARKDOWN_FENCE"),
            ("I will click this button.", "NOT_JSON"),
            ("", "EMPTY_TEXT"),
            (None, "EMPTY_TEXT"),
        )
        for raw, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(classify_raw_text(raw),expected)
                self.assertIn(expected,ALLOWED_FORMATS)

    def test_real_processor_decode_is_passed_through_unchanged(self):
        actual=Mock()
        actual.batch_decode.return_value=["{\"actions\":[{\"kind\":\"abstain\"}]}"]
        fake=ClassifyingProcessor(actual)
        self.assertEqual(fake.format_class,"NO_DECODE")
        self.assertIs(fake.apply_chat_template("hi"),actual.apply_chat_template.return_value)
        returned=fake.batch_decode(["fake_token"],skip_special_tokens=True)
        self.assertEqual(returned, actual.batch_decode.return_value)
        self.assertEqual(fake.format_class,"STRICT_NATIVE_JSON")
        actual.batch_decode.return_value=["sensitive invented text"]
        fake.batch_decode(["fake_token"])
        self.assertEqual(fake.format_class,"NOT_JSON")
        self.assertNotIn("sensitive",str(fake.__dict__))

    def test_prompt_preserves_h199_screenshot_only_contract(self):
        base=source_task_prompt("Add one unit to the first visible stock item.")
        self.assertEqual(prompt_variant(base,ARMS[0]),base)
        self.assertEqual(prompt_variant(base,ARMS[1]),base+STRICT_SUFFIX)
        self.assertLess(len(prompt_variant(base,ARMS[1])),2000)
        for wrong in ("wrong", "", None):
            with self.assertRaises(ValueError):
                prompt_variant(wrong,ARMS[0])
        with self.assertRaises(ValueError):
            prompt_variant(base,"surprise")

    def test_two_arm_aggregate_has_no_predictions_gold_or_media(self):
        report=make_private_report(rows(),{"free_t4_gpu_verified":True,"device_count":2})
        self.assertEqual(report["real_generate_calls"],2)
        self.assertEqual(report["first_party_task_successes"],1)
        self.assertFalse(report["official_train_dev_ood_admitted"])
        self.assertFalse(report["source_independent_blind_ood"])
        self.assertFalse(report["trained_hala_model_gain_proven"])
        self.assertFalse(report["model_weights_in_kaggle_working_outputs"])
        self.assertNotIn("coordinate",json.dumps(report))

    def test_fail_closed_on_fake_model_or_reference_misclassification(self):
        for alteration in (
            {"task_succeeded":True,"native_input_dispatched":False},
            {"generate_calls":0,"native_input_dispatched":True},
            {"format":"NOT_JSON","native_input_dispatched":True},
            {"format":"raw text"},
            {"generate_calls":2},
        ):
            bad=rows()
            bad[1].update(alteration)
            with self.subTest(alteration=alteration),self.assertRaises(ValueError):
                make_private_report(bad,{"free_t4_gpu_verified":True})
        with self.assertRaises(ValueError):
            make_private_report(rows(),{"free_t4_gpu_verified":False})
        with self.assertRaises(ValueError):
            make_private_report(rows()[:1],{"free_t4_gpu_verified":True})

    def test_private_kaggle_payload_no_dataset_no_weight(self):
        source=Path(__file__).resolve().parents[1]/"h267"/"free_kaggle_format_probe.py"
        with TemporaryDirectory() as root:
            destination=Path(root)/"fresh-only"
            proof=make(source,destination)
            metadata=json.loads((destination/META).read_text("utf-8"))
            code=(destination/SCRIPT_FILE).read_text("utf-8")
            self.assertEqual(proof["kernel"],KERNEL_ID)
            self.assertEqual(metadata["is_private"],"true")
            self.assertEqual(metadata["machine_shape"],"NvidiaTeslaT4")
            self.assertEqual(metadata["dataset_sources"],[])
            self.assertEqual(metadata["model_sources"],[])
            self.assertEqual({p.name for p in destination.iterdir()},
                             {SCRIPT_FILE,META})
            self.assertTrue(code.startswith("from __future__ import annotations"))
            self.assertIn("HALA_H267_NONCOMMERCIAL_FREE_GPU",code)
            self.assertIn("--with-deps",code)
            self.assertIn('/tmp/hala-h267-research',code)
            self.assertNotIn('HF_HOME="/kaggle/working',code)
            with self.assertRaises(ValueError):
                make(source,destination)


if __name__=="__main__":
    unittest.main()
