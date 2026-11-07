"""Synthetic adversarial contract tests, not actual model inference."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location(
    "h426_private_aggregate_gate",ROOT/"h426/verify_private_h425_baseline.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def baseline():
    domain={"images":2,"actions":4,"correct":2,"both_correct":0,
            "en_correct":1,"ar_correct":1,
            "en_end_only_error":0,"ar_end_only_error":0}
    return {
        "schema":"hala.h426.h425-pinned-readonly-synthetic-dev.v1",
        "model":m.BASE,"revision":m.REV,
        "actual_h425_adapter_sha256":m.MODEL_SHA,
        "actual_h425_receipt_sha256":m.RECEIPT_SHA,
        "processor_pinned_public_files":8,"device":"Tesla T4",
        "trainable_parameters":0,"new_training_steps":0,
        "source_role":"firstparty_invented_author_visible_dev",
        "images":6,"bilingual_actions":12,
        "domains":{name:dict(domain) for name in m.DOMAIN_SET},
        "raw_model_predictions_persisted":False,
        "model_weights_published":False,
        "independent_external_ood":False,
        "source_rights_review_qualified":False,
        "release_qualified":False,
    }


class H426ReadOnlyReceiptContract(unittest.TestCase):
    def test_synthetic_valid_contract_not_provider_proof(self):
        report=m.verify_aggregate(baseline())
        self.assertEqual(report["scored_images"],6)
        self.assertEqual(report["scored_actions"],12)
        self.assertFalse(report["provider_gpu_receipt_verified_here"])
        self.assertFalse(report["release_qualified"])

    def test_missing_or_extra_private_fields_refused(self):
        for alter in ("missing","extra"):
            value=baseline()
            if alter=="missing":del value["domains"]
            else:value["predictions"]=["private-leak"]
            with self.subTest(alter=alter),self.assertRaises(ValueError):
                m.verify_aggregate(value)

    def test_forged_model_rights_or_release_flag_refused(self):
        for change in (
            {"actual_h425_adapter_sha256":"0"*64},
            {"actual_h425_receipt_sha256":"0"*64},
            {"processor_pinned_public_files":7},
            {"new_training_steps":1},
            {"trainable_parameters":1},
            {"model_weights_published":True},
            {"release_qualified":True},
            {"source_rights_review_qualified":True},
            {"independent_external_ood":True},
            {"device":"cpu"},
            {"bilingual_actions":999},
        ):
            with self.subTest(change=change),self.assertRaises(ValueError):
                m.verify_aggregate({**baseline(),**change})

    def test_inconsistent_domain_metrics_fail_closed(self):
        changes=(
            ("images",3),("actions",3),("correct",5),
            ("en_correct",3),("both_correct",2),
            ("en_end_only_error",3),("ar_end_only_error",-1),
            ("correct",True),
        )
        for key, wrong in changes:
            value=baseline()
            value["domains"]["slider"][key]=wrong
            with self.subTest(key=key,wrong=wrong),self.assertRaises(ValueError):
                m.verify_aggregate(value)

    def test_bad_count_refused(self):
        for count in (0,33,True,2.5):
            with self.subTest(count=count),self.assertRaises(ValueError):
                m.verify_aggregate(baseline(),count_per_domain=count)

if __name__=="__main__":
    unittest.main()
