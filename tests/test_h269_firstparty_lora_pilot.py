"""H269 fail-closed research source contracts; no GPU/data/network activity."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
import json
import unittest

from h269.firstparty_lora_probe import (
    N_STEPS,MODES,RESULT,TASK,training_summary,require,
    check_free_environment,
)
from h269.build_private_payload import (
    build_payload,KERNEL_ID,SCRIPT_NAME,META_NAME,LMFE_GIT_PIN,
)


def good_stats():
    return [
        {"changed_tensors":112,"grad_norm":2.5,"loss":3.5},
        {"changed_tensors":112,"grad_norm":1.2,"loss":3.3},
    ]


def eval_row(success=False):
    return {
        "model_generation_complete":True,
        "grammar_applied_during_generation":True,
        "strict_format":True,
        "action_kind":"click",
        "native_action_dispatched":True,
        "first_party_task_success":success,
    }


class H269FirstpartyLoraTests(unittest.TestCase):
    def test_firstparty_scope_and_two_step_boundaries(self):
        self.assertEqual(N_STEPS,2)
        self.assertEqual(MODES,("default_en","sorted_en","heldout_ar"))
        self.assertNotIn("train",MODES[-1])
        self.assertEqual(RESULT,"hala-h269-private-firstparty-lora-aggregate.json")
        self.assertIn("FIRST visible inventory item",TASK)
        with self.assertRaises(ValueError):
            require(False,"stopped")

    def test_synthetic_step_summary_cannot_claim_generalized_gain(self):
        row=training_summary(good_stats(),eval_row(False),
                eval_row(True),{"free_t4":True,"devices":2})
        self.assertEqual(row["real_optimizer_steps"],2)
        self.assertFalse(row["independent_source_disjoint_ood"])
        self.assertFalse(row["official_train_dev_ood_admitted"])
        self.assertFalse(row["consumed_h173_ood_used"])
        self.assertFalse(row["learned_generalizable_gui_gain_proven"])
        self.assertFalse(row["trained_model_or_weights_published"])
        self.assertTrue(row["heldout_ar_not_in_train_images"])
        self.assertEqual(len(row["steps_aggregate_only"]),2)
        text=json.dumps(row)
        for forbidden in ("image_path","point","model_output","gold","image_sha256"):
            self.assertNotIn(forbidden,text)

    def test_rejects_missing_training_or_fake_gpu_or_unchanged_adapter(self):
        inputs=[
            (good_stats()[:1],{"free_t4":True}),
            ([{"changed_tensors":0,"grad_norm":1.0,"loss":2.0},good_stats()[1]],{"free_t4":True}),
            ([{"changed_tensors":10,"grad_norm":float("nan"),"loss":2.0},good_stats()[1]],{"free_t4":True}),
            (good_stats(),{"free_t4":False}),
        ]
        for steps,env in inputs:
            with self.subTest(steps=steps,env=env),self.assertRaises(ValueError):
                training_summary(steps,eval_row(),eval_row(),env)
        bad=eval_row()
        bad["grammar_applied_during_generation"]=False
        with self.assertRaises(ValueError):
            training_summary(good_stats(),bad,eval_row(),{"free_t4":True})

    def test_private_payload_refuses_media_weights_or_altered_source(self):
        source=Path(__file__).resolve().parents[1]/"h269"/"firstparty_lora_probe.py"
        with TemporaryDirectory() as td:
            stage=Path(td)/"stage"
            result=build_payload(source,stage)
            meta=json.loads((stage/META_NAME).read_text("utf-8"))
            text=(stage/SCRIPT_NAME).read_text("utf-8")
            self.assertEqual(result["kernel"],KERNEL_ID)
            self.assertEqual({p.name for p in stage.iterdir()},
                             {SCRIPT_NAME,META_NAME})
            self.assertEqual(meta["is_private"],"true")
            self.assertEqual(meta["machine_shape"],"NvidiaTeslaT4")
            self.assertEqual(meta["dataset_sources"],[])
            self.assertEqual(meta["model_sources"],[])
            self.assertEqual(meta["kernel_sources"],[])
            self.assertIn("peft==0.20.0",text)
            self.assertIn('if _torchao != "0.10.0":',text)
            self.assertIn('"uninstall",',text)
            self.assertIn('pkg_version("torch") != old_torch',text)
            self.assertIn(LMFE_GIT_PIN,text)
            self.assertIn('"/tmp/hala-h269-private"',text)
            self.assertNotIn('HF_HOME="/kaggle/working',text)
            with self.assertRaises(ValueError):
                build_payload(source,stage)

    def test_no_owner_windows_gpu_model_download(self):
        with self.assertRaises(ValueError):
            check_free_environment(Mock())


if __name__=="__main__":
    unittest.main()
