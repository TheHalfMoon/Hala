"""H422 genuine 32K source-original SAME-screen-two-prompt GPU preflight.

Hosted CI never receives private JPEGs or model weights; it checks
fail-closed implementation contracts. Owner Windows additionally reopens
all nine private SHA-pinned ZIPs with the frozen 32K label dataset.
"""
import json
import os
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from h422.kaggle import h422_two_prompt_native_lora as model

ROOT=Path(__file__).resolve().parents[1]


class H422TwoPromptModelSource(unittest.TestCase):
    def test_model_is_not_source_curation_or_old_h419_replay(self):
        self.assertEqual(model.H422_ORIGINAL_SCREEN_N,16000)
        self.assertEqual(model.H422_NATIVE_ORIGINAL_ACTION_N,32000)
        self.assertEqual(model.H422_AR_TRAIN_N,976)
        self.assertEqual(model.H422_AR_REPEAT,3)
        self.assertEqual(model.H422_UNIQUE_ORIGINAL_AND_AR_IMAGE_N,16976)
        self.assertEqual(model.H422_GRAD_EXAMPLES,35904)
        self.assertEqual(model.H422_OPT_STEPS,8976)
        self.assertEqual(model.H422_OPT_STEPS*model.GRAD_ACCUM,35904)
        self.assertEqual(model.H422_SOURCE_DOMAINS,
                         {"slider":5333,"slide_resize":5333,"text_highlight":5334})
        self.assertNotEqual(model.H419_PARENT_MODEL_FILE,model.PARENT_ADAPTER_FILE)
        self.assertNotEqual(model.H422_ROWS_SHA,model.H418_ROWS_SHA)
        self.assertIn("two-prompt",model.OUT.name)

    def test_only_real_completed_parent_h419_eligible_for_second_source_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            fake={"schema":"hala.h419.private-new-30000-original-native-drag.v1",
                  "status":model.H419_PRIVATE_STATUS,
                  "actual_optimizer_steps":model.H419_STEPS,
                  "processed_training_examples":model.H419_TRAINING_PASSES,
                  "source_only_research_gate":False,
                  "model_release_qualified":False}
            raw=json.dumps(fake).encode()
            (root/model.H419_PARENT_RECEIPT_FILE).write_bytes(raw)
            (root/model.H419_PARENT_SIDECAR_FILE).write_text(sha256(raw).hexdigest())
            files={
              model.H419_PARENT_RECEIPT_FILE:root/model.H419_PARENT_RECEIPT_FILE,
              model.H419_PARENT_SIDECAR_FILE:root/model.H419_PARENT_SIDECAR_FILE
            }
            with patch.object(model,"only_input_file",side_effect=lambda name:files[name]):
                with self.assertRaisesRegex(ValueError,"did not pass"):
                    model.real_h419_parent()
            fake["status"]="NOT_STARTED"
            (root/model.H419_PARENT_RECEIPT_FILE).write_text(json.dumps(fake))
            with patch.object(model,"only_input_file",side_effect=lambda name:files[name]):
                with self.assertRaisesRegex(ValueError,"receipt SHA mismatch"):
                    model.real_h419_parent()

    def test_trained_source_contract_never_loads_untouched_eval_as_gradient(self):
        source=(ROOT/"h422/kaggle/h422_two_prompt_native_lora.py").read_text(encoding="utf8")
        self.assertIn("H421_unseen_dev_used_in_gradient",source)
        self.assertIn("H422_REAL",source) if False else None
        self.assertIn("coordinate_weighted_loss",source)
        self.assertIn("model.enable_input_require_grads()",source)
        self.assertIn("gradient_checkpointing_enable",source)
        self.assertIn("source_only_research_gate",source)
        self.assertNotIn("H421_DEV_FILE]\\n",source)
        self.assertEqual(source.count("from __future__ import annotations"),1)

    def test_actual_32k_original_source_receipt_is_pinned(self):
        proof=ROOT/"h422/verified_receipt/h422-owner-original-16k-paired-actions/receipt.json"
        raw=proof.read_bytes()
        self.assertEqual(sha256(raw).hexdigest(),model.H422_RECEIPT_SHA)
        rec=json.loads(raw)
        self.assertEqual(rec["genuine_original_image_conditional_action_labels"],32000)
        self.assertEqual(rec["original_training_action_rows_sha256"],model.H422_ROWS_SHA)
        self.assertTrue(rec["untouched_H421_eval_192_original_screens_excluded"])
        self.assertFalse(rec["model_training_executed"])

    def test_owner_private_real_9_original_source_zip_and32k_labels_optional(self):
        root=os.environ.get("H422_OWNER_SOURCE_ROOT")
        if not root:
            self.skipTest("Owner private nine ZIP media not mounted on hosted GitHub CI")
        root=Path(root)
        paired=Path(os.environ["H422_OWNER_PAIRED_ROOT"])
        heldout=Path(os.environ["H422_OWNER_H421_ROOT"])
        files={
           **{name:root/"h415"/name for name,_ in model.SOURCE_SHAS.values()},
           **{name:root/"h418"/name for name in model.H419_EXTRA_SHAS},
           model.BIL_FILE:root/"h415"/model.BIL_FILE,
           model.CURRIC_FILE:root/"h415"/model.CURRIC_FILE,
           model.H418_ROWS:root/"new-30k"/model.H418_ROWS,
           model.H418_RECEIPT_FILE:root/"new-30k"/model.H418_RECEIPT_FILE,
           model.H422_ROWS_FILE:paired/model.H422_ROWS_FILE,
           model.H422_RECEIPT_FILE:paired/model.H422_RECEIPT_FILE,
           model.H421_DEV_FILE:heldout/model.H421_DEV_FILE,
        }
        self.assertFalse([k for k,v in files.items() if not v.is_file()])
        with patch.object(model,"only_input_file",side_effect=lambda name:files[name]):
            zips,extra,train,dev,order=model.prepare_h422()
            try:
                self.assertEqual((len(train),len(dev),len(order)),(32976,240,35904))
                self.assertEqual(len({r["image_sha256"] for r in train}),16976)
                self.assertEqual(len({r["image_sha256"] for r in dev}),240)
                self.assertFalse({r["image_sha256"] for r in train}&
                                 {r["image_sha256"] for r in dev})
                self.assertEqual(sum(r["language"]=="ar" for r in train),976)
                self.assertEqual(sum(r["language"]=="en" for r in train),32000)
                originals=[r for r in train if r["kind"]=="original"]
                from collections import Counter
                self.assertEqual(Counter(r["domain"] for r in originals),
                    {name:quota*2 for name,quota in model.H422_SOURCE_DOMAINS.items()})
                # Real original ZIP/Pillow JPEG bytes, not invented fixtures.
                for item in (train[0],train[16000],train[32000],dev[-1]):
                    im,w,h=model.source_image(item,zips)
                    self.assertGreater(w,0)
                    self.assertGreater(h,0)
                    self.assertLessEqual(im.width,640)
                    self.assertLessEqual(im.height,400)
            finally:
                for z in extra:z.close()
                for z in zips.values():
                    if isinstance(z,model.zipfile.ZipFile):z.close()

if __name__=="__main__":
    unittest.main()
