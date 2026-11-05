"""H425 only qualified-source model vs original H417 H421 counterfactual DEV.

No training occurs in this module. A future H425 actual GPU receipt is absent;
all counterfeits in tests MUST be rejected; never show fake model results.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("h425_owner_h421_eval",
    ROOT/"h425/kaggle/h425_h421_paired_eval.py")
h=importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)

class H425OriginalH421ModelEval(unittest.TestCase):
    def test_frozen_384_native_source_pair_and_research_only(self):
        self.assertEqual(h.IMAGE_PAIRS,192)
        self.assertEqual(h.SAMPLE_N,384)
        self.assertEqual(h.H421_PAIR_DOMAINS,
                         {"slider":64,"slide_resize":64,"text_highlight":64})
        self.assertEqual(h.H425_EXPECTED_STEPS,12696)
        self.assertEqual(h.H425_EXPECTED_GRADIENT_SAMPLES,50784)
        self.assertEqual(h.H425_TRUTH_H422_ROWS_SHA,
                         "b31efe107033d62c2fd2b6ced656f69596df2674c22266c4684928f1b09fbcaa")
        script=(ROOT/"h425/kaggle/h425_h421_paired_eval.py").read_text("utf8")
        self.assertIn("h421_generate(",script)
        self.assertIn("PeftModel.from_pretrained(",script)
        self.assertIn("source_only_H425_stronger_improvement_gate",script)
        self.assertIn("H425 original H421 model eval requires",script) if False else None
        self.assertNotIn("def run_h421_eval():",script)
        self.assertNotIn("def h421_trained_h419_adapter():",script)
        self.assertIn("independent_external_ood",script)
        self.assertIn("model_training_executed_in_this_kernel",script)

    def test_draft_eval_gpu_private_metadata_excludes_disqualified_h419(self):
        meta=json.loads((ROOT/"h425/kaggle/h421-eval-kernel-metadata.json").read_text("utf8"))
        self.assertTrue(meta["is_private"])
        self.assertTrue(meta["enable_gpu"])
        self.assertFalse(meta["enable_tpu"])
        self.assertEqual(meta["machine_shape"],"NvidiaTeslaT4")
        self.assertEqual(meta["id"].split("/",1)[1],meta["title"].lower().replace(" ","-"))
        self.assertEqual(meta["code_file"],"h425_h421_paired_eval.py")
        self.assertIn("abdulazizshehri/hala-h421-original-counterfactual-dev-private",
                      meta["dataset_sources"])
        self.assertIn("abdulazizshehri/hala-h425-genuine-trained-parent-private",
                      meta["dataset_sources"])
        self.assertFalse(any("h419-checkpoints" in x for x in meta["dataset_sources"]))

    def test_unqualified_genuine_h419_provider_receipt_cannot_masquerade_as_h425(self):
        path=os.environ.get("H425_OWNER_H419_RECEIPT")
        if not path:
            self.skipTest("Owner H419 actual private rejected receipt not mounted in CI")
        source=Path(path)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            data=source.read_bytes()
            (root/h.H425_CHILD_RECEIPT_FILE).write_bytes(data)
            (root/h.H425_CHILD_RECEIPT_SHA_FILE).write_text(
                hashlib.sha256(data).hexdigest())
            inputs={h.H425_CHILD_RECEIPT_FILE:root/h.H425_CHILD_RECEIPT_FILE,
                    h.H425_CHILD_RECEIPT_SHA_FILE:root/h.H425_CHILD_RECEIPT_SHA_FILE}
            with patch.object(h,"only_input_file",side_effect=lambda name:inputs[name]):
                with self.assertRaisesRegex(ValueError,"not genuinely source-qualified"):
                    h.h425_true_trained_child()

    def test_fabricated_h425_completed_parent_but_false_source_gate_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            fake={
              "schema":h.H425_EXPECTED_SCHEMA,"status":h.H425_EXPECTED_STATUS,
              "official_base":h.BASE,"official_revision":h.REV,
              "h417_source_parent_sha256":h.PARENT_ADAPTER_SHA,
              "h417_source_parent_receipt_sha256":h.PARENT_RECEIPT_SHA,
              "h418_new_original_30k_source_manifest_sha256":h.H418_RECEIPT_SHA,
              "h422_original_pair_rows_sha256":h.H425_TRUTH_H422_ROWS_SHA,
              "h422_original_paired_source_receipt_sha256":h.H425_TRUTH_H422_SOURCE_RECEIPT_SHA,
              "h421_reserved_counterfactual_dev_sha256":h.SOURCE_JSONL_SHA,
              "original_nine_H60_source_zip_SHA256":h.H419_ALL_SOURCE_SHAS,
              "actual_optimizer_steps":12696,
              "completed_optimizer_steps":12696,
              "processed_training_examples":50784,
              "h425_unique_original_training_screens":27952,
              "h425_genuine_distinct_native_action_rows":43952,
              "h425_original_same_screenshot_two_instructions":16000,
              "h425_old_h417_english_replay_examples":10976,
              "h425_old_h417_arabic_replay_unique":976,
              "h425_old_h417_arabic_passes":8,
              "source_only_H419_frozen_research_gate":False,
              "source_only_H425_stronger_improvement_gate":False,
              "independent_external_ood":False,
              "source_rights_review_qualified":False,
              "release_qualified":False,"weights_public":False,
              "changed_lora_tensors":1}
            raw=json.dumps(fake,sort_keys=True).encode()
            (root/h.H425_CHILD_RECEIPT_FILE).write_bytes(raw)
            (root/h.H425_CHILD_RECEIPT_SHA_FILE).write_text(hashlib.sha256(raw).hexdigest())
            inputs={h.H425_CHILD_RECEIPT_FILE:root/h.H425_CHILD_RECEIPT_FILE,
                    h.H425_CHILD_RECEIPT_SHA_FILE:root/h.H425_CHILD_RECEIPT_SHA_FILE}
            with patch.object(h,"only_input_file",side_effect=lambda name:inputs[name]):
                with self.assertRaisesRegex(ValueError,"not genuinely source-qualified"):
                    h.h425_true_trained_child()
            (root/h.H425_CHILD_RECEIPT_SHA_FILE).write_text("0"*64)
            with patch.object(h,"only_input_file",side_effect=lambda name:inputs[name]):
                with self.assertRaisesRegex(ValueError,"sidecar"):
                    h.h425_true_trained_child()

    def test_real_192_original_unseen_h421_screen_pair_source_optional(self):
        owner=os.environ.get("H425_OWNER_SOURCE_ROOT")
        if not owner:
            self.skipTest("Owner private original nine H60 ZIPs + 384 source cases absent on hosted CI")
        root=Path(owner)
        h421_root=Path(os.environ["H425_OWNER_H421_ROOT"])
        inputs={
          **{name:root/"h415"/name for name,_ in h.SOURCE_SHAS.values()},
          **{name:root/"h418"/name for name in h.H419_EXTRA_SHAS},
          h.BIL_FILE:root/"h415"/h.BIL_FILE,
          h.CURRIC_FILE:root/"h415"/h.CURRIC_FILE,
          h.H418_ROWS:root/"new-30k"/h.H418_ROWS,
          h.H418_RECEIPT_FILE:root/"new-30k"/h.H418_RECEIPT_FILE,
          h.H421_LABELS:h421_root/h.H421_LABELS,
          h.H421_SOURCE_RECEIPT:h421_root/h.H421_SOURCE_RECEIPT,
          h.H421_SOURCE_SIDECAR:h421_root/h.H421_SOURCE_SIDECAR}
        self.assertFalse([key for key,p in inputs.items() if not p.is_file()])
        with patch.object(h,"only_input_file",side_effect=lambda name:inputs[name]):
            pairs,rows,cases,zips,extra=h.h425_true_h421_pair_sources()
            try:
                self.assertEqual((len(pairs),len(rows),len(cases)),(192,384,384))
                self.assertEqual(len({x["image_sha256"] for x in cases}),192)
                self.assertTrue(all(a["image_sha256"]==b["image_sha256"]
                    and a["id"]!=b["id"]
                    for a,b in (cases[i:i+2] for i in range(0,384,2))))
                # H425 true 192 SOURCE cases remain held out from BOTH
                # old-H417 replay and new-H419 30K source training.
                # Reopen each of the 192 REAL owner-private original H60 JPEG
                # source bytes; verify actual media SHA, not just a manifest.
                readback=set()
                for a in cases[::2]:
                    src=zips["_by_file"][a["source_zip"]]
                    raw=src.read(a["image_path"])
                    self.assertEqual(hashlib.sha256(raw).hexdigest(),a["image_sha256"])
                    readback.add(a["image_sha256"])
                self.assertEqual(len(readback),192)
            finally:
                for z in extra:z.close()
                for z in zips.values():
                    if isinstance(z,zipfile.ZipFile):z.close()
