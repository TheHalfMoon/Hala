"""H421 real GPU eval-only runner source contract and fail-closed input tests."""
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from h420.kaggle import h421_native_pair_model_eval as m

ROOT=Path(__file__).resolve().parents[1]


def original_case():
    name="h60-dragon-train-slider-0005.hala.bin"
    return {
      "schema":"hala.h421.same-original-image-two-instructions.v1",
      "pair_id":0,"alternative_index":0,
      "id":"h60-slider-unseen-original-task",
      "source_domain":"slider",
      "source_zip":name,
      "source_zip_sha256":m.H419_ALL_SOURCE_SHAS[name],
      "split":"dev","source_original_split":"train",
      "language":"en",
      "image_path":"images/slider/original-unseen.jpg",
      "image_sha256":"a"*64,
      "image_width":1024,"image_height":640,
      "intent":"Move the slider to the other end",
      "native_drag_target":{
        "type":"drag","start_point":[113,210],
        "end_point":[841,720],"ordered":True,
        "coordinate_space":"normalized_1000"},
      "source_original_jpeg_bytes_sha_verified":True,
      "external_independent_ood":False,
    }


class H421NativeRunnerTests(unittest.TestCase):
    def test_actual_model_eval_is_standalone_private_eval_only(self):
        src=(ROOT/"h420/kaggle/h421_native_pair_model_eval.py").read_text()
        self.assertEqual(src.count("from __future__ import annotations"),1)
        self.assertEqual(src.count('if __name__=="__main__":'),1)
        self.assertIn("run_h421_eval()",src)
        self.assertIn("torch.inference_mode()",src)
        self.assertIn("H421_REAL_MODEL_SAME_IMAGE_TWO_PROMPT",src)
        self.assertNotIn("optimizer.step()",src)
        self.assertNotIn(".backward()",src)
        self.assertEqual(m.OUT.name,"h421-private-two-instruction-native-eval")
        self.assertEqual(m.H421_H419_WEIGHTS,"h419_native_adapter.safetensors")
        self.assertNotEqual(m.H421_H419_WEIGHTS,m.PARENT_ADAPTER_FILE)

    def test_native_two_point_original_geometry_preserved_for_model(self):
        source=original_case()
        case=m.h421_model_case(source)
        self.assertEqual(case["kind"],"original")
        self.assertEqual(case["language"],"en")
        self.assertEqual(case["domain"],"slider")
        self.assertEqual(case["source_zip"],source["source_zip"])
        self.assertEqual(case["image_sha256"],source["image_sha256"])
        self.assertEqual(case["body"],source["intent"])
        self.assertEqual(case["start"],[116,134])
        self.assertEqual(case["end"],[861,461])
        normalized=m.target(case,1024,640)
        for key in ("start_point","end_point"):
            self.assertLessEqual(max(abs(x-y) for x,y in zip(
                normalized[key],source["native_drag_target"][key])),1)
        self.assertIn(source["intent"],m.prompt(case))

    def test_reject_unqualified_source_or_schema_as_model_input(self):
        good=original_case()
        for bad in (
            dict(good,source_zip_sha256="b"*64),
            dict(good,language="ar"),
            dict(good,source_domain="sheet"),
            dict(good,image_width=0),
            dict(good,native_drag_target={
                **good["native_drag_target"],"type":"click"}),
        ):
            with self.subTest(reason=list(bad.items())[-1][0]):
                with self.assertRaises(ValueError):
                    m.h421_model_case(bad)

    def test_h419_incomplete_original_receipt_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            doc={"schema":"hala.h419.private-new-30000-original-native-drag.v1",
                 "status":"NOT_STARTED","actual_optimizer_steps":0,
                 "model_release_qualified":False}
            raw=json.dumps(doc).encode()
            (root/m.H421_H419_RECEIPT).write_bytes(raw)
            (root/m.H421_H419_RECEIPT_SIDECAR).write_text(
                sha256(raw).hexdigest())
            files={name:root/name for name in
                   (m.H421_H419_RECEIPT,m.H421_H419_RECEIPT_SIDECAR)}
            with patch.object(m,"only_input_file",side_effect=lambda name:files[name]):
                with self.assertRaisesRegex(ValueError,"unqualified H419"):
                    m.h421_trained_h419_adapter()

    def test_source_384_frozen_owner_receipt_available(self):
        proof=(ROOT/"h420/verified_receipt/h421-original-counterfactual-dev/receipt.json")
        self.assertEqual(sha256(proof.read_bytes()).hexdigest(),
                         m.SOURCE_RECEIPT_SHA)
        self.assertEqual(m.IMAGE_PAIRS,192)
        self.assertEqual(m.SAMPLE_N,384)
        self.assertFalse(m.H421_EVAL_STATUS.endswith("RELEASE"))

if __name__=="__main__":
    unittest.main()
