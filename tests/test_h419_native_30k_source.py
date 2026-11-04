"""H419 genuine original 30K native-drag GPU preflight; fail-closed private."""
from hashlib import sha256
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from h419.kaggle import h419_native_drag_continuation as h

ROOT=Path(__file__).resolve().parents[1]


class H419NativeOriginalPreflight(unittest.TestCase):
    def test_source_only_gpu_and_private_original_data(self):
        meta=json.loads((ROOT/"h419/kaggle/kernel-metadata.json").read_text())
        self.assertTrue(meta["is_private"])
        self.assertTrue(meta["enable_gpu"])
        self.assertEqual(meta["machine_shape"],"NvidiaTeslaT4")
        self.assertFalse(meta["enable_tpu"])
        self.assertEqual(len(meta["dataset_sources"]),6)
        self.assertEqual(set(meta["dataset_sources"]),{
            "abdulazizshehri/hala-h415-dragon-original-4domain-private",
            "abdulazizshehri/hala-h415-frozen-10k-native-drag-private",
            "abdulazizshehri/hala-h415-original-bilingual-drag-2k-private",
            "abdulazizshehri/hala-h418-original-30k-extension-private",
            "abdulazizshehri/hala-h418-verified-new-30k-original-drag-private",
            "abdulazizshehri/hala-h419-h417-native-parent-private",
        })
        self.assertEqual(meta["kernel_sources"],[])
        self.assertEqual(meta["model_sources"],[])
        self.assertEqual(len(h.H419_ALL_SOURCE_SHAS),9)
        self.assertEqual(h.H419_ALL_SOURCE_SHAS["h60-dragon-train-slider-0005.hala.bin"],
                         "cbd55b0a003d0c6aafef2460ce5bc8b0321e8a4237032053a53a077901483c96")

    def test_exact_30000_fresh_original_train_and_arabic_repeats(self):
        self.assertEqual(h.H419_NEW_IMAGE_N,30000)
        self.assertEqual(h.H419_AR_N,976)
        self.assertEqual(h.H419_AR_ADDITIONAL_PASSES,7)
        self.assertEqual(h.H419_UNIQUE_IMAGES,30976)
        self.assertEqual(h.H419_TRAINING_PASSES,37808)
        self.assertEqual(h.H419_STEPS,9452)
        self.assertEqual(h.H419_STEPS*h.GRAD_ACCUM,h.H419_TRAINING_PASSES)
        self.assertEqual(h.H419_NEW_DEV_N,96)
        self.assertEqual(h.H419_OLD_DEV_N,144)
        self.assertEqual(h.H419_DEV_N,240)
        self.assertEqual(h.H418_ROWS_SHA,
            "e7fce082d6ccd931515beacbc6cfc2455a6683c7c92566a245fda9f2182a1605")
        self.assertEqual(h.H418_RECEIPT_SHA,
            "b7b75d6aa2587c3eba83a3934fe4cdf3cacd48a5b970b5a60a640c89ee3054cf")
        self.assertEqual(h.PARENT_ADAPTER_SHA,
            "11f06aac6c225b86d1f30272eb852bca49fd82ce6e8553113b39b4cba4216df1")
        self.assertEqual(h.PARENT_RECEIPT_SHA,
            "227967230362eea759553ae5af892865bae4a3625947c45f30c0d9cb36f9f76a")

    @staticmethod
    def fixture():
        return {
          "schema":"hala.h418.original-native-drag-train.v1",
          "id":"h60-slider-test-original-t001",
          "source_domain":"slider",
          "source_zip":"h60-dragon-train-slider-0005.hala.bin",
          "source_zip_sha256":h.H419_ALL_SOURCE_SHAS[
              "h60-dragon-train-slider-0005.hala.bin"],
          "split":"train",
          "language":"en",
          "intent":"Move the original slider to the requested source position",
          "image_path":"images/slider/00000000-0000-0000-0000-000000000000.jpg",
          "image_sha256":"a"*64,
          "image_width":1024,
          "image_height":640,
          "original_source_task_sha256":"b"*64,
          "ordered":True,
          "native_drag_completion":{
             "type":"drag","start_point":[100,240],
             "end_point":[550,300],
             "ordered":True,"coordinate_space":"source_image_pixels"
          }
        }

    def test_native_drag_converter_preserves_original_two_endpoints_and_orientation(self):
        fixture=self.fixture()
        case=h.case_from_new_original(fixture)
        self.assertEqual((case["start"],case["end"]),([100,240],[550,300]))
        self.assertEqual(case["kind"],"original")
        self.assertTrue(case["ordered"])
        self.assertEqual(case["source_zip"],fixture["source_zip"])
        self.assertEqual(case["image_sha256"],fixture["image_sha256"])
        self.assertEqual(case["body"],fixture["intent"])
        self.assertEqual(h.target(case,1024,640)["coordinate_space"],"normalized_1000")
        reversed_case=dict(fixture,native_drag_completion={
            **fixture["native_drag_completion"],"ordered":False})
        with self.assertRaises(ValueError):
            h.case_from_new_original(reversed_case)
        bad=dict(fixture,native_drag_completion={
            **fixture["native_drag_completion"],"end_point":[3000,240]})
        with self.assertRaises(ValueError):
            h.case_from_new_original(bad)

    def test_train_only_and_source_zip_qualification_refuse_modified_labels(self):
        fixture=self.fixture()
        for bad in (
            dict(fixture,split="test"),
            dict(fixture,source_zip_sha256="0"*64),
            dict(fixture,language="ar"),
            dict(fixture,source_domain="sheet"),
            dict(fixture,native_drag_completion={"type":"click",
                "start_point":[100,240],"end_point":[550,300],
                "ordered":True,"coordinate_space":"source_image_pixels"}),
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    h.case_from_new_original(bad)

    def test_owner_original_30000_receipt_pinned_in_private_git(self):
        orig=ROOT/"h418/verified_receipt/h418-owner-original-30k-source/receipt.json"
        sidecar=orig.with_suffix(".sha256")
        self.assertTrue(orig.is_file())
        raw=orig.read_bytes()
        self.assertEqual(sha256(raw).hexdigest(),h.H418_RECEIPT_SHA)
        self.assertEqual(sidecar.read_text("utf8").strip(),h.H418_RECEIPT_SHA)
        state=json.loads(raw)
        self.assertEqual(state["selected_unique_original_train_images"],30000)
        self.assertTrue(state["h417_dev144_image_sha_disjoint"])
        self.assertTrue(state["h417_train_image_sha_disjoint"])
        self.assertTrue(state["actual_all_30000_original_jpeg_bytes_sha_verified"])
        self.assertFalse(state["model_training_executed"])
        self.assertFalse(state["model_release_qualified"])

    def test_owner_real_private_source_and_parent_optional(self):
        root=os.environ.get("H419_OWNER_SOURCE_ROOT")
        if not root:
            self.skipTest("Owner private raw datasets not mounted on generic GitHub runner")
        root=Path(root)
        roots={
            name:root/"h415"/name
            for name,_ in h.SOURCE_SHAS.values()
        }
        roots.update({name:root/"h418"/name for name in h.H419_EXTRA_SHAS})
        roots.update({
          h.BIL_FILE:root/"h415"/h.BIL_FILE,
          h.CURRIC_FILE:root/"h415"/h.CURRIC_FILE,
          h.H418_ROWS:root/"new-30k"/h.H418_ROWS,
          h.H418_RECEIPT_FILE:root/"new-30k"/h.H418_RECEIPT_FILE,
          h.PARENT_ADAPTER_FILE:root/"parent"/h.PARENT_ADAPTER_FILE,
          h.PARENT_RECEIPT_FILE:root/"parent"/h.PARENT_RECEIPT_FILE,
          "h417_original_receipt.sha256":root/"parent"/"h417_original_receipt.sha256",
        })
        missing=[name for name,p in roots.items() if not p.is_file()]
        self.assertEqual(missing,[],"Owner private source input files missing")
        with patch.object(h,"only_input_file",side_effect=lambda name:roots[name]):
            adapter=h.parent_model_source()
            self.assertEqual(h.sha_file(adapter),h.PARENT_ADAPTER_SHA)
            zips,extra,train,dev,order,old,new=h.h419_prepare_sources()
            try:
                self.assertEqual((len(train),len(dev),len(order)),(30976,240,37808))
                self.assertEqual((len(old),len(new)),(144,96))
                self.assertEqual(sum(r["language"]=="ar" for r in train),976)
                self.assertEqual(sum(r["language"]=="en" for r in train),30000)
                self.assertEqual(len({r["image_sha256"] for r in train}),30976)
                self.assertFalse({r["image_sha256"] for r in train}&
                                 {r["image_sha256"] for r in dev})
                for case in (train[0],train[12000],dev[0],dev[-1]):
                    image,width,height=h.source_image(case,zips)
                    self.assertGreater(width,100)
                    self.assertGreater(height,100)
                    self.assertTrue(image.width<=640)
                    self.assertTrue(image.height<=400)
            finally:
                for z in extra:z.close()
                for value in zips.values():
                    if isinstance(value,h.zipfile.ZipFile):value.close()


if __name__=="__main__":
    unittest.main()
