"""H422 genuine 16K same-JPEG two-prompt original native DRAG source checks."""
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest

from h422 import paired_source as p
from h418 import curate as h418


def row(sha,domain,index,action):
    return {
      "id":f"original-{domain}-{index}-{action}",
      "schema":p.SCHEMA,"pair_id":sha,"alternative_index":action,
      "split":"train","language":"en",
      "source_domain":domain,"source_zip":"x.hala.bin",
      "source_zip_sha256":"f"*64,"image_path":f"images/{domain}/{index}.jpg",
      "image_sha256":sha,"image_width":1024,"image_height":640,
      "instruction":f"Drag the fictional {domain} handle to {action}",
      "native_drag_completion":{"type":"drag","start_point":[100,200],
                                "end_point":[200+400*action,300],
                                "ordered":True,
                                "coordinate_space":"source_image_pixels"},
      "original_source_task_sha256":"b"*64,
      "original_source_JPEG_bytes_SHA_verified":True,
      "H421_unseen_source_image_excluded":True,
      "H417_H419_dev_source_image_task_excluded":True,
    }


class H422SourcePairs(unittest.TestCase):
    def test_budget_genuinely_16k_images_with_two_actions_each(self):
        self.assertEqual(p.ORIGINAL_PAIR_IMAGES,16000)
        self.assertEqual(p.ORIGINAL_TRAIN_ACTIONS,32000)
        self.assertEqual(p.PER_DOMAIN,{"slider":5333,"slide_resize":5333,"text_highlight":5334})
        self.assertEqual(sum(p.PER_DOMAIN.values()),16000)
        self.assertEqual(p.H420_SECOND_ROWS_SHA,"58c88bf09c047973675ca6dd906eae8d6832005f5da338d492a67c8dd35b91ed")
        self.assertEqual(p.H421_UNSEEN_ROWS_SHA,"42d93219ec4083920462d4194b175154f1f12d9b3e7b36ffd1068d8f8a4d96b9")

    def test_only_actual_native_ordered_two_points_admitted(self):
        good=row("a"*64,"slider",1,0)["native_drag_completion"]
        self.assertEqual(p.native_signature(good),((100,200),(200,300),True))
        for bad in ({**good,"type":"click"},
                    {**good,"start_point":[-1,200]},
                    {**good,"end_point":["50",300]},
                    {**good,"ordered":"true"},
                    {**good,"coordinate_space":"normalized_1000"}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    p.native_signature(bad)

    def test_forged_32k_receipt_rejected_on_tiny_sample(self):
        with tempfile.TemporaryDirectory() as temp:
            fake=[row("a"*64,"slider",1,0),row("a"*64,"slider",1,1)]
            with self.assertRaisesRegex(ValueError,"32K native action proof incomplete"):
                p.export(fake,temp,2)
            self.assertEqual(list(Path(temp).iterdir()),[])

    def test_full_fake_16k_pairs_refuse_same_instruction_and_duplicate_target(self):
        # Source-untrusted fixture to verify fail-closed counts before export,
        # NOT evidence any actual media verified.
        pairs=[]
        for domain,quota in p.PER_DOMAIN.items():
            for i in range(quota):
                h=sha256(f"{domain}/{i}".encode()).hexdigest()
                pairs.extend((row(h,domain,i,0),row(h,domain,i,1)))
        self.assertEqual(len(pairs),32000)
        with tempfile.TemporaryDirectory() as temp:
            pairs[1]["instruction"]=pairs[0]["instruction"]
            with self.assertRaisesRegex(ValueError,"not instruction/target paired"):
                p.export(pairs,temp,32000)
            pairs[1]["instruction"]="Drag to DIFFERENT target"
            pairs[1]["native_drag_completion"]=pairs[0]["native_drag_completion"]
            with self.assertRaisesRegex(ValueError,"not instruction/target paired"):
                p.export(pairs,temp,32000)

    def test_private_source_inputs_not_present_in_generic_ci(self):
        root=Path(__file__).resolve().parents[1]/"h420/verified_receipt"
        self.assertTrue((root/"h420-owner-authentic-second-drags/receipt.json").is_file())
        self.assertTrue((root/"h421-original-counterfactual-dev/receipt.json").is_file())
        original=(root/"h420-owner-authentic-second-drags/receipt.json").read_bytes()
        self.assertEqual(sha256(original).hexdigest(),p.H420_SECOND_RECEIPT_SHA)

if __name__=="__main__":
    unittest.main()
