"""H416 expanded native training is not an unsafe automatic promotion."""
from collections import Counter
from hashlib import sha256
import inspect
import json
from pathlib import Path
import unittest

from h416.kaggle import h416_absence_repair_lora as c

ROOT=Path(__file__).resolve().parents[1]

class AbsenceRepairPreflight(unittest.TestCase):
    def test_frozen_training_source_count(self):
        self.assertEqual(c.TRAIN_NEW,8192)
        self.assertEqual(c.TRAIN_ORIGINAL,672)
        self.assertEqual(c.STEPS,2216)
        self.assertEqual(c.STEPS*c.MICROBATCHES,8864)
        self.assertEqual(c.DEV_N,c.AUDIT_N)
        self.assertEqual(c.AUDIT_N,96)

    def test_original_failed_safety_predecessor_is_locked(self):
        r=(ROOT/"h414/verified_receipt/h414-private-thousands-multisource-lora/receipt.json").read_bytes()
        self.assertEqual(sha256(r).hexdigest(),c.PARENT_RECEIPT_SHA)
        state=json.loads(r)
        self.assertEqual(state["adapter_sha256"],c.PARENT_SHA)
        self.assertEqual(state["status"],"COMPLETE_PRIVATE_4768_EXAMPLE_MULTISOURCE_LORA")
        self.assertIs(state["toy_safety_gate"],False)
        self.assertEqual(state["adapted_audit"]["absent"]["unsafe_clicks"],2)
        impl=inspect.getsource(c.private_parent)
        self.assertIn('r.get("toy_safety_gate") is not False',impl)

    def test_absence_heavy_train_and_untouched_eval_strata(self):
        counts=Counter()
        for split,n in (("train",8192),("dev",96),("audit",96)):
            for i in range(n):
                family,kind,lang,_=c.classify(i,split)
                counts[(split,family,kind,lang)]+=1
            for family in c.FAMILIES:
                for kind in c.KINDS:
                    for lang in (0,1):
                        want=({"unique":384,"absent":320,"duplicate":320}[kind]
                              if split=="train" else 4)
                        self.assertEqual(counts[(split,family,kind,lang)],want)
        self.assertGreater(counts[("train","file_tree","absent",0)],192)

    def test_original_private_data_sources_are_same_lawful_train_not_eval(self):
        md=json.loads((ROOT/"h416/kaggle/kernel-metadata.json").read_text())
        self.assertIs(md["is_private"],True)
        self.assertEqual(md["machine_shape"],"NvidiaTeslaT4")
        self.assertEqual(md["dataset_sources"],
                         ["abdulazizshehri/hala-gui-training-library-private"])
        self.assertEqual(md["kernel_sources"],
                         ["abdulazizshehri/hala-h414-private-thousands-multisource-lora"])
        for key,(digest,n) in c.ORIGINAL_DATA.items():
            raw=(ROOT/key/"assets/authoring_sft.jsonl").read_bytes()
            self.assertEqual(sha256(raw).hexdigest(),digest)
            self.assertEqual(len(raw.splitlines()),n)

    def test_new_screens_use_new_seeds_and_refuse_unsafe_alternatives(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow optional for no-site CI")
        from h414.kaggle import h414_multisource_lora as parent
        ids=set()
        old={sha256(parent.compose(i,"train")[0].tobytes()).hexdigest()
             for i in (0,63,384,577,1024,2048,3072,4095)}
        for i in (0,63,768,1408,2048,3072,4096,6144,8191):
            im,prompt,gold,family,kind,lang,boxes=c.compose(i,"train")
            digest=sha256(im.tobytes()).hexdigest()
            self.assertNotIn(digest,old)
            self.assertNotIn(digest,ids)
            ids.add(digest)
            self.assertEqual(len(boxes),{"unique":1,"absent":0,"duplicate":2}[kind])
            if kind=="unique":
                self.assertEqual(c.parse(gold)[0],"point")
            else:
                self.assertEqual(gold,'{"abstain":true}')
        self.assertEqual(c.parse('{"abstain":1}')[0],"invalid")

if __name__=="__main__":
    unittest.main()
