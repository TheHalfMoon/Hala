"""H425 real H417-parent replay + original H422 two-instruction training tests.

Synthetic tests are source/curriculum verification ONLY. Real owner preflight
requires private original nine H60 ZIPs, H422 paired source and H421 unseen DEV.
"""
import collections
import hashlib
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile
import importlib.util

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location(
    "h425_standalone_source",ROOT/"h425/kaggle/h425_h417_replay_paired.py")
h=importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)

def fake_rows():
    rows=[]
    for i in range(16000):
        sha=hashlib.sha256(("pair "+str(i)).encode()).hexdigest()
        for j in range(2):
            rows.append({"id":f"p{i}-{j}","kind":"original","language":"en",
              "image_sha256":sha,"body":f"original instruction {i}-{j}",
              "start":[i%500,j],"end":[i%500,j+2],"domain":"slider"})
    for i in range(11952):
        lang="en" if i<10976 else "ar"
        rows.append({"id":f"old{i}","kind":"firstparty" if lang=="ar" else "original",
           "language":lang,
           "image_sha256":hashlib.sha256(("h417 "+str(i)).encode()).hexdigest(),
           "body":f"authentic old instruction {i}","start":[i%200,20],
           "end":[i%200,40],"domain":"slider"})
    return rows

class H425CurriculumEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train=fake_rows()

    def test_model_has_real_original_h417_lineage(self):
        self.assertEqual(h.H425_OLD_TRAIN,11952)
        self.assertEqual(h.H425_NEW_PAIR_ACTIONS,32000)
        self.assertEqual(h.H425_UNIQUE_TRAIN_SCREENS,27952)
        self.assertEqual(h.H425_SAMPLES,50784)
        self.assertEqual(h.H425_STEPS,12696)
        self.assertEqual(h.H425_AR_TOTAL_PASSES,8)
        self.assertEqual(h.PARENT_ADAPTER_SHA,"11f06aac6c225b86d1f30272eb852bca49fd82ce6e8553113b39b4cba4216df1")

    def test_all_50784_real_future_passes_with_same_jpeg_optimizer_groups(self):
        order=h.h425_curriculum(self.train)
        self.assertEqual(len(order),50784)
        by_index=collections.Counter(order)
        self.assertEqual(len(by_index),len(self.train))
        self.assertTrue(all(n==(8 if row["language"]=="ar" else 1)
            for i,row in enumerate(self.train) for n in [by_index[i]]))
        counts=collections.Counter()
        for off in range(0,len(order),4):
            indices=order[off:off+4]
            batch=[self.train[i] for i in indices]
            kinds=set(i<32000 for i in indices)
            self.assertEqual(len(kinds),1)
            if True in kinds:
                counts["paired"]+=1
                self.assertEqual(sorted(collections.Counter(
                    x["image_sha256"] for x in batch).values()),[2,2])
            elif all(row["language"]=="ar" for row in batch):
                counts["ar"]+=1
                self.assertEqual(len({x["image_sha256"] for x in batch}),4)
            else:
                counts["old_en"]+=1
                self.assertEqual(len({x["image_sha256"] for x in batch}),4)
        self.assertEqual(counts,{"paired":8000,"old_en":2744,"ar":1952})

    def test_mutant_duplicate_real_pair_target_rejected(self):
        mutated=list(self.train)
        mutated[1]=dict(mutated[1],body=mutated[0]["body"],
            start=mutated[0]["start"],end=mutated[0]["end"])
        with self.assertRaises(ValueError):
            h.h425_curriculum(mutated)

    def test_mutant_old_train_pixel_overlap_rejected(self):
        mutated=list(self.train)
        mutated[32000]=dict(mutated[32000],image_sha256=mutated[0]["image_sha256"])
        with self.assertRaises(ValueError):
            h.h425_curriculum(mutated)

    def test_mutant_wrong_old_arabic_provenance_rejected(self):
        mutated=list(self.train)
        mutated[-1]=dict(mutated[-1],kind="original")
        with self.assertRaises(ValueError):
            h.h425_curriculum(mutated)

    def test_standalone_trains_only_real_lora_from_original_parent(self):
        script=(ROOT/"h425/kaggle/h425_h417_replay_paired.py").read_text("utf8")
        self.assertIn("PeftModel.from_pretrained(pretrained,str(original_parent.parent)",script)
        self.assertIn("gradient_checkpointing_enable()",script)
        self.assertIn("coordinate_weighted_loss(output,labels,processor)",script)
        self.assertIn("actual_optimizer_steps",script)
        self.assertIn("h425_after_dev",script)
        self.assertIn("source_only_H425_stronger_improvement_gate",script)
        # H419 final model is NOT an allowed trained parent.
        self.assertNotIn("parent,parent_receipt_sha,weights,config=real_h419_parent()",script)

    def test_owner_original_all_paired_and_h417_train_images_source_preflight(self):
        source=os.environ.get("H425_OWNER_SOURCE_ROOT")
        if not source:
            self.skipTest("Private owner 9 H60 ZIPs inaccessible in CI")
        root=Path(source)
        paired=Path(os.environ["H425_OWNER_PAIRED_ROOT"])
        h421=Path(os.environ["H425_OWNER_H421_ROOT"])
        inputs={
          **{name:root/"h415"/name for name,_ in h.SOURCE_SHAS.values()},
          **{name:root/"h418"/name for name in h.H419_EXTRA_SHAS},
          h.BIL_FILE:root/"h415"/h.BIL_FILE,
          h.CURRIC_FILE:root/"h415"/h.CURRIC_FILE,
          h.H418_ROWS:root/"new-30k"/h.H418_ROWS,
          h.H418_RECEIPT_FILE:root/"new-30k"/h.H418_RECEIPT_FILE,
          h.H422_ROWS_FILE:paired/h.H422_ROWS_FILE,
          h.H422_RECEIPT_FILE:paired/h.H422_RECEIPT_FILE,
          h.H421_DEV_FILE:h421/h.H421_DEV_FILE}
        self.assertFalse([key for key,p in inputs.items() if not p.is_file()])
        with patch.object(h,"only_input_file",side_effect=lambda name:inputs[name]):
            zips,extra,train,dev,order=h.h425_prepare_source()
            try:
                self.assertEqual((len(train),len(dev),len(order)),(43952,240,50784))
                self.assertEqual(len({r["image_sha256"] for r in train}),27952)
                self.assertFalse({r["image_sha256"] for r in train}&
                                 {r["image_sha256"] for r in dev})
                # Explicitly reopen EVERY one of the 27,952 original distinct
                # image BYTES and verify SHA against the authenticated source,
                # including 11,952 original H417 replay images.
                seen=set()
                count=collections.Counter()
                for row in train+dev:
                    digest=row["image_sha256"]
                    if digest in seen:continue
                    seen.add(digest)
                    media=zips["firstparty" if row["kind"]=="firstparty" else row["domain"]]
                    payload=media.read(row["image_path"])
                    self.assertEqual(hashlib.sha256(payload).hexdigest(),digest)
                    count[row["kind"]]+=1
                self.assertEqual(len(seen),28192)
                self.assertEqual(count["firstparty"],2000)
                self.assertEqual(count["original"],26192)
            finally:
                for f in extra:f.close()
                for z in zips.values():
                    if isinstance(z,zipfile.ZipFile):z.close()
